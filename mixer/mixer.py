"""
Mixer: Barramento Principal, Roteamento Auxiliar, Sidechain Ducking e Soma de Alta Precisão.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from plugins.base import BasePlugin
from mixer.channel import MixerChannel


@dataclass
class DuckingRule:
    """Regra de Sidechain Ducking conectando um canal de controle a um canal alvo."""
    source_channel: str
    target_channel: str
    threshold: float = 0.15
    max_attenuation_db: float = 6.0 # Redução em dB
    attack_sec: float = 0.05
    release_sec: float = 0.35
    _envelope_state: float = 0.0


class Mixer:
    """
    Console de Mixagem profissional do Relaxation Audio Engine.
    
    Recursos:
      - Roteamento multi-canal com solo/mute logic.
      - Buses auxiliares compartilhados para efeitos espaciais (Reverb, Delay).
      - Sidechain Ducking com envelope detector suave e sem artefatos.
      - Barramento master com soma de 64 bits para headroom transparente.
    """

    def __init__(self, master_gain: float = 1.0) -> None:
        self.master_gain = float(master_gain)
        self.channels: Dict[str, MixerChannel] = {}
        self.aux_buses: Dict[str, List[BasePlugin]] = {}
        self.ducking_rules: List[DuckingRule] = []
        self.sample_rate: int = 44100

    def add_channel(self, channel: MixerChannel) -> None:
        """Registra um novo canal na mesa."""
        self.channels[channel.channel_id] = channel

    def get_channel(self, channel_id: str) -> Optional[MixerChannel]:
        """Obtém um canal existente."""
        return self.channels.get(channel_id)

    def add_aux_bus(self, bus_name: str, effects: List[BasePlugin]) -> None:
        """Cria um barramento auxiliar (ex: 'reverb_bus', 'delay_bus')."""
        self.aux_buses[bus_name] = effects

    def add_ducking_rule(
        self,
        source_channel: str,
        target_channel: str,
        threshold: float = 0.15,
        max_attenuation_db: float = 6.0,
        attack_sec: float = 0.05,
        release_sec: float = 0.35
    ) -> None:
        """Configura um sidechain ducking entre dois canais."""
        self.ducking_rules.append(DuckingRule(
            source_channel=source_channel,
            target_channel=target_channel,
            threshold=threshold,
            max_attenuation_db=max_attenuation_db,
            attack_sec=attack_sec,
            release_sec=release_sec
        ))

    def prepare(self, context: Any, rng: np.random.Generator) -> None:
        self.sample_rate = context.sample_rate
        # Prepara todos os efeitos nos canais e nos buses
        for ch in self.channels.values():
            for insert in ch.inserts:
                insert.prepare(context, rng)

        for bus_effects in self.aux_buses.values():
            for eff in bus_effects:
                eff.prepare(context, rng)

    def _apply_ducking(
        self,
        channel_buffers: Dict[str, np.ndarray],
        num_samples: int
    ) -> None:
        """Aplica sidechain ducking dinâmico entre canais interconectados."""
        for rule in self.ducking_rules:
            src = channel_buffers.get(rule.source_channel)
            tgt = channel_buffers.get(rule.target_channel)

            if src is None or tgt is None or src.shape[0] == 0:
                continue

            # Detecta o envelope do sinal de controle (RMS / Peak)
            src_mono = np.max(np.abs(src), axis=1)

            # Coeficientes de ataque e release
            alpha_att = float(np.exp(-1.0 / max(1e-4, rule.attack_sec * self.sample_rate)))
            alpha_rel = float(np.exp(-1.0 / max(1e-4, rule.release_sec * self.sample_rate)))

            env = np.empty(num_samples, dtype=np.float32)
            cur_env = rule._envelope_state

            for i in range(num_samples):
                val = src_mono[i]
                if val > cur_env:
                    cur_env = alpha_att * cur_env + (1.0 - alpha_att) * val
                else:
                    cur_env = alpha_rel * cur_env + (1.0 - alpha_rel) * val
                env[i] = cur_env

            rule._envelope_state = cur_env

            # Calcula curva de atenuação (ganho linear de 1.0 down to linear_att)
            max_att_linear = 10.0 ** (-rule.max_attenuation_db / 20.0)
            overshoot = np.maximum(0.0, env - rule.threshold)
            gain_reduction = np.clip(1.0 - (overshoot * 2.0), max_att_linear, 1.0)[:, None]

            # Aplica ducking no canal alvo
            tgt *= gain_reduction

    def mix_chunk(
        self,
        raw_inputs: Dict[str, np.ndarray],
        start_sample: int,
        num_samples: int
    ) -> np.ndarray:
        """
        Mixagem completa de um bloco amostral:
          1. Aplica mute/solo lógica.
          2. Processa cada canal (inserts, automações, pan).
          3. Aplica Sidechain Ducking.
          4. Roteia sends para os buses auxiliares de reverb e delay.
          5. Soma o barramento master com headroom de 64 bits.
        """
        # Verifica se algum canal está com solo ativo
        has_solo = any(ch.solo for ch in self.channels.values())

        channel_direct_outs: Dict[str, np.ndarray] = {}
        bus_inputs: Dict[str, np.ndarray] = {
            b_name: np.zeros((num_samples, 2), dtype=np.float32) for b_name in self.aux_buses
        }

        # 1. Processa canais individuais
        for ch_id, ch in self.channels.items():
            audio_in = raw_inputs.get(ch_id)
            if audio_in is None:
                audio_in = np.zeros((num_samples, 2), dtype=np.float32)

            if has_solo and not ch.solo:
                continue

            direct, sends = ch.process(audio_in, start_sample, self.sample_rate)
            channel_direct_outs[ch_id] = direct

            for bus_name, send_sig in sends.items():
                if bus_name in bus_inputs:
                    bus_inputs[bus_name] += send_sig

        # 2. Sidechain Ducking entre canais
        self._apply_ducking(channel_direct_outs, num_samples)

        # 3. Processa barramentos auxiliares (Reverb, Delay)
        bus_outputs: List[np.ndarray] = []
        for bus_name, effects in self.aux_buses.items():
            bus_signal = bus_inputs[bus_name]
            for eff in effects:
                if hasattr(eff, "process"):
                    bus_signal = eff.process(bus_signal)
            bus_outputs.append(bus_signal)

        # 4. Soma no Master Bus com precisão float64
        master_bus = np.zeros((num_samples, 2), dtype=np.float64)

        for direct in channel_direct_outs.values():
            master_bus += direct.astype(np.float64)

        for aux_out in bus_outputs:
            master_bus += aux_out.astype(np.float64)

        # Aplica fader master
        master_bus *= self.master_gain

        return master_bus.astype(np.float32)

    def reset(self) -> None:
        for ch in self.channels.values():
            ch.reset()
        for effects in self.aux_buses.values():
            for eff in effects:
                eff.reset()
        for rule in self.ducking_rules:
            rule._envelope_state = 0.0
