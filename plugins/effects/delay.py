"""
Efeito DSP: Delay Estéreo e Ping-Pong com Feedback Filtrado e Buffer Circular Persistente.
"""
from typing import Any, Dict, Optional
import numpy as np

from plugins.base import BasePlugin, PluginCategory


class DelayEffect(BasePlugin):
    """
    Linha de atraso estéreo e ping-pong com cauda contínua entre chunks.
    Inclui damping passa-baixas no feedback para simular o calor analógico de eco de fita.
    """
    PLUGIN_ID = "delay_effect"

    def __init__(self, plugin_id: str = "delay_effect", config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(plugin_id=plugin_id, category=PluginCategory.EFFECT, config=config)
        self.mode = str(self.config.get("mode", "ping_pong")).lower().strip() # "stereo" ou "ping_pong"
        self.delay_time_ms_l = float(self.config.get("delay_ms_l", self.config.get("delay_ms", 375.0)))
        self.delay_time_ms_r = float(self.config.get("delay_ms_r", self.config.get("delay_ms", 500.0)))
        self.feedback = float(np.clip(self.config.get("feedback", 0.45), 0.0, 0.95))
        self.mix = float(np.clip(self.config.get("mix", 0.35), 0.0, 1.0))
        self.damping = float(np.clip(self.config.get("damping", 0.25), 0.0, 0.9)) # Atenuação de agudos no feedback
        self.sample_rate: int = 44100

        # Buffers circulares de atraso
        self._max_delay_samples = 0
        self._buffer_l: Optional[np.ndarray] = None
        self._buffer_r: Optional[np.ndarray] = None
        self._write_pos = 0

        # Estado do filtro de damping (1 polo passa-baixas no feedback)
        self._damp_state_l = 0.0
        self._damp_state_r = 0.0

    @property
    def has_state(self) -> bool:
        return True

    def prepare(self, context: Any, rng: np.random.Generator) -> None:
        self.sample_rate = context.sample_rate
        # Buffer com capacidade para até 5 segundos de atraso
        self._max_delay_samples = int(5.0 * self.sample_rate)
        self._buffer_l = np.zeros(self._max_delay_samples, dtype=np.float32)
        self._buffer_r = np.zeros(self._max_delay_samples, dtype=np.float32)
        self._write_pos = 0
        self._damp_state_l = 0.0
        self._damp_state_r = 0.0
        self._is_prepared = True

    def process(self, audio: np.ndarray) -> np.ndarray:
        """
        Processa o áudio aplicando delay e realimentação com preservação de cauda.
        """
        if not self.is_active or self._buffer_l is None or audio.shape[0] == 0:
            return audio

        num_samples = audio.shape[0]
        delay_samples_l = int(round((self.delay_time_ms_l / 1000.0) * self.sample_rate))
        delay_samples_r = int(round((self.delay_time_ms_r / 1000.0) * self.sample_rate))

        delay_samples_l = np.clip(delay_samples_l, 1, self._max_delay_samples - 1)
        delay_samples_r = np.clip(delay_samples_r, 1, self._max_delay_samples - 1)

        dry_l = audio[:, 0].astype(np.float32)
        dry_r = audio[:, 1].astype(np.float32)

        wet_l = np.zeros(num_samples, dtype=np.float32)
        wet_r = np.zeros(num_samples, dtype=np.float32)

        buf_l = self._buffer_l
        buf_r = self._buffer_r
        buf_len = self._max_delay_samples
        w_pos = self._write_pos
        fb = self.feedback
        damp = self.damping
        is_ping_pong = (self.mode == "ping_pong")

        damp_l = self._damp_state_l
        damp_r = self._damp_state_r

        # Processamento sample-a-sample para realimentação exata
        for n in range(num_samples):
            # Leitura do atraso
            read_pos_l = (w_pos - delay_samples_l) % buf_len
            read_pos_r = (w_pos - delay_samples_r) % buf_len

            delayed_l = buf_l[read_pos_l]
            delayed_r = buf_r[read_pos_r]

            wet_l[n] = delayed_l
            wet_r[n] = delayed_r

            # Filtro de damping de 1 polo no feedback
            damp_l = (1.0 - damp) * delayed_l + damp * damp_l
            damp_r = (1.0 - damp) * delayed_r + damp * damp_r

            # Escrita no buffer circular
            if is_ping_pong:
                # Ping-Pong: o eco esquerdo alimenta a direita e vice-versa
                buf_l[w_pos] = dry_l[n] + damp_r * fb
                buf_r[w_pos] = dry_r[n] + damp_l * fb
            else:
                buf_l[w_pos] = dry_l[n] + damp_l * fb
                buf_r[w_pos] = dry_r[n] + damp_r * fb

            w_pos = (w_pos + 1) % buf_len

        self._write_pos = w_pos
        self._damp_state_l = damp_l
        self._damp_state_r = damp_r

        out_l = dry_l * (1.0 - self.mix) + wet_l * self.mix
        out_r = dry_r * (1.0 - self.mix) + wet_r * self.mix

        return np.column_stack((out_l, out_r))

    def render(self, start_sample: int, num_samples: int) -> np.ndarray:
        return np.zeros((num_samples, 2), dtype=np.float32)

    def reset(self) -> None:
        if self._buffer_l is not None:
            self._buffer_l.fill(0.0)
            self._buffer_r.fill(0.0)
            self._write_pos = 0
            self._damp_state_l = 0.0
            self._damp_state_r = 0.0
