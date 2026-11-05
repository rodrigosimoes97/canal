"""
Mixer: Canal de Áudio com Fader de Volume, Pan, Inserção de Efeitos, Sends Auxiliares e Automação.
"""
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from plugins.base import BasePlugin


@dataclass
class AutomationPoint:
    time_sec: float
    value: float


class AutomationCurve:
    """Curva de automação linear contínua para parâmetros no tempo."""
    def __init__(self, default_value: float) -> None:
        self.default_value = float(default_value)
        self.points: List[AutomationPoint] = []

    def add_point(self, time_sec: float, value: float) -> None:
        self.points.append(AutomationPoint(time_sec=float(time_sec), value=float(value)))
        self.points.sort(key=lambda p: p.time_sec)

    def evaluate(self, time_sec: float) -> float:
        if not self.points:
            return self.default_value
        if time_sec <= self.points[0].time_sec:
            return self.points[0].value
        if time_sec >= self.points[-1].time_sec:
            return self.points[-1].value

        # Busca binária/linear do segmento
        for i in range(len(self.points) - 1):
            p1 = self.points[i]
            p2 = self.points[i + 1]
            if p1.time_sec <= time_sec <= p2.time_sec:
                span = p2.time_sec - p1.time_sec
                if span <= 1e-6:
                    return p1.value
                ratio = (time_sec - p1.time_sec) / span
                return float(p1.value + ratio * (p2.value - p1.value))
        return self.default_value

    def evaluate_array(self, time_array: np.ndarray) -> np.ndarray:
        """Gera um array contínuo de valores interpolados para cada amostra do chunk."""
        if not self.points:
            return np.full_like(time_array, self.default_value, dtype=np.float32)
        return np.interp(
            time_array,
            [p.time_sec for p in self.points],
            [p.value for p in self.points],
            left=self.points[0].value,
            right=self.points[-1].value
        ).astype(np.float32)


class MixerChannel:
    """
    Representa uma faixa individual na mesa de mixagem com cadeia de processamento própria.
    """

    def __init__(
        self,
        channel_id: str,
        volume: float = 1.0,
        pan: float = 0.0,
        mute: bool = False,
        solo: bool = False
    ) -> None:
        self.channel_id = channel_id
        self.volume = float(volume)
        self.pan = float(np.clip(pan, -1.0, 1.0))
        self.mute = mute
        self.solo = solo

        self.inserts: List[BasePlugin] = []
        self.sends: Dict[str, float] = {} # ex: {"reverb": 0.3}

        # Automações contínuas
        self.volume_auto = AutomationCurve(self.volume)
        self.pan_auto = AutomationCurve(self.pan)

    def add_insert(self, effect: BasePlugin) -> None:
        """Adiciona um efeito DSP em série no canal."""
        self.inserts.append(effect)

    def set_send(self, bus_name: str, level: float) -> None:
        """Define o ganho de envio auxiliar para um barramento (ex: 'reverb', 'delay')."""
        self.sends[bus_name] = float(np.clip(level, 0.0, 1.0))

    def process(
        self,
        audio: np.ndarray,
        start_sample: int,
        sample_rate: int
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Processa o sinal de áudio do canal:
          1. Aplica efeitos inseridos em série.
          2. Extrai sinais de send auxiliar (pre ou post fader).
          3. Aplica fader de volume, curva de automação e equal-power pan.
          
        Returns:
            Tuple (direct_stereo_out, sends_dict)
        """
        if self.mute or audio.shape[0] == 0:
            zeros = np.zeros_like(audio)
            sends_zero = {bus: np.zeros_like(audio) for bus in self.sends}
            return zeros, sends_zero

        num_samples = audio.shape[0]
        cur_signal = np.copy(audio)

        # 1. Cadeia de inserts em série
        for insert in self.inserts:
            if hasattr(insert, "process"):
                cur_signal = insert.process(cur_signal)

        # 2. Gera envio para sends (post-insert)
        sends_out = {}
        for bus, level in self.sends.items():
            sends_out[bus] = (cur_signal * level).astype(np.float32)

        # 3. Automação e Fader
        t = (np.arange(num_samples, dtype=np.float32) + start_sample) / sample_rate
        vol_curve = self.volume_auto.evaluate_array(t)
        pan_curve = self.pan_auto.evaluate_array(t)

        # Equal-power panning contínuo
        theta = (pan_curve + 1.0) * (np.pi / 4.0)
        gain_l = (np.cos(theta) * 1.41421356 * vol_curve)[:, None]
        gain_r = (np.sin(theta) * 1.41421356 * vol_curve)[:, None]

        out_l = cur_signal[:, 0:1] * gain_l
        out_r = cur_signal[:, 1:2] * gain_r
        direct_out = np.column_stack((out_l[:, 0], out_r[:, 0])).astype(np.float32)

        return direct_out, sends_out

    def reset(self) -> None:
        for ins in self.inserts:
            ins.reset()
