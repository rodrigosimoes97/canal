"""
Efeito DSP: Reverb Algorítmico Schroeder/Freeverb com Delay Lines Persistentes, Difusão Densa e Amortecimento Escuro.
"""
from typing import Any, Dict, List, Optional
import numpy as np
from scipy import signal

from plugins.base import BasePlugin, PluginCategory


# Tempos de atraso base em amostras (para 44.1 kHz) mutualmente primos
COMB_TUNINGS_L = [1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617]
COMB_TUNINGS_R = [1139, 1211, 1300, 1379, 1445, 1514, 1580, 1640]
ALLPASS_TUNINGS_L = [556, 441, 341, 225]
ALLPASS_TUNINGS_R = [579, 464, 364, 248]


class CombFilter:
    """Filtro Comb com feedback e amortecimento (damping)."""
    def __init__(self, size: int) -> None:
        self.size = size
        self.buffer = np.zeros(size, dtype=np.float32)
        self.pos = 0
        self.filter_store = 0.0

    def process(self, input_val: float, feedback: float, damp: float) -> float:
        output = self.buffer[self.pos]
        self.filter_store = output * (1.0 - damp) + self.filter_store * damp
        self.buffer[self.pos] = input_val + self.filter_store * feedback
        self.pos = (self.pos + 1) % self.size
        return output

    def reset(self) -> None:
        self.buffer.fill(0.0)
        self.pos = 0
        self.filter_store = 0.0


class AllpassFilter:
    """Filtro All-pass para difusão espacial densa."""
    def __init__(self, size: int) -> None:
        self.size = size
        self.buffer = np.zeros(size, dtype=np.float32)
        self.pos = 0
        self.feedback = 0.5

    def process(self, input_val: float) -> float:
        buf_out = self.buffer[self.pos]
        out = -input_val + buf_out
        self.buffer[self.pos] = input_val + (buf_out * self.feedback)
        self.pos = (self.pos + 1) % self.size
        return out

    def reset(self) -> None:
        self.buffer.fill(0.0)
        self.pos = 0


class ReverbEffect(BasePlugin):
    """
    Simulador de reverberação algorítmica etérea com difusão estéreo descorrelacionada,
    amortecimento escuro passa-baixa contínuo e preservação estrita de caudas entre chunks contíguos.
    """
    PLUGIN_ID = "reverb_effect"

    def __init__(self, plugin_id: str = "reverb_effect", config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(plugin_id=plugin_id, category=PluginCategory.EFFECT, config=config)
        self.room_size = float(np.clip(self.config.get("room_size", 0.8), 0.1, 0.98))
        self.damping = float(np.clip(self.config.get("damping", 0.4), 0.0, 0.95))
        self.wet_level = float(np.clip(self.config.get("wet_level", 0.35), 0.0, 1.0))
        self.dry_level = float(np.clip(self.config.get("dry_level", 1.0), 0.0, 1.0))
        self.width = float(np.clip(self.config.get("width", 0.8), 0.0, 1.0))
        self.lowpass_cutoff = float(self.config.get("lowpass_cutoff", 3200.0))
        self.sample_rate: int = 44100

        self._combs_l: List[CombFilter] = []
        self._combs_r: List[CombFilter] = []
        self._allpasses_l: List[AllpassFilter] = []
        self._allpasses_r: List[AllpassFilter] = []

        # Filtro passa-baixa analógico para névoa acústica escura
        self._sos = signal.butter(2, min(self.lowpass_cutoff, 18000.0), btype="lowpass", fs=44100, output="sos")
        self._zi_l = signal.sosfilt_zi(self._sos)
        self._zi_r = signal.sosfilt_zi(self._sos)

    @property
    def has_state(self) -> bool:
        return True

    def prepare(self, context: Any, rng: np.random.Generator) -> None:
        self.sample_rate = context.sample_rate
        scale_ratio = self.sample_rate / 44100.0

        self._combs_l = [CombFilter(int(t * scale_ratio)) for t in COMB_TUNINGS_L]
        self._combs_r = [CombFilter(int(t * scale_ratio)) for t in COMB_TUNINGS_R]
        self._allpasses_l = [AllpassFilter(int(t * scale_ratio)) for t in ALLPASS_TUNINGS_L]
        self._allpasses_r = [AllpassFilter(int(t * scale_ratio)) for t in ALLPASS_TUNINGS_R]

        cutoff = min(self.lowpass_cutoff, self.sample_rate * 0.45)
        self._sos = signal.butter(2, cutoff, btype="lowpass", fs=self.sample_rate, output="sos")
        self._zi_l = signal.sosfilt_zi(self._sos)
        self._zi_r = signal.sosfilt_zi(self._sos)

        self._is_prepared = True

    def process(self, audio: np.ndarray) -> np.ndarray:
        if not self.is_active or not self._combs_l or audio.shape[0] == 0:
            return audio

        num_samples = audio.shape[0]
        in_l = audio[:, 0].astype(np.float32)
        in_r = audio[:, 1].astype(np.float32)

        wet_raw_l = np.zeros(num_samples, dtype=np.float32)
        wet_raw_r = np.zeros(num_samples, dtype=np.float32)

        # Ganhos do modelo Freeverb
        fb = self.room_size
        damp = self.damping
        gain_scale = 0.015

        for i in range(num_samples):
            # Sinal mono de entrada somado
            input_mixed = (in_l[i] + in_r[i]) * gain_scale

            # 1. Banco de filtros Comb em paralelo
            comb_l_out = sum(cf.process(input_mixed, fb, damp) for cf in self._combs_l)
            comb_r_out = sum(cf.process(input_mixed, fb, damp) for cf in self._combs_r)

            # 2. Filtros All-pass em cascata
            for ap in self._allpasses_l:
                comb_l_out = ap.process(comb_l_out)
            for ap in self._allpasses_r:
                comb_r_out = ap.process(comb_r_out)

            # 3. Matriz de abertura estéreo (wet1 / wet2)
            wet_1 = self.wet_level * (1.0 + self.width) / 2.0
            wet_2 = self.wet_level * (1.0 - self.width) / 2.0

            wet_raw_l[i] = comb_l_out * wet_1 + comb_r_out * wet_2
            wet_raw_r[i] = comb_r_out * wet_1 + comb_l_out * wet_2

        # 4. Amortecimento escuro suave via passa-baixa contínuo para calor acústico
        wet_filt_l, self._zi_l = signal.sosfilt(self._sos, wet_raw_l, zi=self._zi_l)
        wet_filt_r, self._zi_r = signal.sosfilt(self._sos, wet_raw_r, zi=self._zi_r)

        out_l = in_l * self.dry_level + wet_filt_l
        out_r = in_r * self.dry_level + wet_filt_r

        return np.column_stack((out_l, out_r))

    def render(self, start_sample: int, num_samples: int) -> np.ndarray:
        return np.zeros((num_samples, 2), dtype=np.float32)

    def reset(self) -> None:
        for c in self._combs_l: c.reset()
        for c in self._combs_r: c.reset()
        for a in self._allpasses_l: a.reset()
        for a in self._allpasses_r: a.reset()
        self._zi_l = signal.sosfilt_zi(self._sos)
        self._zi_r = signal.sosfilt_zi(self._sos)
