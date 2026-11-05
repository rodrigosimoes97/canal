"""
Efeito DSP: Saturação Analógica Suave (Calor Harmônico, Soft-Clipping e DC Blocker).
"""
from typing import Any, Dict, Optional
import numpy as np
from scipy.signal import butter, sosfilt

from plugins.base import BasePlugin, PluginCategory


class SaturationEffect(BasePlugin):
    """
    Saturador não-linear baseado em tangente hiperbólica com compensação de ganho
    e filtro DC blocker para calor analógico sem aspereza digital.
    """
    PLUGIN_ID = "saturation_effect"

    def __init__(self, plugin_id: str = "saturation_effect", config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(plugin_id=plugin_id, category=PluginCategory.EFFECT, config=config)
        self.drive = float(np.clip(self.config.get("drive", 1.2), 1.0, 5.0))
        self.mix = float(np.clip(self.config.get("mix", 1.0), 0.0, 1.0))
        self.sample_rate: int = 44100

        self._hp_sos = None
        self._hp_zi_l = None
        self._hp_zi_r = None

    @property
    def has_state(self) -> bool:
        return True

    def prepare(self, context: Any, rng: np.random.Generator) -> None:
        self.sample_rate = context.sample_rate
        # DC Blocker a 15 Hz
        self._hp_sos = butter(2, 15.0 / (self.sample_rate / 2.0), btype="high", output="sos")
        self._hp_zi_l = np.zeros((self._hp_sos.shape[0], 2), dtype=np.float64)
        self._hp_zi_r = np.zeros((self._hp_sos.shape[0], 2), dtype=np.float64)
        self._is_prepared = True

    def process(self, audio: np.ndarray) -> np.ndarray:
        if not self.is_active or audio.shape[0] == 0:
            return audio

        in_l = audio[:, 0].astype(np.float64)
        in_r = audio[:, 1].astype(np.float64)

        # Saturação normalizada via tanh
        norm_factor = 1.0 / np.tanh(self.drive)
        sat_l = np.tanh(in_l * self.drive) * norm_factor
        sat_r = np.tanh(in_r * self.drive) * norm_factor

        # DC Blocker com estado contínuo
        if self._hp_sos is not None:
            sat_l, self._hp_zi_l = sosfilt(self._hp_sos, sat_l, zi=self._hp_zi_l)
            sat_r, self._hp_zi_r = sosfilt(self._hp_sos, sat_r, zi=self._hp_zi_r)

        out_l = in_l * (1.0 - self.mix) + sat_l * self.mix
        out_r = in_r * (1.0 - self.mix) + sat_r * self.mix

        return np.column_stack((out_l, out_r)).astype(np.float32)

    def render(self, start_sample: int, num_samples: int) -> np.ndarray:
        return np.zeros((num_samples, 2), dtype=np.float32)

    def reset(self) -> None:
        if self._hp_zi_l is not None:
            self._hp_zi_l.fill(0.0)
            self._hp_zi_r.fill(0.0)
