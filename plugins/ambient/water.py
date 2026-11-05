"""
Plugin Ambiental: Água Corrente e Riacho (Fluxo Contínuo e Borbulhamento Suave).
"""
from typing import Any, Dict, Optional
import numpy as np
from scipy.signal import butter, sosfilt

from plugins.base import BasePlugin, PluginCategory


class WaterAmbient(BasePlugin):
    """
    Sintetizador do fluxo de riacho e água cristalina corrente.
    """
    PLUGIN_ID = "water_ambient"

    def __init__(self, plugin_id: str = "water_ambient", config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(plugin_id=plugin_id, category=PluginCategory.AMBIENT, config=config)
        self.gain = float(self.config.get("gain", 0.35))
        self.sample_rate: int = 44100
        self.rng: Optional[np.random.Generator] = None

        self._bp_sos = None
        self._bp_zi_l = None
        self._bp_zi_r = None

    @property
    def has_state(self) -> bool:
        return True

    def prepare(self, context: Any, rng: np.random.Generator) -> None:
        self.sample_rate = context.sample_rate
        self.rng = rng

        # Filtro passa-faixa de riacho límpido (350 Hz a 2200 Hz)
        low = 350.0 / (self.sample_rate / 2.0)
        high = min(2200.0, self.sample_rate * 0.45) / (self.sample_rate / 2.0)
        self._bp_sos = butter(2, [low, high], btype="band", output="sos")
        self._bp_zi_l = np.zeros((self._bp_sos.shape[0], 2), dtype=np.float64)
        self._bp_zi_r = np.zeros((self._bp_sos.shape[0], 2), dtype=np.float64)

        self._is_prepared = True

    def render(self, start_sample: int, num_samples: int) -> np.ndarray:
        if not self.is_active or num_samples <= 0 or self.rng is None:
            return np.zeros((num_samples, 2), dtype=np.float32)

        t = (np.arange(num_samples, dtype=np.float64) + start_sample) / self.sample_rate

        # 1. Ruído base de água
        noise_l = self.rng.normal(0.0, 0.5, num_samples).astype(np.float64)
        noise_r = self.rng.normal(0.0, 0.5, num_samples).astype(np.float64)

        # 2. Modulação por pequenas ondulações e borbulhamento
        stream_flow = 0.8 + 0.15 * np.sin(2 * np.pi * 0.25 * t) + 0.05 * np.sin(2 * np.pi * 1.2 * t)

        raw_l = noise_l * stream_flow
        raw_r = noise_r * stream_flow

        # 3. Filtra preservando continuidade
        filt_l, self._bp_zi_l = sosfilt(self._bp_sos, raw_l, zi=self._bp_zi_l)
        filt_r, self._bp_zi_r = sosfilt(self._bp_sos, raw_r, zi=self._bp_zi_r)

        out_l = filt_l * self.gain
        out_r = filt_r * self.gain

        return np.column_stack((out_l, out_r)).astype(np.float32)

    def reset(self) -> None:
        if self._bp_zi_l is not None:
            self._bp_zi_l.fill(0.0)
            self._bp_zi_r.fill(0.0)
