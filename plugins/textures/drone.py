"""
Plugin de Texturas: Drone Textural Contínuo com Modulação de Fase Lenta.
"""
from typing import Any, Dict, Optional
import numpy as np

from plugins.base import BasePlugin, PluginCategory


class DroneTexture(BasePlugin):
    """
    Sintetizador de Drone Textural com osciladores múltiplos de fase contínua.
    """
    PLUGIN_ID = "drone_texture"

    def __init__(self, plugin_id: str = "drone_texture", config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(plugin_id=plugin_id, category=PluginCategory.TEXTURE, config=config)
        self.fundamental = float(self.config.get("fundamental_freq", 108.0)) # 432 Hz / 4
        self.gain = float(self.config.get("gain", 0.25))
        self.detune = float(self.config.get("detune", 0.003))
        self.sample_rate: int = 44100

    def prepare(self, context: Any, rng: np.random.Generator) -> None:
        self.sample_rate = context.sample_rate
        self._is_prepared = True

    def render(self, start_sample: int, num_samples: int) -> np.ndarray:
        if not self.is_active or num_samples <= 0:
            return np.zeros((num_samples, 2), dtype=np.float32)

        t = (np.arange(num_samples, dtype=np.float64) + start_sample) / self.sample_rate

        f = self.fundamental
        # Osciladores múltiplos levemente defasados
        osc_center = np.sin(2 * np.pi * f * t)
        osc_l = np.sin(2 * np.pi * (f * (1.0 + self.detune)) * t) * 0.4
        osc_r = np.sin(2 * np.pi * (f * (1.0 - self.detune)) * t) * 0.4
        sub = np.sin(2 * np.pi * (f * 0.5) * t) * 0.5

        # LFO para respiração orgânica
        lfo = 0.85 + 0.15 * np.sin(2 * np.pi * 0.05 * t)

        left = (osc_center * 0.6 + osc_l + sub) * (self.gain * lfo)
        right = (osc_center * 0.6 + osc_r + sub) * (self.gain * lfo)

        return np.column_stack((left, right)).astype(np.float32)

    def reset(self) -> None:
        pass
