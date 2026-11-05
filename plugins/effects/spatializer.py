"""
Efeito DSP: Espacializador Estéreo (Equal-Power Pan Law, Mid/Side Stereo Width e Órbita Espacial).
"""
from typing import Any, Dict, Optional
import numpy as np

from plugins.base import BasePlugin, PluginCategory


class SpatializerEffect(BasePlugin):
    """
    Controlador de imagem estereofônica com pan de potência constante (Equal-Power),
    largura de campo via codificação Mid/Side e modulação orbital suave.
    """
    PLUGIN_ID = "spatializer_effect"

    def __init__(self, plugin_id: str = "spatializer_effect", config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(plugin_id=plugin_id, category=PluginCategory.EFFECT, config=config)
        self.pan = float(np.clip(self.config.get("pan", 0.0), -1.0, 1.0))
        self.stereo_width = float(np.clip(self.config.get("stereo_width", 1.0), 0.0, 2.5))
        self.orbit_speed_hz = float(self.config.get("orbit_speed_hz", 0.0)) # 0.0 = pan fixo
        self.orbit_depth = float(np.clip(self.config.get("orbit_depth", 0.0), 0.0, 1.0))
        self.sample_rate: int = 44100

    def prepare(self, context: Any, rng: np.random.Generator) -> None:
        self.sample_rate = context.sample_rate
        self._is_prepared = True

    def process(self, audio: np.ndarray, start_sample: int = 0) -> np.ndarray:
        if not self.is_active or audio.shape[0] == 0:
            return audio

        num_samples = audio.shape[0]
        in_l = audio[:, 0].astype(np.float64)
        in_r = audio[:, 1].astype(np.float64)

        # 1. Processamento Mid/Side para largura estéreo
        if self.stereo_width != 1.0:
            mid = (in_l + in_r) * 0.7071067811865475 # 1/sqrt(2)
            side = (in_l - in_r) * 0.7071067811865475
            side *= self.stereo_width
            in_l = (mid + side) * 0.7071067811865475
            in_r = (mid - side) * 0.7071067811865475

        # 2. Equal-Power Panning (com suporte a órbita contínua por LFO)
        if self.orbit_speed_hz > 0.0 and self.orbit_depth > 0.0:
            t = (np.arange(num_samples, dtype=np.float64) + start_sample) / self.sample_rate
            pan_val = self.pan + self.orbit_depth * np.sin(2 * np.pi * self.orbit_speed_hz * t)
            pan_val = np.clip(pan_val, -1.0, 1.0)
            # Ângulo theta entre 0 e pi/2
            theta = (pan_val + 1.0) * (np.pi / 4.0)
            gain_l = np.cos(theta) * 1.4142135623730951 # sqrt(2)
            gain_r = np.sin(theta) * 1.4142135623730951
            out_l = in_l * gain_l
            out_r = in_r * gain_r
        elif self.pan != 0.0:
            theta = (self.pan + 1.0) * (np.pi / 4.0)
            gain_l = np.cos(theta) * 1.4142135623730951
            gain_r = np.sin(theta) * 1.4142135623730951
            out_l = in_l * gain_l
            out_r = in_r * gain_r
        else:
            out_l = in_l
            out_r = in_r

        return np.column_stack((out_l, out_r)).astype(np.float32)

    def render(self, start_sample: int, num_samples: int) -> np.ndarray:
        return np.zeros((num_samples, 2), dtype=np.float32)

    def reset(self) -> None:
        pass
