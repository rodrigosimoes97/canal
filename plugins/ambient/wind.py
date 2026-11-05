"""
Plugin Ambiental: Vento e Brisa (Gravação Real em Loop + Rajadas Procedurais).
"""
from typing import Any, Dict, Optional
import numpy as np

from plugins.base import BasePlugin, PluginCategory
from assets.asset_manager import AudioAssetManager


class WindAmbient(BasePlugin):
    """
    Camada sonora de vento combinando 'vento.mp3' em loop contínuo
    com rajadas aerodinâmicas procedurais.
    """
    PLUGIN_ID = "wind_ambient"

    def __init__(self, plugin_id: str = "wind_ambient", config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(plugin_id=plugin_id, category=PluginCategory.AMBIENT, config=config)
        self.gain = float(self.config.get("gain", 0.38))
        self.sample_rate: int = 44100
        self.asset_mgr: Optional[AudioAssetManager] = None
        self.has_asset: bool = False
        self.rng: Optional[np.random.Generator] = None

    def prepare(self, context: Any, rng: np.random.Generator) -> None:
        self.sample_rate = context.sample_rate
        self.rng = rng
        self.asset_mgr = context.metadata.get("asset_manager")

        if self.asset_mgr is not None and self.asset_mgr.has_asset("vento"):
            self.has_asset = True

        self._is_prepared = True

    def render(self, start_sample: int, num_samples: int) -> np.ndarray:
        if not self.is_active or num_samples <= 0:
            return np.zeros((num_samples, 2), dtype=np.float32)

        t = (np.arange(num_samples, dtype=np.float64) + start_sample) / self.sample_rate
        output = np.zeros((num_samples, 2), dtype=np.float32)

        # 1. Loop contínuo com crossfade do asset real 'vento.mp3'
        if self.has_asset and self.asset_mgr is not None:
            chunk = self.asset_mgr.get_loop_chunk(
                "vento",
                start_sample=start_sample,
                num_samples=num_samples,
                crossfade_sec=2.5
            )
            # LFO lento de rajada natural com shape (num_samples, 1) para broadcast estéreo
            gust = (0.7 + 0.3 * np.sin(2 * np.pi * 0.04 * t))[:, None]
            output += (chunk * (self.gain * gust)).astype(np.float32)
        elif self.rng is not None:
            # Fallback procedural puro se o asset não estiver presente
            white = self.rng.normal(0.0, 1.0, num_samples)
            brown = np.cumsum(white) * 0.03
            gust = 0.5 + 0.5 * np.sin(2 * np.pi * 0.06 * t)
            wave = (brown * gust * self.gain * 0.4).astype(np.float32)
            output += np.column_stack((wave, wave))

        return output

    def reset(self) -> None:
        pass
