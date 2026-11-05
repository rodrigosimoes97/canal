"""
Plugin Ambiental: Lareira Aconchegante (Loop Contínuo com Crossfade e Estalos Vivos).
"""
from typing import Any, Dict, Optional
import numpy as np

from plugins.base import BasePlugin, PluginCategory
from assets.asset_manager import AudioAssetManager


class FireplaceAmbient(BasePlugin):
    """
    Camada sonora de fogueira/lareira utilizando 'lareira.mp3' em loop contínuo
    combinado a micro-estalos procedurais de madeira queimando.
    """
    PLUGIN_ID = "fireplace_ambient"

    def __init__(self, plugin_id: str = "fireplace_ambient", config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(plugin_id=plugin_id, category=PluginCategory.AMBIENT, config=config)
        self.gain = float(self.config.get("gain", 0.45))
        self.pan = float(self.config.get("pan", 0.2)) # Levemente à direita
        self.sample_rate: int = 44100
        self.asset_mgr: Optional[AudioAssetManager] = None
        self.has_asset: bool = False
        self.rng: Optional[np.random.Generator] = None

    def prepare(self, context: Any, rng: np.random.Generator) -> None:
        self.sample_rate = context.sample_rate
        self.rng = rng
        self.asset_mgr = context.metadata.get("asset_manager")

        if self.asset_mgr is not None and self.asset_mgr.has_asset("lareira"):
            self.has_asset = True

        self._is_prepared = True

    def render(self, start_sample: int, num_samples: int) -> np.ndarray:
        if not self.is_active or num_samples <= 0:
            return np.zeros((num_samples, 2), dtype=np.float32)

        output = np.zeros((num_samples, 2), dtype=np.float32)

        # 1. Loop contínuo com crossfade do áudio real
        if self.has_asset and self.asset_mgr is not None:
            chunk = self.asset_mgr.get_loop_chunk(
                "lareira",
                start_sample=start_sample,
                num_samples=num_samples,
                crossfade_sec=4.0
            )
            # Panning
            pan_l = 0.5 - (self.pan * 0.5)
            pan_r = 0.5 + (self.pan * 0.5)
            output[:, 0] += chunk[:, 0] * (self.gain * pan_l * 2.0)
            output[:, 1] += chunk[:, 1] * (self.gain * pan_r * 2.0)

        # 2. Micro-estalos procedurais de madeira
        if self.rng is not None:
            num_pops = self.rng.poisson(lam=1.5 * (num_samples / self.sample_rate))
            if num_pops > 0:
                pop_idx = self.rng.integers(0, num_samples, size=num_pops)
                for idx in pop_idx:
                    pop_len = min(int(0.015 * self.sample_rate), num_samples - idx)
                    if pop_len > 0:
                        t = np.linspace(0, 1, pop_len)
                        pop_wave = (np.sin(2 * np.pi * float(self.rng.uniform(1200, 3500)) * t) * np.exp(-12 * t) * 0.15).astype(np.float32)
                        output[idx : idx + pop_len, 0] += pop_wave * 0.4
                        output[idx : idx + pop_len, 1] += pop_wave * 0.6

        return output

    def reset(self) -> None:
        pass
