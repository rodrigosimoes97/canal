"""
Plugin de Texturas: Batimentos Binaurais Reais com Canais Estéreo Isolados.
"""
from typing import Any, Dict, Optional
import numpy as np

from plugins.base import BasePlugin, PluginCategory


BINAURAL_BANDS: Dict[str, float] = {
    "delta": 2.5,   # 0.5 a 4 Hz  -> sono profundo, regeneração física
    "theta": 6.0,   # 4 a 8 Hz   -> meditação profunda, relaxamento, ASMR
    "alpha": 10.0,  # 8 a 13 Hz  -> relaxamento calmo acordado, criatividade
}


class BinauralBeats(BasePlugin):
    """
    Gera batimentos binaurais autênticos aplicando frequências portadoras
    levemente defasadas nos canais esquerdo e direito independentes.
    """
    PLUGIN_ID = "binaural_beats"

    def __init__(self, plugin_id: str = "binaural_beats", config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(plugin_id=plugin_id, category=PluginCategory.TEXTURE, config=config)
        self.mode = str(self.config.get("mode", "theta")).lower().strip()
        self.carrier_freq = float(self.config.get("carrier_freq", 180.0))
        self.beat_freq = float(self.config.get("beat_freq", BINAURAL_BANDS.get(self.mode, 6.0)))
        self.gain = float(self.config.get("gain", 0.04))
        self.sample_rate: int = 44100

    def prepare(self, context: Any, rng: np.random.Generator) -> None:
        self.sample_rate = context.sample_rate
        self._is_prepared = True

    def render(self, start_sample: int, num_samples: int) -> np.ndarray:
        if not self.is_active or num_samples <= 0:
            return np.zeros((num_samples, 2), dtype=np.float32)

        # Tempo absoluto contínuo garante continuidade perfeita de fase entre chunks
        t = (np.arange(num_samples, dtype=np.float64) + start_sample) / self.sample_rate

        left_freq = self.carrier_freq
        right_freq = self.carrier_freq + self.beat_freq

        left_wave = (np.sin(2 * np.pi * left_freq * t) * self.gain).astype(np.float32)
        right_wave = (np.sin(2 * np.pi * right_freq * t) * self.gain).astype(np.float32)

        return np.column_stack((left_wave, right_wave))

    def reset(self) -> None:
        pass
