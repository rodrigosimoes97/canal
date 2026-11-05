"""
Plugin Ambiental: Chuva Híbrida (Gravação Real em Loop + Gotas Procedurais).
"""
from typing import Any, Dict, Optional
import numpy as np
from scipy.signal import butter, sosfilt

from plugins.base import BasePlugin, PluginCategory
from assets.asset_manager import AudioAssetManager


class RainAmbient(BasePlugin):
    """
    Camada sonora de chuva combinando o sample real 'chuva.mp3' em loop contínuo
    com simulação procedural de gotas d'água de alta frequência.
    """
    PLUGIN_ID = "rain_ambient"

    def __init__(self, plugin_id: str = "rain_ambient", config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(plugin_id=plugin_id, category=PluginCategory.AMBIENT, config=config)
        self.gain = float(self.config.get("gain", 0.45))
        self.procedural_mix = float(self.config.get("procedural_mix", 0.2))
        self.sample_rate: int = 44100
        self.asset_mgr: Optional[AudioAssetManager] = None
        self.has_asset: bool = False
        self.rng: Optional[np.random.Generator] = None

        # Filtro passa-faixa para as gotas procedurais
        self._bp_sos = None
        self._bp_zi_l = None
        self._bp_zi_r = None

    @property
    def has_state(self) -> bool:
        return True

    def prepare(self, context: Any, rng: np.random.Generator) -> None:
        self.sample_rate = context.sample_rate
        self.rng = rng
        self.asset_mgr = context.metadata.get("asset_manager")

        if self.asset_mgr is not None and self.asset_mgr.has_asset("chuva"):
            self.has_asset = True

        # Prepara filtro passa-faixa das gotas (1500 Hz a 4000 Hz)
        low = 1500.0 / (self.sample_rate / 2.0)
        high = min(4000.0, self.sample_rate * 0.45) / (self.sample_rate / 2.0)
        self._bp_sos = butter(2, [low, high], btype="band", output="sos")
        self._bp_zi_l = np.zeros((self._bp_sos.shape[0], 2), dtype=np.float64)
        self._bp_zi_r = np.zeros((self._bp_sos.shape[0], 2), dtype=np.float64)
        self._is_prepared = True

    def render(self, start_sample: int, num_samples: int) -> np.ndarray:
        if not self.is_active or num_samples <= 0:
            return np.zeros((num_samples, 2), dtype=np.float32)

        output = np.zeros((num_samples, 2), dtype=np.float32)

        # 1. Camada de áudio real (chuva.mp3) em streaming contínuo com crossfade
        if self.has_asset and self.asset_mgr is not None:
            sample_chunk = self.asset_mgr.get_loop_chunk(
                "chuva",
                start_sample=start_sample,
                num_samples=num_samples,
                crossfade_sec=3.0
            )
            output += sample_chunk * (self.gain * (1.0 - self.procedural_mix * 0.5))

        # 2. Gotas procedurais pontuais para brilho e presença
        if self.rng is not None and self.procedural_mix > 0:
            drops_raw_l = np.zeros(num_samples, dtype=np.float64)
            drops_raw_r = np.zeros(num_samples, dtype=np.float64)

            # Transientes esparsos
            num_drops = int(num_samples * 0.005) # ~220 gotas por segundo
            idx_l = self.rng.integers(0, num_samples, size=num_drops)
            idx_r = self.rng.integers(0, num_samples, size=num_drops)

            drops_raw_l[idx_l] = self.rng.uniform(0.1, 0.6, size=num_drops)
            drops_raw_r[idx_r] = self.rng.uniform(0.1, 0.6, size=num_drops)

            # Filtra as gotas mantendo estado entre chunks
            drops_l, self._bp_zi_l = sosfilt(self._bp_sos, drops_raw_l, zi=self._bp_zi_l)
            drops_r, self._bp_zi_r = sosfilt(self._bp_sos, drops_raw_r, zi=self._bp_zi_r)

            drops_stereo = np.column_stack((drops_l, drops_r)) * (self.gain * self.procedural_mix)
            output += drops_stereo.astype(np.float32)

        return output

    def reset(self) -> None:
        if self._bp_zi_l is not None:
            self._bp_zi_l.fill(0.0)
            self._bp_zi_r.fill(0.0)
