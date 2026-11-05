"""
Efeito DSP: Filtros IIR Butterworth (Low-Pass, High-Pass e Band-Pass com Preservação de Estado).
"""
from typing import Any, Dict, Optional, Tuple
import numpy as np
from scipy.signal import butter, sosfilt

from plugins.base import BasePlugin, PluginCategory


class FilterEffect(BasePlugin):
    """
    Filtro digital IIR multipropósito com preservação contínua de condições de contorno (zi/zf).
    Garante ausência absoluta de descontinuidades, transientes de chaveamento e cliques entre chunks.
    """
    PLUGIN_ID = "filter_effect"

    def __init__(self, plugin_id: str = "filter_effect", config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(plugin_id=plugin_id, category=PluginCategory.EFFECT, config=config)
        self.filter_type = str(self.config.get("filter_type", "lowpass")).lower().strip()
        self.cutoff_hz = float(self.config.get("cutoff_hz", 2500.0))
        self.cutoff_high_hz = float(self.config.get("cutoff_high_hz", 5000.0)) # Apenas para bandpass
        self.order = int(self.config.get("order", 2))
        self.sample_rate: int = 44100

        self._sos = None
        self._zi_l: Optional[np.ndarray] = None
        self._zi_r: Optional[np.ndarray] = None

    @property
    def has_state(self) -> bool:
        return True

    def _setup_filter(self) -> None:
        nyquist = self.sample_rate / 2.0
        
        if self.filter_type == "highpass":
            cutoff = np.clip(self.cutoff_hz, 10.0, nyquist - 10.0) / nyquist
            self._sos = butter(self.order, cutoff, btype="high", output="sos")
        elif self.filter_type == "bandpass":
            low = np.clip(self.cutoff_hz, 10.0, nyquist - 50.0) / nyquist
            high = np.clip(self.cutoff_high_hz, self.cutoff_hz + 10.0, nyquist - 10.0) / nyquist
            self._sos = butter(self.order, [low, high], btype="band", output="sos")
        else: # "lowpass"
            cutoff = np.clip(self.cutoff_hz, 10.0, nyquist - 10.0) / nyquist
            self._sos = butter(self.order, cutoff, btype="low", output="sos")

        num_sections = self._sos.shape[0]
        self._zi_l = np.zeros((num_sections, 2), dtype=np.float64)
        self._zi_r = np.zeros((num_sections, 2), dtype=np.float64)

    def prepare(self, context: Any, rng: np.random.Generator) -> None:
        self.sample_rate = context.sample_rate
        self._setup_filter()
        self._is_prepared = True

    def process(self, audio: np.ndarray) -> np.ndarray:
        """
        Processa um bloco de áudio estéreo (num_samples, 2), mantendo e atualizando o estado do filtro.
        """
        if not self.is_active or self._sos is None or audio.shape[0] == 0:
            return audio

        left = audio[:, 0].astype(np.float64)
        right = audio[:, 1].astype(np.float64)

        filt_l, self._zi_l = sosfilt(self._sos, left, zi=self._zi_l)
        filt_r, self._zi_r = sosfilt(self._sos, right, zi=self._zi_r)

        return np.column_stack((filt_l, filt_r)).astype(np.float32)

    def render(self, start_sample: int, num_samples: int) -> np.ndarray:
        # Quando acoplado diretamente como plugin gerador, retorna silêncio se não alimentado
        return np.zeros((num_samples, 2), dtype=np.float32)

    def reset(self) -> None:
        if self._zi_l is not None:
            self._zi_l.fill(0.0)
            self._zi_r.fill(0.0)
