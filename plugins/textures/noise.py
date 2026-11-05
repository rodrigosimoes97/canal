"""
Plugin de Texturas: Gerador de Ruídos Procedurais (White, Pink e Brown Noise).
"""
from typing import Any, Dict, Optional
import numpy as np
from scipy.signal import butter, sosfilt

from plugins.base import BasePlugin, PluginCategory


class NoiseTexture(BasePlugin):
    """
    Gerador de ruídos procedurais calibrados para relaxamento e mascaramento acústico.
    Suporta White Noise, Pink Noise (1/f) e Brown Noise (1/f²).
    """
    PLUGIN_ID = "noise_texture"

    def __init__(self, plugin_id: str = "noise_texture", config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(plugin_id=plugin_id, category=PluginCategory.TEXTURE, config=config)
        self.noise_type = str(self.config.get("noise_type", "brown")).lower().strip()
        self.gain = float(self.config.get("gain", 0.25))
        self.sample_rate: int = 44100
        self.rng: Optional[np.random.Generator] = None

        # Estados de filtros IIR para preservação contínua de estado entre chunks
        self._hp_sos = None
        self._hp_zi_l = None
        self._hp_zi_r = None
        self._lp_sos = None
        self._lp_zi_l = None
        self._lp_zi_r = None

        # Estado interno do filtro de ruído rosa (Paul Kellet's filter)
        self._pink_b_l = np.zeros(7, dtype=np.float64)
        self._pink_b_r = np.zeros(7, dtype=np.float64)
        self._brown_state_l = 0.0
        self._brown_state_r = 0.0

    @property
    def has_state(self) -> bool:
        return True

    def prepare(self, context: Any, rng: np.random.Generator) -> None:
        self.sample_rate = context.sample_rate
        self.rng = rng

        # Highpass para remover DC offset (< 20 Hz)
        self._hp_sos = butter(2, 20.0 / (self.sample_rate / 2.0), btype="high", output="sos")
        self._hp_zi_l = np.zeros((self._hp_sos.shape[0], 2), dtype=np.float64)
        self._hp_zi_r = np.zeros((self._hp_sos.shape[0], 2), dtype=np.float64)

        # Lowpass suave para conforto auditivo
        cutoff = float(self.config.get("cutoff_hz", 800.0 if self.noise_type == "brown" else 4000.0))
        cutoff = min(cutoff, self.sample_rate * 0.45)
        self._lp_sos = butter(2, cutoff / (self.sample_rate / 2.0), btype="low", output="sos")
        self._lp_zi_l = np.zeros((self._lp_sos.shape[0], 2), dtype=np.float64)
        self._lp_zi_r = np.zeros((self._lp_sos.shape[0], 2), dtype=np.float64)

        self._pink_b_l.fill(0.0)
        self._pink_b_r.fill(0.0)
        self._brown_state_l = 0.0
        self._brown_state_r = 0.0
        self._is_prepared = True

    def _generate_pink(self, white: np.ndarray, state_b: np.ndarray) -> np.ndarray:
        """Filtro de Paul Kellet para converter ruído branco em rosa (-3dB/oitava)."""
        pink = np.empty_like(white)
        b0, b1, b2, b3, b4, b5, b6 = state_b
        for i in range(len(white)):
            w = white[i]
            b0 = 0.99886 * b0 + w * 0.0555179
            b1 = 0.99332 * b1 + w * 0.0750759
            b2 = 0.96900 * b2 + w * 0.1538520
            b3 = 0.86650 * b3 + w * 0.3104856
            b4 = 0.55000 * b4 + w * 0.5329522
            b5 = -0.7616 * b5 - w * 0.0168980
            pink[i] = b0 + b1 + b2 + b3 + b4 + b5 + b6 + w * 0.5362
            b6 = w * 0.115926
        state_b[:] = [b0, b1, b2, b3, b4, b5, b6]
        return pink * 0.11

    def _generate_brown(self, white: np.ndarray, last_val: float) -> tuple[np.ndarray, float]:
        """Integrador leaky de 1 polo (-6dB/oitava) para ruído marrom estável e sem deriva."""
        brown = np.empty_like(white)
        alpha = 0.985
        gain = 0.15
        cur = last_val
        for i in range(len(white)):
            cur = alpha * cur + (1.0 - alpha) * white[i] * gain
            brown[i] = cur
        return brown * 8.0, cur

    def render(self, start_sample: int, num_samples: int) -> np.ndarray:
        if not self.is_active or num_samples <= 0 or self.rng is None:
            return np.zeros((num_samples, 2), dtype=np.float32)

        white_l = self.rng.normal(0.0, 1.0, num_samples).astype(np.float64)
        white_r = self.rng.normal(0.0, 1.0, num_samples).astype(np.float64)

        if self.noise_type == "pink":
            raw_l = self._generate_pink(white_l, self._pink_b_l)
            raw_r = self._generate_pink(white_r, self._pink_b_r)
        elif self.noise_type == "brown":
            raw_l, self._brown_state_l = self._generate_brown(white_l, self._brown_state_l)
            raw_r, self._brown_state_r = self._generate_brown(white_r, self._brown_state_r)
        else: # "white"
            raw_l = white_l * 0.15
            raw_r = white_r * 0.15

        # Aplica filtro passa-alta para remover DC offset preservando estado entre chunks
        filt_l, self._hp_zi_l = sosfilt(self._hp_sos, raw_l, zi=self._hp_zi_l)
        filt_r, self._hp_zi_r = sosfilt(self._hp_sos, raw_r, zi=self._hp_zi_r)

        # Aplica filtro passa-baixa preservando estado entre chunks
        filt_l, self._lp_zi_l = sosfilt(self._lp_sos, filt_l, zi=self._lp_zi_l)
        filt_r, self._lp_zi_r = sosfilt(self._lp_sos, filt_r, zi=self._lp_zi_r)

        # Normaliza ganho suave
        out_l = filt_l * self.gain
        out_r = filt_r * self.gain

        return np.column_stack((out_l, out_r)).astype(np.float32)

    def reset(self) -> None:
        if self._hp_zi_l is not None:
            self._hp_zi_l.fill(0.0)
            self._hp_zi_r.fill(0.0)
            self._lp_zi_l.fill(0.0)
            self._lp_zi_r.fill(0.0)
            self._pink_b_l.fill(0.0)
            self._pink_b_r.fill(0.0)
