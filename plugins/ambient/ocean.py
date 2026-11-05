"""
Plugin Ambiental: Ondas do Mar e Respiração Oceânica (Síntese Procedural com Movimento Estéreo).
"""
from typing import Any, Dict, Optional
import numpy as np
from scipy.signal import butter, sosfilt

from plugins.base import BasePlugin, PluginCategory


class OceanAmbient(BasePlugin):
    """
    Sintetizador de ondas do mar com ritmo de respiração humana (6 a 8 ciclos/minuto)
    e espacialização contínua entre os canais estéreo.
    """
    PLUGIN_ID = "ocean_ambient"

    def __init__(self, plugin_id: str = "ocean_ambient", config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(plugin_id=plugin_id, category=PluginCategory.AMBIENT, config=config)
        self.gain = float(self.config.get("gain", 0.4))
        self.bpm = float(self.config.get("breathing_bpm", 7.0)) # 7 respirações/minuto
        self.sample_rate: int = 44100
        self.rng: Optional[np.random.Generator] = None

        # Filtro passa-baixa e passa-alta com estado preservado
        self._lp_sos = None
        self._lp_zi_l = None
        self._lp_zi_r = None
        self._hp_sos = None
        self._hp_zi_l = None
        self._hp_zi_r = None
        self._brown_state_l = 0.0
        self._brown_state_r = 0.0

    @property
    def has_state(self) -> bool:
        return True

    def prepare(self, context: Any, rng: np.random.Generator) -> None:
        self.sample_rate = context.sample_rate
        self.rng = rng

        # Filtro passa-baixa suave para simular a água profunda (corte em 850 Hz)
        self._lp_sos = butter(2, 850.0 / (self.sample_rate / 2.0), btype="low", output="sos")
        self._lp_zi_l = np.zeros((self._lp_sos.shape[0], 2), dtype=np.float64)
        self._lp_zi_r = np.zeros((self._lp_sos.shape[0], 2), dtype=np.float64)

        # Filtro passa-alta para remover sub-graves desnecessários (< 35 Hz)
        self._hp_sos = butter(2, 35.0 / (self.sample_rate / 2.0), btype="high", output="sos")
        self._hp_zi_l = np.zeros((self._hp_sos.shape[0], 2), dtype=np.float64)
        self._hp_zi_r = np.zeros((self._hp_sos.shape[0], 2), dtype=np.float64)

        self._is_prepared = True

    def render(self, start_sample: int, num_samples: int) -> np.ndarray:
        if not self.is_active or num_samples <= 0 or self.rng is None:
            return np.zeros((num_samples, 2), dtype=np.float32)

        t = (np.arange(num_samples, dtype=np.float64) + start_sample) / self.sample_rate

        # 1. Gera ruído marrom estável via leaky integrator
        white_l = self.rng.normal(0.0, 1.0, num_samples).astype(np.float64)
        white_r = self.rng.normal(0.0, 1.0, num_samples).astype(np.float64)
        
        brown_l = np.empty_like(white_l)
        brown_r = np.empty_like(white_r)
        alpha = 0.985
        cur_l = self._brown_state_l
        cur_r = self._brown_state_r
        for i in range(num_samples):
            cur_l = alpha * cur_l + (1.0 - alpha) * white_l[i] * 0.15
            cur_r = alpha * cur_r + (1.0 - alpha) * white_r[i] * 0.15
            brown_l[i] = cur_l * 5.0
            brown_r[i] = cur_r * 5.0
        self._brown_state_l = cur_l
        self._brown_state_r = cur_r

        # 2. Filtra preservando estados entre chunks
        filt_l, self._hp_zi_l = sosfilt(self._hp_sos, brown_l, zi=self._hp_zi_l)
        filt_r, self._hp_zi_r = sosfilt(self._hp_sos, brown_r, zi=self._hp_zi_r)
        filt_l, self._lp_zi_l = sosfilt(self._lp_sos, filt_l, zi=self._lp_zi_l)
        filt_r, self._lp_zi_r = sosfilt(self._lp_sos, filt_r, zi=self._lp_zi_r)

        # 3. Ciclo de respiração das ondas (LFO de amplitude)
        wave_freq = self.bpm / 60.0 # ~0.116 Hz
        # Onda com quebra assimétrica (subida mais rápida, descida mais suave)
        phase = 2 * np.pi * wave_freq * t
        swell_l = 0.5 + 0.5 * np.sin(phase)
        swell_r = 0.5 + 0.5 * np.sin(phase - 0.4) # Leve defasagem estéreo de arrebentação

        out_l = filt_l * (swell_l * self.gain)
        out_r = filt_r * (swell_r * self.gain)

        return np.column_stack((out_l, out_r)).astype(np.float32)

    def reset(self) -> None:
        if self._lp_zi_l is not None:
            self._lp_zi_l.fill(0.0)
            self._lp_zi_r.fill(0.0)
            self._hp_zi_l.fill(0.0)
            self._hp_zi_r.fill(0.0)
