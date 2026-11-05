"""
Instrumento: Ambient Pad Aveludado (Textura Analógica Quente, Filtro Passa-Baixa Contínuo e Crossfade Harmônico).
"""
from typing import Any, Dict, List, Optional
import numpy as np
from scipy import signal

from plugins.base import BasePlugin, PluginCategory
from music.scale import midi_to_freq
from music.composer import Composer


class AmbientPad(BasePlugin):
    """
    Sintetizador de Pads Ambientais Aveludados com calor analógico, filtragem passa-baixa contínua,
    detune estéreo desacoplado, saturação suave e LFOs assíncronos de respiração.
    """
    PLUGIN_ID = "ambient_pad"

    def __init__(self, plugin_id: str = "ambient_pad", config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(plugin_id=plugin_id, category=PluginCategory.INSTRUMENT, config=config)
        self.gain = float(self.config.get("gain", 0.28))
        self.detune = float(self.config.get("detune", 0.0025))
        self.cutoff_hz = float(self.config.get("cutoff_hz", 520.0))
        self.sample_rate: int = 44100
        self.composer: Optional[Composer] = None

        # Filtro IIR analógico de 2ª ordem passa-baixa contínuo
        self._sos = signal.butter(2, self.cutoff_hz, btype="lowpass", fs=44100, output="sos")
        self._zi_l = signal.sosfilt_zi(self._sos)
        self._zi_r = signal.sosfilt_zi(self._sos)

    @property
    def has_state(self) -> bool:
        return True

    def prepare(self, context: Any, rng: np.random.Generator) -> None:
        self.sample_rate = context.sample_rate
        self.composer = context.metadata.get("composer")
        self._sos = signal.butter(2, self.cutoff_hz, btype="lowpass", fs=self.sample_rate, output="sos")
        self._zi_l = signal.sosfilt_zi(self._sos)
        self._zi_r = signal.sosfilt_zi(self._sos)
        self._is_prepared = True

    def render(self, start_sample: int, num_samples: int) -> np.ndarray:
        if not self.is_active or num_samples <= 0:
            return np.zeros((num_samples, 2), dtype=np.float32)

        t = (np.arange(num_samples, dtype=np.float64) + start_sample) / self.sample_rate
        t_mid = (start_sample + num_samples / 2) / self.sample_rate

        # Obtém harmonia ativa do Composer
        if self.composer is not None:
            _, voicing, _ = self.composer.get_current_chord(t_mid)
            freqs = [midi_to_freq(m, self.composer.scale.a4_freq) for m in voicing]
        else:
            freqs = [130.81, 164.81, 196.0, 246.94] # C3, E3, G3, B3

        raw_l = np.zeros(num_samples, dtype=np.float64)
        raw_r = np.zeros(num_samples, dtype=np.float64)

        # LFOs assíncronos suaves de respiração espacial
        lfo_l = 0.82 + 0.18 * np.sin(2 * np.pi * 0.023 * t)
        lfo_r = 0.82 + 0.18 * np.sin(2 * np.pi * 0.031 * t + 1.2)

        for i, freq in enumerate(freqs):
            # Onda fundamental + 2º harmônico sutil
            fund_c = np.sin(2 * np.pi * freq * t)
            fund_detune_l = np.sin(2 * np.pi * (freq * (1.0 + self.detune)) * t)
            fund_detune_r = np.sin(2 * np.pi * (freq * (1.0 - self.detune)) * t)

            h2_c = 0.25 * np.sin(2 * np.pi * (freq * 2.0) * t)

            # Abertura estéreo suave
            pan_spread = 0.35 if (i % 2 == 0) else -0.35
            pan_l = np.cos((pan_spread + 1.0) * (np.pi / 4))
            pan_r = np.sin((pan_spread + 1.0) * (np.pi / 4))

            voice_sig_l = (0.55 * fund_c + 0.45 * fund_detune_l + h2_c) * pan_l
            voice_sig_r = (0.55 * fund_c + 0.45 * fund_detune_r + h2_c) * pan_r

            raw_l += voice_sig_l
            raw_r += voice_sig_r

        # Normalização de vozes
        norm = 1.0 / max(1, len(freqs))
        raw_l *= norm * lfo_l
        raw_r *= norm * lfo_r

        # Saturação suave tipo fita analógica quente (warm tape drive)
        sat_l = np.tanh(1.2 * raw_l)
        sat_r = np.tanh(1.2 * raw_r)

        # Filtragem passa-baixa contínua entre blocos para calor aveludado sem estridência
        filt_l, self._zi_l = signal.sosfilt(self._sos, sat_l, zi=self._zi_l)
        filt_r, self._zi_r = signal.sosfilt(self._sos, sat_r, zi=self._zi_r)

        out_l = filt_l * self.gain
        out_r = filt_r * self.gain

        return np.column_stack((out_l, out_r)).astype(np.float32)

    def reset(self) -> None:
        self._zi_l = signal.sosfilt_zi(self._sos)
        self._zi_r = signal.sosfilt_zi(self._sos)
