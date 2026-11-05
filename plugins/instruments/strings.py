"""
Instrumento: Conjunto de Cordas Orquestrais (Textura Acústica Envolvente e Ampla).
"""
from typing import Any, Dict, Optional
import numpy as np

from plugins.base import BasePlugin, PluginCategory
from music.scale import midi_to_freq
from music.composer import Composer


class Strings(BasePlugin):
    """
    Sintetizador de cordas orquestrais com detuning estereofônico e expressividade de arco.
    """
    PLUGIN_ID = "strings"

    def __init__(self, plugin_id: str = "strings", config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(plugin_id=plugin_id, category=PluginCategory.INSTRUMENT, config=config)
        self.gain = float(self.config.get("gain", 0.3))
        self.detune = float(self.config.get("detune", 0.0025))
        self.sample_rate: int = 44100
        self.composer: Optional[Composer] = None

    @property
    def has_state(self) -> bool:
        return True

    def prepare(self, context: Any, rng: np.random.Generator) -> None:
        self.sample_rate = context.sample_rate
        self.composer = context.metadata.get("composer")
        self._is_prepared = True

    def render(self, start_sample: int, num_samples: int) -> np.ndarray:
        if not self.is_active or num_samples <= 0:
            return np.zeros((num_samples, 2), dtype=np.float32)

        t = (np.arange(num_samples, dtype=np.float64) + start_sample) / self.sample_rate
        t_mid = (start_sample + num_samples / 2) / self.sample_rate

        if self.composer is not None:
            _, voicing, _ = self.composer.get_current_chord(t_mid)
            # Eleva o voicing uma oitava para registro de cordas líricas
            freqs = [midi_to_freq(m + 12, self.composer.scale.a4_freq) for m in voicing]
        else:
            freqs = [261.63, 329.63, 392.0] # C4, E4, G4

        out_l = np.zeros(num_samples, dtype=np.float64)
        out_r = np.zeros(num_samples, dtype=np.float64)

        # LFO de vibrato e expressividade do arco
        vibrato = 1.0 + 0.003 * np.sin(2 * np.pi * 4.5 * t)
        bow_breath = 0.8 + 0.2 * np.sin(2 * np.pi * 0.12 * t)

        for i, freq in enumerate(freqs):
            f_vib = freq * vibrato
            # Fundamental + harmônicos característicos de fricção de corda
            fund = np.sin(2 * np.pi * f_vib * t)
            h2 = 0.45 * np.sin(2 * np.pi * (f_vib * 2.0) * t)
            h3 = 0.25 * np.sin(2 * np.pi * (f_vib * 3.0) * t)
            h4 = 0.15 * np.sin(2 * np.pi * (f_vib * 4.0) * t)

            voice = (fund + h2 + h3 + h4)

            # Efeito de conjunto através de detuning bilateral
            detune_l = np.sin(2 * np.pi * (f_vib * (1.0 + self.detune)) * t) * 0.3
            detune_r = np.sin(2 * np.pi * (f_vib * (1.0 - self.detune)) * t) * 0.3

            pan_l = 0.6 if i % 2 == 0 else 0.4
            pan_r = 1.0 - pan_l

            out_l += (voice + detune_l) * pan_l
            out_r += (voice + detune_r) * pan_r

        norm = self.gain / max(1, len(freqs))
        stereo = np.column_stack((out_l * norm * bow_breath, out_r * norm * bow_breath))
        return stereo.astype(np.float32)

    def reset(self) -> None:
        pass
