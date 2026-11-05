"""
Instrumento: Baixo Profundo / Cello (Aterramento Harmônico e Sub-graves Acolhedores).
"""
from typing import Any, Dict, Optional
import numpy as np

from plugins.base import BasePlugin, PluginCategory
from music.scale import midi_to_freq
from music.composer import Composer


class Bass(BasePlugin):
    """
    Sintetizador de Baixo e Sub-graves com fundamental pura e 1º harmônico para presença acústica.
    """
    PLUGIN_ID = "bass"

    def __init__(self, plugin_id: str = "bass", config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(plugin_id=plugin_id, category=PluginCategory.INSTRUMENT, config=config)
        self.gain = float(self.config.get("gain", 0.4))
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

        # Obtém a tônica do acorde atual no registro grave
        if self.composer is not None:
            # Verifica se o arranjo mantém o baixo ativo nesta seção
            if not self.composer.is_role_active("bass", t_mid):
                return np.zeros((num_samples, 2), dtype=np.float32)

            chord, voicing, _ = self.composer.get_current_chord(t_mid)
            root_midi = chord.get_midi_notes()[0]
            # Transpõe para oitava 1 ou 2 (40 Hz a 80 Hz)
            while root_midi > 43: # G2
                root_midi -= 12
            freq = midi_to_freq(root_midi, self.composer.scale.a4_freq)
        else:
            freq = 55.0 # A1

        # Fundamental + primeiro harmônico suave (oitava acima) para definição em fones
        sub = np.sin(2 * np.pi * freq * t)
        octave_harm = 0.35 * np.sin(2 * np.pi * (freq * 2.0) * t)

        # LFO sutil de pulso
        lfo = 0.85 + 0.15 * np.sin(2 * np.pi * 0.08 * t)

        wave = (sub + octave_harm) * (self.gain * lfo)
        # O baixo fica centrado na imagem estéreo para solidez do mix
        stereo = np.column_stack((wave, wave)).astype(np.float32)
        return stereo

    def reset(self) -> None:
        pass
