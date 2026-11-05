"""
Plugin de Texturas: Drone Harmônico Afinado com Acordes e Extensões.
"""
from typing import Any, Dict, Optional
import numpy as np

from plugins.base import BasePlugin, PluginCategory
from music.scale import midi_to_freq
from music.composer import Composer


class HarmonicDrone(BasePlugin):
    """
    Drone tonal que acompanha as notas fundamentais e quintas da harmonia do Composer.
    """
    PLUGIN_ID = "harmonic_drone"

    def __init__(self, plugin_id: str = "harmonic_drone", config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(plugin_id=plugin_id, category=PluginCategory.TEXTURE, config=config)
        self.gain = float(self.config.get("gain", 0.28))
        self.sample_rate: int = 44100
        self.composer: Optional[Composer] = None

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
            chord, _, _ = self.composer.get_current_chord(t_mid)
            notes = chord.get_midi_notes()
            # Usa tônica e quinta para manter estabilidade harmônica
            root_midi = notes[0]
            fifth_midi = notes[2] if len(notes) > 2 else root_midi + 7

            f_root = midi_to_freq(root_midi, self.composer.scale.a4_freq)
            f_fifth = midi_to_freq(fifth_midi, self.composer.scale.a4_freq)
        else:
            f_root = 130.81 # C3
            f_fifth = 196.00 # G3

        # Fundamental + quinta harmônica com tremolo suave
        root_wave = np.sin(2 * np.pi * f_root * t)
        fifth_wave = 0.5 * np.sin(2 * np.pi * f_fifth * t)

        detune_l = 0.3 * np.sin(2 * np.pi * (f_root * 1.002) * t)
        detune_r = 0.3 * np.sin(2 * np.pi * (f_root * 0.998) * t)

        lfo = 0.85 + 0.15 * np.sin(2 * np.pi * 0.03 * t)

        left = (root_wave * 0.5 + fifth_wave * 0.3 + detune_l) * (self.gain * lfo)
        right = (root_wave * 0.5 + fifth_wave * 0.3 + detune_r) * (self.gain * lfo)

        return np.column_stack((left, right)).astype(np.float32)

    def reset(self) -> None:
        pass
