"""
Compositor Algorítmico Integrador: Fachada do MusicEngine.
"""
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from core.context import RenderContext
from music.scale import Scale, ScaleType
from music.harmony import Chord
from music.progression import ProgressionEngine, STANDARD_PROGRESSIONS
from music.rhythm import RhythmEngine
from music.melody import MelodyEngine
from music.arrangement import ArrangementEngine


class Composer:
    """
    Coordena as decisões de teoria musical, harmonia, melodia e arranjo dinâmico.
    Fornece aos instrumentos os dados contextuais necessários em cada instante de renderização.
    """

    def __init__(
        self,
        context: RenderContext,
        root_note: str = "C",
        scale_type: str = "lydian",
        bpm: float = 55.0,
        progression_tokens: Optional[List[str]] = None,
        chord_duration_beats: float = 16.0,
        a4_freq: float = 432.0
    ) -> None:
        self.context = context
        self.scale = Scale(root_note=root_note, scale_type=scale_type, a4_freq=a4_freq)
        self.rhythm = RhythmEngine(bpm=bpm)
        
        tokens = progression_tokens or STANDARD_PROGRESSIONS.get(context.profile_name, ["Iadd9", "IVsus2", "I", "IVadd9"])
        self.progression = ProgressionEngine(
            scale=self.scale,
            progression_tokens=tokens,
            chord_duration_beats=chord_duration_beats,
            base_octave=3
        )
        self.melody = MelodyEngine(scale=self.scale, progression=self.progression, rhythm=self.rhythm)
        self.arrangement = ArrangementEngine(total_duration_sec=context.duration_sec)

        # Voicings pré-calculados para a progressão inteira com smooth voice leading
        self.cached_voicings = self.progression.get_voiced_progression()

    def get_current_chord(self, time_sec: float) -> Tuple[Chord, List[int], int]:
        """
        Retorna (Chord, voiced_midi_notes, chord_index_in_progression).
        """
        chord, idx = self.progression.get_chord_at_time(time_sec, bpm=self.rhythm.bpm)
        voicing = self.cached_voicings[idx % len(self.cached_voicings)] if self.cached_voicings else chord.get_midi_notes()
        return chord, voicing, idx

    def get_density(self, time_sec: float) -> float:
        """Retorna a densidade musical ativa da seção no instante atual."""
        return self.arrangement.get_density_at_time(time_sec)

    def is_role_active(self, role: str, time_sec: float) -> bool:
        """Verifica se o papel instrumental deve tocar no tempo atual."""
        return self.arrangement.is_role_active(role, time_sec)
