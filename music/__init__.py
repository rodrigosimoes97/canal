"""
Módulo MusicEngine do Relaxation Audio Engine.
"""
from music.scale import (
    Scale,
    ScaleType,
    midi_to_freq,
    midi_to_note,
    note_to_freq,
    note_to_midi,
)
from music.harmony import (
    Chord,
    ChordQuality,
    smooth_voice_leading,
)
from music.progression import (
    ProgressionEngine,
    STANDARD_PROGRESSIONS,
)
from music.rhythm import (
    RhythmEngine,
    NoteDuration,
)
from music.melody import (
    MelodyEngine,
    MelodicNote,
)
from music.arrangement import (
    ArrangementEngine,
    ArrangementStage,
)
from music.composer import Composer

__all__ = [
    "Scale",
    "ScaleType",
    "midi_to_freq",
    "midi_to_note",
    "note_to_freq",
    "note_to_midi",
    "Chord",
    "ChordQuality",
    "smooth_voice_leading",
    "ProgressionEngine",
    "STANDARD_PROGRESSIONS",
    "RhythmEngine",
    "NoteDuration",
    "MelodyEngine",
    "MelodicNote",
    "ArrangementEngine",
    "ArrangementStage",
    "Composer",
]
