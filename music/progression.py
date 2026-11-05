"""
Progressões Harmônicas Determinísticas e Sequenciamento de Acordes.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

from music.scale import Scale, ScaleType, SEMITONE_NAMES
from music.harmony import Chord, ChordQuality, smooth_voice_leading


# Mapeamento de numerais romanos para graus da escala (1-indexado)
ROMAN_NUMERALS: Dict[str, Tuple[int, ChordQuality]] = {
    # Maior
    "I": (1, ChordQuality.MAJOR),
    "ii": (2, ChordQuality.MINOR),
    "iii": (3, ChordQuality.MINOR),
    "IV": (4, ChordQuality.MAJOR),
    "V": (5, ChordQuality.MAJOR),
    "vi": (6, ChordQuality.MINOR),
    "vii°": (7, ChordQuality.DIMINISHED),
    
    # Tétrades no campo maior
    "Imaj7": (1, ChordQuality.MAJOR_7TH),
    "iim7": (2, ChordQuality.MINOR_7TH),
    "iiim7": (3, ChordQuality.MINOR_7TH),
    "IVmaj7": (4, ChordQuality.MAJOR_7TH),
    "V7": (5, ChordQuality.DOMINANT_7TH),
    "vim7": (6, ChordQuality.MINOR_7TH),
    "viim7b5": (7, ChordQuality.HALF_DIMINISHED),

    # Menor / Modal
    "i": (1, ChordQuality.MINOR),
    "im7": (1, ChordQuality.MINOR_7TH),
    "bIII": (3, ChordQuality.MAJOR),
    "iv": (4, ChordQuality.MINOR),
    "ivm7": (4, ChordQuality.MINOR_7TH),
    "v": (5, ChordQuality.MINOR),
    "vm7": (5, ChordQuality.MINOR_7TH),
    "bVI": (6, ChordQuality.MAJOR),
    "bVII": (7, ChordQuality.MAJOR),

    # Extensões Ambientais / Calmas
    "Iadd9": (1, ChordQuality.ADD9),
    "IVadd9": (4, ChordQuality.ADD9),
    "viadd9": (6, ChordQuality.MADD9),
    "Isus2": (1, ChordQuality.SUS2),
    "IVsus2": (4, ChordQuality.SUS2),
}


class ProgressionEngine:
    """
    Constrói e sequencia progressões harmônicas determinísticas vinculadas a uma escala.
    """

    def __init__(
        self,
        scale: Scale,
        progression_tokens: Optional[List[str]] = None,
        chord_duration_beats: float = 4.0,
        base_octave: int = 4
    ) -> None:
        self.scale = scale
        self.tokens = progression_tokens or ["I", "IV", "V", "I"]
        self.chord_duration_beats = float(chord_duration_beats)
        self.base_octave = base_octave
        self.chords: List[Chord] = self._build_chords()

    def _build_chords(self) -> List[Chord]:
        """Gera os objetos Chord correspondentes aos tokens romanos informados."""
        chords = []
        for token in self.tokens:
            token_clean = token.strip()
            if token_clean in ROMAN_NUMERALS:
                degree, quality = ROMAN_NUMERALS[token_clean]
                # Pega a nota da escala no grau correspondente
                midi_val = self.scale.get_degree_midi(degree, octave=self.base_octave)
                root_name = SEMITONE_NAMES[midi_val % 12]
                chord = Chord(root_note=root_name, quality=quality, octave=self.base_octave)
                chords.append(chord)
            else:
                # Token não catalogado: tenta usar tônica direta I maior como fallback
                root_name = self.scale.root_note
                chords.append(Chord(root_note=root_name, quality=ChordQuality.MAJOR, octave=self.base_octave))
        return chords

    @property
    def total_beats(self) -> float:
        """Duração total de um ciclo da progressão em tempos (beats)."""
        return len(self.chords) * self.chord_duration_beats

    def get_chord_at_beat(self, beat: float) -> Tuple[Chord, int]:
        """
        Retorna o acorde ativo em um determinado beat e seu índice no ciclo.
        """
        if not self.chords:
            return Chord(root_note=self.scale.root_note, octave=self.base_octave), 0
        
        cycle_beat = beat % self.total_beats
        chord_idx = int(cycle_beat // self.chord_duration_beats)
        chord_idx = min(chord_idx, len(self.chords) - 1)
        return self.chords[chord_idx], chord_idx

    def get_chord_at_time(self, time_sec: float, bpm: float = 60.0) -> Tuple[Chord, int]:
        """Retorna o acorde ativo em um determinado segundo da timeline."""
        beat = (time_sec * bpm) / 60.0
        return self.get_chord_at_beat(beat)

    def get_voiced_progression(self) -> List[List[int]]:
        """
        Gera a sequência completa de voicings com smooth voice leading aplicado
        em todas as transições do ciclo.
        """
        if not self.chords:
            return []

        voiced_list: List[List[int]] = []
        prev_voicing: List[int] = []

        for chord in self.chords:
            current_voicing = smooth_voice_leading(prev_voicing, chord)
            voiced_list.append(current_voicing)
            prev_voicing = current_voicing

        return voiced_list


# Progressões predefinidas por perfil musical
STANDARD_PROGRESSIONS: Dict[str, List[str]] = {
    "deep_sleep": ["Iadd9", "IVsus2", "I", "IVadd9"],
    "meditation": ["I", "IV", "I", "IV"],
    "relaxation": ["Iadd9", "vim7", "IVadd9", "V7"],
    "fantasy_tavern": ["i", "bVII", "bVI", "bVII"],
    "medieval_ambient": ["i", "iv", "v", "i"],
    "cinematic_ambient": ["Imaj7", "IVmaj7", "vim7", "Vsus4"],
    "nature_sounds": ["I", "ii", "IV", "I"],
    "asmr": ["Iadd9", "IVadd9"],
}
