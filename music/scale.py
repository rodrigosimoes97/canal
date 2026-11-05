"""
Teoria Musical: Notas, Escalas, Modos e Conversões de Frequência.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union


SEMITONE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NOTE_TO_SEMITONE: Dict[str, int] = {
    "C": 0, "B#": 0,
    "C#": 1, "Db": 1,
    "D": 2,
    "D#": 3, "Eb": 3,
    "E": 4, "Fb": 4,
    "F": 5, "E#": 5,
    "F#": 6, "Gb": 6,
    "G": 7,
    "G#": 8, "Ab": 8,
    "A": 9,
    "A#": 10, "Bb": 10,
    "B": 11, "Cb": 11,
}


def note_to_midi(note_name: str, octave: int = 4) -> int:
    """
    Converte nome da nota e oitava para número MIDI.
    C4 (Dó central) = MIDI 60. A4 = MIDI 69.
    """
    clean_name = note_name.strip()
    if clean_name not in NOTE_TO_SEMITONE:
        raise ValueError(f"Nome de nota inválido: '{note_name}'")
    semi = NOTE_TO_SEMITONE[clean_name]
    return (octave + 1) * 12 + semi


def midi_to_note(midi_number: int) -> Tuple[str, int]:
    """Converte número MIDI em tupla (nome_da_nota, oitava)."""
    octave = (midi_number // 12) - 1
    semi = midi_number % 12
    return SEMITONE_NAMES[semi], octave


def midi_to_freq(midi_number: Union[int, float], a4_freq: float = 440.0) -> float:
    """
    Converte número MIDI para frequência em Hertz.
    Suporta microtons (números fracionários).
    """
    return float(a4_freq * (2.0 ** ((midi_number - 69.0) / 12.0)))


def note_to_freq(note_name: str, octave: int = 4, a4_freq: float = 440.0) -> float:
    """Converte nome da nota e oitava para frequência em Hertz."""
    midi_num = note_to_midi(note_name, octave)
    return midi_to_freq(midi_num, a4_freq=a4_freq)


class ScaleType(str, Enum):
    """Padrões de intervalos de escalas e modos musicais (em semitons)."""
    MAJOR = "major"                          # 1, 2, 3, 4, 5, 6, 7
    NATURAL_MINOR = "natural_minor"          # 1, 2, b3, 4, 5, b6, b7
    HARMONIC_MINOR = "harmonic_minor"        # 1, 2, b3, 4, 5, b6, 7
    DORIAN = "dorian"                        # 1, 2, b3, 4, 5, 6, b7 (Medieval / Jazz)
    LYDIAN = "lydian"                        # 1, 2, 3, #4, 5, 6, 7 (Etéreo / Místico)
    MIXOLYDIAN = "mixolydian"                # 1, 2, 3, 4, 5, 6, b7 (Celta / Folclore)
    PHRYGIAN = "phrygian"                    # 1, b2, b3, 4, 5, b6, b7
    MAJOR_PENTATONIC = "major_pentatonic"    # 1, 2, 3, 5, 6 (Sem tensão, meditação)
    MINOR_PENTATONIC = "minor_pentatonic"    # 1, b3, 4, 5, b7
    WHOLE_TONE = "whole_tone"                # 1, 2, 3, #4, #5, b7 (Impressionismo)


SCALE_INTERVALS: Dict[ScaleType, List[int]] = {
    ScaleType.MAJOR: [0, 2, 4, 5, 7, 9, 11],
    ScaleType.NATURAL_MINOR: [0, 2, 3, 5, 7, 8, 10],
    ScaleType.HARMONIC_MINOR: [0, 2, 3, 5, 7, 8, 11],
    ScaleType.DORIAN: [0, 2, 3, 5, 7, 9, 10],
    ScaleType.LYDIAN: [0, 2, 4, 6, 7, 9, 11],
    ScaleType.MIXOLYDIAN: [0, 2, 4, 5, 7, 9, 10],
    ScaleType.PHRYGIAN: [0, 1, 3, 5, 7, 8, 10],
    ScaleType.MAJOR_PENTATONIC: [0, 2, 4, 7, 9],
    ScaleType.MINOR_PENTATONIC: [0, 3, 5, 7, 10],
    ScaleType.WHOLE_TONE: [0, 2, 4, 6, 8, 10],
}


class Scale:
    """
    Representa uma escala musical configurada com tônica, modo e afinação de referência.
    """

    def __init__(
        self,
        root_note: str = "C",
        scale_type: Union[ScaleType, str] = ScaleType.MAJOR,
        a4_freq: float = 440.0
    ) -> None:
        clean_root = root_note.strip()
        if clean_root not in NOTE_TO_SEMITONE:
            raise ValueError(f"Tônica inválida: '{root_note}'")
        self.root_note = clean_root
        self.root_semitone = NOTE_TO_SEMITONE[clean_root]

        if isinstance(scale_type, str):
            try:
                self.scale_type = ScaleType(scale_type.lower())
            except ValueError:
                raise ValueError(f"Tipo de escala não suportado: '{scale_type}'")
        else:
            self.scale_type = scale_type

        self.a4_freq = float(a4_freq)
        self.intervals = SCALE_INTERVALS[self.scale_type]

    def get_pitch_classes(self) -> List[int]:
        """Retorna os semitons (0 a 11) pertencentes à escala relativos a C=0."""
        return [(self.root_semitone + interval) % 12 for interval in self.intervals]

    def get_note_names(self) -> List[str]:
        """Retorna os nomes das notas pertencentes à escala na oitava base."""
        return [SEMITONE_NAMES[pc] for pc in self.get_pitch_classes()]

    def get_degree_midi(self, degree: int, octave: int = 4) -> int:
        """
        Retorna o número MIDI do grau da escala (1-indexado).
        Graus além do tamanho da escala sobem para a próxima oitava automaticamente.
        """
        if degree < 1:
            raise ValueError(f"Grau deve ser >= 1, recebido: {degree}")
        
        idx = degree - 1
        num_degrees = len(self.intervals)
        octave_offset = idx // num_degrees
        degree_idx = idx % num_degrees

        interval = self.intervals[degree_idx]
        root_midi = (octave + octave_offset + 1) * 12 + self.root_semitone
        return root_midi + interval

    def get_degree_freq(self, degree: int, octave: int = 4) -> float:
        """Retorna a frequência em Hz do grau da escala."""
        midi_num = self.get_degree_midi(degree, octave)
        return midi_to_freq(midi_num, self.a4_freq)

    def get_scale_midi_range(self, start_octave: int, num_octaves: int) -> List[int]:
        """Gera todos os números MIDI pertencentes à escala em uma faixa de oitavas."""
        notes = []
        for oct_idx in range(start_octave, start_octave + num_octaves):
            for interval in self.intervals:
                midi_val = (oct_idx + 1) * 12 + self.root_semitone + interval
                notes.append(midi_val)
        return notes

    def is_midi_in_scale(self, midi_number: int) -> bool:
        """Verifica se um dado valor MIDI pertence aos graus desta escala."""
        return (midi_number % 12) in self.get_pitch_classes()

    def __repr__(self) -> str:
        return f"<Scale {self.root_note} {self.scale_type.value} A4={self.a4_freq}Hz>"
