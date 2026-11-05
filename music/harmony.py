"""
Teoria Harmônica: Estrutura de Acordes, Extensões, Inversões e Condução de Vozes.
"""
from dataclasses import dataclass, field
from enum import Enum
from itertools import permutations
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from music.scale import NOTE_TO_SEMITONE, SEMITONE_NAMES, midi_to_freq, note_to_midi


class ChordQuality(str, Enum):
    """Qualidades e tipos de acordes com seus respectivos intervalos de semitons."""
    # Tríades Básicas
    MAJOR = "major"                          # 1, 3, 5 -> [0, 4, 7]
    MINOR = "minor"                          # 1, b3, 5 -> [0, 3, 7]
    DIMINISHED = "diminished"                # 1, b3, b5 -> [0, 3, 6]
    AUGMENTED = "augmented"                  # 1, 3, #5 -> [0, 4, 8]
    SUS2 = "sus2"                            # 1, 2, 5 -> [0, 2, 7]
    SUS4 = "sus4"                            # 1, 4, 5 -> [0, 5, 7]

    # Tétrades (Sétimas)
    MAJOR_7TH = "maj7"                       # 1, 3, 5, 7 -> [0, 4, 7, 11]
    MINOR_7TH = "min7"                       # 1, b3, 5, b7 -> [0, 3, 7, 10]
    DOMINANT_7TH = "dom7"                    # 1, 3, 5, b7 -> [0, 4, 7, 10]
    HALF_DIMINISHED = "m7b5"                 # 1, b3, b5, b7 -> [0, 3, 6, 10]
    DIMINISHED_7TH = "dim7"                  # 1, b3, b5, bb7 -> [0, 3, 6, 9]

    # Extensões Ambientais / Relaxamento
    ADD9 = "add9"                            # 1, 3, 5, 9 -> [0, 4, 7, 14]
    MADD9 = "madd9"                          # 1, b3, 5, 9 -> [0, 3, 7, 14]
    MAJOR_9TH = "maj9"                       # 1, 3, 5, 7, 9 -> [0, 4, 7, 11, 14]
    MINOR_9TH = "min9"                       # 1, b3, 5, b7, 9 -> [0, 3, 7, 10, 14]
    ELEVENTH = "11"                          # 1, 5, b7, 9, 11 -> [0, 7, 10, 14, 17]


CHORD_INTERVALS: Dict[ChordQuality, List[int]] = {
    ChordQuality.MAJOR: [0, 4, 7],
    ChordQuality.MINOR: [0, 3, 7],
    ChordQuality.DIMINISHED: [0, 3, 6],
    ChordQuality.AUGMENTED: [0, 4, 8],
    ChordQuality.SUS2: [0, 2, 7],
    ChordQuality.SUS4: [0, 5, 7],
    ChordQuality.MAJOR_7TH: [0, 4, 7, 11],
    ChordQuality.MINOR_7TH: [0, 3, 7, 10],
    ChordQuality.DOMINANT_7TH: [0, 4, 7, 10],
    ChordQuality.HALF_DIMINISHED: [0, 3, 6, 10],
    ChordQuality.DIMINISHED_7TH: [0, 3, 6, 9],
    ChordQuality.ADD9: [0, 4, 7, 14],
    ChordQuality.MADD9: [0, 3, 7, 14],
    ChordQuality.MAJOR_9TH: [0, 4, 7, 11, 14],
    ChordQuality.MINOR_9TH: [0, 3, 7, 10, 14],
    ChordQuality.ELEVENTH: [0, 7, 10, 14, 17],
}


@dataclass
class Chord:
    """
    Representação de um acorde com tônica, qualidade, oitava e inversão.
    """
    root_note: str
    quality: ChordQuality = ChordQuality.MAJOR
    octave: int = 4
    inversion: int = 0
    bass_note: Optional[str] = None

    def __post_init__(self) -> None:
        clean_root = self.root_note.strip()
        if clean_root not in NOTE_TO_SEMITONE:
            raise ValueError(f"Tônica do acorde inválida: '{self.root_note}'")
        self.root_note = clean_root

        if isinstance(self.quality, str):
            self.quality = ChordQuality(self.quality.lower())

    @property
    def root_midi(self) -> int:
        """Número MIDI da tônica no registro base."""
        return note_to_midi(self.root_note, self.octave)

    def get_pitch_classes(self) -> List[int]:
        """Retorna os semitons (0-11) pertencentes ao acorde."""
        semi_root = NOTE_TO_SEMITONE[self.root_note]
        intervals = CHORD_INTERVALS[self.quality]
        return sorted(list(set((semi_root + interval) % 12 for interval in intervals)))

    def get_midi_notes(self) -> List[int]:
        """
        Retorna as notas MIDI do acorde aplicando inversão se configurada.
        """
        base_intervals = CHORD_INTERVALS[self.quality]
        base_notes = [self.root_midi + interval for interval in base_intervals]

        # Aplica a inversão transpondo as notas inferiores em uma oitava (+12)
        inv = self.inversion % len(base_notes)
        for i in range(inv):
            base_notes[i] += 12

        base_notes.sort()

        # Se houver baixo especificado (ex: C/G)
        if self.bass_note is not None:
            clean_bass = self.bass_note.strip()
            bass_semi = NOTE_TO_SEMITONE[clean_bass]
            # Baixo na oitava inferior ao acorde
            bass_midi = (self.octave) * 12 + bass_semi
            if bass_midi >= base_notes[0]:
                bass_midi -= 12
            if bass_midi not in base_notes:
                base_notes.insert(0, bass_midi)

        return base_notes

    def get_frequencies(self, a4_freq: float = 440.0) -> List[float]:
        """Retorna as frequências em Hz de todas as notas do acorde."""
        return [midi_to_freq(m, a4_freq) for m in self.get_midi_notes()]

    def invert(self, inversion: int) -> "Chord":
        """Retorna uma nova instância com a inversão especificada."""
        return Chord(
            root_note=self.root_note,
            quality=self.quality,
            octave=self.octave,
            inversion=inversion,
            bass_note=self.bass_note
        )

    def __repr__(self) -> str:
        inv_str = f" inv={self.inversion}" if self.inversion > 0 else ""
        bass_str = f"/{self.bass_note}" if self.bass_note else ""
        return f"<Chord {self.root_note}{self.quality.value}{bass_str} oct={self.octave}{inv_str}>"


def smooth_voice_leading(
    previous_voicing: List[int],
    next_chord: Chord,
    target_octave_span: int = 12
) -> List[int]:
    """
    Algoritmo de Condução de Vozes Suave (Smooth Voice Leading).
    
    Recebe a disposição de notas anterior (previous_voicing) e encontra a distribuição
    de oitavas/inversões para o próximo acorde que:
      1. Minimiza a soma total do deslocamento em semitons entre as vozes.
      2. Preserva notas comuns nas mesmas alturas.
      3. Evita saltos melódicos abruptos e cruzamentos desagradáveis.
    
    Args:
        previous_voicing: Lista de números MIDI do acorde anterior.
        next_chord: Próximo acorde a ser executado.
        target_octave_span: Dispersão vertical máxima recomendada.
        
    Returns:
        List[int]: Nova lista de notas MIDI ordenadas com o menor esforço harmônico.
    """
    if not previous_voicing:
        return next_chord.get_midi_notes()

    target_pitches = next_chord.get_pitch_classes()
    num_voices = len(previous_voicing)
    
    # Gera candidatos de alturas para cada pitch class nas vizinhanças do voicing anterior
    center_midi = int(np.mean(previous_voicing))
    candidates_per_pitch: Dict[int, List[int]] = {}

    for pc in target_pitches:
        # Gera opções em oitavas próximas ao centro
        opts = []
        for oct_shift in (-2, -1, 0, 1, 2):
            midi_cand = (4 + oct_shift + 1) * 12 + pc
            # Mantém dentro de uma faixa razoável em torno do centro (±18 semitons)
            if abs(midi_cand - center_midi) <= 24:
                opts.append(midi_cand)
        candidates_per_pitch[pc] = opts

    # Testa as diferentes inversões do próximo acorde (0 até len(target_pitches) - 1)
    # e diferentes transposições de oitavas para encontrar a mínima distância L1
    best_voicing = None
    min_distance = float("inf")

    num_notes = len(target_pitches)
    for inv in range(num_notes):
        trial_chord = next_chord.invert(inv)
        base_notes = trial_chord.get_midi_notes()

        # Ajusta base_notes por shifts de oitavas globais (-24 a +24)
        for oct_offset in (-24, -12, 0, 12, 24):
            shifted = [n + oct_offset for n in base_notes]
            
            # Se o número de vozes for diferente, alinha por interpolação/duplicação da tônica
            if len(shifted) == num_voices:
                dist = sum(abs(s - p) for s, p in zip(shifted, previous_voicing))
                # Penaliza se sair da tessitura média
                range_penalty = abs(np.mean(shifted) - center_midi) * 0.5
                total_cost = dist + range_penalty
                if total_cost < min_distance:
                    min_distance = total_cost
                    best_voicing = shifted
            elif len(shifted) < num_voices:
                # Duplica a nota mais grave (fundamental) para preencher vozes
                expanded = [shifted[0] - 12] + shifted
                if len(expanded) == num_voices:
                    dist = sum(abs(s - p) for s, p in zip(expanded, previous_voicing))
                    total_cost = dist + abs(np.mean(expanded) - center_midi) * 0.5
                    if total_cost < min_distance:
                        min_distance = total_cost
                        best_voicing = expanded
            else:
                # Se o acorde tiver mais notas que vozes anteriores, toma as primeiras
                trimmed = shifted[:num_voices]
                dist = sum(abs(s - p) for s, p in zip(trimmed, previous_voicing))
                total_cost = dist + abs(np.mean(trimmed) - center_midi) * 0.5
                if total_cost < min_distance:
                    min_distance = total_cost
                    best_voicing = trimmed

    if best_voicing is None:
        return next_chord.get_midi_notes()

    best_voicing.sort()
    return best_voicing
