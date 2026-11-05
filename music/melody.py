"""
Geração Algorítmica de Melodias Estruturadas, Motivos, Arpejos e Fraseado.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np

from music.scale import Scale, midi_to_freq
from music.harmony import Chord
from music.progression import ProgressionEngine
from music.rhythm import RhythmEngine, NoteDuration


@dataclass
class MelodicNote:
    """Representa uma nota melódica na linha temporal musical."""
    midi_note: int
    start_beat: float
    duration_beats: float
    velocity: float = 0.8
    is_rest: bool = False

    @property
    def end_beat(self) -> float:
        return self.start_beat + self.duration_beats

    def get_frequency(self, a4_freq: float = 440.0) -> float:
        return midi_to_freq(self.midi_note, a4_freq)


class MelodyEngine:
    """
    Gera frases melódicas coerentes com a harmonia ativa, respeitando
    princípios de forma (motivo, variação, notas de repouso, arpejos e silêncios).
    """

    def __init__(
        self,
        scale: Scale,
        progression: ProgressionEngine,
        rhythm: RhythmEngine,
        base_octave: int = 5,
        octave_span: int = 1
    ) -> None:
        self.scale = scale
        self.progression = progression
        self.rhythm = rhythm
        self.base_octave = base_octave
        self.octave_span = octave_span

        # Notas disponíveis na tessitura melódica
        self.available_midi = self.scale.get_scale_midi_range(
            start_octave=self.base_octave,
            num_octaves=self.octave_span + 1
        )

    def generate_arpeggio(
        self,
        chord: Chord,
        start_beat: float,
        duration_beats: float,
        pattern: str = "up",
        note_duration: NoteDuration = NoteDuration.EIGHTH,
        octave_offset: int = 0
    ) -> List[MelodicNote]:
        """
        Gera um arpejo fluído sobre as notas do acorde especificado.
        
        Args:
            pattern: 'up', 'down', 'up_down'
        """
        notes = chord.get_midi_notes()
        if octave_offset != 0:
            notes = [n + (octave_offset * 12) for n in notes]

        if pattern == "down":
            seq = sorted(notes, reverse=True)
        elif pattern == "up_down":
            seq = notes + sorted(notes[1:-1], reverse=True)
        else: # "up"
            seq = sorted(notes)

        arpeggiated: List[MelodicNote] = []
        cur_beat = start_beat
        step = float(note_duration)
        idx = 0

        while (cur_beat + step) <= (start_beat + duration_beats):
            pitch = seq[idx % len(seq)]
            arpeggiated.append(MelodicNote(
                midi_note=pitch,
                start_beat=cur_beat,
                duration_beats=step * 0.95, # Leve staccato natural entre notas
                velocity=0.75
            ))
            cur_beat += step
            idx += 1

        return arpeggiated

    def generate_phrase(
        self,
        start_beat: float,
        phrase_length_beats: float,
        rng: np.random.Generator,
        density: float = 0.5,
        prefer_chord_tones: bool = True
    ) -> List[MelodicNote]:
        """
        Gera uma frase melódica com motivo temático, variação e nota de repouso final.
        
        Args:
            start_beat: Beat inicial da frase.
            phrase_length_beats: Extensão total da frase em tempos musicais (ex: 8 ou 16).
            rng: Gerador aleatório determinístico do instrumento.
            density: Frequência de notas (0.1 = muito espaçado/zen, 1.0 = contínuo).
            prefer_chord_tones: Força notas coincidentes com a tríade do acorde corrente.
        """
        melody: List[MelodicNote] = []
        cur_beat = start_beat
        end_beat = start_beat + phrase_length_beats

        # Cria um motivo semente de 2 a 3 notas (intervalos em graus da escala)
        motif_steps = [0, int(rng.choice([1, 2, -1, -2])), int(rng.choice([0, 2, 4, -1]))]
        motif_rhythm = [NoteDuration.HALF, NoteDuration.QUARTER, NoteDuration.HALF]

        step_idx = 0

        while cur_beat < end_beat:
            # Chance de pausa/silêncio expressivo baseado na densidade
            if rng.random() > density:
                rest_dur = float(rng.choice([NoteDuration.QUARTER, NoteDuration.HALF, NoteDuration.WHOLE]))
                cur_beat += min(rest_dur, end_beat - cur_beat)
                continue

            active_chord, _ = self.progression.get_chord_at_beat(cur_beat)
            chord_pitches = active_chord.get_pitch_classes()

            # Se estiver nos últimos 2 tempos da frase, busca nota de repouso (resolução)
            is_resolution = (end_beat - cur_beat) <= 3.0
            
            if is_resolution or (prefer_chord_tones and rng.random() < 0.7):
                # Filtra as notas da escala que também pertencem ao acorde
                chord_candidates = [m for m in self.available_midi if (m % 12) in chord_pitches]
                if chord_candidates:
                    chosen_pitch = int(rng.choice(chord_candidates))
                else:
                    chosen_pitch = int(rng.choice(self.available_midi))
                dur = float(NoteDuration.WHOLE if is_resolution else NoteDuration.HALF)
            else:
                # Nota melódica guiada pelo motivo com variação
                base_idx = len(self.available_midi) // 2
                delta = motif_steps[step_idx % len(motif_steps)]
                target_idx = int(np.clip(base_idx + delta, 0, len(self.available_midi) - 1))
                chosen_pitch = self.available_midi[target_idx]
                dur = float(motif_rhythm[step_idx % len(motif_rhythm)])

            # Ajusta para não ultrapassar a frase
            actual_dur = min(dur, end_beat - cur_beat)
            if actual_dur > 0.1:
                velocity = float(rng.uniform(0.65, 0.85))
                melody.append(MelodicNote(
                    midi_note=chosen_pitch,
                    start_beat=cur_beat,
                    duration_beats=actual_dur * 0.95,
                    velocity=velocity
                ))
            
            cur_beat += actual_dur
            step_idx += 1

        return melody

    def generate_sparse_gestures(
        self,
        start_beat: float,
        duration_beats: float,
        rng: np.random.Generator,
        min_interval_beats: float = 16.0,
        max_interval_beats: float = 32.0,
        density: float = 0.3
    ) -> List[MelodicNote]:
        """
        Gera gestos musicais esparsos, meditativos e contemplativos (1 nota longa ou díade suave),
        com amplos intervalos de silêncio para integrar-se pacificamente à natureza.
        """
        gestures: List[MelodicNote] = []
        cur_beat = start_beat + float(rng.uniform(2.0, 6.0))
        end_beat = start_beat + duration_beats

        while cur_beat < end_beat - 4.0:
            if rng.random() <= density:
                active_chord, _ = self.progression.get_chord_at_beat(cur_beat)
                chord_pitches = active_chord.get_pitch_classes()
                candidates = [m for m in self.available_midi if (m % 12) in chord_pitches]
                if not candidates:
                    candidates = self.available_midi

                pitch = int(rng.choice(candidates))
                note_dur_beats = float(rng.choice([3.0, 4.0, 6.0, 8.0]))
                vel = float(rng.uniform(0.35, 0.58)) # Toque pianíssimo aveludado

                gestures.append(MelodicNote(
                    midi_note=pitch,
                    start_beat=cur_beat,
                    duration_beats=min(note_dur_beats, end_beat - cur_beat),
                    velocity=vel
                ))

                # Chance de 30% de uma segunda nota em harmonia (díade suave)
                if rng.random() < 0.30 and (cur_beat + 3.0) < end_beat:
                    second_candidates = [m for m in candidates if m != pitch]
                    if second_candidates:
                        sec_pitch = int(rng.choice(second_candidates))
                        gestures.append(MelodicNote(
                            midi_note=sec_pitch,
                            start_beat=cur_beat + float(rng.uniform(1.0, 2.5)),
                            duration_beats=min(note_dur_beats * 0.8, end_beat - cur_beat),
                            velocity=vel * float(rng.uniform(0.7, 0.9))
                        ))

            cur_beat += float(rng.uniform(min_interval_beats, max_interval_beats))

        return gestures

