"""
Instrumento: Piano Acústico de Feltro (Felt Piano com Inarmonicidade, Uníssono e Ressonância de Corpo).
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np

from plugins.base import BasePlugin, PluginCategory
from music.scale import midi_to_freq


@dataclass
class PianoNoteEvent:
    midi_note: int
    start_sec: float
    duration_sec: float
    velocity: float = 0.6
    pan: float = 0.0


class Piano(BasePlugin):
    """
    Sintetizador físico-acústico de Piano Vertical com feltro (Felt Piano) para relaxamento profundo.
    Implementa inarmonicidade de cordas (stiffness), uníssono triplo com micro-detune acústico,
    decaimento harmônico seletivo e ressonância cálida do corpo do instrumento.
    """
    PLUGIN_ID = "piano"

    def __init__(self, plugin_id: str = "piano", config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(plugin_id=plugin_id, category=PluginCategory.INSTRUMENT, config=config)
        self.gain = float(self.config.get("gain", 0.32))
        self.sparse_mode = bool(self.config.get("sparse_mode", True))
        self.sample_rate: int = 44100
        self.notes: List[PianoNoteEvent] = []

    def prepare(self, context: Any, rng: np.random.Generator) -> None:
        self.sample_rate = context.sample_rate
        self.notes.clear()

        composer = context.metadata.get("composer")
        if composer is not None:
            rhythm = composer.rhythm
            total_beats = rhythm.seconds_to_beat(context.duration_sec)

            if self.sparse_mode:
                # Gestos contemplativos esparsos para relaxamento e sono (notas a cada 20-35s)
                notes = composer.melody.generate_sparse_gestures(
                    start_beat=4.0,
                    duration_beats=total_beats - 4.0,
                    rng=rng,
                    min_interval_beats=18.0,
                    max_interval_beats=36.0,
                    density=float(self.config.get("density", 0.35))
                )
                for n in notes:
                    start_s = rhythm.beat_to_seconds(n.start_beat)
                    # Adiciona rubato natural (micro-deslocamento orgânico de tempo)
                    start_s += float(rng.uniform(-0.15, 0.15))
                    start_s = max(0.5, start_s)
                    dur_s = rhythm.duration_to_seconds(n.duration_beats) + float(rng.uniform(1.0, 3.0))
                    pan = float(rng.uniform(-0.25, 0.25))
                    self.notes.append(PianoNoteEvent(
                        midi_note=n.midi_note,
                        start_sec=start_s,
                        duration_sec=dur_s,
                        velocity=n.velocity,
                        pan=pan
                    ))
            else:
                # Modo melódico clássico com frases estruturadas
                phrase_step_beats = 16.0
                cur_beat = 4.0
                while cur_beat < total_beats:
                    phrase = composer.melody.generate_phrase(
                        start_beat=cur_beat,
                        phrase_length_beats=8.0,
                        rng=rng,
                        density=float(self.config.get("density", 0.35)),
                        prefer_chord_tones=True
                    )
                    for note in phrase:
                        start_s = rhythm.beat_to_seconds(note.start_beat)
                        dur_s = rhythm.duration_to_seconds(note.duration_beats)
                        pan = float(rng.uniform(-0.25, 0.25))
                        self.notes.append(PianoNoteEvent(
                            midi_note=note.midi_note,
                            start_sec=start_s,
                            duration_sec=dur_s,
                            velocity=note.velocity * 0.7,
                            pan=pan
                        ))
                    cur_beat += phrase_step_beats

        self._is_prepared = True

    def add_note(self, note: PianoNoteEvent) -> None:
        """Permite adicionar notas explicitamente via API ou Timeline."""
        self.notes.append(note)

    def render(self, start_sample: int, num_samples: int) -> np.ndarray:
        if not self.is_active or num_samples <= 0:
            return np.zeros((num_samples, 2), dtype=np.float32)

        chunk_start_s = start_sample / self.sample_rate
        chunk_end_s = (start_sample + num_samples) / self.sample_rate

        out_l = np.zeros(num_samples, dtype=np.float64)
        out_r = np.zeros(num_samples, dtype=np.float64)

        # Inarmonicidade típica de corda de piano vertical de feltro
        B = 0.00028

        for ev in self.notes:
            ev_start = ev.start_sec
            # Sustentação longa e ressonância natural da caixa de ressonância
            ev_decay_total = ev.duration_sec + 5.0
            ev_end = ev_start + ev_decay_total

            if ev_end > chunk_start_s and ev_start < chunk_end_s:
                sample_start = max(0, int(round((ev_start - chunk_start_s) * self.sample_rate)))
                sample_end = min(num_samples, int(round((ev_end - chunk_start_s) * self.sample_rate)))

                if sample_end > sample_start:
                    len_seg = sample_end - sample_start
                    t_rel = (np.arange(len_seg, dtype=np.float64) + (start_sample + sample_start) - int(round(ev_start * self.sample_rate))) / self.sample_rate

                    f0 = midi_to_freq(ev.midi_note)

                    # Amortecimento dependente de pitch (notas mais agudas decaem mais rápido)
                    pitch_decay_mult = 1.0 + (ev.midi_note - 60) * 0.015

                    # Síntese inarmônica com 3 cordas em uníssono (batimento estéreo natural)
                    # Corda central, corda esquerda (-detune), corda direita (+detune)
                    detune_hz = 0.32

                    note_sig_l = np.zeros(len_seg, dtype=np.float64)
                    note_sig_r = np.zeros(len_seg, dtype=np.float64)

                    # Harmônicos com inarmonicidade real: f_n = n * f0 * sqrt(1 + B * n^2)
                    harmonic_weights = [1.0, 0.45, 0.18, 0.08, 0.03, 0.01]
                    for n_idx, weight in enumerate(harmonic_weights, start=1):
                        fn = n_idx * f0 * np.sqrt(1.0 + B * (n_idx ** 2))
                        # Decaimento exponencial rápido nos harmônicos agudos (som quente e aveludado)
                        decay_rate = (0.45 + (n_idx - 1) * 0.75) * pitch_decay_mult
                        env_h = np.exp(-decay_rate * t_rel)

                        # Osciladores de uníssono
                        osc_center = np.sin(2 * np.pi * fn * t_rel)
                        osc_left = np.sin(2 * np.pi * (fn - detune_hz) * t_rel)
                        osc_right = np.sin(2 * np.pi * (fn + detune_hz) * t_rel)

                        note_sig_l += weight * env_h * (0.6 * osc_center + 0.4 * osc_left)
                        note_sig_r += weight * env_h * (0.6 * osc_center + 0.4 * osc_right)

                    # Ataque macio de feltro amortecido (20 ms)
                    felt_attack = np.clip(t_rel / 0.020, 0.0, 1.0)
                    felt_attack = np.sin(felt_attack * (np.pi / 2)) # Curva convexa macia

                    # Ressonância cálida do corpo (soundboard boom nos primeiros 80 ms)
                    body_thump = 0.08 * np.sin(2 * np.pi * 92.0 * t_rel) * np.exp(-25.0 * t_rel)

                    wave_l = (note_sig_l * felt_attack + body_thump) * (ev.velocity * self.gain)
                    wave_r = (note_sig_r * felt_attack + body_thump) * (ev.velocity * self.gain)

                    pan_l = np.cos((ev.pan + 1.0) * (np.pi / 4))
                    pan_r = np.sin((ev.pan + 1.0) * (np.pi / 4))

                    out_l[sample_start:sample_end] += wave_l * pan_l
                    out_r[sample_start:sample_end] += wave_r * pan_r

        return np.column_stack((out_l, out_r)).astype(np.float32)

    def reset(self) -> None:
        pass
