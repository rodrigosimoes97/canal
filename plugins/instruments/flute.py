"""
Instrumento: Flauta de Bambu Meditativa / Shakuhachi (Sopro Orgânico, Turbulência Acústica e Respiração Livre).
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np

from plugins.base import BasePlugin, PluginCategory
from music.scale import midi_to_freq


@dataclass
class FluteNoteEvent:
    midi_note: int
    start_sec: float
    duration_sec: float
    velocity: float = 0.6
    pan: float = 0.0


class Flute(BasePlugin):
    """
    Sintetizador físico-acústico de Flauta de Bambu (Shakuhachi / Bansuri) para meditação.
    Utiliza modelagem de turbulência de embocadura (ruído de sopro passa-faixa), ressonância de
    coluna de ar de tubo aberto, ataque aveludado e micro-flutuação orgânica de respiração.
    """
    PLUGIN_ID = "flute"

    def __init__(self, plugin_id: str = "flute", config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(plugin_id=plugin_id, category=PluginCategory.INSTRUMENT, config=config)
        self.gain = float(self.config.get("gain", 0.28))
        self.sample_rate: int = 44100
        self.notes: List[FluteNoteEvent] = []

    def prepare(self, context: Any, rng: np.random.Generator) -> None:
        self.sample_rate = context.sample_rate
        self.notes.clear()

        composer = context.metadata.get("composer")
        if composer is not None:
            rhythm = composer.rhythm
            cur_sec = float(rng.uniform(10.0, 18.0))
            total_sec = context.duration_sec

            # Frases esparsas contemplativas com longos silêncios meditativos
            while cur_sec < total_sec - 15.0:
                start_beat = rhythm.seconds_to_beat(cur_sec)
                # Seleciona nota harmônica consonante do acorde ativo
                chord, _, _ = composer.get_current_chord(cur_sec)
                notes = chord.get_midi_notes()
                # Oitava aguda expressiva para a flauta (D4 a A5)
                flute_pitch = notes[0] + 24 if notes[0] < 60 else notes[0] + 12

                dur_s = float(rng.uniform(4.5, 7.5))
                pan = float(rng.uniform(-0.25, 0.25))

                self.notes.append(FluteNoteEvent(
                    midi_note=flute_pitch,
                    start_sec=cur_sec,
                    duration_sec=dur_s,
                    velocity=float(rng.uniform(0.40, 0.60)),
                    pan=pan
                ))

                # Ocasionalmente adiciona uma segunda nota de resolução 3 segundos depois
                if rng.random() < 0.35 and (cur_sec + dur_s + 4.0) < total_sec:
                    next_pitch = flute_pitch - int(rng.choice([2, 4, 5]))
                    self.notes.append(FluteNoteEvent(
                        midi_note=next_pitch,
                        start_sec=cur_sec + float(rng.uniform(3.5, 4.5)),
                        duration_sec=float(rng.uniform(4.0, 6.0)),
                        velocity=float(rng.uniform(0.35, 0.50)),
                        pan=pan * 0.8
                    ))
                    cur_sec += float(rng.uniform(30.0, 50.0))
                else:
                    cur_sec += float(rng.uniform(22.0, 42.0))

        self._is_prepared = True

    def add_note(self, event: FluteNoteEvent) -> None:
        self.notes.append(event)

    def render(self, start_sample: int, num_samples: int) -> np.ndarray:
        if not self.is_active or num_samples <= 0:
            return np.zeros((num_samples, 2), dtype=np.float32)

        chunk_start_s = start_sample / self.sample_rate
        chunk_end_s = (start_sample + num_samples) / self.sample_rate

        out_l = np.zeros(num_samples, dtype=np.float64)
        out_r = np.zeros(num_samples, dtype=np.float64)

        for ev in self.notes:
            ev_start = ev.start_sec
            ev_end = ev_start + ev.duration_sec

            if ev_end > chunk_start_s and ev_start < chunk_end_s:
                sample_start = max(0, int(round((ev_start - chunk_start_s) * self.sample_rate)))
                sample_end = min(num_samples, int(round((ev_end - chunk_start_s) * self.sample_rate)))

                if sample_end > sample_start:
                    len_seg = sample_end - sample_start
                    t_rel = (np.arange(len_seg, dtype=np.float64) + (start_sample + sample_start) - int(round(ev_start * self.sample_rate))) / self.sample_rate

                    f0 = midi_to_freq(ev.midi_note)

                    # Micro-flutuação orgânica de respiração humana lenta (sem vibrato mecânico)
                    human_breath = 1.0 + 0.0012 * np.sin(2 * np.pi * 0.35 * t_rel)

                    # Queda de pitch expressiva no release final (típica do sopro de bambu)
                    release_start = max(0.0, ev.duration_sec - 1.2)
                    pitch_bend = np.where(t_rel > release_start, 1.0 - 0.015 * ((t_rel - release_start) / 1.2) ** 2, 1.0)
                    freq_inst = f0 * human_breath * pitch_bend

                    # Parciais da coluna de ar de tubo de madeira aberto (predomínio de harmônicos ímpares)
                    fundamental = np.sin(2 * np.pi * freq_inst * t_rel)
                    h2 = 0.22 * np.sin(2 * np.pi * (freq_inst * 2.0) * t_rel + 0.4)
                    h3 = 0.12 * np.sin(2 * np.pi * (freq_inst * 3.0) * t_rel + 0.9)
                    h4 = 0.03 * np.sin(2 * np.pi * (freq_inst * 4.0) * t_rel)
                    pipe_tone = fundamental + h2 + h3 + h4

                    # Turbulência acústica do sopro (breath noise sintetizado organicamente)
                    # Modulação com ruído de ar centrado na frequência da nota
                    phase_noise = np.sin(2 * np.pi * (freq_inst * 1.002) * t_rel) * np.cos(2 * np.pi * 18.0 * t_rel)
                    breath_noise = 0.14 * phase_noise * np.sin(2 * np.pi * freq_inst * t_rel)

                    # Envelope expressivo de ar: sopro suave nos primeiros 350 ms, corpo pleno e release aéreo
                    attack_s = 0.40
                    release_s = 1.2
                    attack_env = np.clip(t_rel / attack_s, 0.0, 1.0)
                    attack_env = np.sin(attack_env * (np.pi / 2)) # Subida suave senoidal

                    release_env = np.clip(1.0 - (t_rel - release_start) / release_s, 0.0, 1.0) if release_start > 0 else np.ones(len_seg)
                    overall_env = attack_env * release_env

                    wave = (pipe_tone + breath_noise) * overall_env * (ev.velocity * self.gain)

                    pan_l = np.cos((ev.pan + 1.0) * (np.pi / 4))
                    pan_r = np.sin((ev.pan + 1.0) * (np.pi / 4))

                    out_l[sample_start:sample_end] += wave * pan_l
                    out_r[sample_start:sample_end] += wave * pan_r

        return np.column_stack((out_l, out_r)).astype(np.float32)

    def reset(self) -> None:
        pass
