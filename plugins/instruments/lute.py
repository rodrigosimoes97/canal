"""
Instrumento: Alaúde Medieval (Corda Pinçada de Madeira e Atmosfera de Taverna).
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np

from plugins.base import BasePlugin, PluginCategory
from music.scale import midi_to_freq


@dataclass
class LuteNoteEvent:
    midi_note: int
    start_sec: float
    duration_sec: float
    velocity: float = 0.75
    pan: float = -0.2


class Lute(BasePlugin):
    """
    Sintetizador de Alaúde com pares de cordas em coro e ressonância de corpo acústico.
    """
    PLUGIN_ID = "lute"

    def __init__(self, plugin_id: str = "lute", config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(plugin_id=plugin_id, category=PluginCategory.INSTRUMENT, config=config)
        self.gain = float(self.config.get("gain", 0.35))
        self.sample_rate: int = 44100
        self.events: List[LuteNoteEvent] = []

    def prepare(self, context: Any, rng: np.random.Generator) -> None:
        self.sample_rate = context.sample_rate
        self.events.clear()

        composer = context.metadata.get("composer")
        if composer is not None:
            rhythm = composer.rhythm
            cur_sec = 4.0
            total_sec = context.duration_sec

            while cur_sec < total_sec - 5.0:
                if rng.random() < 0.6:
                    start_beat = rhythm.seconds_to_beat(cur_sec)
                    phrase = composer.melody.generate_phrase(
                        start_beat=start_beat,
                        phrase_length_beats=4.0,
                        rng=rng,
                        density=0.4,
                        prefer_chord_tones=True
                    )
                    for note in phrase:
                        s_sec = rhythm.beat_to_seconds(note.start_beat)
                        dur_s = rhythm.duration_to_seconds(note.duration_beats)
                        self.events.append(LuteNoteEvent(
                            midi_note=note.midi_note - 12, # Oitava média de alaúde
                            start_sec=s_sec,
                            duration_sec=dur_s + 1.2,
                            velocity=note.velocity,
                            pan=float(rng.uniform(-0.35, 0.1))
                        ))
                cur_sec += float(rng.uniform(6.0, 10.0))

        self._is_prepared = True

    def add_note(self, event: LuteNoteEvent) -> None:
        self.events.append(event)

    def render(self, start_sample: int, num_samples: int) -> np.ndarray:
        if not self.is_active or num_samples <= 0:
            return np.zeros((num_samples, 2), dtype=np.float32)

        chunk_start_s = start_sample / self.sample_rate
        chunk_end_s = (start_sample + num_samples) / self.sample_rate

        out_l = np.zeros(num_samples, dtype=np.float64)
        out_r = np.zeros(num_samples, dtype=np.float64)

        for ev in self.events:
            ev_start = ev.start_sec
            ev_end = ev_start + ev.duration_sec

            if ev_end > chunk_start_s and ev_start < chunk_end_s:
                sample_start = max(0, int(round((ev_start - chunk_start_s) * self.sample_rate)))
                sample_end = min(num_samples, int(round((ev_end - chunk_start_s) * self.sample_rate)))

                if sample_end > sample_start:
                    len_seg = sample_end - sample_start
                    t_rel = (np.arange(len_seg, dtype=np.float64) + (start_sample + sample_start) - int(round(ev_start * self.sample_rate))) / self.sample_rate

                    freq = midi_to_freq(ev.midi_note)

                    # Síntese de Alaúde: par de cordas (chorus leve de 0.2%) + parciais ímpares de madeira
                    carrier1 = np.sin(2 * np.pi * freq * t_rel)
                    carrier2 = np.sin(2 * np.pi * (freq * 1.003) * t_rel) * 0.4 # Curso duplo desafinado
                    wood_h3 = -0.15 * np.sin(2 * np.pi * (3.0 * freq) * t_rel)

                    tremolo = 1.0 + 0.1 * np.sin(2 * np.pi * 4.0 * t_rel)
                    env = np.exp(-1.8 * t_rel)
                    attack = np.clip(t_rel / 0.008, 0.0, 1.0)

                    wave = (carrier1 * 0.6 + carrier2 + wood_h3) * env * attack * tremolo * (ev.velocity * self.gain)

                    pan_l = 0.5 - (ev.pan * 0.5)
                    pan_r = 0.5 + (ev.pan * 0.5)

                    out_l[sample_start:sample_end] += wave * pan_l
                    out_r[sample_start:sample_end] += wave * pan_r

        return np.column_stack((out_l, out_r)).astype(np.float32)

    def reset(self) -> None:
        pass
