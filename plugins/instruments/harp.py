"""
Instrumento: Harpa Dedilhada (Arpejos Etéreos e Ressonância de Corda Livre).
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np

from plugins.base import BasePlugin, PluginCategory
from music.scale import midi_to_freq
from music.rhythm import NoteDuration


@dataclass
class HarpPluckEvent:
    midi_note: int
    start_sec: float
    duration_sec: float
    velocity: float = 0.75
    pan: float = 0.0


class Harp(BasePlugin):
    """
    Sintetizador de Harpa etérea com arpejos melódicos baseados na harmonia.
    """
    PLUGIN_ID = "harp"

    def __init__(self, plugin_id: str = "harp", config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(plugin_id=plugin_id, category=PluginCategory.INSTRUMENT, config=config)
        self.gain = float(self.config.get("gain", 0.35))
        self.sample_rate: int = 44100
        self.events: List[HarpPluckEvent] = []

    def prepare(self, context: Any, rng: np.random.Generator) -> None:
        self.sample_rate = context.sample_rate
        self.events.clear()

        composer = context.metadata.get("composer")
        if composer is not None:
            rhythm = composer.rhythm
            # Agenda arpejos espaçados a cada 12 segundos
            total_sec = context.duration_sec
            cur_sec = 6.0

            while cur_sec < total_sec - 8.0:
                # 65% de chance de arpejo na janela
                if rng.random() < 0.65:
                    chord, _, _ = composer.get_current_chord(cur_sec)
                    start_beat = rhythm.seconds_to_beat(cur_sec)
                    arp_notes = composer.melody.generate_arpeggio(
                        chord=chord,
                        start_beat=start_beat,
                        duration_beats=4.0,
                        pattern=str(rng.choice(["up", "up_down"])),
                        note_duration=NoteDuration.QUARTER,
                        octave_offset=1
                    )
                    for i, anote in enumerate(arp_notes):
                        s_sec = rhythm.beat_to_seconds(anote.start_beat)
                        pan = float(-0.4 + (i / max(1, len(arp_notes) - 1)) * 0.8)
                        self.events.append(HarpPluckEvent(
                            midi_note=anote.midi_note,
                            start_sec=s_sec,
                            duration_sec=3.5,
                            velocity=anote.velocity,
                            pan=pan
                        ))
                cur_sec += float(rng.uniform(10.0, 16.0))

        self._is_prepared = True

    def add_pluck(self, event: HarpPluckEvent) -> None:
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

                    # Síntese de corda de harpa: fundamental pura + harmônicos com decaimento exponencial
                    fund = np.sin(2 * np.pi * freq * t_rel) * np.exp(-0.8 * t_rel)
                    h1 = 0.45 * np.sin(2 * np.pi * (freq * 2.0) * t_rel) * np.exp(-1.4 * t_rel)
                    h2 = 0.20 * np.sin(2 * np.pi * (freq * 3.0) * t_rel) * np.exp(-2.2 * t_rel)
                    h3 = 0.10 * np.sin(2 * np.pi * (freq * 4.0) * t_rel) * np.exp(-3.5 * t_rel)

                    # Ataque suave de polpa de dedo (12 ms)
                    attack = np.clip(t_rel / 0.012, 0.0, 1.0)
                    wave = (fund + h1 + h2 + h3) * attack * (ev.velocity * self.gain)

                    pan_l = 0.5 - (ev.pan * 0.5)
                    pan_r = 0.5 + (ev.pan * 0.5)

                    out_l[sample_start:sample_end] += wave * pan_l
                    out_r[sample_start:sample_end] += wave * pan_r

        return np.column_stack((out_l, out_r)).astype(np.float32)

    def reset(self) -> None:
        pass
