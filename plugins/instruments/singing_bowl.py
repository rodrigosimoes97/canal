"""
Instrumento: Tigela Tibetana / Singing Bowl (Ressonância Meditativa Profunda).
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np

from plugins.base import BasePlugin, PluginCategory
from music.scale import midi_to_freq


@dataclass
class SingingBowlStrike:
    midi_note: int
    start_sec: float
    duration_sec: float = 18.0
    velocity: float = 0.8
    pan: float = 0.0


class SingingBowl(BasePlugin):
    """
    Sintetizador de Tigela Tibetana (Singing Bowl) com parciais inarmônicas e batimento natural.
    """
    PLUGIN_ID = "singing_bowl"

    def __init__(self, plugin_id: str = "singing_bowl", config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(plugin_id=plugin_id, category=PluginCategory.INSTRUMENT, config=config)
        self.gain = float(self.config.get("gain", 0.45))
        self.sample_rate: int = 44100
        self.strikes: List[SingingBowlStrike] = []

    def prepare(self, context: Any, rng: np.random.Generator) -> None:
        self.sample_rate = context.sample_rate
        self.strikes.clear()

        composer = context.metadata.get("composer")
        # Toques esparsos a cada 30 a 50 segundos
        step_sec = float(self.config.get("interval_sec", 40.0))
        cur_sec = 1.0
        total_sec = context.duration_sec

        while cur_sec < total_sec:
            if composer is not None:
                chord, _, _ = composer.get_current_chord(cur_sec)
                # Tônica do acorde na oitava grave (C2 ou C3)
                pitch = chord.get_midi_notes()[0]
                if pitch > 55:
                    pitch -= 12
            else:
                pitch = 48 # C3

            dur = float(rng.uniform(14.0, 22.0))
            pan = float(rng.uniform(-0.25, 0.25))
            self.strikes.append(SingingBowlStrike(
                midi_note=pitch,
                start_sec=cur_sec,
                duration_sec=dur,
                velocity=float(rng.uniform(0.7, 0.9)),
                pan=pan
            ))
            cur_sec += float(rng.uniform(step_sec * 0.8, step_sec * 1.3))

        self._is_prepared = True

    def add_strike(self, strike: SingingBowlStrike) -> None:
        self.strikes.append(strike)

    def render(self, start_sample: int, num_samples: int) -> np.ndarray:
        if not self.is_active or num_samples <= 0:
            return np.zeros((num_samples, 2), dtype=np.float32)

        chunk_start_s = start_sample / self.sample_rate
        chunk_end_s = (start_sample + num_samples) / self.sample_rate

        out_l = np.zeros(num_samples, dtype=np.float64)
        out_r = np.zeros(num_samples, dtype=np.float64)

        for strike in self.strikes:
            ev_start = strike.start_sec
            ev_end = ev_start + strike.duration_sec

            if ev_end > chunk_start_s and ev_start < chunk_end_s:
                sample_start = max(0, int(round((ev_start - chunk_start_s) * self.sample_rate)))
                sample_end = min(num_samples, int(round((ev_end - chunk_start_s) * self.sample_rate)))

                if sample_end > sample_start:
                    len_seg = sample_end - sample_start
                    t_rel = (np.arange(len_seg, dtype=np.float64) + (start_sample + sample_start) - int(round(ev_start * self.sample_rate))) / self.sample_rate

                    freq = midi_to_freq(strike.midi_note)

                    # Síntese inarmônica da tigela tibetana
                    fund = np.sin(2 * np.pi * freq * t_rel) * np.exp(-0.15 * t_rel)
                    h2 = 0.45 * np.sin(2 * np.pi * (freq * 2.0) * t_rel) * np.exp(-0.25 * t_rel)
                    h3 = 0.22 * np.sin(2 * np.pi * (freq * 3.97) * t_rel) * np.exp(-0.40 * t_rel)
                    h4 = 0.12 * np.sin(2 * np.pi * (freq * 5.43) * t_rel) * np.exp(-0.65 * t_rel)

                    # Batimento de tremor de amplitude (7 Hz característico)
                    shimmer = 1.0 + 0.04 * np.sin(2 * np.pi * 7.0 * t_rel)
                    # Ataque suave de baqueta acolchoada (40 ms)
                    attack = np.clip(t_rel / 0.04, 0.0, 1.0)
                    # Decaimento suave final
                    release_start = strike.duration_sec - 3.0
                    release_mask = t_rel > release_start
                    decay_tail = np.ones(len_seg, dtype=np.float64)
                    decay_tail[release_mask] = np.clip(1.0 - (t_rel[release_mask] - release_start) / 3.0, 0.0, 1.0)

                    wave = (fund + h2 + h3 + h4) * shimmer * attack * decay_tail * (strike.velocity * self.gain)

                    pan_l = 0.5 - (strike.pan * 0.5)
                    pan_r = 0.5 + (strike.pan * 0.5)

                    out_l[sample_start:sample_end] += wave * pan_l
                    out_r[sample_start:sample_end] += wave * pan_r

        return np.column_stack((out_l, out_r)).astype(np.float32)

    def reset(self) -> None:
        pass
