"""
Instrumento: Sinos de Vento e Cristais (Brilho e Espacialidade Metálica).
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np

from plugins.base import BasePlugin, PluginCategory
from music.scale import midi_to_freq


@dataclass
class BellChimeEvent:
    midi_note: int
    start_sec: float
    duration_sec: float
    velocity: float = 0.7
    pan: float = 0.0


class Bells(BasePlugin):
    """
    Sintetizador de Sinos Metálicos e Wind Chimes com parciais inarmônicas e panning estéreo amplo.
    """
    PLUGIN_ID = "bells"

    def __init__(self, plugin_id: str = "bells", config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(plugin_id=plugin_id, category=PluginCategory.INSTRUMENT, config=config)
        self.gain = float(self.config.get("gain", 0.3))
        self.sample_rate: int = 44100
        self.chimes: List[BellChimeEvent] = []

    def prepare(self, context: Any, rng: np.random.Generator) -> None:
        self.sample_rate = context.sample_rate
        self.chimes.clear()

        composer = context.metadata.get("composer")
        cur_sec = 8.0
        total_sec = context.duration_sec

        while cur_sec < total_sec - 5.0:
            # 55% de chance de soar um sino na janela
            if rng.random() < 0.55:
                if composer is not None:
                    chord, _, _ = composer.get_current_chord(cur_sec)
                    # Escolhe uma nota do acorde em oitava alta (oitava 6 ou 7)
                    base_pitch = int(rng.choice(chord.get_midi_notes()))
                    pitch = base_pitch + 24 if base_pitch < 72 else base_pitch + 12
                else:
                    pitch = int(rng.choice([72, 76, 79, 83, 84])) # C5 a C6

                dur = float(rng.uniform(6.0, 11.0))
                pan = float(rng.uniform(-0.75, 0.75)) # Amplo estéreo
                self.chimes.append(BellChimeEvent(
                    midi_note=pitch,
                    start_sec=cur_sec,
                    duration_sec=dur,
                    velocity=float(rng.uniform(0.55, 0.8)),
                    pan=pan
                ))
            cur_sec += float(rng.uniform(12.0, 20.0))

        self._is_prepared = True

    def add_chime(self, event: BellChimeEvent) -> None:
        self.chimes.append(event)

    def render(self, start_sample: int, num_samples: int) -> np.ndarray:
        if not self.is_active or num_samples <= 0:
            return np.zeros((num_samples, 2), dtype=np.float32)

        chunk_start_s = start_sample / self.sample_rate
        chunk_end_s = (start_sample + num_samples) / self.sample_rate

        out_l = np.zeros(num_samples, dtype=np.float64)
        out_r = np.zeros(num_samples, dtype=np.float64)

        for chime in self.chimes:
            ev_start = chime.start_sec
            ev_end = ev_start + chime.duration_sec

            if ev_end > chunk_start_s and ev_start < chunk_end_s:
                sample_start = max(0, int(round((ev_start - chunk_start_s) * self.sample_rate)))
                sample_end = min(num_samples, int(round((ev_end - chunk_start_s) * self.sample_rate)))

                if sample_end > sample_start:
                    len_seg = sample_end - sample_start
                    t_rel = (np.arange(len_seg, dtype=np.float64) + (start_sample + sample_start) - int(round(ev_start * self.sample_rate))) / self.sample_rate

                    freq = midi_to_freq(chime.midi_note)

                    # Síntese inarmônica metálica pura
                    fund = np.sin(2 * np.pi * freq * t_rel) * np.exp(-0.6 * t_rel)
                    h2 = 0.35 * np.sin(2 * np.pi * (freq * 2.52) * t_rel) * np.exp(-1.2 * t_rel)
                    h3 = 0.18 * np.sin(2 * np.pi * (freq * 4.10) * t_rel) * np.exp(-2.0 * t_rel)
                    h4 = 0.08 * np.sin(2 * np.pi * (freq * 6.30) * t_rel) * np.exp(-3.5 * t_rel)

                    # Ataque rápido sem clique (4 ms)
                    attack = np.clip(t_rel / 0.004, 0.0, 1.0)
                    wave = (fund + h2 + h3 + h4) * attack * (chime.velocity * self.gain)

                    pan_l = 0.5 - (chime.pan * 0.5)
                    pan_r = 0.5 + (chime.pan * 0.5)

                    out_l[sample_start:sample_end] += wave * pan_l
                    out_r[sample_start:sample_end] += wave * pan_r

        return np.column_stack((out_l, out_r)).astype(np.float32)

    def reset(self) -> None:
        pass
