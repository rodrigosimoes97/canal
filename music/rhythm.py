"""
Métrica, Divisões Rítmicas, Andamento (BPM) e Temporalidade Musical.
"""
from enum import Enum
from typing import List, Union


class NoteDuration(float, Enum):
    """Durações relativas em tempos (beats). Em compasso quaternário (4/4):"""
    DOUBLE_WHOLE = 8.0     # Breve (2 compassos)
    WHOLE = 4.0            # Semibreve (1 compasso)
    DOTTED_HALF = 3.0      # Mínima pontuada
    HALF = 2.0             # Mínima
    DOTTED_QUARTER = 1.5   # Semínima pontuada
    QUARTER = 1.0          # Semínima (1 beat)
    EIGHTH = 0.5           # Colcheia (meio beat)
    TRIPLET_EIGHTH = 1.0 / 3.0  # Tercina de colcheia
    SIXTEENTH = 0.25       # Semicolcheia


class RhythmEngine:
    """
    Gerencia conversões entre tempos musicais (beats), compassos e segundos físicos.
    """

    def __init__(self, bpm: float = 60.0, beats_per_bar: int = 4) -> None:
        if bpm <= 0:
            raise ValueError(f"BPM inválido: {bpm}")
        self.bpm = float(bpm)
        self.beats_per_bar = int(beats_per_bar)

    @property
    def seconds_per_beat(self) -> float:
        """Duração de um beat em segundos."""
        return 60.0 / self.bpm

    @property
    def seconds_per_bar(self) -> float:
        """Duração de um compasso completo em segundos."""
        return self.seconds_per_beat * self.beats_per_bar

    def beat_to_seconds(self, beat: float) -> float:
        """Converte um número de beat em tempo em segundos."""
        return float(beat * self.seconds_per_beat)

    def seconds_to_beat(self, time_sec: float) -> float:
        """Converte tempo em segundos para o beat musical correspondente."""
        return float(time_sec / self.seconds_per_beat)

    def duration_to_seconds(self, duration: Union[NoteDuration, float]) -> float:
        """Converte uma duração relativa em tempos para segundos."""
        beats = float(duration)
        return self.beat_to_seconds(beats)

    def quantize_to_beat(self, beat: float, grid: float = 0.25) -> float:
        """Quantiza um beat para o grid rítmico mais próximo."""
        return round(beat / grid) * grid
