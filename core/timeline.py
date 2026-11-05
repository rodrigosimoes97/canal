"""
Linha do Tempo e Gestão de Eventos Absolutos.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TimelineEvent:
    """
    Representa um evento sonoro posicionado na linha do tempo absoluta.
    """
    event_id: str
    plugin_id: str
    start_sample: int
    duration_samples: int
    data: Dict[str, Any] = field(default_factory=dict)
    tag: str = ""

    @property
    def end_sample(self) -> int:
        """Amostra de término do evento."""
        return self.start_sample + self.duration_samples

    def overlaps(self, start_sample: int, end_sample: int) -> bool:
        """Verifica se este evento tem sobreposição com a janela especificada."""
        return self.end_sample > start_sample and self.start_sample < end_sample

    def get_intersection(self, chunk_start: int, chunk_end: int) -> Optional[tuple[int, int, int, int]]:
        """
        Calcula os índices relativos de leitura do evento e escrita no chunk.
        
        Returns:
            tuple (read_start, read_length, write_start, write_length) ou None se não houver sobreposição.
        """
        if not self.overlaps(chunk_start, chunk_end):
            return None
        
        overlap_start = max(self.start_sample, chunk_start)
        overlap_end = min(self.end_sample, chunk_end)
        length = overlap_end - overlap_start

        read_start = overlap_start - self.start_sample
        write_start = overlap_start - chunk_start

        return read_start, length, write_start, length


class Timeline:
    """
    Armazena e indexa todos os eventos da renderização.
    """

    def __init__(self) -> None:
        self._events: List[TimelineEvent] = []

    def add_event(self, event: TimelineEvent) -> None:
        """Adiciona um evento à linha do tempo."""
        self._events.append(event)

    def get_events_overlapping(self, start_sample: int, end_sample: int) -> List[TimelineEvent]:
        """Retorna todos os eventos que possuem interseção com o intervalo amostral."""
        return [ev for ev in self._events if ev.overlaps(start_sample, end_sample)]

    def get_events_for_plugin(self, plugin_id: str) -> List[TimelineEvent]:
        """Filtra eventos vinculados a um plugin específico."""
        return [ev for ev in self._events if ev.plugin_id == plugin_id]

    def clear(self) -> None:
        """Limpa todos os eventos."""
        self._events.clear()

    @property
    def total_events(self) -> int:
        return len(self._events)
