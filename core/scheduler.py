"""
Agendador de Eventos para Processamento em Chunks.
"""
from typing import Dict, List, Tuple
from core.timeline import Timeline, TimelineEvent


class Scheduler:
    """
    Coordena o agendamento de eventos temporais e sua projeção nos limites de cada chunk.
    """

    def __init__(self, timeline: Timeline) -> None:
        self.timeline = timeline

    def schedule_chunk(
        self,
        chunk_start: int,
        chunk_end: int
    ) -> List[Tuple[TimelineEvent, int, int, int]]:
        """
        Calcula as fatias de eventos que devem ser processadas no chunk atual.
        
        Args:
            chunk_start: Início do bloco de amostras.
            chunk_end: Fim do bloco de amostras.
            
        Returns:
            Lista de tuplas: (evento, read_start, write_start, length)
        """
        scheduled = []
        events = self.timeline.get_events_overlapping(chunk_start, chunk_end)

        for event in events:
            inter = event.get_intersection(chunk_start, chunk_end)
            if inter is not None:
                read_start, read_length, write_start, write_length = inter
                scheduled.append((event, read_start, write_start, read_length))

        return scheduled
