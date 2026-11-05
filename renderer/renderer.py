"""
Motor de Renderização em Streaming por Chunks.
"""
from pathlib import Path
from typing import Callable, Optional, Union
import numpy as np

from core.engine import Engine
from renderer.chunk_writer import ChunkWriter


ProgressCallback = Callable[[int, int, float, float], None]


class Renderer:
    """
    Executa o ciclo de renderização chunk-a-chunk sem retenção de memória global.
    """

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self.context = engine.context

    def render_to_file(
        self,
        output_path: Union[str, Path],
        on_progress: Optional[ProgressCallback] = None
    ) -> float:
        """
        Renderiza o áudio gerado pelo Engine diretamente para um arquivo WAV em disco.
        
        Args:
            output_path: Caminho do arquivo de saída (.wav).
            on_progress: Função de callback opcional (chunk_idx, total_chunks, current_sec, total_sec).
            
        Returns:
            float: Pico máximo absoluto registrado na renderização.
        """
        self.engine.prepare()
        total_chunks = self.context.total_chunks
        total_duration = self.context.duration_sec

        with ChunkWriter(output_path, self.context) as writer:
            for chunk_idx in range(total_chunks):
                chunk = self.engine.render_chunk(chunk_idx)
                writer.write_chunk(chunk)

                if on_progress:
                    current_sec = min((chunk_idx + 1) * self.context.chunk_duration_sec, total_duration)
                    on_progress(chunk_idx + 1, total_chunks, current_sec, total_duration)

            return writer.max_peak
