"""
Escritor de Áudio em Chunks para Streaming Contínuo em Disco.
"""
from pathlib import Path
from typing import Optional, Union
import numpy as np
import soundfile as sf

from core.context import RenderContext


class ChunkWriterError(Exception):
    """Exceção levantada quando ocorre erro de I/O na escrita de chunks."""
    pass


class ChunkWriter:
    """
    Grava blocos sucessivos de áudio diretamente em disco sem acumular dados na RAM.
    Garante consumo de memória constante O(1) mesmo para renderizações de 8 a 10 horas.
    """

    def __init__(
        self,
        output_path: Union[str, Path],
        context: RenderContext,
        subtype: Optional[str] = None
    ) -> None:
        self.output_path = Path(output_path)
        self.context = context
        self.subtype = subtype or context.export_subtype
        self._file: Optional[sf.SoundFile] = None
        self.samples_written: int = 0
        self.max_peak: float = 0.0

        # Garante que a pasta pai exista
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def open(self) -> "ChunkWriter":
        """Abre o arquivo para escrita contínua."""
        try:
            self._file = sf.SoundFile(
                str(self.output_path),
                mode="w",
                samplerate=self.context.sample_rate,
                channels=self.context.channels,
                subtype=self.subtype,
                format=self.context.export_format
            )
            self.samples_written = 0
            self.max_peak = 0.0
            return self
        except Exception as exc:
            raise ChunkWriterError(f"Falha ao abrir {self.output_path} para escrita: {exc}") from exc

    def write_chunk(self, chunk: np.ndarray) -> None:
        """
        Escreve um bloco de áudio no arquivo em disco.
        
        Args:
            chunk: Matriz numpy de áudio float32 com formato (num_samples, channels).
        """
        if self._file is None or self._file.closed:
            raise ChunkWriterError("Tentativa de escrita com arquivo fechado.")

        if chunk.ndim != 2 or chunk.shape[1] != self.context.channels:
            raise ChunkWriterError(
                f"Formato de chunk inválido: {chunk.shape}, esperado (*, {self.context.channels})"
            )

        # Atualiza o pico máximo registrado para telemetria/relatório
        peak = float(np.max(np.abs(chunk))) if chunk.size > 0 else 0.0
        if peak > self.max_peak:
            self.max_peak = peak

        try:
            self._file.write(chunk.astype(np.float32))
            self.samples_written += chunk.shape[0]
            self._file.flush()
        except Exception as exc:
            raise ChunkWriterError(f"Erro ao gravar chunk em {self.output_path}: {exc}") from exc

    def close(self) -> None:
        """Fecha o arquivo de áudio e descarrega todos os buffers."""
        if self._file is not None and not self._file.closed:
            self._file.flush()
            self._file.close()
            self._file = None

    def __enter__(self) -> "ChunkWriter":
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    @property
    def written_duration_sec(self) -> float:
        """Duração gravada até o momento em segundos."""
        return self.samples_written / self.context.sample_rate
