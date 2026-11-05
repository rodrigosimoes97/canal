"""
Contexto de Renderização do Relaxation Audio Engine.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class RenderContext:
    """
    Mantém os parâmetros imutáveis e compartilhados de uma sessão de renderização.
    """
    sample_rate: int = 44100
    channels: int = 2
    chunk_duration_sec: float = 10.0
    duration_sec: float = 120.0
    seed: int = 42
    profile_name: str = "default"
    internal_dtype: str = "float32"
    export_format: str = "WAV"
    export_subtype: str = "PCM_24"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError(f"Taxa de amostragem inválida: {self.sample_rate}")
        if self.channels not in (1, 2):
            raise ValueError(f"Número de canais não suportado: {self.channels} (suporta 1 ou 2)")
        if self.duration_sec <= 0:
            raise ValueError(f"Duração inválida: {self.duration_sec}")
        if self.chunk_duration_sec <= 0:
            raise ValueError(f"Duração de chunk inválida: {self.chunk_duration_sec}")

    @property
    def total_samples(self) -> int:
        """Total de amostras na renderização inteira."""
        return int(round(self.duration_sec * self.sample_rate))

    @property
    def chunk_samples(self) -> int:
        """Número de amostras contidas em um chunk padrão."""
        return int(round(self.chunk_duration_sec * self.sample_rate))

    @property
    def total_chunks(self) -> int:
        """Quantidade total de chunks para cobrir a duração."""
        return (self.total_samples + self.chunk_samples - 1) // self.chunk_samples

    def sample_to_time(self, sample_idx: int) -> float:
        """Converte o índice da amostra em segundos."""
        return sample_idx / self.sample_rate

    def time_to_sample(self, time_sec: float) -> int:
        """Converte tempo em segundos para índice de amostra."""
        return int(round(time_sec * self.sample_rate))

    def get_chunk_bounds(self, chunk_idx: int) -> tuple[int, int]:
        """
        Retorna (start_sample, end_sample) para o chunk especificado,
        respeitando o limite total_samples.
        """
        start = chunk_idx * self.chunk_samples
        end = min(start + self.chunk_samples, self.total_samples)
        return start, end
