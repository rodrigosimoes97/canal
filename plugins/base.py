"""
Contrato Base e Interfaces para o Sistema de Plugins do Relaxation Audio Engine.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional
import numpy as np


class PluginCategory(str, Enum):
    """Categorias oficiais de plugins suportadas pelo motor."""
    INSTRUMENT = "instrument"
    AMBIENT = "ambient"
    TEXTURE = "texture"
    EFFECT = "effect"
    MASTER = "master"


@dataclass
class PluginMetadata:
    """Metadados descritivos de um plugin."""
    plugin_id: str
    name: str
    category: PluginCategory
    description: str = ""
    version: str = "1.0.0"
    author: str = "Relaxation Audio Engine Team"


class BasePlugin(ABC):
    """
    Contrato base obrigatório para todos os plugins de áudio.
    
    Garante que qualquer instrumento, ambiência, textura ou efeito possa
    ser instanciado, configurado, preparado e renderizado de maneira
    uniforme e desacoplada pelo Core Engine.
    """

    def __init__(
        self,
        plugin_id: str,
        category: PluginCategory,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        self.plugin_id = plugin_id
        self.category = category
        self.config: Dict[str, Any] = config or {}
        self.is_active: bool = True
        self._is_prepared: bool = False

    @property
    def has_state(self) -> bool:
        """Indica se o plugin mantém estado interno entre chunks contíguos."""
        return False

    def set_active(self, active: bool) -> None:
        """Ativa ou desativa a geração de áudio deste plugin."""
        self.is_active = active

    @abstractmethod
    def prepare(self, context: Any, rng: np.random.Generator) -> None:
        """
        Prepara o plugin antes da renderização.
        
        Args:
            context: Contexto de renderização contendo taxa de amostragem,
                     duração total, timeline, etc.
            rng: Gerador determinístico de números pseudo-aleatórios isolado para este plugin.
        """
        pass

    @abstractmethod
    def render(self, start_sample: int, num_samples: int) -> np.ndarray:
        """
        Renderiza um bloco contínuo de áudio no intervalo [start_sample, start_sample + num_samples).
        
        Args:
            start_sample: Índice absoluto da amostra inicial na linha do tempo.
            num_samples: Quantidade de amostras a gerar para o bloco.
            
        Returns:
            np.ndarray: Matriz de áudio float32 com formato (num_samples, 2) estéreo.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Reinicializa o estado interno do plugin para as condições iniciais
        sem alterar suas configurações.
        """
        pass

    def teardown(self) -> None:
        """Libera recursos externos se houver (arquivos abertos, buffers de cache)."""
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} id='{self.plugin_id}' category='{self.category.value}' active={self.is_active}>"
