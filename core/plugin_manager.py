"""
Gerenciador de Ciclo de Vida e Registro de Plugins.
"""
from typing import Any, Dict, List, Optional, Type
import numpy as np

from plugins.base import BasePlugin, PluginCategory


class PluginManagerError(Exception):
    """Exceção levantada em caso de falha no registro ou ciclo de vida de plugins."""
    pass


class PluginManager:
    """
    Gerencia o registro de classes de plugins e o ciclo de vida das instâncias ativas.
    """

    def __init__(self) -> None:
        self._registry: Dict[str, Type[BasePlugin]] = {}
        self._active_plugins: Dict[str, BasePlugin] = {}

    def register(self, plugin_class: Type[BasePlugin], identifier: Optional[str] = None) -> None:
        """
        Registra uma classe de plugin no sistema.
        """
        if not issubclass(plugin_class, BasePlugin):
            raise PluginManagerError(
                f"Classe {plugin_class} deve herdar de BasePlugin para ser registrada."
            )
        
        reg_id = identifier or getattr(plugin_class, "PLUGIN_ID", None) or plugin_class.__name__.lower()
        self._registry[reg_id] = plugin_class

    def unregister(self, identifier: str) -> None:
        """Remove o registro de uma classe de plugin."""
        self._registry.pop(identifier, None)

    def is_registered(self, identifier: str) -> bool:
        """Verifica se um identificador está registrado."""
        return identifier in self._registry

    def get_registered_ids(self) -> List[str]:
        """Retorna todos os identificadores de plugins registrados."""
        return sorted(list(self._registry.keys()))

    def create_instance(
        self,
        identifier: str,
        instance_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> BasePlugin:
        """
        Instancia um plugin a partir de seu identificador registrado.
        """
        if identifier not in self._registry:
            raise PluginManagerError(f"Plugin '{identifier}' não está registrado no sistema.")
        
        plugin_class = self._registry[identifier]
        inst_id = instance_id or identifier
        instance = plugin_class(plugin_id=inst_id, config=config or {})
        self._active_plugins[inst_id] = instance
        return instance

    def get_instance(self, instance_id: str) -> Optional[BasePlugin]:
        """Retorna uma instância ativa pelo seu ID."""
        return self._active_plugins.get(instance_id)

    def get_active_plugins(self) -> List[BasePlugin]:
        """Retorna todas as instâncias ativas."""
        return list(self._active_plugins.values())

    def get_plugins_by_category(self, category: PluginCategory) -> List[BasePlugin]:
        """Filtra instâncias ativas pela categoria."""
        return [p for p in self._active_plugins.values() if p.category == category]

    def prepare_all(self, context: Any, random_manager: Any) -> None:
        """
        Prepara todos os plugins ativos atribuindo seu contexto e RNG determinístico isolado.
        """
        for plugin in self._active_plugins.values():
            rng = random_manager.get_plugin_rng(plugin.plugin_id)
            plugin.prepare(context, rng)

    def reset_all(self) -> None:
        """Reinicializa o estado de todos os plugins ativos."""
        for plugin in self._active_plugins.values():
            plugin.reset()

    def teardown_all(self) -> None:
        """Finaliza e libera recursos de todos os plugins ativos."""
        for plugin in self._active_plugins.values():
            plugin.teardown()
        self._active_plugins.clear()
