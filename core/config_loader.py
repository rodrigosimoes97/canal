"""
Carregador e Validador de Configurações YAML.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


class ConfigError(Exception):
    """Exceção levantada quando uma configuração YAML é inválida ou inacessível."""
    pass


class ConfigLoader:
    """
    Responsável por carregar e validar as configurações do motor e dos perfis de áudio.
    """

    @staticmethod
    def load_yaml(file_path: Path | str) -> Dict[str, Any]:
        """Lê e analisa um arquivo YAML."""
        path = Path(file_path)
        if not path.exists():
            raise ConfigError(f"Arquivo de configuração não encontrado: {path}")
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if not isinstance(data, dict):
                raise ConfigError(f"O arquivo {path} deve conter um mapeamento YAML (dicionário).")
            return data
        except yaml.YAMLError as exc:
            raise ConfigError(f"Erro de sintaxe no arquivo YAML {path}: {exc}") from exc
        except Exception as exc:
            raise ConfigError(f"Erro ao ler {path}: {exc}") from exc

    @staticmethod
    def validate_engine_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Valida e aplica defaults à configuração principal do motor."""
        if "engine" not in config or not isinstance(config["engine"], dict):
            raise ConfigError("Configuração do motor deve conter a seção 'engine'.")
        
        eng = config["engine"]
        sample_rate = eng.get("sample_rate", 44100)
        channels = eng.get("channels", 2)
        chunk_sec = eng.get("chunk_duration_sec", 10.0)

        if not isinstance(sample_rate, int) or sample_rate <= 0:
            raise ConfigError(f"'sample_rate' deve ser um inteiro positivo, recebido: {sample_rate}")
        if channels not in (1, 2):
            raise ConfigError(f"'channels' deve ser 1 ou 2, recebido: {channels}")
        if not isinstance(chunk_sec, (int, float)) or chunk_sec <= 0:
            raise ConfigError(f"'chunk_duration_sec' deve ser um número positivo, recebido: {chunk_sec}")

        return config

    @staticmethod
    def validate_profile_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Valida a estrutura de um perfil de áudio (estilo).
        Garante a presença das seções mínimas esperadas.
        """
        required_keys = ["name", "category"]
        for k in required_keys:
            if k not in config:
                raise ConfigError(f"Perfil inválido: campo obrigatório '{k}' ausente.")
        
        if "plugins" in config and not isinstance(config["plugins"], list):
            raise ConfigError("A seção 'plugins' no perfil deve ser uma lista de configurações de plugins.")
            
        return config

    @classmethod
    def list_profiles(cls, profiles_dir: Path | str) -> List[str]:
        """Retorna os nomes de todos os perfis disponíveis na pasta especificada."""
        path = Path(profiles_dir)
        if not path.is_dir():
            return []
        return sorted([p.stem for p in path.glob("*.yaml")])
