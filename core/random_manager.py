"""
Gerenciador Centralizado de Randomização e Reprodutibilidade.
"""
import hashlib
from typing import Dict, Union
import numpy as np


class RandomManager:
    """
    Controla a geração de números pseudo-aleatórios determinísticos.
    
    Cada plugin e subsistema recebe seu próprio `numpy.random.Generator` (PCG64)
    derivado deterministicamente da semente mestre através de hashing criptográfico seguro,
    garantindo que:
      1. Os geradores são totalmente isolados e nunca colidem.
      2. Adicionar ou remover um plugin não corrompe a sequência dos outros plugins.
      3. A mesma semente produzirá exatamente o mesmo resultado bit-a-bit em qualquer ambiente.
    """

    def __init__(self, master_seed: int) -> None:
        self.master_seed: int = int(master_seed)
        self._plugin_generators: Dict[str, np.random.Generator] = {}

    def _derive_seed_int(self, *tokens: Union[str, int]) -> int:
        """
        Deriva um inteiro de 64 bits a partir da seed mestre e de tokens identificadores.
        """
        payload = f"{self.master_seed}:" + ":".join(str(t) for t in tokens)
        digest = hashlib.sha256(payload.encode("utf-8")).digest()
        # Converte os primeiros 8 bytes em um inteiro de 64 bits sem sinal
        return int.from_bytes(digest[:8], byteorder="big", signed=False)

    def get_plugin_rng(self, plugin_id: str, sub_stream: int = 0) -> np.random.Generator:
        """
        Retorna um Generator determinístico e exclusivo para um plugin.
        """
        key = f"{plugin_id}_{sub_stream}"
        if key not in self._plugin_generators:
            derived_seed = self._derive_seed_int("plugin", plugin_id, sub_stream)
            seed_seq = np.random.SeedSequence(derived_seed)
            self._plugin_generators[key] = np.random.default_rng(seed_seq)
        return self._plugin_generators[key]

    def get_time_event_rng(self, plugin_id: str, event_time_key: int) -> np.random.Generator:
        """
        Retorna um Generator específico para um evento pontual no tempo absoluto.
        Isso garante que a decisão de um evento (ex: tom de um sino às 12.0s) seja idêntica
        independente do tamanho do chunk em que ela foi processada.
        """
        derived_seed = self._derive_seed_int("event", plugin_id, event_time_key)
        seed_seq = np.random.SeedSequence(derived_seed)
        return np.random.default_rng(seed_seq)

    def reset(self) -> None:
        """Limpa o cache de geradores, reiniciando todas as sequências a partir da seed mestre."""
        self._plugin_generators.clear()
