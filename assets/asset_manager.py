"""
Audio Asset Manager: Decodificação, Resampling Polifásico, Cache e Looping Contínuo com Crossfade.
"""
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


class AssetError(Exception):
    """Exceção levantada para erros de carregamento ou processamento de assets de áudio."""
    pass


class AudioAssetManager:
    """
    Gerenciador centralizado de gravações externas (MP3, WAV, FLAC, OGG).
    
    Características:
      - Resampling polifásico exato via scipy.signal.resample_poly para 44.1 kHz.
      - Conversão automática mono -> estéreo.
      - Looping contínuo sem emendas utilizando crossfade circular pré-calculado.
      - Cache em memória compartilhado e reutilizável entre plugins.
      - Preservação estrita dos arquivos de áudio originais.
    """

    def __init__(
        self,
        assets_dir: Union[str, Path] = "audios",
        target_sample_rate: int = 44100
    ) -> None:
        self.assets_dir = Path(assets_dir)
        self.target_sample_rate = int(target_sample_rate)

        # Cache de áudios brutos resampleados para 44.1 kHz estéreo float32
        self._raw_cache: Dict[str, np.ndarray] = {}
        # Cache de buffers de loop com crossfade aplicado
        self._loop_cache: Dict[str, np.ndarray] = {}
        # Índice de arquivos disponíveis
        self._file_index: Dict[str, Path] = {}

        self._index_files()

    def _index_files(self) -> None:
        """Localiza todos os arquivos de áudio suportados na pasta de assets."""
        if not self.assets_dir.exists():
            return
        
        valid_exts = {".mp3", ".wav", ".ogg", ".flac"}
        for f in self.assets_dir.iterdir():
            if f.is_file() and f.suffix.lower() in valid_exts:
                # Registra pelo nome com extensão e também pelo radical (sem extensão)
                self._file_index[f.name.lower()] = f
                self._file_index[f.stem.lower()] = f

    def list_available_assets(self) -> List[str]:
        """Retorna os nomes dos arquivos encontrados na pasta de assets."""
        return sorted(list(set(p.name for p in self._file_index.values())))

    def has_asset(self, name: str) -> bool:
        """Verifica se um determinado asset existe no índice."""
        return name.lower().strip() in self._file_index

    def load_asset(self, name: str) -> np.ndarray:
        """
        Carrega um arquivo de áudio, convertendo para target_sample_rate e estéreo float32.
        Utiliza cache para evitar reprocessamento.
        """
        key = name.lower().strip()
        if key in self._raw_cache:
            return self._raw_cache[key]

        if key not in self._file_index:
            raise AssetError(f"Asset '{name}' não encontrado em {self.assets_dir}")

        path = self._file_index[key]

        try:
            data, fs = sf.read(str(path), dtype="float32")
        except Exception as exc:
            raise AssetError(f"Falha ao ler arquivo de áudio {path}: {exc}") from exc

        # Garante 2 canais (estéreo)
        if data.ndim == 1:
            # Mono para estéreo (duplicação de canal)
            data = np.column_stack((data, data))
        elif data.shape[1] == 1:
            data = np.column_stack((data[:, 0], data[:, 0]))

        # Resampling polifásico se a taxa for diferente da taxa alvo
        if fs != self.target_sample_rate:
            gcd = math.gcd(int(fs), self.target_sample_rate)
            up = self.target_sample_rate // gcd
            down = int(fs) // gcd
            # resample_poly aplica filtro FIR anti-aliasing de fase linear
            data = resample_poly(data, up=up, down=down, axis=0)

        # Armazena em float32 contíguo
        processed = np.ascontiguousarray(data, dtype=np.float32)
        self._raw_cache[key] = processed
        return processed

    def get_loop_buffer(self, name: str, crossfade_sec: float = 2.0) -> np.ndarray:
        """
        Prepara um buffer cíclico contínuo aplicando crossfade entre o fim e o início do áudio.
        
        Isso cria um loop fechado sem descontinuidades onde a amostra final emenda perfeitamente
        com a amostra inicial, permitindo indexação circular direta O(1).
        """
        key = f"{name.lower().strip()}_xfade_{crossfade_sec}"
        if key in self._loop_cache:
            return self._loop_cache[key]

        audio = self.load_asset(name)
        total_samples = audio.shape[0]
        xfade_samples = int(round(crossfade_sec * self.target_sample_rate))

        # Se o áudio for mais curto que o dobro do crossfade, reduz o tamanho do crossfade
        if total_samples <= xfade_samples * 2:
            xfade_samples = total_samples // 4

        if xfade_samples <= 0:
            self._loop_cache[key] = audio
            return audio

        # Cria a emenda circular:
        # Pega a cauda de tamanho xfade_samples e sobrepõe no início via fade cruzado com curva equal-power
        loop_len = total_samples - xfade_samples
        loop_buf = np.copy(audio[:loop_len])

        tail = audio[loop_len : loop_len + xfade_samples]
        head = loop_buf[:xfade_samples]

        t = np.linspace(0.0, 1.0, xfade_samples, dtype=np.float32)[:, None]
        # Equal-power crossfade (seno e cosseno) para preservar energia acústica
        fade_in = np.sin(t * (np.pi / 2.0))
        fade_out = np.cos(t * (np.pi / 2.0))

        blended = (head * fade_in) + (tail * fade_out)
        loop_buf[:xfade_samples] = blended

        self._loop_cache[key] = loop_buf
        return loop_buf

    def get_loop_chunk(
        self,
        name: str,
        start_sample: int,
        num_samples: int,
        crossfade_sec: float = 2.0
    ) -> np.ndarray:
        """
        Extrai um bloco de áudio contínuo em loop para o intervalo [start_sample, start_sample + num_samples).
        Garante continuidade de fase e ausência total de cliques em qualquer transição de chunk.
        """
        loop_buf = self.get_loop_buffer(name, crossfade_sec=crossfade_sec)
        l_len = loop_buf.shape[0]
        if l_len == 0:
            return np.zeros((num_samples, 2), dtype=np.float32)

        # Indexação modular circular ultrarrápida
        indices = (start_sample + np.arange(num_samples, dtype=np.int64)) % l_len
        return loop_buf[indices]

    def clear_cache(self) -> None:
        """Limpa o cache em memória de áudios processados."""
        self._raw_cache.clear()
        self._loop_cache.clear()
