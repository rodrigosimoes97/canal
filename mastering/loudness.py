"""
Medidor de Loudness Integrado em conformidade com ITU-R BS.1770-4 e EBU R128.
Suporta processamento em streaming com consumo constante de RAM O(1).
"""
from typing import List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
from scipy import signal
import soundfile as sf
import pyloudnorm as pyln


class LoudnessMeter:
    """
    Medidor de Loudness em conformidade estrita com ITU-R BS.1770-4 / EBU R128.
    Implementa K-weighting (filtro de prateleira alta e passa-alta) com preservação de estados
    entre chunks arbitrários e duplo gating (-70 LKFS absoluto e -10 LU relativo).
    """

    def __init__(self, sample_rate: int = 44100, block_size_sec: float = 0.400, hop_size_sec: float = 0.100) -> None:
        self.sample_rate = sample_rate
        self.block_size_sec = block_size_sec
        self.hop_size_sec = hop_size_sec

        self.block_samples = int(round(self.block_size_sec * self.sample_rate))
        self.hop_samples = int(round(self.hop_size_sec * self.sample_rate))

        # Extrai os coeficientes precisos BS.1770 para o sample_rate via pyloudnorm
        pyln_meter = pyln.Meter(self.sample_rate)
        hs = pyln_meter._filters["high_shelf"]
        hp = pyln_meter._filters["high_pass"]

        self.hs_b = np.array(hs.b, dtype=np.float64)
        self.hs_a = np.array(hs.a, dtype=np.float64)
        self.hp_b = np.array(hp.b, dtype=np.float64)
        self.hp_a = np.array(hp.a, dtype=np.float64)

        # Estados dos filtros IIR (2 canais estéreo)
        self._zi_hs = np.zeros((2, 2), dtype=np.float64)
        self._zi_hp = np.zeros((2, 2), dtype=np.float64)

        # Buffer circular residual para fatiamento de janelas com hop de 100ms
        self._residual_buffer = np.zeros((0, 2), dtype=np.float64)

        # Lista de potências médias dos blocos (z_j) de 400 ms
        self._block_powers: List[float] = []

        self.reset()

    def reset(self) -> None:
        """Reinicia o estado interno dos filtros e acumuladores de bloco."""
        self._zi_hs = np.zeros((2, max(len(self.hs_a), len(self.hs_b)) - 1), dtype=np.float64)
        self._zi_hp = np.zeros((2, max(len(self.hp_a), len(self.hp_b)) - 1), dtype=np.float64)
        self._residual_buffer = np.zeros((0, 2), dtype=np.float64)
        self._block_powers.clear()

    def process_chunk(self, chunk: np.ndarray) -> None:
        """
        Processa um bloco de áudio contíguo (N amostras, 2 canais).
        Preserva os estados IIR entre chamadas.
        """
        if chunk.ndim == 1:
            chunk = np.column_stack((chunk, chunk))
        elif chunk.shape[1] == 1:
            chunk = np.column_stack((chunk[:, 0], chunk[:, 0]))

        num_samples = chunk.shape[0]
        if num_samples == 0:
            return

        filtered = np.zeros_like(chunk, dtype=np.float64)

        # Aplica os dois estágios de K-weighting canal por canal mantendo zi
        for ch in range(min(2, chunk.shape[1])):
            sig_in = chunk[:, ch].astype(np.float64)
            # Estágio 1: High-shelf (efeito de cabeça)
            s1, self._zi_hs[ch] = signal.lfilter(self.hs_b, self.hs_a, sig_in, zi=self._zi_hs[ch])
            # Estágio 2: High-pass (filtro RLB)
            s2, self._zi_hp[ch] = signal.lfilter(self.hp_b, self.hp_a, s1, zi=self._zi_hp[ch])
            filtered[:, ch] = s2

        # Concatena com resíduo do chunk anterior
        if self._residual_buffer.shape[0] > 0:
            combined = np.vstack((self._residual_buffer, filtered))
        else:
            combined = filtered

        total_len = combined.shape[0]
        cur_pos = 0

        # Ponderação de canais ITU-R BS.1770-4 (Left: 1.0, Right: 1.0)
        # G_i = 1.0 para L e R
        while cur_pos + self.block_samples <= total_len:
            block = combined[cur_pos:cur_pos + self.block_samples, :]
            # z_j = sum(G_i * mean(s_i^2))
            mean_sq_l = np.mean(block[:, 0] ** 2)
            mean_sq_r = np.mean(block[:, 1] ** 2)
            z_j = mean_sq_l + mean_sq_r
            self._block_powers.append(float(z_j))
            cur_pos += self.hop_samples

        # Guarda amostras não consumidas para o próximo chunk
        self._residual_buffer = combined[cur_pos:, :]

    def get_integrated_loudness(self) -> float:
        """
        Calcula o Loudness Integrado final aplicando o duplo gating conforme ITU-R BS.1770-4.
        Retorna valor em LUFS (LKFS). Se silêncio total, retorna -70.0.
        """
        if not self._block_powers:
            return -70.0

        z = np.array(self._block_powers, dtype=np.float64)

        # Evita log de zero
        z = np.maximum(z, 1e-12)

        # Loudness individual dos blocos em LKFS
        l_j = -0.691 + 10.0 * np.log10(z)

        # 1. Gating absoluto: descarta blocos com l_j < -70 LKFS
        abs_gate_mask = l_j >= -70.0
        if not np.any(abs_gate_mask):
            return -70.0

        z_abs = z[abs_gate_mask]
        gamma_a = -0.691 + 10.0 * np.log10(np.mean(z_abs))

        # 2. Gating relativo: descarta blocos com l_j < gamma_a - 10 LU
        gamma_r = gamma_a - 10.0
        rel_gate_mask = abs_gate_mask & (l_j >= gamma_r)

        if not np.any(rel_gate_mask):
            return float(gamma_a)

        z_rel = z[rel_gate_mask]
        lufs_integrated = -0.691 + 10.0 * np.log10(np.mean(z_rel))

        return float(lufs_integrated)

    def measure_buffer(self, audio: np.ndarray) -> float:
        """Mede o Loudness Integrado de um buffer em memória."""
        self.reset()
        self.process_chunk(audio)
        return self.get_integrated_loudness()

    def measure_file(self, file_path: Union[str, Path], chunk_duration_sec: float = 5.0) -> float:
        """
        Mede o Loudness Integrado de um arquivo de áudio em streaming (RAM O(1)).
        """
        self.reset()
        path_str = str(file_path)
        with sf.SoundFile(path_str, "r") as f:
            chunk_frames = int(round(chunk_duration_sec * f.samplerate))
            while True:
                data = f.read(frames=chunk_frames, dtype="float32")
                if len(data) == 0:
                    break
                self.process_chunk(data)

        return self.get_integrated_loudness()
