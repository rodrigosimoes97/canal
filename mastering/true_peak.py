"""
Medidor de True Peak com Oversampling 4x conforme ITU-R BS.1770-4.
Detecta picos interamostrais com streaming por blocos e RAM O(1).
"""
from typing import Optional, Tuple, Union
from pathlib import Path
import numpy as np
from scipy import signal
import soundfile as sf


class TruePeakMeter:
    """
    Medidor de True Peak com interpolação polifásica 4x (oversampling).
    Identifica picos que ocorrem entre as amostras discretas que causariam clipping
    nos estágios de conversão digital para analógico (DAC) ou compressão com perdas.
    """

    def __init__(self, sample_rate: int = 44100, oversample_factor: int = 4) -> None:
        self.sample_rate = sample_rate
        self.oversample_factor = oversample_factor
        self.max_peak_linear: float = 0.0
        self._overlap_samples = 64
        self._prev_tail: Optional[np.ndarray] = None

    def reset(self) -> None:
        """Reinicia o pico máximo registrado."""
        self.max_peak_linear = 0.0
        self._prev_tail = None

    def process_chunk(self, chunk: np.ndarray) -> Tuple[float, float]:
        """
        Processa um bloco de áudio, calcula o oversampling 4x e atualiza o pico máximo global.
        Retorna (pico_linear_deste_bloco, pico_dbtp_deste_bloco).
        """
        if chunk.ndim == 1:
            chunk = np.column_stack((chunk, chunk))
        elif chunk.shape[1] == 1:
            chunk = np.column_stack((chunk[:, 0], chunk[:, 0]))

        num_samples = chunk.shape[0]
        if num_samples == 0:
            return 0.0, -120.0

        # Concatena cauda anterior para evitar transiente de borda na interpolação sinc
        if self._prev_tail is not None:
            work_chunk = np.vstack((self._prev_tail, chunk))
            skip_os_samples = self._prev_tail.shape[0] * self.oversample_factor
        else:
            work_chunk = chunk
            skip_os_samples = 0

        # Atualiza a cauda para a próxima chamada
        if num_samples >= self._overlap_samples:
            self._prev_tail = chunk[-self._overlap_samples:, :].copy()
        else:
            self._prev_tail = chunk.copy()

        # Oversampling polifásico 4x canal por canal
        os_l = signal.resample_poly(work_chunk[:, 0], self.oversample_factor, 1)
        os_r = signal.resample_poly(work_chunk[:, 1], self.oversample_factor, 1)

        # Descarta região de sobreposição anterior
        valid_l = os_l[skip_os_samples:]
        valid_r = os_r[skip_os_samples:]

        chunk_peak_l = float(np.max(np.abs(valid_l))) if len(valid_l) > 0 else 0.0
        chunk_peak_r = float(np.max(np.abs(valid_r))) if len(valid_r) > 0 else 0.0
        chunk_peak = max(chunk_peak_l, chunk_peak_r)

        if chunk_peak > self.max_peak_linear:
            self.max_peak_linear = chunk_peak

        chunk_dbtp = 20.0 * np.log10(chunk_peak + 1e-12)
        return chunk_peak, float(chunk_dbtp)

    def get_max_true_peak(self) -> Tuple[float, float]:
        """
        Retorna o True Peak máximo acumulado como (valor_linear, valor_dbtp).
        """
        peak_lin = self.max_peak_linear
        peak_dbtp = 20.0 * np.log10(peak_lin + 1e-12)
        return peak_lin, float(peak_dbtp)

    def measure_buffer(self, audio: np.ndarray) -> Tuple[float, float]:
        """Mede o True Peak de um array estéreo em memória."""
        self.reset()
        self.process_chunk(audio)
        return self.get_max_true_peak()

    def measure_file(self, file_path: Union[str, Path], chunk_duration_sec: float = 5.0) -> Tuple[float, float]:
        """
        Mede o True Peak de um arquivo em streaming com consumo constante de RAM O(1).
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

        return self.get_max_true_peak()
