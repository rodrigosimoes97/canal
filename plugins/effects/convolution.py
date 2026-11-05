"""
Efeito DSP: Reverb de Convolução com Particionamento Overlap-Add e Preservação de Cauda.
"""
from typing import Any, Dict, Optional, Tuple
import numpy as np
from scipy.signal import fftconvolve

from plugins.base import BasePlugin, PluginCategory


class ConvolutionReverb(BasePlugin):
    """
    Reverb de convolução por blocos utilizando o método Overlap-Add (OLA).
    Garante caudas reverberantes de qualquer comprimento sem truncamento entre chunks.
    """
    PLUGIN_ID = "convolution_reverb"

    def __init__(self, plugin_id: str = "convolution_reverb", config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(plugin_id=plugin_id, category=PluginCategory.EFFECT, config=config)
        self.ir_duration_sec = float(self.config.get("ir_duration_sec", 2.0))
        self.mix = float(np.clip(self.config.get("mix", 0.3), 0.0, 1.0))
        self.sample_rate: int = 44100

        self._ir_l: Optional[np.ndarray] = None
        self._ir_r: Optional[np.ndarray] = None
        self._overlap_tail_l: Optional[np.ndarray] = None
        self._overlap_tail_r: Optional[np.ndarray] = None

    @property
    def has_state(self) -> bool:
        return True

    def _generate_synthetic_ir(self, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        """Gera uma resposta ao impulso (IR) sintética com decaimento exponencial suave e difusão estéreo."""
        ir_len = int(round(self.ir_duration_sec * self.sample_rate))
        t = np.linspace(0.0, self.ir_duration_sec, ir_len, dtype=np.float64)

        # Decaimento exponencial típico de sala grande / catedral
        decay = np.exp(-3.0 * t)

        # Ruído branco estéreo filtrado como reflexões difusas
        noise_l = rng.normal(0.0, 1.0, ir_len) * decay
        noise_r = rng.normal(0.0, 1.0, ir_len) * decay

        # Normaliza a energia da IR
        norm_l = np.sqrt(np.sum(noise_l ** 2)) + 1e-9
        norm_r = np.sqrt(np.sum(noise_r ** 2)) + 1e-9

        ir_l = (noise_l / norm_l).astype(np.float32)
        ir_r = (noise_r / norm_r).astype(np.float32)
        return ir_l, ir_r

    def prepare(self, context: Any, rng: np.random.Generator) -> None:
        self.sample_rate = context.sample_rate
        self._ir_l, self._ir_r = self._generate_synthetic_ir(rng)

        ir_len = len(self._ir_l)
        # O buffer de cauda tem tamanho igual ao comprimento da IR
        self._overlap_tail_l = np.zeros(ir_len, dtype=np.float32)
        self._overlap_tail_r = np.zeros(ir_len, dtype=np.float32)
        self._is_prepared = True

    def process(self, audio: np.ndarray) -> np.ndarray:
        if not self.is_active or self._ir_l is None or audio.shape[0] == 0:
            return audio

        num_samples = audio.shape[0]
        in_l = audio[:, 0].astype(np.float32)
        in_r = audio[:, 1].astype(np.float32)

        # Convolução FFT rápida: tamanho de saída = num_samples + ir_len - 1
        conv_l = fftconvolve(in_l, self._ir_l, mode="full").astype(np.float32)
        conv_r = fftconvolve(in_r, self._ir_r, mode="full").astype(np.float32)

        # 1. Soma a cauda do chunk anterior à cabeça do chunk atual
        tail_len_l = len(self._overlap_tail_l)
        add_len = min(num_samples, tail_len_l)

        conv_l[:add_len] += self._overlap_tail_l[:add_len]
        conv_r[:add_len] += self._overlap_tail_r[:add_len]

        # 2. O áudio do chunk atual são as primeiras `num_samples` amostras
        wet_l = conv_l[:num_samples]
        wet_r = conv_r[:num_samples]

        # 3. A cauda restante vira o novo `_overlap_tail` para o próximo chunk
        rem_l = conv_l[num_samples:]
        rem_r = conv_r[num_samples:]

        # Se houver cauda do chunk anterior além de `num_samples`, soma também
        if tail_len_l > num_samples:
            old_rem_l = self._overlap_tail_l[num_samples:]
            old_rem_r = self._overlap_tail_r[num_samples:]
            max_len = max(len(rem_l), len(old_rem_l))
            new_tail_l = np.zeros(max_len, dtype=np.float32)
            new_tail_r = np.zeros(max_len, dtype=np.float32)
            new_tail_l[:len(rem_l)] += rem_l
            new_tail_l[:len(old_rem_l)] += old_rem_l
            new_tail_r[:len(rem_r)] += rem_r
            new_tail_r[:len(old_rem_r)] += old_rem_r
            self._overlap_tail_l = new_tail_l
            self._overlap_tail_r = new_tail_r
        else:
            self._overlap_tail_l = rem_l
            self._overlap_tail_r = rem_r

        out_l = in_l * (1.0 - self.mix) + wet_l * self.mix
        out_r = in_r * (1.0 - self.mix) + wet_r * self.mix

        return np.column_stack((out_l, out_r))

    def render(self, start_sample: int, num_samples: int) -> np.ndarray:
        return np.zeros((num_samples, 2), dtype=np.float32)

    def reset(self) -> None:
        if self._overlap_tail_l is not None:
            self._overlap_tail_l.fill(0.0)
            self._overlap_tail_r.fill(0.0)
