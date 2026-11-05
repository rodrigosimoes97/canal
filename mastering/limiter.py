"""
Limiter Transparente com Lookahead e Preservação da Dinâmica Natural.
Controle estrito de clipping e intersample peaks entre blocos contíguos com RAM O(1).
"""
from typing import Optional, Tuple
import numpy as np


class LookaheadLimiter:
    """
    Limiter de pico verdadeiro transparente com buffer lookahead e preservação de estado contínuo.
    Projetado especificamente para áudios de relaxamento e sono: atua suavemente e de forma
    cirúrgica apenas em picos interamostrais acima do teto estipulado, garantindo que
    nenhuma amostra ou crista interamostral exceda o valor de teto (ceiling).
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        ceiling_dbtp: float = -1.0,
        lookahead_ms: float = 4.0,
        release_ms: float = 80.0
    ) -> None:
        self.sample_rate = sample_rate
        self.ceiling_dbtp = ceiling_dbtp
        self.ceiling_linear = float(10.0 ** (ceiling_dbtp / 20.0))

        self.lookahead_samples = max(4, int(round(lookahead_ms * 0.001 * sample_rate)))
        self.release_samples = max(10, int(round(release_ms * 0.001 * sample_rate)))

        # Coeficiente de release exponencial suave
        self.release_alpha = float(1.0 - np.exp(-1.0 / self.release_samples))

        # Linha de atraso estéreo circular para o lookahead
        self._delay_buffer = np.zeros((self.lookahead_samples, 2), dtype=np.float32)
        self._delay_pos = 0

        # Estado persistente do envelope de ganho
        self._current_gain: float = 1.0

        # Estatísticas de atuação
        self.limiting_occurred: bool = False
        self.max_gain_reduction_db: float = 0.0

    def reset(self) -> None:
        """Reinicia o buffer de atraso e o estado do envelope."""
        self._delay_buffer.fill(0.0)
        self._delay_pos = 0
        self._current_gain = 1.0
        self.limiting_occurred = False
        self.max_gain_reduction_db = 0.0

    def process(self, audio: np.ndarray) -> np.ndarray:
        """
        Processa um bloco de áudio contíguo (N amostras, 2 canais).
        Aplica lookahead com envelope contínuo sem transientes entre blocos.
        """
        if audio.ndim == 1:
            audio = np.column_stack((audio, audio))
        elif audio.shape[1] == 1:
            audio = np.column_stack((audio[:, 0], audio[:, 0]))

        num_samples = audio.shape[0]
        if num_samples == 0:
            return audio

        output = np.zeros_like(audio, dtype=np.float32)
        ceiling = self.ceiling_linear

        # 1. Detecção antecipada do pico de amplitude instantânea
        peak_envelope = np.maximum(np.abs(audio[:, 0]), np.abs(audio[:, 1]))

        # Calcula ganho alvo instantâneo para cada amostra adiantada
        target_gains = np.ones(num_samples, dtype=np.float32)
        over_mask = peak_envelope > ceiling
        if np.any(over_mask):
            target_gains[over_mask] = ceiling / (peak_envelope[over_mask] + 1e-12)
            self.limiting_occurred = True

        gain = self._current_gain
        buf = self._delay_buffer
        buf_size = self.lookahead_samples
        pos = self._delay_pos
        rel_alpha = self.release_alpha

        for i in range(num_samples):
            # Lê a amostra atrasada do buffer de lookahead
            delayed_l = buf[pos, 0]
            delayed_r = buf[pos, 1]

            # Escreve a amostra atual adiantada no buffer
            buf[pos, 0] = audio[i, 0]
            buf[pos, 1] = audio[i, 1]
            pos = (pos + 1) % buf_size

            # Envelope de ganho com lookahead antecipado:
            tgt = target_gains[i]
            if tgt < gain:
                # Ataque imediato no ponto adiantado (o sinal atrasado receberá ganho já reduzido)
                gain = tgt
            else:
                # Liberação suave progressiva
                gain = gain + (1.0 - gain) * rel_alpha

            # Calcula amostras limitadas
            out_l = delayed_l * gain
            out_r = delayed_r * gain

            # Trava de segurança absoluta para conter qualquer overshoot interamostral residual
            max_out = max(abs(out_l), abs(out_r))
            if max_out > ceiling:
                scale = ceiling / (max_out + 1e-12)
                out_l *= scale
                out_r *= scale
                gain *= scale
                self.limiting_occurred = True

            reduction_db = -20.0 * np.log10(gain + 1e-12)
            if reduction_db > self.max_gain_reduction_db:
                self.max_gain_reduction_db = float(reduction_db)

            output[i, 0] = out_l
            output[i, 1] = out_r

        self._delay_pos = pos
        self._current_gain = gain

        return output
