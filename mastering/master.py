"""
Motor de Masterização em Duas Passagens com Consumo de RAM O(1).
Integra Medição BS.1770-4, True Peak 4x, Limiter Lookahead e Relatório Técnico.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import numpy as np
import soundfile as sf

from mastering.loudness import LoudnessMeter
from mastering.true_peak import TruePeakMeter
from mastering.limiter import LookaheadLimiter
from mastering.profiles import MasteringProfile, get_mastering_profile
from renderer.chunk_writer import ChunkWriter


@dataclass
class MasteringReport:
    """Relatório técnico completo da masterização de áudio."""
    profile_name: str
    target_lufs: float
    target_true_peak_dbtp: float
    # Antes da masterização
    lufs_before: float
    true_peak_dbtp_before: float
    sample_peak_before: float
    rms_db_before: float
    # Decisões do mastering
    gain_applied_db: float
    limiting_occurred: bool
    max_gain_reduction_db: float
    # Depois da masterização (auditado no arquivo final em disco)
    lufs_after: float
    true_peak_dbtp_after: float
    sample_peak_after: float
    rms_db_after: float
    duration_sec: float
    format_info: str

    def format_summary(self) -> str:
        """Gera resumo formatado para logs e relatórios."""
        return (
            f"=== RELATÓRIO DE MASTERIZAÇÃO: {self.profile_name.upper()} ===\n"
            f"Alvos: {self.target_lufs:.1f} LUFS | True Peak Máx: {self.target_true_peak_dbtp:.1f} dBTP\n"
            f"--------------------------------------------------\n"
            f"Antes:  Loudness: {self.lufs_before:.2f} LUFS | True Peak: {self.true_peak_dbtp_before:.2f} dBTP | RMS: {self.rms_db_before:.2f} dBFS | Peak: {self.sample_peak_before:.4f}\n"
            f"Ganho:  {self.gain_applied_db:+.2f} dB | Limiter Atuou: {'Sim' if self.limiting_occurred else 'Não'} (Redução máx: {self.max_gain_reduction_db:.2f} dB)\n"
            f"Depois: Loudness: {self.lufs_after:.2f} LUFS | True Peak: {self.true_peak_dbtp_after:.2f} dBTP | RMS: {self.rms_db_after:.2f} dBFS | Peak: {self.sample_peak_after:.4f}\n"
            f"Duração: {self.duration_sec:.2f}s | Formato: {self.format_info}\n"
            f"=================================================="
        )


class AudioMasterer:
    """
    Processador de Masterização Profissional com arquitetura de duas passagens.
    Preserva a dinâmica orgânica sem esmagamento do áudio e opera em streaming O(1) de RAM.
    """

    def __init__(self, sample_rate: int = 44100) -> None:
        self.sample_rate = sample_rate

    def master_file(
        self,
        input_file: Union[str, Path],
        output_file: Union[str, Path],
        profile: Union[str, MasteringProfile] = "deep_relaxation",
        chunk_duration_sec: float = 5.0,
        **overrides
    ) -> MasteringReport:
        """
        Executa o pipeline completo de masterização em duas passagens para um arquivo de áudio.
        
        Args:
            input_file: Arquivo pré-mixado de entrada.
            output_file: Destino do arquivo masterizado PCM 24-bit.
            profile: Nome do perfil ou instância de MasteringProfile.
            chunk_duration_sec: Tamanho da janela de streaming em segundos.
            overrides: Parâmetros pontuais para sobrepor o perfil.
        """
        in_path = Path(input_file)
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(profile, str):
            master_prof = get_mastering_profile(profile, **overrides)
        else:
            master_prof = profile

        # =====================================================================
        # PASSAGEM 1: ANÁLISE COMPLETA EM STREAMING (RAM O(1))
        # =====================================================================
        loudness_meter = LoudnessMeter(sample_rate=self.sample_rate)
        true_peak_meter = TruePeakMeter(sample_rate=self.sample_rate)

        total_frames = 0
        sum_sq = 0.0
        max_sample_peak = 0.0

        with sf.SoundFile(str(in_path), "r") as f_in:
            actual_sr = f_in.samplerate
            chunk_frames = int(round(chunk_duration_sec * actual_sr))

            while True:
                chunk = f_in.read(frames=chunk_frames, dtype="float32")
                if len(chunk) == 0:
                    break

                num_samples = chunk.shape[0]
                total_frames += num_samples

                # Acumula métricas
                loudness_meter.process_chunk(chunk)
                true_peak_meter.process_chunk(chunk)

                sum_sq += np.sum(chunk ** 2)
                max_sample_peak = max(max_sample_peak, float(np.max(np.abs(chunk))))

        lufs_before = loudness_meter.get_integrated_loudness()
        tp_lin_before, tp_dbtp_before = true_peak_meter.get_max_true_peak()

        rms_before = np.sqrt(sum_sq / max(1, total_frames * 2))
        rms_db_before = 20.0 * np.log10(rms_before + 1e-12)
        total_duration_sec = total_frames / actual_sr

        # =====================================================================
        # CÁLCULO DINÂMICO DE GANHO E HEADROOM
        # =====================================================================
        # Delta necessário para atingir o target LUFS
        needed_gain_db = master_prof.target_lufs - lufs_before

        # Limita o ganho positivo para evitar amplificação agressiva de ruído
        gain_applied_db = min(needed_gain_db, master_prof.max_gain_boost_db)

        # Se o ganho aplicado for projetar o True Peak além do teto aceitável com margem de segurança,
        # limita o ganho estático e delega a contenção fina ao limiter transparente
        proj_tp = tp_dbtp_before + gain_applied_db
        if proj_tp > (master_prof.target_true_peak_dbtp + 2.5):
            gain_applied_db = (master_prof.target_true_peak_dbtp + 2.5) - tp_dbtp_before

        linear_gain = float(10.0 ** (gain_applied_db / 20.0))

        # =====================================================================
        # PASSAGEM 2: APLICAÇÃO DE GANHO, FADES E LIMITER LOOKAHEAD (RAM O(1))
        # =====================================================================
        limiter = LookaheadLimiter(
            sample_rate=actual_sr,
            ceiling_dbtp=master_prof.target_true_peak_dbtp,
            lookahead_ms=master_prof.lookahead_ms,
            release_ms=master_prof.release_ms
        )

        fade_in_samples = int(round(master_prof.fade_in_sec * actual_sr))
        fade_out_samples = int(round(master_prof.fade_out_sec * actual_sr))
        fade_out_start_frame = max(0, total_frames - fade_out_samples)

        with sf.SoundFile(str(out_path), mode="w", samplerate=actual_sr, channels=2, subtype="PCM_24") as f_out:
            current_frame = 0

            with sf.SoundFile(str(in_path), "r") as f_in:
                chunk_frames = int(round(chunk_duration_sec * actual_sr))

                while True:
                    chunk = f_in.read(frames=chunk_frames, dtype="float32")
                    if len(chunk) == 0:
                        break

                    num_samples = chunk.shape[0]
                    frame_indices = np.arange(current_frame, current_frame + num_samples, dtype=np.float64)

                    # 1. Aplica ganho de normalização transparente
                    processed = chunk * linear_gain

                    # 2. Fade-in suave na introdução (curva senoidal convexa sem cliques)
                    if current_frame < fade_in_samples:
                        fi_env = np.clip(frame_indices / max(1, fade_in_samples), 0.0, 1.0)
                        fi_curve = np.sin(fi_env * (np.pi / 2)) ** 2
                        processed *= fi_curve[:, None]

                    # 3. Fade-out suave na cauda (preservando o decaimento natural)
                    if (current_frame + num_samples) > fade_out_start_frame:
                        fo_env = np.clip((total_frames - frame_indices) / max(1, fade_out_samples), 0.0, 1.0)
                        fo_curve = np.sin(fo_env * (np.pi / 2)) ** 2
                        processed *= fo_curve[:, None]

                    # 4. Limiter de True Peak transparente com lookahead
                    limited = limiter.process(processed)

                    # 5. Escrita streaming direta em disco PCM 24-bit
                    f_out.write(limited)

                    current_frame += num_samples

        # =====================================================================
        # PASSAGEM 3: AUDITORIA REAL DO ARQUIVO MASTERIZADO EM DISCO
        # =====================================================================
        audit_loudness = LoudnessMeter(sample_rate=actual_sr)
        audit_true_peak = TruePeakMeter(sample_rate=actual_sr)

        lufs_after = audit_loudness.measure_file(out_path, chunk_duration_sec=chunk_duration_sec)
        _, tp_dbtp_after = audit_true_peak.measure_file(out_path, chunk_duration_sec=chunk_duration_sec)

        info = sf.info(str(out_path))

        # Medição de RMS e Sample Peak pós-mastering
        sum_sq_after = 0.0
        max_sample_peak_after = 0.0
        total_frames_after = 0

        with sf.SoundFile(str(out_path), "r") as f_out:
            while True:
                chunk = f_out.read(frames=chunk_frames, dtype="float32")
                if len(chunk) == 0:
                    break
                sum_sq_after += np.sum(chunk ** 2)
                max_sample_peak_after = max(max_sample_peak_after, float(np.max(np.abs(chunk))))
                total_frames_after += chunk.shape[0]

        rms_after = np.sqrt(sum_sq_after / max(1, total_frames_after * 2))
        rms_db_after = 20.0 * np.log10(rms_after + 1e-12)

        return MasteringReport(
            profile_name=master_prof.name,
            target_lufs=master_prof.target_lufs,
            target_true_peak_dbtp=master_prof.target_true_peak_dbtp,
            lufs_before=lufs_before,
            true_peak_dbtp_before=tp_dbtp_before,
            sample_peak_before=max_sample_peak,
            rms_db_before=rms_db_before,
            gain_applied_db=gain_applied_db,
            limiting_occurred=limiter.limiting_occurred,
            max_gain_reduction_db=limiter.max_gain_reduction_db,
            lufs_after=lufs_after,
            true_peak_dbtp_after=tp_dbtp_after,
            sample_peak_after=max_sample_peak_after,
            rms_db_after=rms_db_after,
            duration_sec=total_duration_sec,
            format_info=f"{info.format} {info.subtype} {actual_sr}Hz {info.channels}ch"
        )
