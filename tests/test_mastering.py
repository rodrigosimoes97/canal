"""
Testes Automatizados da Fase 5: Masterização Profissional, Conformidade EBU R128 / ITU-R BS.1770-4,
True Peak 4x Oversampling, Limiter Lookahead e Processamento em Duas Passagens com RAM O(1).
"""
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf
import pyloudnorm as pyln

from mastering.loudness import LoudnessMeter
from mastering.true_peak import TruePeakMeter
from mastering.limiter import LookaheadLimiter
from mastering.profiles import MASTERING_PROFILES, get_mastering_profile
from mastering.master import AudioMasterer, MasteringReport

OUTPUT_DIR = Path("output")


def test_loudness_meter_compliance():
    """Valida a medição de LUFS integrado comparando o streaming O(1) com pyloudnorm."""
    sr = 44100
    duration_sec = 6.0
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)

    # Onda senoidal estéreo a 1000 Hz com amplitude 0.1
    sig = np.column_stack((
        0.1 * np.sin(2 * np.pi * 1000 * t),
        0.1 * np.cos(2 * np.pi * 1000 * t)
    )).astype(np.float32)

    # Medição direta com pyloudnorm
    pyln_meter = pyln.Meter(sr)
    expected_lufs = pyln_meter.integrated_loudness(sig)

    # Medição em streaming por chunks (3 chunks de 2 segundos)
    our_meter = LoudnessMeter(sample_rate=sr)
    chunk_size = sr * 2
    for i in range(0, len(sig), chunk_size):
        our_meter.process_chunk(sig[i:i + chunk_size])

    actual_lufs = our_meter.get_integrated_loudness()

    # Deve bater com precisão milimétrica com o padrão BS.1770
    assert pytest.approx(actual_lufs, abs=0.15) == expected_lufs


def test_true_peak_detection():
    """Valida a detecção de True Peak (intersample peaks) com oversampling 4x."""
    sr = 44100
    # Frequência em Nyquist/2 (11025 Hz) com fase calibrada para gerar intersample peaks
    t = np.arange(2000)
    sig_intersample = np.sin(2 * np.pi * 11025 * t / sr + (np.pi / 4))
    stereo_sig = np.column_stack((sig_intersample, sig_intersample)).astype(np.float32)

    sample_peak = float(np.max(np.abs(stereo_sig)))

    tp_meter = TruePeakMeter(sample_rate=sr, oversample_factor=4)
    tp_lin, tp_dbtp = tp_meter.measure_buffer(stereo_sig)

    # O True Peak deve capturar a crista analógica que excede o pico das amostras discretas
    assert tp_lin > sample_peak
    assert tp_dbtp > (20.0 * np.log10(sample_peak))


def test_lookahead_limiter_containment():
    """Valida que o LookaheadLimiter atua suavemente e contém picos acima do teto estipulado."""
    sr = 44100
    limiter = LookaheadLimiter(sample_rate=sr, ceiling_dbtp=-1.5, lookahead_ms=4.0, release_ms=50.0)

    # Sinal que excede o teto (-1.5 dBTP = 0.8414 linear) com picos de 1.2
    t = np.linspace(0, 1.0, sr, endpoint=False)
    sig = np.column_stack((
        1.2 * np.sin(2 * np.pi * 200 * t),
        1.2 * np.sin(2 * np.pi * 200 * t)
    )).astype(np.float32)

    # Processa em dois chunks contíguos
    chunk1 = limiter.process(sig[:sr // 2])
    chunk2 = limiter.process(sig[sr // 2:])
    processed = np.vstack((chunk1, chunk2))

    ceiling_lin = float(10.0 ** (-1.5 / 20.0))

    assert limiter.limiting_occurred is True
    assert limiter.max_gain_reduction_db > 0.0
    # Nenhum valor processado deve exceder o teto além de tolerância numérica ínfima
    assert np.max(np.abs(processed)) <= (ceiling_lin + 1e-4)


def test_mastering_profiles_catalog():
    """Valida a existência e coerência dos perfis de relaxamento solicitados."""
    required = ["nature_ambient", "meditation", "deep_relaxation", "deep_sleep"]
    for name in required:
        prof = get_mastering_profile(name)
        assert prof.target_lufs < 0
        assert prof.target_true_peak_dbtp <= -1.0
        assert prof.max_gain_boost_db > 0

    # Confere alvos do usuário
    assert get_mastering_profile("nature_ambient").target_lufs == -20.0
    assert get_mastering_profile("meditation").target_lufs == -18.0
    assert get_mastering_profile("deep_relaxation").target_lufs == -21.0
    assert get_mastering_profile("deep_sleep").target_lufs == -23.0


def test_master_all_four_profiles_production():
    """
    Executa a masterização completa em 2 passagens para as produções refinadas,
    gerando relatórios técnicos detalhados e validando conformidade EBU R128.
    """
    masterer = AudioMasterer(sample_rate=44100)

    test_matrix = [
        ("output/sample1_refined_ambient_music.wav", "output/sample1_mastered_deep_relaxation.wav", "deep_relaxation"),
        ("output/sample2_refined_nature_zen.wav", "output/sample2_mastered_nature_ambient.wav", "nature_ambient"),
        ("output/sample3_refined_deep_relaxation.wav", "output/sample3_mastered_deep_sleep.wav", "deep_sleep"),
        ("output/sample1_refined_ambient_music.wav", "output/sample1_mastered_meditation.wav", "meditation"),
    ]

    for in_file, out_file, prof_name in test_matrix:
        in_p = Path(in_file)
        out_p = Path(out_file)

        assert in_p.exists(), f"Arquivo pré-mixado não encontrado: {in_file}"

        report = masterer.master_file(
            input_file=in_p,
            output_file=out_p,
            profile=prof_name,
            chunk_duration_sec=4.0
        )

        assert out_p.exists()
        assert isinstance(report, MasteringReport)

        # Validações obrigatórias
        # 1. True Peak após mastering deve respeitar o teto estipulado (com tolerância de 0.1 dBTP)
        assert report.true_peak_dbtp_after <= (report.target_true_peak_dbtp + 0.1)

        # 2. Ausência total de clipping de amostra
        assert report.sample_peak_after <= 1.0

        # 3. Formato PCM 24-bit preservado
        info = sf.info(str(out_p))
        assert info.subtype == "PCM_24"
        assert info.channels == 2
        assert info.samplerate == 44100

        # Imprime o sumário técnico do relatório
        summary = report.format_summary()
        assert len(summary) > 50
