"""
Relaxation Sound Generator — Professional Edition
==================================================
Gera sons terapêuticos de alta qualidade para ansiedade, meditação e sleep.

Dependências:
    pip install numpy soundfile scipy

Uso rápido:
    python relax_sound_generator.py

Uso avançado (CLI):
    python relax_sound_generator.py --preset deep_sleep --duration 3600 --output sleep.wav
    python relax_sound_generator.py --preset anxiety_relief --duration 1800 --binaural --output anxiety.wav
    python relax_sound_generator.py --list-presets
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.signal import butter, sosfilt

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constantes técnicas
# ---------------------------------------------------------------------------
SAMPLE_RATE = 44100          # Hz — qualidade CD
BIT_DEPTH = "PCM_24"         # 24-bit para distribuição
CHANNELS = 2                 # Stereo


# ---------------------------------------------------------------------------
# Tipos de ruído base
# ---------------------------------------------------------------------------

def white_noise(n: int) -> np.ndarray:
    """Ruído branco — todos os harmônicos com amplitude igual."""
    return np.random.normal(0, 1, n).astype(np.float64)


def pink_noise(n: int) -> np.ndarray:
    """
    Ruído rosa (1/f) via filtro de Voss-McCartney.
    Soa mais natural e menos fatigante que o branco.
    Ideal para foco e relaxamento.
    """
    # Método de estados (mais estável que o recursivo ingênuo)
    num_rows = 16
    array = np.empty((num_rows, n))
    array[:] = np.random.randn(num_rows, n)
    reshaped = array.cumsum(axis=1)
    reshaped -= reshaped[:, 0:1]
    pink = reshaped.sum(axis=0)
    pink /= np.max(np.abs(pink))
    return pink.astype(np.float64)

def stereo_pan_dynamic(left: np.ndarray, right: np.ndarray,
                       rate_hz: float = 0.035,
                       depth: float = 0.35) -> tuple[np.ndarray, np.ndarray]:
    """
    Movimento estéreo lento L ↔ R.
    rate_hz: velocidade (0.02–0.06 ideal)
    depth: intensidade do movimento (0–1)
    """
    t = np.linspace(0, len(left) / SAMPLE_RATE, len(left))
    
    pan = np.sin(2 * np.pi * rate_hz * t)  # -1 a 1
    
    l = left * (1 - depth * (pan + 1)/2)
    r = right * (1 - depth * (1 - (pan + 1)/2))
    
    return l, r


def brown_noise(n: int) -> np.ndarray:
    """
    Ruído marrom (1/f²) — integração do branco.
    Tom mais grave e profundo. Excelente para sono profundo.
    """
    white = np.random.normal(0, 1, n)
    brown = np.cumsum(white)
    # Remove drift DC com filtro passa-alta suave
    sos = butter(2, 20.0 / (SAMPLE_RATE / 2), btype="high", output="sos")
    brown = sosfilt(sos, brown)
    brown /= np.max(np.abs(brown))
    return brown.astype(np.float64)


def green_noise(n: int) -> np.ndarray:
    """
    Ruído verde — rosa com corte nas extremidades de frequência.
    Concentrado na faixa 500 Hz – 2 kHz (mais parecido com natureza).
    """
    raw = pink_noise(n)
    sos = butter(4, [500 / (SAMPLE_RATE / 2), 2000 / (SAMPLE_RATE / 2)],
                 btype="bandpass", output="sos")
    green = sosfilt(sos, raw)
    green /= np.max(np.abs(green))
    return green.astype(np.float64)


def grey_noise(n: int) -> np.ndarray:
    """
    Ruído cinza — ruído branco equalizado pela curva igual-loudness (ISO 226).
    Percebido como igualmente intenso em todas as frequências pelo ouvido humano.
    """
    white = white_noise(n)
    freqs = np.fft.rfftfreq(n, d=1.0 / SAMPLE_RATE)
    spectrum = np.fft.rfft(white)

    # Curva de equal-loudness simplificada (A-weighting invertida)
    freqs_safe = np.where(freqs > 0, freqs, 1.0)
    f2 = freqs_safe ** 2
    a_weight = (
        (12194 ** 2 * f2 ** 2)
        / ((f2 + 20.6 ** 2) * np.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2)) * (f2 + 12194 ** 2))
    )
    a_weight = np.where(freqs > 0, a_weight, 0.0)
    grey = np.fft.irfft(spectrum / (a_weight + 1e-12))
    grey = grey[:n]
    grey /= np.max(np.abs(grey))
    return grey.astype(np.float64)


# ---------------------------------------------------------------------------
# Efeitos e camadas adicionais
# ---------------------------------------------------------------------------

def binaural_beat(n: int, base_freq: float = 200.0, beat_freq: float = 6.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Gera um batimento binaural para induzir estados cerebrais específicos.

    beat_freq:
        0.5–4 Hz  → Delta (sono profundo)
        4–8 Hz    → Theta (meditação, criatividade)
        8–14 Hz   → Alpha (relaxamento, calma)
        14–30 Hz  → Beta  (foco, atenção)

    Retorna (canal_esquerdo, canal_direito) para mixagem stereo.
    """
    t = np.linspace(0, n / SAMPLE_RATE, n, endpoint=False)
    left  = np.sin(2 * np.pi * base_freq * t) * 0.15
    right = np.sin(2 * np.pi * (base_freq + beat_freq) * t) * 0.15
    return left, right


def isochronic_pulse(n: int, pulse_freq: float = 8.0, carrier_freq: float = 200.0) -> np.ndarray:
    """
    Pulso isocrônico — alternativa aos binaurais que funciona com um único canal.
    pulse_freq define a frequência de entrainment (Hz).
    """
    t = np.linspace(0, n / SAMPLE_RATE, n, endpoint=False)
    carrier = np.sin(2 * np.pi * carrier_freq * t)
    # Envelope de pulso suave (senoidal retificada)
    envelope = np.clip(np.sin(2 * np.pi * pulse_freq * t), 0, 1)
    return carrier * envelope * 0.2


def solfeggio_tone(n: int, freq: float = 528.0) -> np.ndarray:
    """
    Tom solfeggio — frequências usadas em terapia de som.
    Frequências comuns: 174, 285, 396, 417, 528, 639, 741, 852, 963 Hz.
    528 Hz = 'frequência do amor / cura DNA'.
    """
    t = np.linspace(0, n / SAMPLE_RATE, n, endpoint=False)
    # Harmônicos suaves para soar mais orgânico
    tone = (
        0.5 * np.sin(2 * np.pi * freq * t)
        + 0.25 * np.sin(2 * np.pi * freq * 2 * t)
        + 0.1  * np.sin(2 * np.pi * freq * 3 * t)
    )
    tone /= np.max(np.abs(tone))
    return tone * 0.12


def nature_layer(n: int, texture: str = "water") -> np.ndarray:
    """
    Adiciona textura de natureza simulada por filtragem.
    texture: 'water', 'forest', 'rain'
    """
    base = pink_noise(n)
    if texture == "water":
        # Bandpass centrado na frequência de água corrente ~300-800 Hz
        sos = butter(6, [300 / (SAMPLE_RATE / 2), 800 / (SAMPLE_RATE / 2)],
                     btype="bandpass", output="sos")
        filtered = sosfilt(sos, base)
        # Modulação lenta para simular ondulação
        t = np.linspace(0, n / SAMPLE_RATE, n)
        mod = 0.85 + 0.15 * np.sin(2 * np.pi * 0.07 * t)
        return filtered * mod

    elif texture == "forest":
        # Corte de graves — floresta tem mais médios-agudos
        sos = butter(4, 400 / (SAMPLE_RATE / 2), btype="high", output="sos")
        return sosfilt(sos, base) * 0.9

    elif texture == "rain":
        # Ruído de chuva tem espectro mais branco com suave queda nos agudos
        raw = white_noise(n) * 0.6 + base * 0.4
        sos = butter(2, 8000 / (SAMPLE_RATE / 2), btype="low", output="sos")
        return sosfilt(sos, raw)

    return base


# ---------------------------------------------------------------------------
# Envelope dinâmico e processamento de áudio
# ---------------------------------------------------------------------------

def make_envelope(n: int, fade_in_s: float = 8.0, fade_out_s: float = 10.0,
                  shape: str = "cosine") -> np.ndarray:
    """
    Cria envelope de amplitude com fade in/out profissional.
    shape: 'linear' | 'cosine' (mais natural)
    """
    envelope = np.ones(n)
    fi = int(fade_in_s * SAMPLE_RATE)
    fo = int(fade_out_s * SAMPLE_RATE)

    if shape == "cosine":
        fade_in_curve  = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, fi)))
        fade_out_curve = 0.5 * (1 + np.cos(np.pi * np.linspace(0, 1, fo)))
    else:
        fade_in_curve  = np.linspace(0, 1, fi)
        fade_out_curve = np.linspace(1, 0, fo)

    envelope[:fi]  = fade_in_curve
    envelope[-fo:] = fade_out_curve
    return envelope


def slow_volume_modulation(n: int, depth: float = 0.04, rate_hz: float = 0.025) -> np.ndarray:
    """
    Modulação de volume muito lenta e sutil (tipo respiração).
    Evita fadiga auditiva em sessões longas.
    depth: profundidade da modulação (0–1)
    rate_hz: velocidade da oscilação (~0.025 = ciclo de ~40s)
    """
    t = np.linspace(0, n / SAMPLE_RATE, n)
    return 1.0 - depth + depth * np.sin(2 * np.pi * rate_hz * t)


def stereo_spread(mono: np.ndarray, width: float = 0.4, decorr_ms: float = 18.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Cria imagem stereo a partir de sinal mono por decorrelação temporal.
    Soa muito mais imersivo que duplicar o canal.
    width: largura stereo (0 = mono, 1 = máximo)
    decorr_ms: delay de decorrelação em ms (~15–25 ms é perceptualmente natural)
    """
    delay_samples = int(decorr_ms * SAMPLE_RATE / 1000)
    right_delayed = np.roll(mono, delay_samples)
    right_delayed[:delay_samples] = mono[:delay_samples]  # sem silêncio no início

    left  = mono * (1 - width * 0.5) + right_delayed * (width * 0.5)
    right = mono * (1 - width * 0.5) - right_delayed * (width * 0.5) + right_delayed

    return left, right


def normalize_lufs(audio: np.ndarray, target_lufs: float = -23.0) -> np.ndarray:
    """
    Normalização de loudness baseada em RMS (aproximação de LUFS).
    -23 LUFS = padrão EBU R128 (broadcast).
    -16 LUFS = padrão streaming (Spotify, YouTube).
    """
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-9:
        return audio
    # Converte target LUFS em amplitude RMS alvo
    target_rms = 10 ** (target_lufs / 20)
    gain = target_rms / rms
    normalized = audio * gain
    # Limiter de segurança (evita clipping)
    peak = np.max(np.abs(normalized))
    if peak > 0.98:
        normalized = normalized * (0.98 / peak)
    return normalized


def soft_limiter(audio: np.ndarray, threshold: float = 0.95) -> np.ndarray:
    """Soft clipper suave para evitar distorção digital."""
    return np.tanh(audio / threshold) * threshold


# ---------------------------------------------------------------------------
# Presets terapêuticos
# ---------------------------------------------------------------------------

class BeatType(str, Enum):
    NONE        = "none"
    BINAURAL    = "binaural"
    ISOCHRONIC  = "isochronic"


@dataclass
class Preset:
    name: str
    description: str
    noise_fn: Callable[[int], np.ndarray]
    volume: float = 0.65
    binaural_base: float = 200.0
    binaural_beat: float = 8.0          # Alpha por padrão
    beat_type: BeatType = BeatType.NONE
    solfeggio_freq: float | None = None
    nature_texture: str | None = None
    nature_volume: float = 0.15
    stereo_width: float = 0.45
    fade_in_s: float = 8.0
    fade_out_s: float = 12.0
    lufs_target: float = -23.0
    tags: list[str] = field(default_factory=list)


PRESETS: dict[str, Preset] = {
    "anxiety_relief": Preset(
        name="Anxiety Relief",
        description="Ruído rosa + batimento alpha (10 Hz) + tom solfeggio 528 Hz. "
                    "Ideal para crises de ansiedade e ataques de pânico.",
        noise_fn=pink_noise,
        volume=0.60,
        beat_type=BeatType.BINAURAL,
        binaural_base=220.0,
        binaural_beat=10.0,          # Alpha: relaxamento consciente
        solfeggio_freq=528.0,
        stereo_width=0.5,
        tags=["ansiedade", "pânico", "relaxamento"],
    ),
    "deep_sleep": Preset(
        name="Deep Sleep",
        description="Ruído marrom + batimento delta (2 Hz) + textura de chuva. "
                    "Induz sono profundo e restaurador.",
        noise_fn=brown_noise,
        volume=0.55,
        beat_type=BeatType.BINAURAL,
        binaural_base=180.0,
        binaural_beat=2.0,           # Delta: sono profundo
        nature_texture="rain",
        nature_volume=0.12,
        fade_in_s=15.0,
        fade_out_s=20.0,
        lufs_target=-25.0,
        tags=["sono", "insônia", "delta"],
    ),
    "focus": Preset(
        name="Focus & Flow",
        description="Ruído rosa + batimento beta (18 Hz). "
                    "Aumenta concentração e estado de flow.",
        noise_fn=pink_noise,
        volume=0.60,
        beat_type=BeatType.BINAURAL,
        binaural_base=240.0,
        binaural_beat=18.0,          # Beta baixo: foco sem tensão
        fade_in_s=6.0,
        fade_out_s=8.0,
        lufs_target=-20.0,
        tags=["foco", "estudo", "trabalho", "beta"],
    ),
    "meditation": Preset(
        name="Deep Meditation",
        description="Ruído verde + batimento theta (6 Hz) + tom solfeggio 432 Hz. "
                    "Meditação profunda e introspecção.",
        noise_fn=green_noise,
        volume=0.55,
        beat_type=BeatType.BINAURAL,
        binaural_base=200.0,
        binaural_beat=6.0,           # Theta: meditação e criatividade
        solfeggio_freq=432.0,
        stereo_width=0.6,
        fade_in_s=12.0,
        fade_out_s=15.0,
        tags=["meditação", "theta", "mindfulness"],
    ),
    "stress_relief": Preset(
        name="Stress Relief",
        description="Ruído cinza equalizado + textura de água + pulso isocrônico alpha. "
                    "Desaceleração do sistema nervoso simpático.",
        noise_fn=grey_noise,
        volume=0.58,
        beat_type=BeatType.ISOCHRONIC,
        binaural_beat=9.0,
        nature_texture="water",
        nature_volume=0.18,
        stereo_width=0.55,
        tags=["estresse", "cortisol", "sistema nervoso"],
    ),
    "white_pure": Preset(
        name="Pure White Noise",
        description="Ruído branco clássico — mascara sons externos. "
                    "Excelente para bebês, foco e masking de tinnitus.",
        noise_fn=white_noise,
        volume=0.50,
        stereo_width=0.35,
        lufs_target=-22.0,
        tags=["masking", "bebê", "tinnitus", "foco"],
    ),
    "nature_forest": Preset(
        name="Forest Ambience",
        description="Ruído rosa + textura de floresta + solfeggio 963 Hz (consciência). "
                    "Sensação de estar em meio à natureza.",
        noise_fn=pink_noise,
        volume=0.50,
        nature_texture="forest",
        nature_volume=0.30,
        solfeggio_freq=963.0,
        stereo_width=0.65,
        fade_in_s=10.0,
        fade_out_s=15.0,
        tags=["natureza", "floresta", "grounding"],
    ),
}


# ---------------------------------------------------------------------------
# Engine de renderização
# ---------------------------------------------------------------------------

def render(
    preset_key: str = "anxiety_relief",
    duration_s: float = 3600.0,
    output_path: str | Path = "output.wav",
    seed: int | None = None,
    verbose: bool = True,
) -> Path:
    """
    Renderiza um arquivo de áudio terapêutico completo.

    Args:
        preset_key:   Chave do preset (ver PRESETS).
        duration_s:   Duração em segundos.
        output_path:  Caminho do arquivo WAV de saída.
        seed:         Semente aleatória para reprodutibilidade.
        verbose:      Exibe progresso.

    Returns:
        Path do arquivo gerado.
    """
    if preset_key not in PRESETS:
        raise ValueError(f"Preset '{preset_key}' não encontrado. Disponíveis: {list(PRESETS.keys())}")

    preset = PRESETS[preset_key]
    output_path = Path(output_path)
    n = int(duration_s * SAMPLE_RATE)

    if seed is not None:
        np.random.seed(seed)

    if verbose:
        log.info("Preset     : %s — %s", preset_key, preset.name)
        log.info("Duração    : %.0f min (%.0f s)", duration_s / 60, duration_s)
        log.info("Saída      : %s", output_path)
        t0 = time.time()

    # ------------------------------------------------------------------
    # 1. Ruído base
    # ------------------------------------------------------------------
    if verbose:
        log.info("[1/6] Gerando ruído base (%s)...", preset.noise_fn.__name__)
    base = preset.noise_fn(n)

    # ------------------------------------------------------------------
    # 2. Camada de natureza (opcional)
    # ------------------------------------------------------------------
    if preset.nature_texture:
        if verbose:
            log.info("[2/6] Adicionando textura de natureza (%s)...", preset.nature_texture)
        nature = nature_layer(n, preset.nature_texture)
        base = base * (1 - preset.nature_volume) + nature * preset.nature_volume
    else:
        if verbose:
            log.info("[2/6] Sem textura de natureza.")

    # ------------------------------------------------------------------
    # 3. Criação stereo com decorrelação
    # ------------------------------------------------------------------
    if verbose:
        log.info("[3/6] Criando imagem stereo (width=%.2f)...", preset.stereo_width)
    left, right = stereo_spread(base, width=preset.stereo_width)
    left, right = stereo_pan_dynamic(left, right)

    # ------------------------------------------------------------------
    # 4. Batimentos binaurais / isocrônicos
    # ------------------------------------------------------------------
    if preset.beat_type == BeatType.BINAURAL:
        if verbose:
            log.info("[4/6] Adicionando binaural %.1f Hz (base=%.0f Hz)...",
                     preset.binaural_beat, preset.binaural_base)
        bl, br = binaural_beat(n, preset.binaural_base, preset.binaural_beat)
        left  = left  + bl
        right = right + br
    elif preset.beat_type == BeatType.ISOCHRONIC:
        if verbose:
            log.info("[4/6] Adicionando isocrônico %.1f Hz...", preset.binaural_beat)
        iso = isochronic_pulse(n, preset.binaural_beat)
        left  = left  + iso
        right = right + iso
    else:
        if verbose:
            log.info("[4/6] Sem entrainment cerebral.")

    # Tom solfeggio
    if preset.solfeggio_freq:
        if verbose:
            log.info("       + Tom solfeggio %.0f Hz", preset.solfeggio_freq)
        sol = solfeggio_tone(n, preset.solfeggio_freq)
        left  = left  + sol
        right = right + sol

    # ------------------------------------------------------------------
    # 5. Modulação de volume lenta + envelope fade in/out
    # ------------------------------------------------------------------
    if verbose:
        log.info("[5/6] Aplicando envelope e modulação de volume...")
    mod = slow_volume_modulation(n)
    envelope = make_envelope(n, preset.fade_in_s, preset.fade_out_s, shape="cosine")

    left  = left  * mod * envelope * preset.volume
    right = right * mod * envelope * preset.volume

    # Soft limiter anti-clipping
    left  = soft_limiter(left)
    right = soft_limiter(right)

    # ------------------------------------------------------------------
    # 6. Normalização LUFS e exportação
    # ------------------------------------------------------------------
    if verbose:
        log.info("[6/6] Normalizando (alvo %.0f LUFS) e exportando...", preset.lufs_target)
    left  = normalize_lufs(left,  preset.lufs_target)
    right = normalize_lufs(right, preset.lufs_target)

    stereo = np.column_stack([left, right])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), stereo, SAMPLE_RATE, subtype=BIT_DEPTH)

    if verbose:
        elapsed = time.time() - t0
        size_mb = output_path.stat().st_size / 1_048_576
        log.info("✓ Concluído em %.1f s — arquivo: %.1f MB → %s", elapsed, size_mb, output_path)

    return output_path


# ---------------------------------------------------------------------------
# Renderização em lote
# ---------------------------------------------------------------------------

def render_batch(
    presets: list[str] | None = None,
    duration_s: float = 3600.0,
    output_dir: str | Path = "output",
) -> list[Path]:
    """
    Renderiza múltiplos presets em sequência.
    presets=None renderiza todos os presets disponíveis.
    """
    output_dir = Path(output_dir)
    keys = presets or list(PRESETS.keys())
    results = []
    for key in keys:
        out = output_dir / f"{key}.wav"
        try:
            results.append(render(key, duration_s, out))
        except Exception as exc:
            log.error("Erro no preset '%s': %s", key, exc)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Relaxation Sound Generator — Professional Edition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--preset",    default="anxiety_relief",
                   help="Preset terapêutico (default: anxiety_relief)")
    p.add_argument("--duration",  type=float, default=3600.0,
                   help="Duração em segundos (default: 3600 = 1h)")
    p.add_argument("--output",    default=None,
                   help="Arquivo WAV de saída (default: <preset>.wav)")
    p.add_argument("--seed",      type=int, default=None,
                   help="Semente aleatória para reprodutibilidade")
    p.add_argument("--batch",     action="store_true",
                   help="Renderiza todos os presets em lote")
    p.add_argument("--output-dir", default="output",
                   help="Diretório de saída para renderização em lote")
    p.add_argument("--list-presets", action="store_true",
                   help="Lista todos os presets disponíveis e sai")
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.list_presets:
        print("\n=== Presets disponíveis ===\n")
        for key, preset in PRESETS.items():
            tags = ", ".join(preset.tags)
            print(f"  {key:<20} {preset.name}")
            print(f"  {'':20} {preset.description}")
            print(f"  {'':20} Tags: {tags}\n")
        sys.exit(0)

    if args.batch:
        render_batch(duration_s=args.duration, output_dir=args.output_dir)
        return

    output = args.output or f"{args.preset}.wav"
    render(
        preset_key=args.preset,
        duration_s=args.duration,
        output_path=output,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()