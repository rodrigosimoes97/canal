"""
Zen Automator — Versão Refinada (Som Natural & Profissional)
"""

import numpy as np
import soundfile as sf
import subprocess
import argparse
from scipy.signal import butter, sosfilt

SAMPLE_RATE = 44100

# ---------------------------------------------------------------------------
# UTILIDADES DSP
# ---------------------------------------------------------------------------

def soft_saturation(x, drive=1.2):
    return np.tanh(x * drive)

def simple_reverb(signal, decay=0.4, delay_ms=120):
    delay = int(SAMPLE_RATE * delay_ms / 1000)
    out = np.copy(signal)
    for i in range(delay, len(signal)):
        out[i] += signal[i - delay] * decay
    return out

def lowpass(data, cutoff=600):
    sos = butter(4, cutoff / (SAMPLE_RATE / 2), btype='low', output='sos')
    return sosfilt(sos, data)

def highcut(data, cutoff=40):
    sos = butter(2, cutoff / (SAMPLE_RATE / 2), btype='high', output='sos')
    return sosfilt(sos, data)

# ---------------------------------------------------------------------------
# NOISE MAIS SUAVE
# ---------------------------------------------------------------------------

def generate_brown_noise(n):
    white = np.random.normal(0, 1, n)
    brown = np.cumsum(white)
    brown = highcut(brown, 30)
    brown = lowpass(brown, 500)
    return brown / (np.max(np.abs(brown)) + 1e-9)

# ---------------------------------------------------------------------------
# BINAURAL MAIS SUAVE
# ---------------------------------------------------------------------------

def apply_binaural(n, base_freq=180, beat_freq=4):
    t = np.linspace(0, n / SAMPLE_RATE, n)
    left = np.sin(2 * np.pi * base_freq * t)
    right = np.sin(2 * np.pi * (base_freq + beat_freq) * t)
    return left * 0.05, right * 0.05

# ---------------------------------------------------------------------------
# MOTOR MUSICAL MELHORADO
# ---------------------------------------------------------------------------

class MusicEngine:

    @staticmethod
    def smooth_pad(t, freq):
        # Osciladores múltiplos levemente desafinados
        osc1 = np.sin(2 * np.pi * freq * t)
        osc2 = np.sin(2 * np.pi * (freq * 1.01) * t)
        osc3 = np.sin(2 * np.pi * (freq * 0.99) * t)

        pad = (osc1 + osc2 + osc3) / 3

        # Envelope ultra suave
        env = np.exp(-t * 0.05)
        return pad * env * 0.3

    @staticmethod
    def generate(n, root=110):
        t = np.linspace(0, n / SAMPLE_RATE, n)

        # Frequências consonantes
        freqs = [
            root,
            root * 1.25,
            root * 1.5,
            root * 2.0
        ]

        out = np.zeros(n)

        for f in freqs:
            out += MusicEngine.smooth_pad(t, f)

        # Melodia lenta mais previsível
        step = int(SAMPLE_RATE * 6)
        for i in range(0, n, step):
            dur = int(SAMPLE_RATE * 4)
            if i + dur >= n:
                break

            f = np.random.choice(freqs) * 2
            tt = np.linspace(0, dur / SAMPLE_RATE, dur)
            note = np.sin(2 * np.pi * f * tt)

            env = np.exp(-tt * 0.8)
            out[i:i+dur] += note * env * 0.15

        out = lowpass(out, 700)
        return out

# ---------------------------------------------------------------------------
# RENDER FINAL MELHORADO
# ---------------------------------------------------------------------------

def render(output="zen.wav", duration=120):
    n = duration * SAMPLE_RATE

    print("[ZEN] Gerando base sonora refinada...")

    noise = generate_brown_noise(n) * 0.15
    music = MusicEngine.generate(n)
    bl, br = apply_binaural(n)

    mix = noise + music

    # Reverb ESSENCIAL
    mix = simple_reverb(mix, decay=0.5)

    # Saturação leve (calor analógico)
    mix = soft_saturation(mix, 1.1)

    # Stereo
    left = mix + bl
    right = mix + br

    stereo = np.column_stack((left, right))

    # Fade
    fade = int(SAMPLE_RATE * 8)
    stereo[:fade] *= np.linspace(0, 1, fade)[:, None]
    stereo[-fade:] *= np.linspace(1, 0, fade)[:, None]

    # Normalização suave
    stereo /= np.max(np.abs(stereo)) + 1e-9
    stereo *= 0.85

    sf.write(output, stereo, SAMPLE_RATE)

    print(f"[ZEN] Finalizado: {output}")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=120)
    parser.add_argument("--output", default="zen_v1.wav")
    args = parser.parse_args()

    render(args.output, args.duration)

if __name__ == "__main__":
    main()