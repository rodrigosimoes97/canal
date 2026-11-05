"""
Zen Automator V2 — Ultimate Sleep Edition (Natureza + Áudios Externos)
"""

import numpy as np
import soundfile as sf
import argparse
import os
import random
from scipy.signal import butter, sosfilt, resample

SAMPLE_RATE = 44100

# ---------------------------------------------------------------------------
# UTILIDADES DSP
# ---------------------------------------------------------------------------

def soft_saturation(x, drive=1.1):
    return np.tanh(x * drive)

def simple_reverb(signal, decay=0.5, delay_ms=180):
    delay = int(SAMPLE_RATE * delay_ms / 1000)
    out = np.copy(signal)
    for i in range(delay, len(signal)):
        out[i] += signal[i - delay] * decay
    return out

def lowpass(data, cutoff=600):
    sos = butter(4, cutoff / (SAMPLE_RATE / 2), btype='low', output='sos')
    return sosfilt(sos, data)

def bandpass(data, low=500, high=3000):
    sos = butter(4, [low / (SAMPLE_RATE / 2), high / (SAMPLE_RATE / 2)], btype='band', output='sos')
    return sosfilt(sos, data)

# ---------------------------------------------------------------------------
# CARREGAMENTO DE ÁUDIOS EXTERNOS (NOVO)
# ---------------------------------------------------------------------------

def load_external_audios(duration_samples, folder="audios"):
    mix = np.zeros(duration_samples)
    if not os.path.exists(folder):
        return mix

    files = [f for f in os.listdir(folder) if f.endswith(('.mp3', '.wav', '.ogg'))]
    if not files:
        return mix

    # Tenta mixar até 2 áudios da pasta aleatoriamente
    selected_files = random.sample(files, min(len(files), 2))

    for file_name in selected_files:
        try:
            file_path = os.path.join(folder, file_name)
            data, fs = sf.read(file_path)
            
            # Converter para mono se for stereo para facilitar a mixagem base
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)
            
            # Resample se necessário
            if fs != SAMPLE_RATE:
                num_samples = int(len(data) * SAMPLE_RATE / fs)
                data = resample(data, num_samples)

            # Ajustar duração
            if len(data) < duration_samples:
                repeats = (duration_samples // len(data)) + 1
                data = np.tile(data, repeats)[:duration_samples]
            else:
                start = random.randint(0, len(data) - duration_samples)
                data = data[start:start+duration_samples]
            
            mix += data * 0.2 # Volume moderado para arquivos externos
            print(f"[ZEN] Mixado áudio externo: {file_name}")
        except Exception as e:
            print(f"[ERRO] Falha ao carregar {file_name}: {e}")
            
    return mix

# ---------------------------------------------------------------------------
# GERADORES AMBIENTAIS
# ---------------------------------------------------------------------------

def generate_rain(n):
    white = np.random.normal(0, 1, n)
    rain = np.cumsum(white) 
    rain = lowpass(rain, 1500)
    
    drops = np.zeros(n)
    num_drops = n // (SAMPLE_RATE // 10)
    drop_indices = np.random.randint(0, n, num_drops)
    drops[drop_indices] = np.random.uniform(0, 0.5, num_drops)
    drops = bandpass(drops, 1500, 4000) * 0.3
    
    mix = (rain / (np.max(np.abs(rain)) + 1e-9)) * 0.8 + drops
    return mix * 0.4

def generate_thunder(n):
    out = np.zeros(n)
    step = int(SAMPLE_RATE * 30)
    for i in range(0, n, step):
        if np.random.rand() < 0.5:
            pos = i + np.random.randint(0, step // 2)
            dur = int(SAMPLE_RATE * 10) 
            if pos + dur >= n: break
            t_noise = np.cumsum(np.random.normal(0, 1, dur))
            t_noise = lowpass(t_noise, 150) 
            tt = np.linspace(0, 1, dur)
            env = np.exp(-tt * 2) * (0.9 + 0.1 * np.sin(2 * np.pi * 3 * tt))
            out[pos:pos+dur] += (t_noise / (np.max(np.abs(t_noise)) + 1e-9)) * env * 0.6
    return out

# ---------------------------------------------------------------------------
# MOTOR MUSICAL
# ---------------------------------------------------------------------------

class SleepMusicEngine:
    @staticmethod
    def evolution_pad(t, freq):
        osc1 = np.sin(2 * np.pi * freq * t)
        osc2 = np.sin(2 * np.pi * (freq * 1.008) * t)
        lfo = 0.5 + 0.5 * np.sin(2 * np.pi * 0.08 * t) 
        return (osc1 + osc2) * lfo * 0.2

    @staticmethod
    def generate(n, root=110):
        t = np.linspace(0, n / SAMPLE_RATE, n)
        out = np.zeros(n)
        scale = [1.0, 1.25, 1.333, 1.5, 1.75, 2.0]
        
        # Pads que mudam de nota
        chunk = int(SAMPLE_RATE * 15)
        for i in range(0, n, chunk):
            dur = min(chunk, n - i)
            tt = np.linspace(0, dur / SAMPLE_RATE, dur)
            f_pad = root * np.random.choice([0.5, 0.75, 1.0])
            out[i:i+dur] += SleepMusicEngine.evolution_pad(tt, f_pad)

        # Melodias esporádicas
        current = 0
        while current < n:
            wait = int(np.random.uniform(4, 8) * SAMPLE_RATE)
            current += wait
            if current >= n: break
            dur_s = np.random.uniform(3, 6)
            dur_samples = int(dur_s * SAMPLE_RATE)
            if current + dur_samples >= n: break
            f = root * np.random.choice(scale) * np.random.choice([2, 3])
            tt = np.linspace(0, dur_s, dur_samples)
            note = np.sin(2 * np.pi * f * tt)
            env = np.exp(-tt * 1.0) * np.sin(np.pi * (tt/dur_s))
            out[current:current+dur_samples] += note * env * 0.12
            
        return lowpass(out, 700)

# ---------------------------------------------------------------------------
# RENDER FINAL
# ---------------------------------------------------------------------------

def render(output="zen_sleep.wav", duration=120):
    n = duration * SAMPLE_RATE
    print(f"[ZEN] Iniciando renderização final ({duration}s)...")

    # Gerar componentes procedurais
    rain = generate_rain(n)
    thunder = generate_thunder(n)
    music = SleepMusicEngine.generate(n, root=110)
    
    # Carregar áudios externos da pasta 'audios'
    external = load_external_audios(n, folder="audios")
    
    # Binaural Delta (2Hz)
    t = np.linspace(0, duration, n)
    bl_tone = np.sin(2 * np.pi * 100 * t) * 0.04
    br_tone = np.sin(2 * np.pi * 102 * t) * 0.04

    # Mixagem (Base Mono)
    mix = (rain * 0.7) + (music * 0.6) + (thunder * 0.8) + (external * 1.0)
    mix = simple_reverb(mix, decay=0.4)

    # Stereo + Binaural
    left = mix + bl_tone
    right = mix + br_tone
    stereo = np.column_stack((left, right))

    # Fade e Normalização
    fade = int(SAMPLE_RATE * 10)
    fade_in = np.linspace(0, 1, fade)
    fade_out = np.linspace(1, 0, fade)
    for c in range(2):
        stereo[:fade, c] *= fade_in
        stereo[-fade:, c] *= fade_out

    max_v = np.max(np.abs(stereo))
    if max_v > 0:
        stereo = (stereo / max_v) * 0.85

    sf.write(output, stereo, SAMPLE_RATE)
    print(f"[ZEN] Finalizado com sucesso: {output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=120)
    parser.add_argument("--output", default="zen_v3"
    ".wav")
    args = parser.parse_args()
    render(args.output, args.duration)
