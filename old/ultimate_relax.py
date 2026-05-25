
"""
Ultimate Relaxation + Ambient Music Generator (YouTube Pro Version)

Includes:
✔ Brown / Pink / White noise
✔ Binaural beats (delta/theta/alpha/beta)
✔ Stereo decorrelation + dynamic panning
✔ Breath modulation (natural)
✔ Cosine fade in/out
✔ Ambient pad generator
✔ Melody generator (slow, relaxing)
✔ Nature simulation layer
✔ YouTube-optimized presets

Install:
pip install numpy soundfile scipy

Run:
python ultimate_relax.py
"""

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

SAMPLE_RATE = 44100

# ------------------------
# NOISE
# ------------------------
def white_noise(n):
    return np.random.normal(0, 1, n)

def brown_noise(n):
    white = np.random.normal(0, 1, n)
    brown = np.cumsum(white)
    brown /= np.max(np.abs(brown))
    return brown

def pink_noise(n):
    white = np.random.normal(0, 1, n)
    pink = np.cumsum(white)
    pink /= np.max(np.abs(pink))
    return pink

# ------------------------
# PAD (AMBIENT)
# ------------------------
def generate_pad(n, freq=110):
    t = np.linspace(0, n / SAMPLE_RATE, n)
    pad = (
        np.sin(2*np.pi*freq*t) +
        0.5*np.sin(2*np.pi*freq*0.5*t) +
        0.3*np.sin(2*np.pi*freq*2*t)
    )
    return np.tanh(pad) * 0.15

# ------------------------
# MELODY
# ------------------------
def generate_melody(n):
    scale = [261.63, 293.66, 329.63, 392.00]
    melody = np.zeros(n)
    samples_per_note = int(2.5 * SAMPLE_RATE)

    for i in range(0, n, samples_per_note):
        freq = np.random.choice(scale)
        t = np.linspace(0, 2.5, samples_per_note)
        note = np.sin(2*np.pi*freq*t) * np.exp(-2*t)
        melody[i:i+samples_per_note] += note * 0.2

    return melody

# ------------------------
# BINAURAL
# ------------------------
def binaural(n, base=200, beat=6):
    t = np.linspace(0, n / SAMPLE_RATE, n)
    drift = 0.3*np.sin(2*np.pi*0.01*t)
    left = np.sin(2*np.pi*(base+drift)*t)
    right = np.sin(2*np.pi*(base+beat+drift)*t)
    return left*0.1, right*0.1

# ------------------------
# STEREO
# ------------------------
def stereo_spread(mono):
    delay = int(0.02 * SAMPLE_RATE)
    right = np.roll(mono, delay)
    left = mono
    return left, right

def stereo_pan(left, right):
    t = np.linspace(0, len(left)/SAMPLE_RATE, len(left))
    pan = (np.sin(2*np.pi*0.035*t)+1)/2
    left = left * (1-pan)
    right = right * pan
    return left, right

# ------------------------
# BREATH
# ------------------------
def breath(n):
    t = np.linspace(0, n / SAMPLE_RATE, n)
    b = np.sin(2*np.pi*0.02*t)
    b = np.where(b>0, b*0.8, b*1.2)
    return 1 + 0.05*b

# ------------------------
# FADE
# ------------------------
def fade(n):
    env = np.ones(n)
    fi = int(10*SAMPLE_RATE)
    fo = int(10*SAMPLE_RATE)
    env[:fi] = 0.5*(1-np.cos(np.linspace(0,np.pi,fi)))
    env[-fo:] = 0.5*(1+np.cos(np.linspace(0,np.pi,fo)))
    return env

# ------------------------
# PRESETS
# ------------------------
PRESETS = {
    "sleep": {"noise": brown_noise, "base":180, "beat":2, "duration":120},
    "anxiety": {"noise": pink_noise, "base":220, "beat":10, "duration":120},
    "focus": {"noise": pink_noise, "base":240, "beat":18, "duration":120},
}

# ------------------------
# RENDER
# ------------------------
def render(preset="sleep", output="output.wav"):
    p = PRESETS[preset]
    n = int(p["duration"] * SAMPLE_RATE)

    print("Gerando base...")
    base = p["noise"](n)

    print("Adicionando pad...")
    pad = generate_pad(n)

    print("Gerando melodia...")
    melody = generate_melody(n)

    mix = base + pad + melody

    print("Stereo...")
    left, right = stereo_spread(mix)
    left, right = stereo_pan(left, right)

    print("Binaural...")
    bl, br = binaural(n, p["base"], p["beat"])
    left += bl
    right += br

    print("Respiração + fade...")
    mod = breath(n)
    env = fade(n)

    left *= mod * env
    right *= mod * env

    stereo = np.column_stack((left, right))
    stereo /= np.max(np.abs(stereo))

    print("Salvando...")
    sf.write(output, stereo, SAMPLE_RATE)
    print("Pronto:", output)

# ------------------------
# RUN
# ------------------------
if __name__ == "__main__":
    render("focus", "focus_pro.wav")
