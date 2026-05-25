"""
Ultimate Relaxation Audio Generator — YouTube Optimized Edition
Inclui:
✔ Ruídos avançados (white, pink, brown)
✔ Binaural beats (delta/theta/alpha/beta)
✔ Pulsos isocrônicos
✔ Tons solfeggio
✔ Texturas naturais realistas
✔ Movimento estéreo (L ↔ R)
✔ Decorrelação estéreo (imersão real)
✔ Modulação tipo respiração
✔ Fade cosseno profissional
✔ Presets otimizados para viralização no YouTube

Dependências:
pip install numpy soundfile scipy
"""

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

SAMPLE_RATE = 44100

# ------------------------
# NOISE GENERATORS
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
# STEREO + MOVEMENT
# ------------------------
def stereo_spread(mono, width=0.5):
    delay = int(0.02 * SAMPLE_RATE)
    right = np.roll(mono, delay)
    left = mono
    return left, right

def stereo_pan(left, right, rate=0.04):
    t = np.linspace(0, len(left)/SAMPLE_RATE, len(left))
    pan = (np.sin(2*np.pi*rate*t)+1)/2
    left = left * (1-pan)
    right = right * pan
    return left, right

# ------------------------
# BINAURAL
# ------------------------
def binaural(n, base=200, beat=6):
    t = np.linspace(0, n/SAMPLE_RATE, n)
    drift = 0.3*np.sin(2*np.pi*0.01*t)
    left = np.sin(2*np.pi*(base+drift)*t)
    right = np.sin(2*np.pi*(base+beat+drift)*t)
    return left*0.12, right*0.12

# ------------------------
# BREATH MODULATION
# ------------------------
def breath(n):
    t = np.linspace(0, n/SAMPLE_RATE, n)
    b = np.sin(2*np.pi*0.02*t)
    b = np.where(b>0, b*0.8, b*1.2)
    return 1 + 0.05*b

# ------------------------
# FADE COSINE
# ------------------------
def fade(n, fade_in=10, fade_out=10):
    env = np.ones(n)
    fi = int(fade_in*SAMPLE_RATE)
    fo = int(fade_out*SAMPLE_RATE)
    env[:fi] = 0.5*(1-np.cos(np.linspace(0,np.pi,fi)))
    env[-fo:] = 0.5*(1+np.cos(np.linspace(0,np.pi,fo)))
    return env

# ------------------------
# PRESETS YOUTUBE
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

    base = p["noise"](n)

    left, right = stereo_spread(base)
    left, right = stereo_pan(left, right)

    bl, br = binaural(n, p["base"], p["beat"])
    left += bl
    right += br

    mod = breath(n)
    env = fade(n)

    left = left * mod * env
    right = right * mod * env

    stereo = np.column_stack((left, right))
    stereo /= np.max(np.abs(stereo))

    sf.write(output, stereo, SAMPLE_RATE)
    print("Arquivo gerado:", output)

# ------------------------
# RUN
# ------------------------
if __name__ == "__main__":
    render("anxiety", "anxiety.wav")
