"""
Zen Automator Final — Premium Ethereal Edition (V8)
Correção de audibilidade: Filtros abertos, volumes ajustados e estéreo real.
"""

import numpy as np
import soundfile as sf
import argparse
import os
import random
from scipy.signal import butter, sosfilt, resample

SAMPLE_RATE = 44100
CHUNK_DURATION = 10 
CHUNK_SAMPLES = CHUNK_DURATION * SAMPLE_RATE

# ---------------------------------------------------------------------------
# UTILIDADES DSP (ESTÉREO COM ESTADO)
# ---------------------------------------------------------------------------

class StereoDSP:
    def __init__(self, lp_cutoff=2000, final_cutoff=3000):
        # Filtro para os instrumentos (mais aberto para ouvir flauta e sinos)
        self.lp_sos = butter(4, lp_cutoff / (SAMPLE_RATE / 2), btype='low', output='sos')
        self.lp_zi_l = np.zeros((self.lp_sos.shape[0], 2))
        self.lp_zi_r = np.zeros((self.lp_sos.shape[0], 2))
        
        # Filtro final de mixagem
        self.final_sos = butter(4, final_cutoff / (SAMPLE_RATE / 2), btype='low', output='sos')
        self.final_zi_l = np.zeros((self.final_sos.shape[0], 2))
        self.final_zi_r = np.zeros((self.final_sos.shape[0], 2))

    def process_instr(self, l, r):
        l_filt, self.lp_zi_l = sosfilt(self.lp_sos, l, zi=self.lp_zi_l)
        r_filt, self.lp_zi_r = sosfilt(self.lp_sos, r, zi=self.lp_zi_r)
        return l_filt, r_filt

    def process_final(self, l, r):
        l_filt, self.final_zi_l = sosfilt(self.final_sos, l, zi=self.final_zi_l)
        r_filt, self.final_zi_r = sosfilt(self.final_sos, r, zi=self.final_zi_r)
        return l_filt, r_filt

def soft_saturation(x, drive=1.02):
    return np.tanh(x * drive)

# ---------------------------------------------------------------------------
# SÍNTESE DE INSTRUMENTOS
# ---------------------------------------------------------------------------

class PremiumSynth:
    @staticmethod
    def apply_envelope(signal, note_start_abs, note_dur, chunk_start_abs, attack=3.0, release=4.0):
        n = len(signal)
        t_abs = np.arange(n) / SAMPLE_RATE + chunk_start_abs
        t_rel = t_abs - note_start_abs
        env = np.ones(n)
        attack_mask = (t_rel >= 0) & (t_rel < attack)
        env[attack_mask] = t_rel[attack_mask] / attack
        release_start = note_dur - release
        release_mask = (t_rel > release_start)
        env[release_mask] = 1.0 - (t_rel[release_mask] - release_start) / release
        env[t_rel < 0] = 0
        env[t_rel > note_dur] = 0
        return signal * np.clip(env, 0, 1)

    @staticmethod
    def synth_flute(t, freq):
        """Flauta FM melhorada: mais presente e soprada"""
        # Modulação leve para dar corpo
        modulator = 1.5 * np.sin(2 * np.pi * (freq * 1.001) * t)
        carrier = np.sin(2 * np.pi * freq * t + modulator)
        # Adiciona um pouco de 'madeira' (harmônicos ímpares)
        wood = 0.2 * np.sin(2 * np.pi * freq * 3 * t)
        # Vibrato e Sopro
        vibrato = 1.0 + 0.005 * np.sin(2 * np.pi * 5 * t)
        breath = 0.03 * np.random.normal(0, 0.1, len(t))
        return (carrier + wood) * vibrato + breath

    @staticmethod
    def synth_bell(t, freq):
        """Sino cristalino"""
        out = np.sin(2 * np.pi * freq * t)
        out += 0.4 * np.sin(2 * np.pi * freq * 2.51 * t)
        return out

    @staticmethod
    def apply_panning(signal, t_abs, speed=0.05, depth=0.8):
        pan = depth * np.sin(2 * np.pi * speed * t_abs)
        l = signal * (0.5 - pan/2)
        r = signal * (0.5 + pan/2)
        return l, r

# ---------------------------------------------------------------------------
# MOTOR ZEN FINAL
# ---------------------------------------------------------------------------

class ZenEngineV8:
    def __init__(self, root=110):
        self.root = root
        self.scale = [1.0, 1.25, 1.333, 1.5, 1.875, 2.0] # Pentatônica / Lídia

    def generate_chunk(self, start_s, duration_s):
        n = int(duration_s * SAMPLE_RATE)
        t = np.arange(n) / SAMPLE_RATE + start_s
        mix_l = np.zeros(n, dtype=np.float32)
        mix_r = np.zeros(n, dtype=np.float32)

        # 1. PADS (Oceano de fundo)
        for i, f_mult in enumerate([0.5, 1.0, 1.5]):
            pad = np.sin(2 * np.pi * self.root * f_mult * t)
            pad += 0.3 * np.sin(2 * np.pi * (self.root * f_mult * 1.002) * t)
            l, r = PremiumSynth.apply_panning(pad, t, speed=0.02 + i*0.01, depth=0.4)
            mix_l += l * 0.15
            mix_r += r * 0.15

        # 2. FLAUTA (Melodia Principal) - VOLUME AUMENTADO
        step = 8
        for i in range(int((start_s - 10) // step) * step, int((start_s + duration_s) // step + 1) * step, step):
            random.seed(i + 777)
            if random.random() > 0.4:
                freq = self.root * random.choice(self.scale) * 2
                note_start = i + random.uniform(0, 3)
                note_dur = random.uniform(6, 10)
                if note_start < start_s + duration_s and note_start + note_dur > start_s:
                    n_start = max(0, int((note_start - start_s) * SAMPLE_RATE))
                    n_end = min(n, int((note_start + note_dur - start_s) * SAMPLE_RATE))
                    if n_end > n_start:
                        t_n = np.arange(n_end - n_start) / SAMPLE_RATE + max(0, start_s - note_start)
                        wave = PremiumSynth.synth_flute(t_n, freq)
                        wave = PremiumSynth.apply_envelope(wave, note_start, note_dur, max(start_s, note_start), attack=3.0, release=4.0)
                        l, r = PremiumSynth.apply_panning(wave, t[n_start:n_end], speed=0.04, depth=0.7)
                        mix_l[n_start:n_end] += l * 0.25 # Aumentado de 0.12 para 0.25
                        mix_r[n_start:n_end] += r * 0.25

        # 3. SINOS (Brilho) - VOLUME AUMENTADO
        step_bell = 15
        for i in range(int((start_s - 15) // step_bell) * step_bell, int((start_s + duration_s) // step_bell + 1) * step_bell, step_bell):
            random.seed(i + 123)
            if random.random() > 0.5:
                freq = self.root * random.choice(self.scale) * 6
                note_start = i + random.uniform(0, 10)
                note_dur = random.uniform(5, 9)
                if note_start < start_s + duration_s and note_start + note_dur > start_s:
                    n_start = max(0, int((note_start - start_s) * SAMPLE_RATE))
                    n_end = min(n, int((note_start + note_dur - start_s) * SAMPLE_RATE))
                    if n_end > n_start:
                        t_c = np.arange(n_end - n_start) / SAMPLE_RATE + max(0, start_s - note_start)
                        bell = PremiumSynth.synth_bell(t_c, freq)
                        bell = PremiumSynth.apply_envelope(bell, note_start, note_dur, max(start_s, note_start), attack=0.2, release=8.0)
                        mix_l[n_start:n_end] += bell * 0.12 # Aumentado de 0.05 para 0.12
                        mix_r[n_start:n_end] += bell * 0.12

        # 4. PULSO (Tum-Tum)
        p_period = 60 / 50
        for i in range(int(start_s // p_period), int((start_s + duration_s) // p_period + 1)):
            p_time = i * p_period
            if p_time >= start_s and p_time < start_s + duration_s:
                idx = int((p_time - start_s) * SAMPLE_RATE)
                if idx + int(0.5*SAMPLE_RATE) < n:
                    t_p = np.arange(int(0.5*SAMPLE_RATE)) / SAMPLE_RATE
                    heart = np.sin(2 * np.pi * 40 * t_p) * np.exp(-15 * t_p)
                    mix_l[idx : idx + len(heart)] += heart * 0.1
                    mix_r[idx : idx + len(heart)] += heart * 0.1

        return mix_l, mix_r

# ---------------------------------------------------------------------------
# RENDERIZADOR
# ---------------------------------------------------------------------------

def render(output="zen_teste.wav", duration_s=120):
    total_samples = int(duration_s * SAMPLE_RATE)
    print(f"[ZEN V8] Renderizando com Melhoria de Audibilidade ({duration_s}s)...")
    
    dsp = StereoDSP(lp_cutoff=2000, final_cutoff=3000)
    ext_manager = ExternalAudioManagerV7(total_duration_s=duration_s)
    music_engine = ZenEngineV8(root=110)
    
    with sf.SoundFile(output, mode='w', samplerate=SAMPLE_RATE, channels=2) as file:
        current_sample = 0
        while current_sample < total_samples:
            chunk_end = min(current_sample + CHUNK_SAMPLES, total_samples)
            actual_len = chunk_end - current_sample
            t_start = current_sample / SAMPLE_RATE
            d_chunk = actual_len / SAMPLE_RATE
            
            print(f" > Progresso: {t_start:.1f}s / {duration_s}s", end='\r')
            
            # Gerar Camadas
            music_l, music_r = music_engine.generate_chunk(t_start, d_chunk)
            music_l, music_r = dsp.process_instr(music_l, music_r)
            
            external = ext_manager.get_mix(current_sample, chunk_end)
            
            # Mixagem
            final_l = music_l + external
            final_r = music_r + external
            
            # DSP Final
            final_l = soft_saturation(final_l)
            final_r = soft_saturation(final_r)
            final_l, final_r = dsp.process_final(final_l, final_r)
            
            # Binaural Delta (2.5Hz)
            t_arr = np.arange(actual_len) / SAMPLE_RATE + t_start
            stereo = np.column_stack((
                final_l + np.sin(2 * np.pi * 100 * t_arr) * 0.015,
                final_r + np.sin(2 * np.pi * 102.5 * t_arr) * 0.015
            ))
            
            # Fades Globais
            fade_len = int(SAMPLE_RATE * 15)
            if current_sample < fade_len:
                f_in = np.linspace(0, 1, fade_len)
                for i in range(actual_len):
                    if current_sample + i < fade_len: stereo[i] *= f_in[current_sample + i]
            if chunk_end > total_samples - fade_len:
                f_out = np.linspace(1, 0, fade_len)
                for i in range(actual_len):
                    idx = current_sample + i
                    if idx > total_samples - fade_len: stereo[i] *= f_out[idx - (total_samples - fade_len)]

            max_v = np.max(np.abs(stereo))
            if max_v > 1.0: stereo /= max_v
            file.write(stereo * 0.88)
            current_sample = chunk_end
            
    print(f"\n[ZEN V8] Sucesso: {output}")

# Reutilizando a classe anterior para não quebrar dependência
class ExternalAudioManagerV7:
    def __init__(self, folder="audios", total_duration_s=120):
        self.events = []
        if not os.path.exists(folder): return
        files = [f for f in os.listdir(folder) if f.endswith(('.mp3', '.wav', '.ogg'))]
        for file_name in files:
            try:
                path = os.path.join(folder, file_name)
                data, fs = sf.read(path)
                if len(data.shape) > 1: data = np.mean(data, axis=1)
                if fs != SAMPLE_RATE:
                    num = int(len(data) * SAMPLE_RATE / fs)
                    data = resample(data, num)
                num_occ = max(3, int((total_duration_s / 120) * random.randint(3, 6)))
                for _ in range(num_occ):
                    start_s = random.uniform(0, max(0, total_duration_s - (len(data)/SAMPLE_RATE)))
                    self.events.append({
                        'data': data.astype(np.float32),
                        'start_sample': int(start_s * SAMPLE_RATE),
                        'end_sample': int(start_s * SAMPLE_RATE) + len(data)
                    })
            except: pass

    def get_mix(self, start_sample, end_sample):
        chunk_len = end_sample - start_sample
        mix = np.zeros(chunk_len, dtype=np.float32)
        for ev in self.events:
            ev_s = ev['start_sample']; ev_e = ev['end_sample']
            if ev_e > start_sample and ev_s < end_sample:
                read_start = max(0, start_sample - ev_s)
                write_start = max(0, ev_s - start_sample)
                length = min(len(ev['data']) - read_start, chunk_len - write_start)
                if length > 0:
                    mix[write_start : write_start + length] += ev['data'][read_start : read_start + length] * 0.15
        return mix

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=120)
    parser.add_argument("--output", default="zen_teste.wav")
    args = parser.parse_args()
    render(args.output, args.duration)
