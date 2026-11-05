"""
Zen Automator Final — Ultra Smooth Melodic Edition
Baseado na V7, com foco em melodias em escala, sinos de vento e elementos calmantes.
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
# UTILIDADES DSP (COM ESTADO)
# ---------------------------------------------------------------------------

class DSPState:
    def __init__(self):
        # Filtro de instrumentos (Pads)
        self.lp_sos = butter(4, 500 / (SAMPLE_RATE / 2), btype='low', output='sos')
        self.lp_zi = np.zeros((self.lp_sos.shape[0], 2))
        
        # Filtro final de mixagem
        self.final_lp_sos = butter(4, 1200 / (SAMPLE_RATE / 2), btype='low', output='sos')
        self.final_lp_zi = np.zeros((self.final_lp_sos.shape[0], 2))

    def apply_lowpass(self, data, final=False):
        sos = self.final_lp_sos if final else self.lp_sos
        zi = self.final_lp_zi if final else self.lp_zi
        y, zf = sosfilt(sos, data, zi=zi)
        if final: self.final_lp_zi = zf
        else: self.lp_zi = zf
        return y

def soft_saturation(x, drive=1.01):
    return np.tanh(x * drive)

# ---------------------------------------------------------------------------
# SÍNTESE DE INSTRUMENTOS
# ---------------------------------------------------------------------------

class CalmingSynth:
    @staticmethod
    def apply_envelope(signal, note_start_abs, note_dur, chunk_start_abs, attack=3.0, release=4.0):
        n = len(signal)
        t_abs = np.arange(n) / SAMPLE_RATE + chunk_start_abs
        t_rel = t_abs - note_start_abs
        
        env = np.ones(n)
        # Attack
        attack_mask = (t_rel >= 0) & (t_rel < attack)
        env[attack_mask] = t_rel[attack_mask] / attack
        # Release
        release_start = note_dur - release
        release_mask = (t_rel > release_start)
        env[release_mask] = 1.0 - (t_rel[release_mask] - release_start) / release
        # Fora da nota
        env[t_rel < 0] = 0
        env[t_rel > note_dur] = 0
        
        return signal * np.clip(env, 0, 1)

    @staticmethod
    def synth_wind_chime(t, freq):
        """Simula um sino metálico suave (senoide com harmônicos puros)"""
        out = np.sin(2 * np.pi * freq * t)
        out += 0.4 * np.sin(2 * np.pi * freq * 2.52 * t) # Harmônico não-inteiro para metal
        out += 0.2 * np.sin(2 * np.pi * freq * 4.1 * t)
        return out

    @staticmethod
    def synth_pad(t, freq):
        """Pad ultra suave"""
        out = np.sin(2 * np.pi * freq * t)
        out += 0.3 * np.sin(2 * np.pi * (freq * 1.002) * t)
        out += 0.3 * np.sin(2 * np.pi * (freq * 0.998) * t)
        lfo = 0.8 + 0.2 * np.sin(2 * np.pi * 0.04 * t)
        return out * lfo

# ---------------------------------------------------------------------------
# GESTOR DE ÁUDIOS EXTERNOS
# ---------------------------------------------------------------------------

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
                
                num_occ = max(2, int((total_duration_s / 120) * random.randint(2, 5)))
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
                    mix[write_start : write_start + length] += ev['data'][read_start : read_start + length] * 0.12
        return mix

# ---------------------------------------------------------------------------
# MOTOR ZEN FINAL
# ---------------------------------------------------------------------------

class ZenEngineFinal:
    def __init__(self, root=110):
        self.root = root
        # Escala Lídia (muito usada para sons etéreos/sonho)
        self.scale = [1.0, 1.125, 1.25, 1.406, 1.5, 1.687, 1.875, 2.0]

    def generate_chunk(self, start_s, duration_s, dsp):
        n = int(duration_s * SAMPLE_RATE)
        t = np.arange(n) / SAMPLE_RATE + start_s
        mix = np.zeros(n, dtype=np.float32)

        # 1. PADS (Oceano de som)
        for f_mult in [0.5, 1.0, 1.5]:
            pad = CalmingSynth.synth_pad(t, self.root * f_mult)
            mix += (pad * 0.18).astype(np.float32)

        # 2. MELODIA NA ESCALA (Mais presente)
        melody_step = 6 # Intervalos menores
        for i in range(int((start_s - 10) // melody_step) * melody_step, int((start_s + duration_s) // melody_step + 1) * melody_step, melody_step):
            random.seed(i + 888)
            if random.random() > 0.4: # 60% de chance
                f_mult = random.choice(self.scale)
                freq = self.root * f_mult * 2
                note_start = i + random.uniform(0, 2)
                note_dur = random.uniform(5, 8)
                
                if note_start < start_s + duration_s and note_start + note_dur > start_s:
                    n_start = max(0, int((note_start - start_s) * SAMPLE_RATE))
                    n_end = min(n, int((note_start + note_dur - start_s) * SAMPLE_RATE))
                    if n_end > n_start:
                        t_n = np.arange(n_end - n_start) / SAMPLE_RATE + max(0, start_s - note_start)
                        wave = np.sin(2 * np.pi * freq * t_n)
                        wave = CalmingSynth.apply_envelope(wave, note_start, note_dur, max(start_s, note_start), attack=2.5, release=3.5)
                        mix[n_start:n_end] += (wave * 0.12).astype(np.float32)

        # 3. SINOS DE VENTO (Wind Chimes) - Elemento Espacial
        chime_step = 15
        for i in range(int((start_s - 20) // chime_step) * chime_step, int((start_s + duration_s) // chime_step + 1) * chime_step, chime_step):
            random.seed(i + 111)
            if random.random() > 0.3:
                f_mult = random.choice(self.scale)
                freq = self.root * f_mult * 8 # Bem agudo
                note_start = i + random.uniform(0, 10)
                note_dur = random.uniform(8, 12)
                
                if note_start < start_s + duration_s and note_start + note_dur > start_s:
                    n_start = max(0, int((note_start - start_s) * SAMPLE_RATE))
                    n_end = min(n, int((note_start + note_dur - start_s) * SAMPLE_RATE))
                    if n_end > n_start:
                        t_c = np.arange(n_end - n_start) / SAMPLE_RATE + max(0, start_s - note_start)
                        chime = CalmingSynth.synth_wind_chime(t_c, freq)
                        # Envelope ultra longo para os sinos
                        chime = CalmingSynth.apply_envelope(chime, note_start, note_dur, max(start_s, note_start), attack=4.0, release=6.0)
                        mix[n_start:n_end] += (chime * 0.05).astype(np.float32)

        # 4. PULSO DE RELAXAMENTO (Soft Heartbeat)
        pulse_bpm = 50
        pulse_period = 60 / pulse_bpm
        pulse_start_idx = int(start_s // pulse_period)
        for i in range(pulse_start_idx, int((start_s + duration_s) // pulse_period + 1)):
            p_time = i * pulse_period
            if p_time >= start_s and p_time < start_s + duration_s:
                idx = int((p_time - start_s) * SAMPLE_RATE)
                # Pequeno burst de senoide grave (40Hz)
                burst_len = int(0.5 * SAMPLE_RATE)
                if idx + burst_len < n:
                    t_p = np.arange(burst_len) / SAMPLE_RATE
                    # Dois batimentos (tum-tum)
                    heart = np.sin(2 * np.pi * 40 * t_p) * np.exp(-15 * t_p)
                    heart += 0.6 * np.sin(2 * np.pi * 40 * (t_p - 0.2)) * np.exp(-15 * (t_p - 0.2)) * (t_p > 0.2)
                    mix[idx : idx + burst_len] += (heart * 0.08).astype(np.float32)

        return dsp.apply_lowpass(mix)

# ---------------------------------------------------------------------------
# RENDERIZADOR
# ---------------------------------------------------------------------------

def render(output="zen_teste.wav", duration_s=120):
    total_samples = int(duration_s * SAMPLE_RATE)
    print(f"[ZEN FINAL] Renderizando Experiência Melódica ({duration_s}s)...")
    
    dsp = DSPState()
    ext_manager = ExternalAudioManagerV7(total_duration_s=duration_s)
    music_engine = ZenEngineFinal(root=110)
    
    with sf.SoundFile(output, mode='w', samplerate=SAMPLE_RATE, channels=2) as file:
        current_sample = 0
        while current_sample < total_samples:
            chunk_end = min(current_sample + CHUNK_SAMPLES, total_samples)
            actual_len = chunk_end - current_sample
            t_start = current_sample / SAMPLE_RATE
            d_chunk = actual_len / SAMPLE_RATE
            
            print(f" > Progresso: {t_start:.1f}s / {duration_s}s", end='\r')
            
            music = music_engine.generate_chunk(t_start, d_chunk, dsp)
            external = ext_manager.get_mix(current_sample, chunk_end)
            
            # Mixagem Final
            mix = music + external
            mix = soft_saturation(mix, 1.01)
            mix = dsp.apply_lowpass(mix, final=True)
            
            # Binaural (Theta 4Hz - Relaxamento profundo e meditação)
            t_arr = np.arange(actual_len) / SAMPLE_RATE + t_start
            left = mix + np.sin(2 * np.pi * 100 * t_arr) * 0.015
            right = mix + np.sin(2 * np.pi * 104 * t_arr) * 0.015
            stereo = np.column_stack((left, right))
            
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
            file.write(stereo * 0.85)
            current_sample = chunk_end
            
    print(f"\n[ZEN FINAL] Sucesso: {output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=120)
    parser.add_argument("--output", default="zen_teste.wav")
    args = parser.parse_args()
    render(args.output, args.duration)
