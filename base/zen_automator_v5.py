"""
Zen Automator V2 — Ultimate Fusion Edition (Instrumental Streaming)
Consolidação de: zen_automator, zen_automator_v2 e zen_automator_v3
Otimizado para longa duração com síntese de instrumentos (Pad, Cordas, Teclado)
"""

import numpy as np
import soundfile as sf
import argparse
import os
import random
from scipy.signal import butter, sosfilt, resample

SAMPLE_RATE = 44100
CHUNK_DURATION = 20  # Reduzido levemente para maior agilidade no streaming
CHUNK_SAMPLES = CHUNK_DURATION * SAMPLE_RATE

# ---------------------------------------------------------------------------
# UTILIDADES DSP
# ---------------------------------------------------------------------------

def soft_saturation(x, drive=1.2):
    return np.tanh(x * drive)

def simple_reverb(signal, decay=0.55, delay_ms=250):
    delay = int(SAMPLE_RATE * delay_ms / 1000)
    out = np.copy(signal)
    if len(signal) > delay:
        out[delay:] += signal[:-delay] * decay
    return out

def lowpass(data, cutoff=800):
    sos = butter(4, cutoff / (SAMPLE_RATE / 2), btype='low', output='sos')
    return sosfilt(sos, data)

# ---------------------------------------------------------------------------
# SÍNTESE DE INSTRUMENTOS (VIRTUAL INSTRUMENTS)
# ---------------------------------------------------------------------------

class InstrumentSynth:
    @staticmethod
    def apply_envelope(signal, attack_s, decay_s, sustain_level, release_s, duration_s):
        n = len(signal)
        t = np.linspace(0, duration_s, n)
        env = np.ones(n)
        
        # Attack
        attack_n = int(attack_s * SAMPLE_RATE)
        if attack_n > 0:
            env[:min(attack_n, n)] = np.linspace(0, 1, min(attack_n, n))
            
        # Release
        release_n = int(release_s * SAMPLE_RATE)
        if release_n > 0:
            env[max(0, n-release_n):] = np.linspace(1, 0, min(release_n, n))
            
        return signal * env

    @staticmethod
    def synth_pad(t, freq, richness=0.5):
        """Simula um Pad de Teclado/Cordas usando síntese aditiva"""
        # Fundamental + Harmônicos leves para textura de cordas
        out = np.sin(2 * np.pi * freq * t)
        out += 0.4 * np.sin(2 * np.pi * (freq * 1.002) * t) # Chorus/Detune
        out += 0.2 * np.sin(2 * np.pi * freq * 2 * t)      # Oitava
        out += 0.1 * np.sin(2 * np.pi * freq * 3 * t)      # Quinta harmônica
        
        # LFO para movimento (Vibrato/Tremolo)
        lfo = 1.0 + richness * np.sin(2 * np.pi * 0.5 * t)
        return out * lfo

    @staticmethod
    def synth_lead(t, freq):
        """Simula um som de Piano/Teclado Elétrico suave"""
        # Onda mais quadrada/triangular suavizada para harmônicos de instrumento de tecla
        out = np.sin(2 * np.pi * freq * t)
        out += 0.3 * np.sin(2 * np.pi * freq * 2 * t)
        out += 0.1 * np.random.normal(0, 0.01, len(t)) # Ruído de ataque metálico
        return out

# ---------------------------------------------------------------------------
# GESTOR DE ÁUDIOS EXTERNOS
# ---------------------------------------------------------------------------

class ExternalAudioManager:
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
                
                num_occ = max(1, int((total_duration_s / 120) * random.randint(1, 2)))
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
                    mix[write_start : write_start + length] += ev['data'][read_start : read_start + length] * 0.25
        return mix

# ---------------------------------------------------------------------------
# NOVO MOTOR MUSICAL (INSTRUMENTAL)
# ---------------------------------------------------------------------------

class ZenInstrumentEngine:
    def __init__(self, root=110):
        self.root = root
        # Escala Pentatônica Maior para harmonia sempre agradável
        self.scale = [1.0, 1.125, 1.25, 1.5, 1.666, 2.0] 

    def generate_chunk(self, start_s, duration_s):
        n = int(duration_s * SAMPLE_RATE)
        t = np.linspace(start_s, start_s + duration_s, n, dtype=np.float32)
        mix = np.zeros(n, dtype=np.float32)

        # 1. PAD DE FUNDO (TEXTURA DE CORDAS/TECLADO)
        # Notas longas que mudam a cada 15 segundos
        pad_cycle = 15
        for f_mult in [1.0, 1.25, 1.5]: # Tríade base
            # Calcula fase e intensidade baseado no tempo absoluto
            lfo_slow = 0.5 + 0.5 * np.sin(2 * np.pi * 0.05 * t)
            pad_wave = InstrumentSynth.synth_pad(t, self.root * f_mult)
            mix += (pad_wave * 0.15 * lfo_slow).astype(np.float32)

        # 2. MELODIA INSTRUMENTAL ALEATÓRIA (PIANO/LEAD)
        # Gera notas baseadas em sementes de tempo para manter consistência entre chunks
        step_s = 4 
        for i in range(int(start_s // step_s) * step_s, int((start_s + duration_s) // step_s + 1) * step_s, step_s):
            random.seed(i + 999) # Semente única para a melodia
            if random.random() > 0.6: # 40% de chance de ter uma nota neste tempo
                f_mult = random.choice(self.scale)
                octave = random.choice([2, 3])
                freq = self.root * f_mult * octave
                
                note_start = i + random.uniform(0, 1)
                note_dur = random.uniform(2, 5)
                
                if note_start < start_s + duration_s and note_start + note_dur > start_s:
                    n_start = max(0, int((note_start - start_s) * SAMPLE_RATE))
                    n_end = min(n, int((note_start + note_dur - start_s) * SAMPLE_RATE))
                    
                    if n_end > n_start:
                        local_dur = (n_end - n_start) / SAMPLE_RATE
                        t_local = np.linspace(max(0, start_s - note_start), note_dur, n_end - n_start)
                        
                        note_wave = InstrumentSynth.synth_lead(t_local, freq)
                        note_wave = InstrumentSynth.apply_envelope(note_wave, 0.5, 0, 1, 1.5, note_dur)
                        
                        mix[n_start:n_end] += (note_wave * 0.12).astype(np.float32)

        # 3. BAIXO PROFUNDO (CELLOS/SYNTH BASS)
        bass_cycle = 20
        if int(start_s // bass_cycle) % 2 == 0:
            bass_freq = self.root / 2
            mix += (np.sin(2 * np.pi * bass_freq * t) * 0.2 * (0.6 + 0.4 * np.sin(2 * np.pi * 0.1 * t))).astype(np.float32)

        return lowpass(mix, 1200) # Filtro um pouco mais aberto para os instrumentos aparecerem

# ---------------------------------------------------------------------------
# RENDERIZADOR
# ---------------------------------------------------------------------------

def render(output="zen_instrumental.wav", duration_s=120):
    total_samples = int(duration_s * SAMPLE_RATE)
    print(f"[ZEN] Renderizando Música Instrumental ({duration_s}s)...")
    
    ext_manager = ExternalAudioManager(total_duration_s=duration_s)
    music_engine = ZenInstrumentEngine(root=110) # Root em A2
    
    with sf.SoundFile(output, mode='w', samplerate=SAMPLE_RATE, channels=2) as file:
        current_sample = 0
        while current_sample < total_samples:
            chunk_end = min(current_sample + CHUNK_SAMPLES, total_samples)
            actual_len = chunk_end - current_sample
            t_start = current_sample / SAMPLE_RATE
            d_chunk = actual_len / SAMPLE_RATE
            
            print(f" > Progresso: {t_start:.1f}s / {duration_s}s", end='\r')
            
            # Gerar Camadas
            music = music_engine.generate_chunk(t_start, d_chunk)
            external = ext_manager.get_mix(current_sample, chunk_end)
            
            # Binaural Relax
            t_arr = np.linspace(t_start, t_start + d_chunk, actual_len, dtype=np.float32)
            bl_tone = np.sin(2 * np.pi * 100 * t_arr) * 0.03
            br_tone = np.sin(2 * np.pi * 103 * t_arr) * 0.03 # 3Hz Delta profundo
            
            # Mixagem e DSP
            mix = music + external
            mix = simple_reverb(mix)
            mix = soft_saturation(mix, 1.1)
            
            left = mix + bl_tone
            right = mix + br_tone
            stereo = np.column_stack((left, right))
            
            # Fades Globais
            fade_len = int(SAMPLE_RATE * 10)
            if current_sample < fade_len:
                f_in = np.linspace(0, 1, fade_len)
                for i in range(actual_len):
                    if current_sample + i < fade_len: stereo[i] *= f_in[current_sample + i]
            if chunk_end > total_samples - fade_len:
                f_out = np.linspace(1, 0, fade_len)
                for i in range(actual_len):
                    idx = current_sample + i
                    if idx > total_samples - fade_len: stereo[i] *= f_out[idx - (total_samples - fade_len)]

            # Normalização e Escrita
            max_v = np.max(np.abs(stereo))
            if max_v > 1.0: stereo /= max_v
            file.write(stereo * 0.85)
            
            current_sample = chunk_end
            
    print(f"\n[ZEN] Música finalizada com sucesso: {output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=120)
    parser.add_argument("--output", default="zen_automator_v5.wav")
    args = parser.parse_args()
    render(args.output, args.duration)
