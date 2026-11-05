"""
Zen Automator V2 — Ultimate Fusion Edition (Streaming Render)
Consolidação de: zen_automator, zen_automator_v2 e zen_automator_v3
Otimizado para longa duração (consumo de RAM constante)
"""

import numpy as np
import soundfile as sf
import argparse
import os
import random
from scipy.signal import butter, sosfilt, resample

SAMPLE_RATE = 44100
CHUNK_DURATION = 30  # segundos por bloco de renderização
CHUNK_SAMPLES = CHUNK_DURATION * SAMPLE_RATE

# ---------------------------------------------------------------------------
# UTILIDADES DSP (Processamento em Tempo Real)
# ---------------------------------------------------------------------------

def soft_saturation(x, drive=1.15):
    return np.tanh(x * drive)

def simple_reverb(signal, decay=0.45, delay_ms=200):
    delay = int(SAMPLE_RATE * delay_ms / 1000)
    out = np.copy(signal)
    # Reverb simplificado para processamento em bloco (cauda ignorada entre blocos para estabilidade)
    if len(signal) > delay:
        out[delay:] += signal[:-delay] * decay
    return out

def lowpass(data, cutoff=600):
    sos = butter(4, cutoff / (SAMPLE_RATE / 2), btype='low', output='sos')
    return sosfilt(sos, data)

def highcut(data, cutoff=40):
    sos = butter(2, cutoff / (SAMPLE_RATE / 2), btype='high', output='sos')
    return sosfilt(sos, data)

def bandpass(data, low=500, high=3000):
    sos = butter(4, [low / (SAMPLE_RATE / 2), high / (SAMPLE_RATE / 2)], btype='band', output='sos')
    return sosfilt(sos, data)

# ---------------------------------------------------------------------------
# GESTOR DE ÁUDIOS EXTERNOS (ESPORÁDICO)
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
                
                # Agendar aparições
                num_occ = max(1, int((total_duration_s / 60) * random.randint(1, 2)))
                for _ in range(num_occ):
                    start_s = random.uniform(0, max(0, total_duration_s - (len(data)/SAMPLE_RATE)))
                    self.events.append({
                        'data': data.astype(np.float32),
                        'start_sample': int(start_s * SAMPLE_RATE),
                        'end_sample': int(start_s * SAMPLE_RATE) + len(data)
                    })
                print(f"[ZEN] Som agendado: {file_name} ({num_occ}x)")
            except: pass

    def get_mix(self, start_sample, end_sample):
        chunk_len = end_sample - start_sample
        mix = np.zeros(chunk_len, dtype=np.float32)
        for ev in self.events:
            # Verifica sobreposição do evento com o chunk atual
            ev_s = ev['start_sample']
            ev_e = ev['end_sample']
            
            if ev_e > start_sample and ev_s < end_sample:
                # Calcular intersecção
                read_start = max(0, start_sample - ev_s)
                write_start = max(0, ev_s - start_sample)
                length = min(len(ev['data']) - read_start, chunk_len - write_start)
                
                if length > 0:
                    clip = ev['data'][read_start : read_start + length]
                    # Fade simplificado se for o início ou fim do clipe
                    mix[write_start : write_start + length] += clip * 0.3
        return mix

# ---------------------------------------------------------------------------
# MOTORES PROCEDURAIS (OTIMIZADOS PARA CHUNKS)
# ---------------------------------------------------------------------------

class ProceduralEngine:
    @staticmethod
    def brown_noise(n):
        white = np.random.normal(0, 1, n).astype(np.float32)
        brown = np.cumsum(white).astype(np.float32)
        return lowpass(highcut(brown, 30), 500)

    @staticmethod
    def rain(n):
        rain = ProceduralEngine.brown_noise(n)
        drops = np.zeros(n, dtype=np.float32)
        num_drops = n // (SAMPLE_RATE // 10)
        indices = np.random.randint(0, n, num_drops)
        drops[indices] = np.random.uniform(0, 0.4, num_drops)
        return (rain / (np.max(np.abs(rain)) + 1e-9)) * 0.3 + bandpass(drops, 1500, 4000) * 0.1

class ZenMusicEngine:
    @staticmethod
    def generate_chunk(start_s, duration_s, root=110):
        n = int(duration_s * SAMPLE_RATE)
        t = np.linspace(start_s, start_s + duration_s, n, dtype=np.float32)
        out = np.zeros(n, dtype=np.float32)
        
        freqs = [root, root * 1.25, root * 1.5, root * 2.0, root * (4/3)]
        for f in freqs:
            out += (np.sin(2 * np.pi * f * t) * 0.1 * np.exp(-0.02 * t)).astype(np.float32)
            
        # Melodia (Geração baseada no tempo absoluto para consistência)
        step_s = 6
        for i in range(int(start_s // step_s) * step_s, int((start_s + duration_s) // step_s + 1) * step_s, step_s):
            random.seed(i) # Semente fixa por tempo para ser consistente entre chunks
            f_mel = root * 2 * random.choice(freqs) / root
            note_start = i
            note_dur = 4
            if note_start < start_s + duration_s and note_start + note_dur > start_s:
                # Renderizar parte da nota que cai neste chunk
                n_start = max(0, int((note_start - start_s) * SAMPLE_RATE))
                n_end = min(n, int((note_start + note_dur - start_s) * SAMPLE_RATE))
                if n_end > n_start:
                    tt = np.linspace(max(0, start_s - note_start), (n_end - n_start)/SAMPLE_RATE, n_end - n_start)
                    out[n_start:n_end] += (np.sin(2 * np.pi * f_mel * root * tt) * 0.12 * np.exp(-tt * 0.8)).astype(np.float32)
        
        return lowpass(out, 800)

# ---------------------------------------------------------------------------
# RENDERIZADOR POR FLUXO (STREAMING)
# ---------------------------------------------------------------------------

def render(output="zen_fusion.wav", duration_s=120):
    total_samples = int(duration_s * SAMPLE_RATE)
    
    print(f"[ZEN] Iniciando Renderização Otimizada ({duration_s}s)...")
    
    # Preparar gestor de áudios externos
    ext_manager = ExternalAudioManager(total_duration_s=duration_s)
    
    # Abrir arquivo para escrita em modo streaming
    with sf.SoundFile(output, mode='w', samplerate=SAMPLE_RATE, channels=2) as file:
        current_sample = 0
        
        while current_sample < total_samples:
            chunk_end = min(current_sample + CHUNK_SAMPLES, total_samples)
            actual_chunk_len = chunk_end - current_sample
            t_start = current_sample / SAMPLE_RATE
            d_chunk = actual_chunk_len / SAMPLE_RATE
            
            print(f" > Processando: {t_start:.1f}s / {duration_s}s", end='\r')
            
            # 1. Gerar Camadas
            noise = ProceduralEngine.brown_noise(actual_chunk_len) * 0.1
            rain = ProceduralEngine.rain(actual_chunk_len) * 0.4
            music = ZenMusicEngine.generate_chunk(t_start, d_chunk) * 0.6
            external = ext_manager.get_mix(current_sample, chunk_end)
            
            # Binaural (Delta/Theta)
            t_arr = np.linspace(t_start, t_start + d_chunk, actual_chunk_len, dtype=np.float32)
            bl_tone = np.sin(2 * np.pi * 100 * t_arr) * 0.04
            br_tone = np.sin(2 * np.pi * 104 * t_arr) * 0.04
            
            # 2. Mixagem
            mix = noise + rain + music + external
            mix = simple_reverb(mix)
            mix = soft_saturation(mix)
            
            # 3. Stereo e Fade Final (apenas no último chunk)
            left = mix + bl_tone
            right = mix + br_tone
            stereo = np.column_stack((left, right))
            
            # Fade global (Início e Fim do arquivo total)
            fade_len = int(SAMPLE_RATE * 8)
            # Fade In
            if current_sample < fade_len:
                f_in = np.linspace(0, 1, fade_len)
                for i in range(actual_chunk_len):
                    if current_sample + i < fade_len:
                        stereo[i] *= f_in[current_sample + i]
            # Fade Out
            if chunk_end > total_samples - fade_len:
                f_out = np.linspace(1, 0, fade_len)
                for i in range(actual_chunk_len):
                    idx_total = current_sample + i
                    if idx_total > total_samples - fade_len:
                        stereo[i] *= f_out[idx_total - (total_samples - fade_len)]

            # 4. Escrita Direta
            # Normalização leve por chunk
            max_v = np.max(np.abs(stereo))
            if max_v > 1.0: stereo /= max_v
            
            file.write(stereo * 0.85)
            current_sample = chunk_end
            
    print(f"\n[ZEN] Concluído com sucesso: {output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=1800) # Padrão 30 min
    parser.add_argument("--output", default="zen_automator_v4.wav")
    args = parser.parse_args()
    render(args.output, args.duration)
