import numpy as np
import soundfile as sf
import argparse
import random
import os
from scipy.signal import butter, sosfilt, lfilter, resample

SAMPLE_RATE = 44100
CHUNK_DURATION = 10 
CHUNK_SAMPLES = CHUNK_DURATION * SAMPLE_RATE

# ---------------------------------------------------------------------------
# DSP & EFFECTS
# ---------------------------------------------------------------------------
class TavernDSP:
    @staticmethod
    def filter_audio(signal, cutoff, btype='low', order=2):
        sos = butter(order, cutoff / (SAMPLE_RATE / 2), btype=btype, output='sos')
        return sosfilt(sos, signal)

    @staticmethod
    def apply_room_reverb(signal, mix=0.3):
        """Reverb curto, simulando uma taverna de madeira (sala média)."""
        delay_ms = 40
        delay_samples = int((delay_ms / 1000.0) * SAMPLE_RATE)
        out = np.copy(signal)
        for i in range(1, 4):
            shift = int(delay_samples * i * 1.3) 
            if shift < len(out):
                out[shift:] += signal[:-shift] * (0.4 ** i)
        return (signal * (1 - mix)) + (out * mix)

# ---------------------------------------------------------------------------
# SÍNTESE ASMR E MEDIEVAL
# ---------------------------------------------------------------------------
class ASMRSynth:
    @staticmethod
    def heavy_rain(t_abs):
        """Som de chuva contínua batendo no telhado."""
        white = np.random.normal(0, 1, len(t_abs))
        # Filtro passa-baixa para dar som de telhado / ambiente fechado
        rain = TavernDSP.filter_audio(white, cutoff=1500, btype='low')
        # LFO para simular rajadas de vento na chuva
        wind_gusts = 0.7 + 0.3 * np.sin(2 * np.pi * 0.05 * t_abs)
        return rain * wind_gusts * 0.15

    # Lareira artificial removida: agora usamos o áudio real lareira.mp3

    @staticmethod
    def thunder_rumble(t, duration_s):
        """Trovão abafado (Longe da taverna)."""
        white = np.random.normal(0, 1, len(t))
        # O trovão começa agudo e vai ficando mais grave
        env = np.exp(-1.5 * t)
        thunder = TavernDSP.filter_audio(white, cutoff=150, btype='low') * env
        return thunder

    @staticmethod
    def lute_pluck(t, freq):
        """Harpa/Alaúde Mágica super suave, com delay e chorus embutidos."""
        env = np.exp(-1.5 * t) # Mais sustentação e corpo
        
        # Mistura de Sino puro (Sine) com corda de harpa (aproximação via Triângulo suave)
        carrier1 = np.sin(2 * np.pi * freq * t)
        carrier2 = np.sin(2 * np.pi * freq * t) - (1/9) * np.sin(2 * np.pi * 3 * freq * t)
        
        # Efeito Chorus e Vibrato para soar mágico e etéreo
        chorus = np.sin(2 * np.pi * (freq * 1.004) * t) * 0.4
        tremolo = 1.0 + 0.15 * np.sin(2 * np.pi * 3.5 * t)
        
        # Ataque extremamente macio (fade-in gradual)
        attack = np.minimum(t * 70.0, 1.0)
        
        out = (carrier1 * 0.5 + carrier2 * 0.5 + chorus) * env * attack * tremolo
        
        # Adiciona um eco/delay místico dentro da própria nota (meio segundo)
        delay_idx = int(0.5 * SAMPLE_RATE)
        if delay_idx < len(out):
            out[delay_idx:] += out[:-delay_idx] * 0.45
            
        return out

# ---------------------------------------------------------------------------
# MOTOR DA TAVERNA
# ---------------------------------------------------------------------------
class TavernEngine:
    def __init__(self):
        # Escala Dórica menor (Clássica sonoridade medieval) - D, E, F, G, A, B, C
        self.base_freq = 146.83 # Nota D3
        self.scale_ratios = [1.0, 1.122, 1.189, 1.334, 1.498, 1.681, 1.781, 2.0]
        
        # Carregar áudio real da lareira
        self.fireplace_audio = None
        try:
            path = os.path.join("audios", "lareira.mp3")
            if os.path.exists(path):
                data, fs = sf.read(path)
                if len(data.shape) > 1: data = np.mean(data, axis=1)
                if fs != SAMPLE_RATE:
                    data = resample(data, int(len(data) * SAMPLE_RATE / fs))
                self.fireplace_audio = data.astype(np.float32)
        except Exception as e:
            print(f"Não foi possível carregar a lareira real: {e}")

    def generate_chunk(self, start_s, duration_s):
        n = int(duration_s * SAMPLE_RATE)
        t = np.arange(n) / SAMPLE_RATE + start_s
        t_local = np.arange(n) / SAMPLE_RATE
        
        mix_l = np.zeros(n, dtype=np.float32)
        mix_r = np.zeros(n, dtype=np.float32)

        # 1. LAREIRA REAL (audios/lareira.mp3) em Loop Contínuo
        if self.fireplace_audio is not None and len(self.fireplace_audio) > 0:
            f_len = len(self.fireplace_audio)
            start_idx = int(start_s * SAMPLE_RATE) % f_len
            end_idx = start_idx + n
            
            if end_idx <= f_len:
                fire_chunk = self.fireplace_audio[start_idx:end_idx]
            else:
                part1 = self.fireplace_audio[start_idx:]
                part2_len = n - len(part1)
                repeats = int(np.ceil(part2_len / f_len))
                part2 = np.tile(self.fireplace_audio, repeats)[:part2_len]
                fire_chunk = np.concatenate((part1, part2))
                
            mix_l += fire_chunk * 0.7
            mix_r += fire_chunk * 1.2 # Fogo focado na direita

        # 2. CHUVA FORTE (Ambiente estéreo amplo)
        rain_l = ASMRSynth.heavy_rain(t)
        rain_r = ASMRSynth.heavy_rain(t + 10) # Offset para estéreo natural
        mix_l += rain_l * 0.6
        mix_r += rain_r * 0.6

        # 3. TROVÕES ESPARSOS
        thunder_step = 25 # Trovão a cada ~25 segs
        for i in range(int(start_s // thunder_step) * thunder_step, int((start_s + duration_s) // thunder_step + 1) * thunder_step, thunder_step):
            np.random.seed(i + 123)
            if np.random.random() > 0.4: # 60% chance de trovão
                note_start = i + np.random.uniform(0, thunder_step - 5)
                note_dur = 8.0
                
                if note_start < start_s + duration_s and note_start + note_dur > start_s:
                    n_start = max(0, int((note_start - start_s) * SAMPLE_RATE))
                    n_end = min(n, int((note_start + note_dur - start_s) * SAMPLE_RATE))
                    if n_end > n_start:
                        t_n = np.arange(n_end - n_start) / SAMPLE_RATE + max(0, start_s - note_start)
                        thunder = ASMRSynth.thunder_rumble(t_n, note_dur)
                        
                        pan = np.random.uniform(0.3, 0.7) # Trovão centralizado
                        mix_l[n_start:n_end] += thunder * (1 - pan) * 0.8
                        mix_r[n_start:n_end] += thunder * pan * 0.8

        # 4. O BARDO (Alaúde)
        # Notas esparsas, como se um bardo estivesse praticando lentamente
        lute_step = 3
        for i in range(int(start_s // lute_step) * lute_step, int((start_s + duration_s) // lute_step + 1) * lute_step, lute_step):
            np.random.seed(i + 456)
            # Arpejo rápido de 3 notas
            if np.random.random() > 0.3:
                num_notes = np.random.randint(1, 4)
                for note_idx in range(num_notes):
                    note_start = i + np.random.uniform(0, 1) + (note_idx * 0.3)
                    note_dur = 3.0
                    
                    if note_start < start_s + duration_s and note_start + note_dur > start_s:
                        n_start = max(0, int((note_start - start_s) * SAMPLE_RATE))
                        n_end = min(n, int((note_start + note_dur - start_s) * SAMPLE_RATE))
                        if n_end > n_start:
                            t_n = np.arange(n_end - n_start) / SAMPLE_RATE + max(0, start_s - note_start)
                            
                            freq = self.base_freq * np.random.choice(self.scale_ratios)
                            # Ocasionalmente uma oitava acima
                            if np.random.random() > 0.7: freq *= 2.0
                            
                            lute = ASMRSynth.lute_pluck(t_n, freq)
                            
                            # Alaúde centralizado levemente à esquerda
                            mix_l[n_start:n_end] += lute * 0.25
                            mix_r[n_start:n_end] += lute * 0.15

        # Reverb da taverna no instrumento (Bardo) e na lareira
        mix_l = TavernDSP.apply_room_reverb(mix_l, mix=0.2)
        mix_r = TavernDSP.apply_room_reverb(mix_r, mix=0.2)

        return mix_l, mix_r

# ---------------------------------------------------------------------------
# RENDERIZAÇÃO
# ---------------------------------------------------------------------------
def render(output="zen_tavern.wav", duration_s=120):
    total_samples = int(duration_s * SAMPLE_RATE)
    print(f"[TAVERN ASMR] Iniciando cenário Taverna Chuvosa ({duration_s}s)...")
    
    engine = TavernEngine()
    
    with sf.SoundFile(output, mode='w', samplerate=SAMPLE_RATE, channels=2) as file:
        current_sample = 0
        while current_sample < total_samples:
            chunk_end = min(current_sample + CHUNK_SAMPLES, total_samples)
            actual_len = chunk_end - current_sample
            t_start = current_sample / SAMPLE_RATE
            d_chunk = actual_len / SAMPLE_RATE
            
            print(f"Bebendo Hidromel: {t_start:.1f}s / {duration_s}s ({(current_sample/total_samples)*100:.1f}%)", end='\r')
            
            # Geração
            l, r = engine.generate_chunk(t_start, d_chunk)
            
            # Limiter para evitar que os estalos da lareira passem do limite
            l = np.tanh(l * 1.3)
            r = np.tanh(r * 1.3)
            
            stereo = np.column_stack((l, r))
            
            # Fades
            fade_len = min(int(SAMPLE_RATE * 10), int(total_samples * 0.1))
            if fade_len < SAMPLE_RATE * 2: fade_len = int(SAMPLE_RATE * 2)
            
            if current_sample < fade_len:
                f_in = np.linspace(0, 1, fade_len)
                for i in range(actual_len):
                    if current_sample + i < fade_len: stereo[i] *= f_in[current_sample + i]
            
            if chunk_end > total_samples - fade_len:
                f_out = np.linspace(1, 0, fade_len)
                for i in range(actual_len):
                    idx = current_sample + i
                    if idx > total_samples - fade_len: stereo[i] *= f_out[idx - (total_samples - fade_len)]

            stereo *= 0.90 
            
            file.write(stereo)
            current_sample = chunk_end
            
    print(f"\n[TAVERN ASMR] A noite caiu na taverna! Arquivo salvo: {output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=120, help="Duração em segundos")
    parser.add_argument("--output", default="zen_tavern.wav", help="Nome do arquivo")
    args = parser.parse_args()
    render(args.output, args.duration)
