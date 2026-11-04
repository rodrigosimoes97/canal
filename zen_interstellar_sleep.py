import numpy as np
import soundfile as sf
import argparse
from scipy.signal import butter, sosfilt

SAMPLE_RATE = 44100
CHUNK_DURATION = 10 
CHUNK_SAMPLES = CHUNK_DURATION * SAMPLE_RATE

# ---------------------------------------------------------------------------
# DSP & EFFECTS (Filtros, Reverb Cósmico)
# ---------------------------------------------------------------------------

class SpaceDSP:
    @staticmethod
    def apply_infinite_reverb(signal, delay_ms=800, decay=0.7, mix=0.5):
        """Um delay gigantesco em feedback que cria um 'buraco negro' sonoro."""
        delay_samples = int((delay_ms / 1000.0) * SAMPLE_RATE)
        out = np.copy(signal)
        # 6 delays retroalimentados para criar uma nuvem sonora
        for i in range(1, 7):
            shift = int(delay_samples * (i * 1.618)) # Golden ratio espalha os ecos
            if shift < len(out):
                out[shift:] += signal[:-shift] * (decay ** i)
        return (signal * (1 - mix)) + (out * mix)

    @staticmethod
    def lowpass_filter(signal, cutoff, order=2):
        sos = butter(order, cutoff / (SAMPLE_RATE / 2), btype='low', output='sos')
        return sosfilt(sos, signal)

# ---------------------------------------------------------------------------
# SÍNTESE "DEEP SPACE" (Interestelar)
# ---------------------------------------------------------------------------

class InterstellarSynth:
    @staticmethod
    def envelope(signal, t_abs, start_s, dur_s, attack=5.0, release=10.0):
        t_rel = t_abs - start_s
        env = np.ones(len(signal))
        
        attack_mask = (t_rel >= 0) & (t_rel < attack)
        env[attack_mask] = t_rel[attack_mask] / attack
        
        release_start = dur_s - release
        release_mask = (t_rel > release_start)
        env[release_mask] = np.maximum(0, 1.0 - (t_rel[release_mask] - release_start) / release)
        
        env[t_rel < 0] = 0
        env[t_rel > dur_s] = 0
        return signal * env

    @staticmethod
    def deep_space_drone(t_abs, base_freq=55.0):
        """Um sub-grave colossal, como a vibração de um planeta."""
        # Modulação FM ultra lenta para dar movimento interno ao som
        modulator = np.sin(2 * np.pi * 0.05 * t_abs) # 1 ciclo a cada 20 segundos
        carrier1 = np.sin(2 * np.pi * base_freq * t_abs + (modulator * 0.5))
        # Oitava abaixo para o sub
        sub = np.sin(2 * np.pi * (base_freq / 2.0) * t_abs) 
        
        # NOVO: Pad na região média (fundamental x 1.5) para audibilidade em fones simples
        mid_pad = np.sin(2 * np.pi * (base_freq * 1.5) * t_abs) * 0.5
        
        # Leve desafinação no canal estéreo simulada depois no panning
        drone = (carrier1 * 0.4) + (sub * 0.6) + mid_pad
        
        # Filtra frequências agudas para manter o som abafado e "escuro"
        drone = SpaceDSP.lowpass_filter(drone, cutoff=500)
        return drone * 2.0 # Dobra o volume base do drone

    @staticmethod
    def solar_wind(t_abs):
        """Ventos solares: Ruído marrom com um filtro que 'varre' as frequências lentamente."""
        white = np.random.normal(0, 1, len(t_abs))
        brown = np.cumsum(white)
        brown -= np.mean(brown)
        brown /= np.max(np.abs(brown)) if np.max(np.abs(brown)) > 0 else 1
        
        # O LFO (Low Frequency Oscillator) move o cutoff do filtro, parecendo vento
        lfo = 0.5 + 0.5 * np.sin(2 * np.pi * 0.03 * t_abs) # Muito lento
        cutoff_freqs = 100 + (lfo * 800) # O filtro sobe e desce entre 100Hz e 900Hz
        
        # Como o filtro dinâmico é pesado em Python puro, faremos uma aproximação tática
        # Misturando duas versões filtradas de forma estática, controladas pelo LFO
        wind_low = SpaceDSP.lowpass_filter(brown, 150)
        wind_high = SpaceDSP.lowpass_filter(brown, 900)
        
        dynamic_wind = (wind_low * (1 - lfo)) + (wind_high * lfo)
        return dynamic_wind * 0.35 # Aumentado de 0.08 para 0.35 para ser mais audível

    @staticmethod
    def star_ping(t, freq=800.0):
        """Som de gota/sonar de estrela que reverbera no infinito."""
        # A frequência cai ligeiramente no início (efeito ping/gota)
        freq_env = np.exp(-10 * t) 
        actual_freq = freq + (freq * 0.1 * freq_env)
        
        ping = np.sin(2 * np.pi * actual_freq * t)
        
        # Envelope super percussivo (ataque zero, decay rápido)
        amp_env = np.exp(-3 * t)
        return ping * amp_env

# ---------------------------------------------------------------------------
# MOTOR INTERESTELAR
# ---------------------------------------------------------------------------

class CosmicEngine:
    def __init__(self):
        self.base_freq = 432.0 / 4.0 # 108Hz (Região grave audível em qualquer fone)
        self.ping_intervals = [20, 35, 45] # Segundos entre os 'pings'

    def generate_chunk(self, start_s, duration_s):
        n = int(duration_s * SAMPLE_RATE)
        t = np.arange(n) / SAMPLE_RATE + start_s
        t_local = np.arange(n) / SAMPLE_RATE
        
        mix_l = np.zeros(n, dtype=np.float32)
        mix_r = np.zeros(n, dtype=np.float32)

        # 1. MOTOR SUB-GRAVE (DRONE CONTÍNUO)
        drone = InterstellarSynth.deep_space_drone(t, self.base_freq)
        # Estéreo sutil no drone
        mix_l += drone * 0.4
        mix_r += drone * 0.4 * np.cos(2 * np.pi * 0.01 * t) # Leve defasagem natural

        # 2. VENTO SOLAR ESPACIAL
        wind = InterstellarSynth.solar_wind(t)
        mix_l += wind
        # Invertendo o LFO da direita para estéreo massivo
        mix_r += InterstellarSynth.solar_wind(t + 10) 

        # 3. PINGS CÓSMICOS (Apenas de vez em quando)
        for interval in self.ping_intervals:
            for i in range(int(start_s // interval) * interval, int((start_s + duration_s) // interval + 1) * interval, interval):
                np.random.seed(i)
                if np.random.random() > 0.4: # 60% chance de tocar
                    note_start = i + np.random.uniform(0, interval - 5)
                    note_dur = 5.0 # O ping em si é curto, o reverb dura muito
                    
                    if note_start < start_s + duration_s and note_start + note_dur > start_s:
                        n_start = max(0, int((note_start - start_s) * SAMPLE_RATE))
                        n_end = min(n, int((note_start + note_dur - start_s) * SAMPLE_RATE))
                        if n_end > n_start:
                            t_n = t_local[:n_end - n_start]
                            
                            # Escolhe frequências agudas e cristalinas (harmônicos de 432Hz)
                            freq_multiplier = np.random.choice([4, 6, 8, 9])
                            ping = InterstellarSynth.star_ping(t_n, self.base_freq * freq_multiplier)
                            
                            # Aplica um reverb absolutamente massivo
                            ping_verb = SpaceDSP.apply_infinite_reverb(ping, delay_ms=1200, decay=0.75, mix=0.8)
                            
                            pan = np.random.uniform(0.1, 0.9)
                            mix_l[n_start:n_end] += ping_verb * (1 - pan) * 0.15
                            mix_r[n_start:n_end] += ping_verb * pan * 0.15

        return mix_l, mix_r

# ---------------------------------------------------------------------------
# RENDERIZAÇÃO E MASTERIZAÇÃO
# ---------------------------------------------------------------------------

def render(output="zen_interstellar.wav", duration_s=120):
    total_samples = int(duration_s * SAMPLE_RATE)
    print(f"[INTERSTELLAR] Iniciando geração de Áudio Cosmic Sleep ({duration_s}s)...")
    
    engine = CosmicEngine()
    
    with sf.SoundFile(output, mode='w', samplerate=SAMPLE_RATE, channels=2) as file:
        current_sample = 0
        while current_sample < total_samples:
            chunk_end = min(current_sample + CHUNK_SAMPLES, total_samples)
            actual_len = chunk_end - current_sample
            t_start = current_sample / SAMPLE_RATE
            d_chunk = actual_len / SAMPLE_RATE
            
            print(f"Progresso Espacial: {t_start:.1f}s / {duration_s}s ({(current_sample/total_samples)*100:.1f}%)", end='\r')
            
            # Geração
            l, r = engine.generate_chunk(t_start, d_chunk)
            
            # Masterização Suave: Soft Clipping
            l = np.tanh(l * 1.5)
            r = np.tanh(r * 1.5)
            
            stereo = np.column_stack((l, r))
            
            # Fades dinâmicos: 30s ou apenas 10% do áudio (evita áudios curtos ficarem mudos)
            fade_len = min(int(SAMPLE_RATE * 30), int(total_samples * 0.1))
            if fade_len < SAMPLE_RATE * 2: fade_len = int(SAMPLE_RATE * 2) # Mínimo de 2 segs
            
            if current_sample < fade_len:
                f_in = np.linspace(0, 1, fade_len)
                for i in range(actual_len):
                    if current_sample + i < fade_len: stereo[i] *= f_in[current_sample + i]
            
            if chunk_end > total_samples - fade_len:
                f_out = np.linspace(1, 0, fade_len)
                for i in range(actual_len):
                    idx = current_sample + i
                    if idx > total_samples - fade_len: stereo[i] *= f_out[idx - (total_samples - fade_len)]

            # Normalização suave
            stereo *= 0.85 
            
            file.write(stereo)
            current_sample = chunk_end
            
    print(f"\n[INTERSTELLAR] Viagem concluída! Arquivo salvo: {output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=120, help="Duração em segundos")
    parser.add_argument("--output", default="zen_interstellar.wav", help="Nome do arquivo")
    args = parser.parse_args()
    render(args.output, args.duration)
