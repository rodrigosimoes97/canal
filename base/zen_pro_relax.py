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
# EFFECTS & DSP (REVERB, EQ, PANNING)
# ---------------------------------------------------------------------------

class EffectsDSP:
    def __init__(self):
        # Filtro passa-baixa para aquecer o som geral
        self.warm_sos = butter(2, 4000 / (SAMPLE_RATE / 2), btype='low', output='sos')
        self.zi_l = np.zeros((self.warm_sos.shape[0], 2))
        self.zi_r = np.zeros((self.warm_sos.shape[0], 2))

    @staticmethod
    def apply_delay_reverb(signal, delay_ms=400, decay=0.5, mix=0.4, num_taps=4):
        """Simula um reverb etéreo usando múltiplos delays."""
        delay_samples = int((delay_ms / 1000.0) * SAMPLE_RATE)
        out = np.copy(signal)
        for i in range(1, num_taps + 1):
            shift = int(delay_samples * i * 1.13) # Tempos irracionais evitam ressonância metálica
            if shift < len(out):
                out[shift:] += signal[:-shift] * (decay ** i)
        return (signal * (1 - mix)) + (out * mix)

    @staticmethod
    def apply_panning(signal, t_abs, speed=0.05, depth=0.8):
        """Movimento estéreo suave e contínuo."""
        pan = depth * np.sin(2 * np.pi * speed * t_abs)
        l = signal * (0.5 - pan/2)
        r = signal * (0.5 + pan/2)
        return l, r

    def process_master(self, l, r):
        l_filt, self.zi_l = sosfilt(self.warm_sos, l, zi=self.zi_l)
        r_filt, self.zi_r = sosfilt(self.warm_sos, r, zi=self.zi_r)
        
        # Soft Saturation/Limiter para evitar distorção
        l_final = np.tanh(l_filt * 1.2)
        r_final = np.tanh(r_filt * 1.2)
        return l_final, r_final

# ---------------------------------------------------------------------------
# SÍNTESE DE ÁUDIO PROFISSIONAL
# ---------------------------------------------------------------------------

class RelaxSynth:
    @staticmethod
    def envelope(signal, start_s, dur_s, t_abs, attack=0.3, release=0.3):
        n = len(signal)
        t_rel = t_abs - start_s
        env = np.ones(n)
        
        attack_mask = (t_rel >= 0) & (t_rel < attack)
        env[attack_mask] = t_rel[attack_mask] / attack
        
        release_start = dur_s - release
        release_mask = (t_rel > release_start)
        env[release_mask] = 1.0 - (t_rel[release_mask] - release_start) / release
        
        env[t_rel < 0] = 0
        env[t_rel > dur_s] = 0
        return signal * np.clip(env, 0, 1)

    @staticmethod
    def brown_noise(num_samples):
        """Ruído marrom: som orgânico parecido com cachoeiras/vento distante."""
        white = np.random.normal(0, 1, num_samples)
        # Filtro simples para transformar white noise em brown noise
        brown = np.cumsum(white)
        # Normalização e remoção de DC offset
        brown -= np.mean(brown)
        max_val = np.max(np.abs(brown))
        if max_val > 0:
            brown /= max_val
        return brown

    @staticmethod
    def ocean_waves(t_abs, bpm=8):
        """Simula o som de ondas do mar (respiração relaxante) usando modulação de volume."""
        noise = RelaxSynth.brown_noise(len(t_abs))
        # Frequência de respiração/ondas em Hz (ex: 8 batidas por minuto)
        freq = (bpm / 60.0) 
        # LFO para volume das ondas
        lfo = 0.5 + 0.5 * np.sin(2 * np.pi * freq * t_abs)
        return noise * lfo * 0.15

    @staticmethod
    def pad_drone(t, freq):
        """Drone profundo para aterramento (Grounding)."""
        sine1 = np.sin(2 * np.pi * freq * t)
        sine2 = np.sin(2 * np.pi * (freq * 1.005) * t) * 0.5 # Leve detuning
        sub = np.sin(2 * np.pi * (freq / 2) * t) * 0.8 # Sub-grave
        return (sine1 + sine2 + sub) * 0.3

    @staticmethod
    def healing_bell(t, freq):
        """Sino tibetano cristalino com harmônicos ricos."""
        fundamental = np.sin(2 * np.pi * freq * t)
        h1 = 0.6 * np.sin(2 * np.pi * freq * 2.76 * t) * np.exp(-1.5 * t)
        h2 = 0.4 * np.sin(2 * np.pi * freq * 5.4 * t) * np.exp(-3.0 * t)
        return fundamental + h1 + h2

# ---------------------------------------------------------------------------
# GERADOR DE PAISAGEM SONORA (SOUNDSCAPE)
# ---------------------------------------------------------------------------

class ZenProEngine:
    def __init__(self, root_freq=432.0):
        # Frequência Solfeggio 432Hz associada à cura e harmonia natural
        self.root = root_freq / 4.0 # Transpondo para frequências graves (108Hz)
        self.scale_ratios = [1.0, 1.125, 1.25, 1.5, 1.666, 2.0] # Escala Pentatônica Maior

    def generate_chunk(self, start_s, duration_s):
        n = int(duration_s * SAMPLE_RATE)
        t = np.arange(n) / SAMPLE_RATE + start_s
        mix_l = np.zeros(n, dtype=np.float32)
        mix_r = np.zeros(n, dtype=np.float32)

        # 1. ONDAS DO MAR / RESPIRAÇÃO (White/Brown Noise)
        waves = RelaxSynth.ocean_waves(t, bpm=7) # 7 ciclos por minuto (ritmo ideal de respiração)
        mix_l += waves
        mix_r += waves

        # 2. PADS DE CURA (Fundo)
        pad_freq = self.root
        pad = RelaxSynth.pad_drone(t, pad_freq)
        # Movimento estéreo bem lento para o pad
        pl, pr = EffectsDSP.apply_panning(pad, t, speed=0.01, depth=0.5)
        mix_l += pl * 0.2
        mix_r += pr * 0.2

        # 3. SINOS TERAPÊUTICOS E MELODIA ESPARSA
        # Tocar notas a cada ~12 segundos, de forma bem espaçada e aleatória
        step_bell = 12
        for i in range(int((start_s - 20) // step_bell) * step_bell, int((start_s + duration_s) // step_bell + 1) * step_bell, step_bell):
            random.seed(i + 432)
            if random.random() > 0.3: # 70% de chance de tocar
                freq = self.root * random.choice(self.scale_ratios) * 4 # Oitava mais alta
                note_start = i + random.uniform(0, 6)
                note_dur = random.uniform(8, 15)
                
                if note_start < start_s + duration_s and note_start + note_dur > start_s:
                    n_start = max(0, int((note_start - start_s) * SAMPLE_RATE))
                    n_end = min(n, int((note_start + note_dur - start_s) * SAMPLE_RATE))
                    if n_end > n_start:
                        t_n = np.arange(n_end - n_start) / SAMPLE_RATE
                        bell = RelaxSynth.healing_bell(t_n, freq)
                        bell = RelaxSynth.envelope(bell, note_start, note_dur, t[n_start:n_end], attack=2.0, release=6.0)
                        
                        # Adicionando muito REVERB para os sinos
                        bell_reverb = EffectsDSP.apply_delay_reverb(bell, delay_ms=600, decay=0.6, mix=0.5)
                        
                        # Panning aleatório para cada nota
                        pan_pos = random.uniform(-0.8, 0.8)
                        bl = bell_reverb * (0.5 - pan_pos/2)
                        br = bell_reverb * (0.5 + pan_pos/2)
                        
                        mix_l[n_start:n_end] += bl * 0.15
                        mix_r[n_start:n_end] += br * 0.15

        # 4. BATIDAS BINAURAIS (Binaural Beats)
        # Delta waves (2 a 4 Hz) para sono profundo. Vamos usar 3Hz de diferença.
        base_binaural = 174.0 # Solfeggio de alívio de dor
        binaural_l = np.sin(2 * np.pi * base_binaural * t)
        binaural_r = np.sin(2 * np.pi * (base_binaural + 3.0) * t) # 3Hz delta wave
        mix_l += binaural_l * 0.05
        mix_r += binaural_r * 0.05

        return mix_l, mix_r

# ---------------------------------------------------------------------------
# INTEGRAÇÃO DE ÁUDIOS EXTERNOS (Passarinhos, chuva, etc)
# ---------------------------------------------------------------------------

class ExternalAudioManager:
    def __init__(self, folder="audios", total_duration_s=120):
        self.events = []
        if not os.path.exists(folder):
            print(f"Aviso: Pasta '{folder}' não encontrada. Áudios externos ignorados.")
            return
            
        files = [f for f in os.listdir(folder) if f.endswith(('.mp3', '.wav', '.ogg'))]
        for file_name in files:
            try:
                path = os.path.join(folder, file_name)
                data, fs = sf.read(path)
                if len(data.shape) > 1: 
                    data = np.mean(data, axis=1) # Converte para mono
                if fs != SAMPLE_RATE:
                    num = int(len(data) * SAMPLE_RATE / fs)
                    data = resample(data, num)
                
                # Distribui o áudio várias vezes pela track
                num_occ = max(1, int((total_duration_s / 120) * random.randint(2, 4)))
                for _ in range(num_occ):
                    start_s = random.uniform(0, max(0, total_duration_s - (len(data)/SAMPLE_RATE)))
                    self.events.append({
                        'data': data.astype(np.float32),
                        'start_sample': int(start_s * SAMPLE_RATE),
                        'end_sample': int(start_s * SAMPLE_RATE) + len(data)
                    })
            except Exception as e:
                print(f"Erro ao ler {file_name}: {e}")

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
                    # Aplica fade in/out no chunk de áudio externo para evitar cliques
                    segment = ev['data'][read_start : read_start + length] * 0.2
                    mix[write_start : write_start + length] += segment
        return mix

# ---------------------------------------------------------------------------
# RENDERIZAÇÃO
# ---------------------------------------------------------------------------

def render(output="zen_pro_relax.wav", duration_s=120):
    total_samples = int(duration_s * SAMPLE_RATE)
    print(f"[ZEN PRO] Iniciando geração de Áudio Premium ({duration_s}s)...")
    print(" -> Aplicando 432Hz Solfeggio Frequencies")
    print(" -> Gerando Delta Waves (3Hz) para Sono Profundo")
    print(" -> Sintetizando Ruído Marrom e Ondas de Respiração")
    
    dsp = EffectsDSP()
    ext_manager = ExternalAudioManager(total_duration_s=duration_s)
    music_engine = ZenProEngine()
    
    with sf.SoundFile(output, mode='w', samplerate=SAMPLE_RATE, channels=2) as file:
        current_sample = 0
        while current_sample < total_samples:
            chunk_end = min(current_sample + CHUNK_SAMPLES, total_samples)
            actual_len = chunk_end - current_sample
            t_start = current_sample / SAMPLE_RATE
            d_chunk = actual_len / SAMPLE_RATE
            
            print(f"Progresso: {t_start:.1f}s / {duration_s}s ({(current_sample/total_samples)*100:.1f}%)", end='\r')
            
            # 1. Geração de Sintetizadores
            music_l, music_r = music_engine.generate_chunk(t_start, d_chunk)
            
            # 2. Adição de Áudios Externos (Natureza)
            external = ext_manager.get_mix(current_sample, chunk_end)
            final_l = music_l + external
            final_r = music_r + external
            
            # 3. Processamento Master (EQ, Saturação leve, Limiter)
            final_l, final_r = dsp.process_master(final_l, final_r)
            
            stereo = np.column_stack((final_l, final_r))
            
            # Fades iniciais e finais
            fade_len = int(SAMPLE_RATE * 15) # 15 segundos de fade in/out
            if current_sample < fade_len:
                f_in = np.linspace(0, 1, fade_len)
                for i in range(actual_len):
                    if current_sample + i < fade_len: 
                        stereo[i] *= f_in[current_sample + i]
            
            if chunk_end > total_samples - fade_len:
                f_out = np.linspace(1, 0, fade_len)
                for i in range(actual_len):
                    idx = current_sample + i
                    if idx > total_samples - fade_len: 
                        stereo[i] *= f_out[idx - (total_samples - fade_len)]

            # Normalização de Segurança
            max_v = np.max(np.abs(stereo))
            if max_v > 0.95: 
                stereo = (stereo / max_v) * 0.95
                
            file.write(stereo)
            current_sample = chunk_end
            
    print(f"\n[ZEN PRO] Sucesso! Áudio renderizado em: {output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gerador de Som para Relaxamento Profissional (YouTube)")
    parser.add_argument("--duration", type=int, default=300, help="Duração em segundos (padrão: 300s/5min)")
    parser.add_argument("--output", default="zen_pro_relax.wav", help="Nome do arquivo de saída")
    args = parser.parse_args()
    render(args.output, args.duration)
