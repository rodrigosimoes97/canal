import numpy as np
import soundfile as sf
import argparse
from scipy.signal import butter, sosfilt

SAMPLE_RATE = 44100
CHUNK_DURATION = 15 # Blocos maiores para aguentar o reverb longo
CHUNK_SAMPLES = int(CHUNK_DURATION * SAMPLE_RATE)

# ---------------------------------------------------------------------------
# DSP & EFFECTS (Reverb de Catedral e Filtros Formantes)
# ---------------------------------------------------------------------------
class ChoirDSP:
    @staticmethod
    def bandpass_filter(signal, center_freq, bandwidth=100):
        """Filtro passa-banda usado para ressoar frequências específicas (formantes)."""
        low = max(20.0, center_freq - bandwidth / 2.0)
        high = min(SAMPLE_RATE / 2.0 - 1.0, center_freq + bandwidth / 2.0)
        sos = butter(2, [low / (SAMPLE_RATE / 2.0), high / (SAMPLE_RATE / 2.0)], btype='band', output='sos')
        return sosfilt(sos, signal)
        
    @staticmethod
    def lowpass_filter(signal, cutoff=1200):
        """Amacia o som, tirando os agudos rasgados do sintetizador."""
        sos = butter(2, cutoff / (SAMPLE_RATE / 2.0), btype='low', output='sos')
        return sosfilt(sos, signal)

    @staticmethod
    def cathedral_reverb(signal, mix=0.7):
        """Simula a reverberação massiva e angelical de uma catedral de pedra."""
        out = np.copy(signal)
        # Delays com tempos primos/irracionais para não gerar ressonâncias metálicas
        delays_ms = [277, 431, 683, 1097, 1709]
        for i, d_ms in enumerate(delays_ms):
            shift = int((d_ms / 1000.0) * SAMPLE_RATE)
            if shift < len(out):
                out[shift:] += signal[:-shift] * (0.65 ** (i + 1))
        return (signal * (1 - mix)) + (out * mix)

# ---------------------------------------------------------------------------
# SÍNTESE DO CORAL HUMANO (Formant Synthesis)
# ---------------------------------------------------------------------------
class AngelicSynth:
    @staticmethod
    def envelope(n_samples, t_abs, start_s, dur_s, attack=8.0, release=12.0):
        """Envelope super longo de respiração do coral (Slow Attack)."""
        t_rel = t_abs - start_s
        env = np.ones(n_samples)
        
        attack_mask = (t_rel >= 0) & (t_rel < attack)
        env[attack_mask] = t_rel[attack_mask] / attack
        
        release_start = dur_s - release
        release_mask = (t_rel > release_start)
        env[release_mask] = np.maximum(0, 1.0 - (t_rel[release_mask] - release_start) / release)
        
        env[t_rel < 0] = 0
        env[t_rel > dur_s] = 0
        return env

    @staticmethod
    def generate_sawtooth(t, freq):
        """Gera uma onda Dente de Serra (rica em harmônicos, como as cordas vocais)."""
        return 2.0 * (t * freq - np.floor(t * freq + 0.5))

    @staticmethod
    def choir_voice(t, freq, vowel='Ooo'):
        """Sintetiza um grupo de cantores usando Filtros de Formante (Ressonância da Boca)."""
        # 1. Cordas Vocais: 4 vozes ligeiramente desafinadas cantando a mesma nota (Chorus natural)
        s1 = AngelicSynth.generate_sawtooth(t, freq)
        s2 = AngelicSynth.generate_sawtooth(t, freq * 1.002)
        s3 = AngelicSynth.generate_sawtooth(t, freq * 0.998)
        s4 = AngelicSynth.generate_sawtooth(t, freq * 1.004)
        
        # LFO simulando o "Vibrato" natural dos cantores líricos (5Hz)
        vibrato = 1.0 + 0.05 * np.sin(2 * np.pi * 5.0 * t)
        source = ((s1 + s2 + s3 + s4) / 4.0) * vibrato

        # 2. Trato Vocal (Boca/Garganta): Formantes
        if vowel == 'Ooo':
            # Vogal mais fechada e relaxante
            f1, f2, f3 = 300, 870, 2240
        else: 
            # Vogal 'Ahhh' (mais aberta e brilhante)
            f1, f2, f3 = 730, 1090, 2440

        # Filtramos a "buzina" do sawtooth para soar como voz humana
        v1 = ChoirDSP.bandpass_filter(source, f1, 80)
        v2 = ChoirDSP.bandpass_filter(source, f2, 100) * 0.5
        v3 = ChoirDSP.bandpass_filter(source, f3, 120) * 0.2
        
        voice = v1 + v2 + v3
        
        # Tira a estridência eletrônica
        voice = ChoirDSP.lowpass_filter(voice, cutoff=1500)
        return voice

# ---------------------------------------------------------------------------
# MOTOR ANGELICAL (Solfeggio 528Hz)
# ---------------------------------------------------------------------------
class SolfeggioEngine:
    def __init__(self, root=528.0):
        # 528Hz é a famosa Frequência do Milagre (Solfeggio).
        # Vamos dividir por 4 para criar o tom fundamental dos cantores baixos/tenores (132Hz).
        self.base_freq = root / 4.0 
        
        # Acordes celestiais construídos a partir de 528Hz
        # Usaremos intervalos simples e puros: Tônica, Terça Maior (1.25), Quinta (1.5)
        self.chords = [
            [1.0, 1.5, 2.0],          # Acorde Base (Vazio e meditativo)
            [1.0, 1.25, 1.5, 2.0],    # Acorde Maior Luminoso
            [1.0, 1.333, 1.5, 2.0],   # Acorde Sus4 (Sensação de flutuação)
            [0.75, 1.125, 1.5, 2.0]   # Subdominante grave
        ]

    def generate_chunk(self, start_s, duration_s):
        n = int(duration_s * SAMPLE_RATE)
        t = np.arange(n) / SAMPLE_RATE + start_s
        
        mix_l = np.zeros(n, dtype=np.float32)
        mix_r = np.zeros(n, dtype=np.float32)

        # O coral respira e troca de acorde bem devagar (a cada 20 segundos)
        chord_duration = 20.0
        
        # Encontra quais acordes tocam neste pedaço de tempo
        for i in range(int(start_s // chord_duration), int((start_s + duration_s) // chord_duration) + 1):
            chord_start = i * chord_duration
            # Intercala os acordes celestialmente
            chord_idx = i % len(self.chords)
            current_chord = self.chords[chord_idx]
            
            # Alterna vogais entre 'Ooo' e 'Ahhh'
            vowel = 'Ooo' if i % 2 == 0 else 'Ahhh'
            
            if chord_start < start_s + duration_s and chord_start + chord_duration > start_s:
                # Extrai apenas o tempo correspondente ao bloco atual
                idx_start = max(0, int((chord_start - start_s) * SAMPLE_RATE))
                idx_end = min(n, int((chord_start + chord_duration - start_s) * SAMPLE_RATE))
                
                if idx_end > idx_start:
                    # Tempo absoluto para o Envelope não cortar e dar estalos (Glitch-free)
                    t_n = np.arange(idx_end - idx_start) / SAMPLE_RATE + max(0, start_s - chord_start)
                    t_abs_n = t[idx_start:idx_end]
                    
                    # Envelope de "Respiração" longo
                    env = AngelicSynth.envelope(len(t_n), t_abs_n, chord_start, chord_duration, attack=7.0, release=8.0)
                    
                    for ratio in current_chord:
                        freq = self.base_freq * ratio
                        voice = AngelicSynth.choir_voice(t_n, freq, vowel=vowel)
                        
                        # Espalha os cantores pelo estéreo (Panning)
                        pan = np.sin(freq) * 0.4 + 0.5 # Posição semi-aleatória baseada na nota
                        
                        mix_l[idx_start:idx_end] += voice * env * (1 - pan) * 0.4
                        mix_r[idx_start:idx_end] += voice * env * pan * 0.4

        # Efeito Sub-grave Isocrônico (Aterramento) - Frequência contínua muito baixa
        sub = np.sin(2 * np.pi * (self.base_freq / 2.0) * t) * 0.15
        mix_l += sub
        mix_r += sub

        # Aplica o Reverb de Catedral Massivo (A magia acontece aqui)
        mix_l = ChoirDSP.cathedral_reverb(mix_l, mix=0.6)
        mix_r = ChoirDSP.cathedral_reverb(mix_r, mix=0.6)

        return mix_l, mix_r

# ---------------------------------------------------------------------------
# RENDERIZAÇÃO
# ---------------------------------------------------------------------------
def render(output="zen_angelic.wav", duration_s=60):
    total_samples = int(duration_s * SAMPLE_RATE)
    print(f"[ANGELIC CHOIR] Invocando Coral Solfeggio 528Hz ({duration_s}s)...")
    
    engine = SolfeggioEngine(root=528.0)
    
    with sf.SoundFile(output, mode='w', samplerate=SAMPLE_RATE, channels=2) as file:
        current_sample = 0
        while current_sample < total_samples:
            chunk_end = min(current_sample + CHUNK_SAMPLES, total_samples)
            actual_len = chunk_end - current_sample
            t_start = current_sample / SAMPLE_RATE
            d_chunk = actual_len / SAMPLE_RATE
            
            print(f"Sintetizando Vozes Angélicas: {t_start:.1f}s / {duration_s}s ({(current_sample/total_samples)*100:.1f}%)", end='\r')
            
            # Geração
            l, r = engine.generate_chunk(t_start, d_chunk)
            
            # Equalização Master e Soft Limiter para evitar distorções
            l = np.tanh(l * 1.2)
            r = np.tanh(r * 1.2)
            
            stereo = np.column_stack((l, r))
            
            # Fades dinâmicos (para o áudio não começar nem terminar bruscamente)
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

            stereo *= 0.85 
            file.write(stereo)
            current_sample = chunk_end
            
    print(f"\n[ANGELIC CHOIR] Frequência de Cura gerada com sucesso! Arquivo salvo: {output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=60, help="Duração em segundos")
    parser.add_argument("--output", default="zen_angelic.wav", help="Nome do arquivo")
    args = parser.parse_args()
    render(args.output, args.duration)
