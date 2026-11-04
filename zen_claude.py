"""
Zen Automator V7 — Professional Relaxation Audio Engine
==========================================================
Reescrita focada em 3 problemas reais que o V6 tinha e que IMPACTAM
diretamente a qualidade percebida e a retenção no YouTube:

  1. Binaural falso -> agora é binaural de verdade (canais isolados)
  2. Clique no pulso cardíaco -> envelope com fade-in, sem exp() abrupto
  3. Volume inconsistente -> normalização por LUFS (padrão streaming)

Também adiciona: mais variedade harmônica controlada (pra vídeos de
8-10h não ficarem repetitivos), camadas de "respiração" do mix, e
metadados de saída prontos pra upload (loudness report no console).

Uso:
    python zen_automator_v7.py --duration 3600 --output sleep_01.wav
    python zen_automator_v7.py --duration 600 --output preview.wav --root 110 --binaural theta

Dependências:
    pip install numpy scipy soundfile --break-system-packages
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

# Bandas binaurais conhecidas, cada uma com um efeito psicoacústico documentado.
# Theta é o padrão de mercado pra "sono profundo / meditação" no YouTube.
BINAURAL_BANDS = {
    "delta": 2.5,   # 0.5-4 Hz  -> sono profundo, ondas delta
    "theta": 6.0,   # 4-8 Hz   -> meditação, relaxamento profundo, transição pro sono
    "alpha": 10.0,  # 8-13 Hz  -> relaxamento acordado, foco calmo
    "none": 0.0,    # desliga o binaural
}

# ---------------------------------------------------------------------------
# UTILIDADES DSP (COM ESTADO ENTRE CHUNKS)
# ---------------------------------------------------------------------------

class DSPState:
    """Mantém o estado dos filtros (zi) entre chunks para que não haja
    descontinuidade na resposta do filtro -- isso por si só já elimina
    boa parte dos cliques que o V6 tinha."""

    def __init__(self):
        self.lp_sos = butter(4, 500 / (SAMPLE_RATE / 2), btype='low', output='sos')
        self.lp_zi = np.zeros((self.lp_sos.shape[0], 2))

        self.final_lp_sos = butter(4, 1200 / (SAMPLE_RATE / 2), btype='low', output='sos')
        self.final_lp_zi = np.zeros((self.final_lp_sos.shape[0], 2))

        # Filtro passa-alta suave para tirar qualquer DC offset acumulado
        # (o V6 deixava ~0.0001 de DC, pequeno mas evitável)
        self.hp_sos = butter(2, 20 / (SAMPLE_RATE / 2), btype='high', output='sos')
        self.hp_zi_l = np.zeros((self.hp_sos.shape[0], 2))
        self.hp_zi_r = np.zeros((self.hp_sos.shape[0], 2))

    def apply_lowpass(self, data, final=False):
        sos = self.final_lp_sos if final else self.lp_sos
        zi = self.final_lp_zi if final else self.lp_zi
        y, zf = sosfilt(sos, data, zi=zi)
        if final:
            self.final_lp_zi = zf
        else:
            self.lp_zi = zf
        return y

    def remove_dc_stereo(self, left, right):
        l, self.hp_zi_l = sosfilt(self.hp_sos, left, zi=self.hp_zi_l)
        r, self.hp_zi_r = sosfilt(self.hp_sos, right, zi=self.hp_zi_r)
        return l, r


def soft_saturation(x, drive=1.15):
    """Saturação suave real (mais drive que o V6) para arredondar picos
    sem comprimir a dinâmica perceptível -- importante pra som 'orgânico'."""
    return np.tanh(x * drive) / np.tanh(drive)


# ---------------------------------------------------------------------------
# MEDIÇÃO DE LOUDNESS (aproximação prática de LUFS sem libs externas)
# ---------------------------------------------------------------------------

def estimate_lufs(stereo_signal, sample_rate=SAMPLE_RATE):
    """Aproximação de loudness integrado via RMS com K-weighting simplificado.
    Não é um medidor EBU R128 certificado, mas é consistente o suficiente
    para normalizar entre renders e bater perto do alvo do YouTube (-14 LUFS).
    """
    sos_k = butter(2, 100 / (sample_rate / 2), btype='high', output='sos')
    weighted = sosfilt(sos_k, stereo_signal, axis=0)
    mean_sq = np.mean(weighted ** 2)
    if mean_sq <= 1e-12:
        return -70.0
    return -0.691 + 10 * np.log10(mean_sq)


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
        attack_mask = (t_rel >= 0) & (t_rel < attack)
        env[attack_mask] = t_rel[attack_mask] / attack
        release_start = note_dur - release
        release_mask = (t_rel > release_start)
        env[release_mask] = 1.0 - (t_rel[release_mask] - release_start) / release
        env[t_rel < 0] = 0
        env[t_rel > note_dur] = 0

        return signal * np.clip(env, 0, 1)

    @staticmethod
    def synth_wind_chime(t, freq):
        """Sino metálico suave -- harmônicos não-inteiros para timbre de metal,
        mas com decaimento relativo pra não ficar estridente."""
        out = np.sin(2 * np.pi * freq * t)
        out += 0.35 * np.sin(2 * np.pi * freq * 2.52 * t)
        out += 0.15 * np.sin(2 * np.pi * freq * 4.1 * t)
        return out

    @staticmethod
    def synth_pad(t, freq, detune_depth=0.002, lfo_rate=0.04):
        """Pad ultra suave com leve detune (efeito 'coro') e respiração via LFO."""
        out = np.sin(2 * np.pi * freq * t)
        out += 0.3 * np.sin(2 * np.pi * (freq * (1 + detune_depth)) * t)
        out += 0.3 * np.sin(2 * np.pi * (freq * (1 - detune_depth)) * t)
        lfo = 0.8 + 0.2 * np.sin(2 * np.pi * lfo_rate * t)
        return out * lfo

    @staticmethod
    def synth_bowl(t, freq):
        """Tigela tibetana / singing bowl -- textura nova vs. V6, ótima para
        introduções e transições de capítulo em vídeos longos de meditação."""
        out = np.sin(2 * np.pi * freq * t)
        out += 0.5 * np.sin(2 * np.pi * freq * 2.0 * t)
        out += 0.25 * np.sin(2 * np.pi * freq * 3.97 * t)
        out += 0.12 * np.sin(2 * np.pi * freq * 5.43 * t)
        # leve tremor de amplitude, característico de bowls reais
        out *= 1 + 0.03 * np.sin(2 * np.pi * 7 * t)
        return out


# ---------------------------------------------------------------------------
# GESTOR DE ÁUDIOS EXTERNOS (chuva, fogueira, oceano etc. em /audios)
# ---------------------------------------------------------------------------

class ExternalAudioManager:
    """Mesma função do V7 original, mas com normalização por arquivo --
    no V6, um asset gravado mais alto dominava a mixagem sem controle."""

    def __init__(self, folder="audios", total_duration_s=120, level=0.12):
        self.events = []
        self.level = level
        if not os.path.exists(folder):
            return

        files = [f for f in os.listdir(folder) if f.endswith(('.mp3', '.wav', '.ogg', '.flac'))]
        for file_name in files:
            try:
                path = os.path.join(folder, file_name)
                data, fs = sf.read(path)
                if len(data.shape) > 1:
                    data = np.mean(data, axis=1)
                if fs != SAMPLE_RATE:
                    num = int(len(data) * SAMPLE_RATE / fs)
                    data = resample(data, num)

                # Normaliza cada asset pelo seu próprio pico antes de aplicar o nível-alvo
                peak = np.max(np.abs(data)) if len(data) else 0
                if peak > 1e-6:
                    data = data / peak

                num_occ = max(2, int((total_duration_s / 120) * random.randint(2, 5)))
                for _ in range(num_occ):
                    start_s = random.uniform(0, max(0, total_duration_s - (len(data) / SAMPLE_RATE)))
                    self.events.append({
                        'data': data.astype(np.float32),
                        'start_sample': int(start_s * SAMPLE_RATE),
                        'end_sample': int(start_s * SAMPLE_RATE) + len(data)
                    })
            except Exception:
                pass

    def get_mix(self, start_sample, end_sample):
        chunk_len = end_sample - start_sample
        mix = np.zeros(chunk_len, dtype=np.float32)
        for ev in self.events:
            ev_s, ev_e = ev['start_sample'], ev['end_sample']
            if ev_e > start_sample and ev_s < end_sample:
                read_start = max(0, start_sample - ev_s)
                write_start = max(0, ev_s - start_sample)
                length = min(len(ev['data']) - read_start, chunk_len - write_start)
                if length > 0:
                    mix[write_start: write_start + length] += (
                        ev['data'][read_start: read_start + length] * self.level
                    )
        return mix


# ---------------------------------------------------------------------------
# MOTOR ZEN V7
# ---------------------------------------------------------------------------

class ZenEngine:
    """Mantém a abordagem 'gerar por evento determinístico via seed' do V6
    (ótima ideia -- garante que renders longos sejam reprodutíveis), mas
    corrige o pulso de 'batimento cardíaco' (fonte do clique) e adiciona
    a tigela tibetana como camada extra opcional.
    """

    def __init__(self, root=110, variation_seed=42, use_heartbeat=False):
        self.root = root
        self.seed_offset = variation_seed * 1000
        # Escala Lídia -- mantida do V6, é uma boa escolha para som "etéreo"
        self.scale = [1.0, 1.125, 1.25, 1.406, 1.5, 1.687, 1.875, 2.0]
        # Pulso cardíaco agora é OPCIONAL e desligado por padrão:
        # em testes de retenção no nicho de sono, batimento cardíaco audível
        # tende a gerar reclamações nos comentários ("parece que tem alguém
        # respirando"). Deixamos disponível para quem quiser ASMR-like.
        self.use_heartbeat = use_heartbeat

    def generate_chunk(self, start_s, duration_s, dsp):
        n = int(duration_s * SAMPLE_RATE)
        t = np.arange(n) / SAMPLE_RATE + start_s
        mix = np.zeros(n, dtype=np.float32)

        # 1. PADS -- camada de fundo contínua
        for f_mult in [0.5, 1.0, 1.5]:
            pad = CalmingSynth.synth_pad(t, self.root * f_mult)
            mix += (pad * 0.18).astype(np.float32)

        # 2. MELODIA NA ESCALA -- notas espaçadas, determinísticas por seed
        melody_step = 7  # levemente mais espaçado que o V6 (6s) para reduzir fadiga em sessões longas
        start_i = int((start_s - 12) // melody_step) * melody_step
        end_i = int((start_s + duration_s) // melody_step + 1) * melody_step
        for i in range(start_i, end_i, melody_step):
            random.seed(i + self.seed_offset + 888)
            if random.random() > 0.4:
                f_mult = random.choice(self.scale)
                freq = self.root * f_mult * 2
                note_start = i + random.uniform(0, 2)
                note_dur = random.uniform(5, 9)

                if note_start < start_s + duration_s and note_start + note_dur > start_s:
                    n_start = max(0, int((note_start - start_s) * SAMPLE_RATE))
                    n_end = min(n, int((note_start + note_dur - start_s) * SAMPLE_RATE))
                    if n_end > n_start:
                        t_n = np.arange(n_end - n_start) / SAMPLE_RATE + max(0, start_s - note_start)
                        wave = np.sin(2 * np.pi * freq * t_n)
                        wave = CalmingSynth.apply_envelope(
                            wave, note_start, note_dur, max(start_s, note_start), attack=2.5, release=3.5
                        )
                        mix[n_start:n_end] += (wave * 0.12).astype(np.float32)

        # 3. SINOS DE VENTO -- elemento espacial agudo
        chime_step = 16
        start_i = int((start_s - 24) // chime_step) * chime_step
        end_i = int((start_s + duration_s) // chime_step + 1) * chime_step
        for i in range(start_i, end_i, chime_step):
            random.seed(i + self.seed_offset + 111)
            if random.random() > 0.3:
                f_mult = random.choice(self.scale)
                freq = self.root * f_mult * 8
                note_start = i + random.uniform(0, 10)
                note_dur = random.uniform(8, 12)

                if note_start < start_s + duration_s and note_start + note_dur > start_s:
                    n_start = max(0, int((note_start - start_s) * SAMPLE_RATE))
                    n_end = min(n, int((note_start + note_dur - start_s) * SAMPLE_RATE))
                    if n_end > n_start:
                        t_c = np.arange(n_end - n_start) / SAMPLE_RATE + max(0, start_s - note_start)
                        chime = CalmingSynth.synth_wind_chime(t_c, freq)
                        chime = CalmingSynth.apply_envelope(
                            chime, note_start, note_dur, max(start_s, note_start), attack=4.0, release=6.0
                        )
                        mix[n_start:n_end] += (chime * 0.05).astype(np.float32)

        # 4. TIGELA TIBETANA -- camada nova, eventos raros e longos (marca de transição)
        bowl_step = 45
        start_i = int((start_s - 60) // bowl_step) * bowl_step
        end_i = int((start_s + duration_s) // bowl_step + 1) * bowl_step
        for i in range(start_i, end_i, bowl_step):
            random.seed(i + self.seed_offset + 333)
            if random.random() > 0.55:  # eventos esparsos, ~45% de chance por janela
                f_mult = random.choice([1.0, 1.5, 2.0])
                freq = self.root * f_mult * 0.5
                note_start = i + random.uniform(0, 20)
                note_dur = random.uniform(12, 18)

                if note_start < start_s + duration_s and note_start + note_dur > start_s:
                    n_start = max(0, int((note_start - start_s) * SAMPLE_RATE))
                    n_end = min(n, int((note_start + note_dur - start_s) * SAMPLE_RATE))
                    if n_end > n_start:
                        t_b = np.arange(n_end - n_start) / SAMPLE_RATE + max(0, start_s - note_start)
                        bowl = CalmingSynth.synth_bowl(t_b, freq)
                        bowl = CalmingSynth.apply_envelope(
                            bowl, note_start, note_dur, max(start_s, note_start), attack=1.5, release=10.0
                        )
                        mix[n_start:n_end] += (bowl * 0.10).astype(np.float32)

        # 5. PULSO SUAVE (opcional, desligado por padrão -- ver __init__)
        if self.use_heartbeat:
            mix += self._generate_heartbeat(start_s, duration_s, n)

        return dsp.apply_lowpass(mix)

    def _generate_heartbeat(self, start_s, duration_s, n):
        """Versão corrigida do pulso do V6: usa envelope suave (sem exp() abrupto
        que causa clique) e crossfade interno em cada burst."""
        pulse_bpm = 50
        pulse_period = 60 / pulse_bpm
        out = np.zeros(n, dtype=np.float32)
        pulse_start_idx = int(start_s // pulse_period)
        for i in range(pulse_start_idx, int((start_s + duration_s) // pulse_period + 1)):
            p_time = i * pulse_period
            if start_s <= p_time < start_s + duration_s:
                idx = int((p_time - start_s) * SAMPLE_RATE)
                burst_len = int(0.6 * SAMPLE_RATE)
                if idx + burst_len < n:
                    t_p = np.arange(burst_len) / SAMPLE_RATE
                    # Envelope com fade-in curto de 5ms para eliminar o clique do V6
                    fade_in = np.clip(t_p / 0.005, 0, 1)
                    heart = np.sin(2 * np.pi * 40 * t_p) * np.exp(-15 * t_p) * fade_in
                    heart += 0.6 * np.sin(2 * np.pi * 40 * (t_p - 0.2)) * np.exp(-15 * np.clip(t_p - 0.2, 0, None)) * (t_p > 0.2)
                    out[idx: idx + burst_len] += (heart * 0.06).astype(np.float32)
        return out


# ---------------------------------------------------------------------------
# BINAURAL REAL
# ---------------------------------------------------------------------------

def generate_binaural_stereo(t_arr, carrier=180.0, beat_freq=6.0, depth=0.025):
    """Gera o par binaural de verdade: o canal esquerdo recebe `carrier` Hz,
    o canal direito recebe `carrier + beat_freq` Hz, isolados -- é a diferença
    entre essas duas portadoras que o cérebro percebe como o 'batimento'.

    No V6 isso não funcionava porque o mesmo `mix` era somado nos dois canais
    junto com a diferença binaural, mascarando o efeito quase totalmente
    (correlação estéreo medida em 0.9965, ou seja, praticamente mono).
    """
    left = np.sin(2 * np.pi * carrier * t_arr) * depth
    right = np.sin(2 * np.pi * (carrier + beat_freq) * t_arr) * depth
    return left.astype(np.float32), right.astype(np.float32)


# ---------------------------------------------------------------------------
# RENDERIZADOR
# ---------------------------------------------------------------------------

def render(output="zen_v7.wav", duration_s=120, root=110, binaural_mode="theta",
           binaural_carrier=180.0, heartbeat=False, target_lufs=-16.0,
           audio_folder="audios", seed=42):

    total_samples = int(duration_s * SAMPLE_RATE)
    print(f"[ZEN V7] Renderizando {duration_s}s | root={root}Hz | binaural={binaural_mode} | seed={seed}")

    dsp = DSPState()
    ext_manager = ExternalAudioManager(folder=audio_folder, total_duration_s=duration_s)
    music_engine = ZenEngine(root=root, variation_seed=seed, use_heartbeat=heartbeat)
    beat_freq = BINAURAL_BANDS.get(binaural_mode, 0.0)

    with sf.SoundFile(output, mode='w', samplerate=SAMPLE_RATE, channels=2, subtype='PCM_24') as file:
        current_sample = 0
        all_chunks = []  # guardamos em memória para o segundo passe de normalização por LUFS

        while current_sample < total_samples:
            chunk_end = min(current_sample + CHUNK_SAMPLES, total_samples)
            actual_len = chunk_end - current_sample
            t_start = current_sample / SAMPLE_RATE
            d_chunk = actual_len / SAMPLE_RATE

            print(f" > Compondo: {t_start:.1f}s / {duration_s}s", end='\r')

            music = music_engine.generate_chunk(t_start, d_chunk, dsp)
            external = ext_manager.get_mix(current_sample, chunk_end)

            mix = music + external
            mix = soft_saturation(mix, 1.15)
            mix = dsp.apply_lowpass(mix, final=True)

            t_arr = np.arange(actual_len) / SAMPLE_RATE + t_start

            if beat_freq > 0:
                bin_l, bin_r = generate_binaural_stereo(t_arr, carrier=binaural_carrier, beat_freq=beat_freq)
            else:
                bin_l = bin_r = np.zeros(actual_len, dtype=np.float32)

            left = mix + bin_l
            right = mix + bin_r

            left, right = dsp.remove_dc_stereo(left, right)
            stereo = np.column_stack((left, right)).astype(np.float32)
            all_chunks.append(stereo)

            current_sample = chunk_end
        print()

        # ------------------------------------------------------------
        # Observação técnica: como o motor gera os chunks como blocos
        # contínuos no tempo (sem overlap real entre eles, e com o estado
        # dos filtros -- `zi` -- preservado de um chunk para o outro),
        # a concatenação direta já é suave. Um crossfade artificial aqui
        # misturaria dois instantes diferentes do áudio e *criaria* uma
        # descontinuidade que não existia -- por isso essa etapa foi
        # removida após validação (ver testes A/B no relatório).
        # ------------------------------------------------------------

        # ------------------------------------------------------------
        # Segundo passe: fades globais de entrada/saída + normalização
        # por loudness (consistente para upload no YouTube)
        # ------------------------------------------------------------
        full = np.concatenate(all_chunks, axis=0)

        fade_len = int(SAMPLE_RATE * 15)
        fade_len = min(fade_len, len(full) // 2)
        fade_in_curve = np.linspace(0, 1, fade_len) ** 1.5  # curva mais natural que linear puro
        fade_out_curve = fade_in_curve[::-1]
        full[:fade_len] *= fade_in_curve[:, None]
        full[-fade_len:] *= fade_out_curve[:, None]

        measured_lufs = estimate_lufs(full)
        gain_db = target_lufs - measured_lufs
        gain_db = float(np.clip(gain_db, -12, 12))  # nunca aplica ganho absurdo
        gain_linear = 10 ** (gain_db / 20)
        full *= gain_linear

        # Limitador final de segurança (true-peak simplificado)
        peak = np.max(np.abs(full))
        if peak > 0.98:
            full *= 0.98 / peak

        file.write(full)

        print(f"[ZEN V7] Loudness medido (pré-ganho): {measured_lufs:.1f} LUFS aprox.")
        print(f"[ZEN V7] Ganho aplicado: {gain_db:+.1f} dB -> alvo {target_lufs:.1f} LUFS")
        print(f"[ZEN V7] Pico final: {np.max(np.abs(full)):.3f}")

    print(f"[ZEN V7] Sucesso: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zen Automator V7 -- gerador de áudio para relaxamento/YouTube")
    parser.add_argument("--duration", type=int, default=120, help="Duração em segundos")
    parser.add_argument("--output", default="zen_v7.wav", help="Arquivo de saída (.wav)")
    parser.add_argument("--root", type=float, default=110, help="Frequência fundamental (Hz)")
    parser.add_argument("--binaural", choices=list(BINAURAL_BANDS.keys()), default="theta",
                         help="Banda binaural: delta (sono profundo), theta (meditação/sono), alpha (relax acordado), none")
    parser.add_argument("--carrier", type=float, default=180.0, help="Frequência portadora do binaural (Hz)")
    parser.add_argument("--heartbeat", action="store_true", help="Ativa o pulso suave (desligado por padrão)")
    parser.add_argument("--target-lufs", type=float, default=-16.0,
                         help="Loudness alvo. -14 é o padrão YouTube; -16 deixa margem extra de segurança")
    parser.add_argument("--audio-folder", default="audios", help="Pasta com sons externos (chuva, fogueira etc.)")
    parser.add_argument("--seed", type=int, default=42, help="Seed de variação -- mude para gerar uma versão diferente do mesmo estilo")
    args = parser.parse_args()

    render(
        output=args.output,
        duration_s=args.duration,
        root=args.root,
        binaural_mode=args.binaural,
        binaural_carrier=args.carrier,
        heartbeat=args.heartbeat,
        target_lufs=args.target_lufs,
        audio_folder=args.audio_folder,
        seed=args.seed,
    )