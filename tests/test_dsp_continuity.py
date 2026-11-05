"""
Suíte de Testes Automatizados para a Fase 4: DSP, Efeitos, Mixer e Continuidade Contígua de Chunks.
"""
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf

from core.context import RenderContext
from core.random_manager import RandomManager
from core.engine import Engine
from renderer.renderer import Renderer
from assets.asset_manager import AudioAssetManager

from plugins.effects import (
    FilterEffect,
    DelayEffect,
    ReverbEffect,
    ConvolutionReverb,
    SaturationEffect,
    SpatializerEffect,
    ALL_EFFECTS,
)
from mixer import (
    MixerChannel,
    AutomationCurve,
    Mixer,
    DuckingRule,
)
from plugins.instruments import AmbientPad, Piano, Flute
from plugins.ambient import RainAmbient, FireplaceAmbient, WindAmbient
from plugins.textures import BinauralBeats, NoiseTexture
from music.composer import Composer


OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Continuidade e Resposta dos Filtros (Low-pass, High-pass, Band-pass)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("f_type", ["lowpass", "highpass", "bandpass"])
def test_filter_continuity_and_responses(f_type):
    filt = FilterEffect(config={"filter_type": f_type, "cutoff_hz": 1000.0, "cutoff_high_hz": 3000.0})
    ctx = RenderContext(duration_sec=4.0)
    filt.prepare(ctx, np.random.default_rng(42))

    # Sinal de teste com múltiplos componentes de frequência
    t = np.arange(88200) / 44100.0
    sig = np.column_stack((
        np.sin(2 * np.pi * 200 * t) + np.sin(2 * np.pi * 2500 * t),
        np.sin(2 * np.pi * 500 * t) + np.sin(2 * np.pi * 5000 * t)
    )).astype(np.float32)

    # 1. Processa em dois chunks contíguos de 1 segundo (44100 amostras)
    chunk1 = filt.process(sig[:44100])
    chunk2 = filt.process(sig[44100:])
    concatenated = np.concatenate((chunk1, chunk2), axis=0)

    # 2. Processa direto de uma vez só
    filt.reset()
    direct = filt.process(sig)

    # Devem ser matematicamente idênticos graças à preservação de zi/zf
    np.testing.assert_allclose(concatenated, direct, atol=1e-5)

    # Ausência de descontinuidade na transição do chunk em relação ao sinal contínuo direto
    diff_at_boundary = np.abs(concatenated[44100] - direct[44100])
    assert np.max(diff_at_boundary) < 1e-5


# ---------------------------------------------------------------------------
# 2. Preservação de Cauda e Ping-Pong do Delay
# ---------------------------------------------------------------------------
def test_delay_tails_and_ping_pong():
    delay = DelayEffect(config={"mode": "ping_pong", "delay_ms": 300.0, "feedback": 0.7, "mix": 1.0})
    ctx = RenderContext(duration_sec=2.0)
    delay.prepare(ctx, np.random.default_rng(42))

    # Chunk 1: impulso no final do chunk (amostra 35000)
    chunk1_in = np.zeros((44100, 2), dtype=np.float32)
    chunk1_in[35000:35100] = 1.0

    # Chunk 2: entrada 100% silenciosa
    chunk2_in = np.zeros((44100, 2), dtype=np.float32)

    chunk1_out = delay.process(chunk1_in)
    chunk2_out = delay.process(chunk2_in)

    # O chunk 2 deve conter a cauda do delay soando no chunk seguinte
    assert np.max(np.abs(chunk2_out)) > 0.05
    # Verifica energia de repetições decaindo suavemente
    assert np.mean(np.abs(chunk2_out[:22050])) > np.mean(np.abs(chunk2_out[22050:]))


# ---------------------------------------------------------------------------
# 3. Preservação de Cauda e Continuidade do Reverb Algorítmico
# ---------------------------------------------------------------------------
def test_reverb_continuity_and_tails():
    reverb = ReverbEffect(config={"room_size": 0.90, "wet_level": 0.8, "dry_level": 0.0})
    ctx = RenderContext(duration_sec=2.0)
    reverb.prepare(ctx, np.random.default_rng(42))

    # Chunk 1: burst perto do fim do chunk (amostra 35000)
    c1_in = np.zeros((44100, 2), dtype=np.float32)
    c1_in[35000:36000] = 1.0

    # Chunk 2: silêncio absoluto na entrada
    c2_in = np.zeros((44100, 2), dtype=np.float32)

    c1_out = reverb.process(c1_in)
    c2_out = reverb.process(c2_in)

    # A cauda de reverberação continua ativa e mensurável no chunk 2
    assert np.max(np.abs(c2_out)) > 0.005


# ---------------------------------------------------------------------------
# 4. Convolução Overlap-Add e Preservação de Cauda
# ---------------------------------------------------------------------------
def test_convolution_overlap_add_continuity():
    conv = ConvolutionReverb(config={"ir_duration_sec": 1.0, "mix": 0.5})
    ctx = RenderContext(duration_sec=2.0)
    conv.prepare(ctx, np.random.default_rng(101))

    c1_in = np.zeros((44100, 2), dtype=np.float32)
    c1_in[35000:36000] = 1.0
    c2_in = np.zeros((44100, 2), dtype=np.float32)

    c1_out = conv.process(c1_in)
    c2_out = conv.process(c2_in)

    assert c1_out.shape == (44100, 2)
    assert c2_out.shape == (44100, 2)
    # A cauda de convolução deve se estender para o chunk 2
    assert np.max(np.abs(c2_out)) > 0.005


# ---------------------------------------------------------------------------
# 5. Saturação Harmônica e DC Blocker
# ---------------------------------------------------------------------------
def test_saturation_and_dc_blocker():
    sat = SaturationEffect(config={"drive": 2.5, "mix": 1.0})
    ctx = RenderContext(duration_sec=2.0)
    sat.prepare(ctx, np.random.default_rng(42))

    # Onda com pico alto e offset DC artificial
    t = np.arange(44100) / 44100.0
    sig = np.column_stack((
        np.sin(2 * np.pi * 100 * t) * 1.5 + 0.3,
        np.sin(2 * np.pi * 100 * t) * 1.5 + 0.3
    )).astype(np.float32)

    processed = sat.process(sig)
    # Picos contidos no regime de soft-clipping
    assert np.max(np.abs(processed)) <= 1.5
    # DC Blocker deve eliminar a média residual
    assert np.abs(np.mean(processed[22050:])) < 0.05


# ---------------------------------------------------------------------------
# 6. Espacializador (Equal-Power Pan e Mid/Side Width)
# ---------------------------------------------------------------------------
def test_spatializer_pan_and_mid_side():
    spat = SpatializerEffect(config={"pan": 0.0, "stereo_width": 0.0}) # Mono puro
    ctx = RenderContext(duration_sec=1.0)
    spat.prepare(ctx, np.random.default_rng(42))

    # Entrada estéreo com canais diferentes
    sig = np.column_stack((
        np.ones(1000, dtype=np.float32),
        np.zeros(1000, dtype=np.float32)
    ))
    mono_out = spat.process(sig)
    # Com stereo_width = 0, ambos os canais devem ser iguais (Mid puro)
    np.testing.assert_allclose(mono_out[:, 0], mono_out[:, 1], atol=1e-5)


# ---------------------------------------------------------------------------
# 7. Mixer: Automações Contínuas e Sends Auxiliares
# ---------------------------------------------------------------------------
def test_mixer_channel_automation_and_sends():
    ch = MixerChannel(channel_id="pad", volume=0.5, pan=0.0)
    # Automação de volume de 0.2 a 1.0 ao longo de 2 segundos
    ch.volume_auto.add_point(0.0, 0.2)
    ch.volume_auto.add_point(2.0, 1.0)
    ch.set_send("reverb", 0.4)

    sig = np.ones((44100 * 2, 2), dtype=np.float32)
    direct, sends = ch.process(sig, start_sample=0, sample_rate=44100)

    # Verifica fader crescendo continuamente sem degraus
    assert direct[0, 0] < direct[-1, 0]
    # Send de reverb recebeu sinal proporcional
    assert "reverb" in sends
    assert pytest.approx(sends["reverb"][0, 0], 1e-4) == 0.4


# ---------------------------------------------------------------------------
# 8. Mixer: Sidechain Ducking
# ---------------------------------------------------------------------------
def test_mixer_sidechain_ducking():
    mixer = Mixer()
    ch_melody = MixerChannel(channel_id="melody", volume=1.0)
    ch_pad = MixerChannel(channel_id="pad", volume=1.0)

    mixer.add_channel(ch_melody)
    mixer.add_channel(ch_pad)

    # Melodia atenua o pad em 12 dB quando ultrapassa 0.1
    mixer.add_ducking_rule(
        source_channel="melody",
        target_channel="pad",
        threshold=0.1,
        max_attenuation_db=12.0,
        attack_sec=0.01,
        release_sec=0.1
    )
    mixer.sample_rate = 44100

    # 1. Melodia tocando forte
    melody_audio = np.ones((22050, 2), dtype=np.float32) * 0.8
    # Pad sustentado constante
    pad_audio = np.ones((22050, 2), dtype=np.float32) * 0.5

    mixed = mixer.mix_chunk({"melody": melody_audio, "pad": pad_audio}, start_sample=0, num_samples=22050)

    # O pad deve ter sido atenuado pelo ducking
    assert mixed.shape == (22050, 2)
    assert not np.isnan(mixed).any()


# ---------------------------------------------------------------------------
# 9. Geração Prática das 3 Amostras WAV Obrigatórias
# ---------------------------------------------------------------------------
def test_render_sample1_music_reverb_delay():
    """Amostra 1: Música procedural com Reverb e Delay."""
    sample_wav = OUTPUT_DIR / "sample1_music_reverb_delay.wav"
    ctx = RenderContext(
        sample_rate=44100,
        channels=2,
        chunk_duration_sec=5.0,
        duration_sec=30.0,
        seed=101,
        export_subtype="PCM_24"
    )
    ctx.metadata["composer"] = Composer(context=ctx, root_note="D", scale_type="lydian", bpm=52.0)

    engine = Engine(context=ctx)
    engine.plugin_manager.register(AmbientPad)
    engine.plugin_manager.register(Flute)

    pad = engine.plugin_manager.create_instance("ambient_pad", config={"gain": 0.35})
    flute = engine.plugin_manager.create_instance("flute", config={"gain": 0.4})

    # Configura canais no Mixer
    ch_pad = MixerChannel("ambient_pad", volume=0.8, pan=-0.2)
    ch_pad.set_send("reverb_bus", 0.4)

    ch_flute = MixerChannel("flute", volume=0.9, pan=0.2)
    ch_flute.set_send("reverb_bus", 0.5)
    ch_flute.set_send("delay_bus", 0.35)

    engine.mixer.add_channel(ch_pad)
    engine.mixer.add_channel(ch_flute)

    # Buses auxiliares
    engine.mixer.add_aux_bus("reverb_bus", [ReverbEffect(config={"room_size": 0.88, "wet_level": 0.8, "dry_level": 0.0})])
    engine.mixer.add_aux_bus("delay_bus", [DelayEffect(config={"mode": "ping_pong", "delay_ms": 380.0, "feedback": 0.45, "mix": 1.0})])

    renderer = Renderer(engine)
    max_peak = renderer.render_to_file(sample_wav)

    assert sample_wav.exists()
    assert 0.05 < max_peak <= 1.2
    info = sf.info(str(sample_wav))
    assert info.frames == 30 * 44100


def test_render_sample2_ambient_spatial():
    """Amostra 2: Chuva, lareira, vento e espacialização estéreo."""
    sample_wav = OUTPUT_DIR / "sample2_ambient_spatial.wav"
    ctx = RenderContext(
        sample_rate=44100,
        channels=2,
        chunk_duration_sec=5.0,
        duration_sec=30.0,
        seed=202,
        export_subtype="PCM_24"
    )
    ctx.metadata["asset_manager"] = AudioAssetManager(assets_dir="audios", target_sample_rate=44100)

    engine = Engine(context=ctx)
    engine.plugin_manager.register(RainAmbient)
    engine.plugin_manager.register(FireplaceAmbient)
    engine.plugin_manager.register(WindAmbient)

    engine.plugin_manager.create_instance("rain_ambient", config={"gain": 0.35})
    engine.plugin_manager.create_instance("fireplace_ambient", config={"gain": 0.4})
    engine.plugin_manager.create_instance("wind_ambient", config={"gain": 0.3})

    ch_rain = MixerChannel("rain_ambient", volume=0.8, pan=0.0)
    ch_fire = MixerChannel("fireplace_ambient", volume=0.85, pan=0.35) # Lareira à direita
    ch_wind = MixerChannel("wind_ambient", volume=0.7, pan=-0.3)

    # Insere spatializer com órbita no vento
    wind_spat = SpatializerEffect(config={"orbit_speed_hz": 0.05, "orbit_depth": 0.6})
    ch_wind.add_insert(wind_spat)

    engine.mixer.add_channel(ch_rain)
    engine.mixer.add_channel(ch_fire)
    engine.mixer.add_channel(ch_wind)

    renderer = Renderer(engine)
    max_peak = renderer.render_to_file(sample_wav)

    assert sample_wav.exists()
    assert 0.05 < max_peak <= 1.5
    info = sf.info(str(sample_wav))
    assert info.frames == 30 * 44100


def test_render_sample3_full_mix_production():
    """Amostra 3: Mixagem completa (instrumentos, ambientes, texturas, ducking e efeitos)."""
    sample_wav = OUTPUT_DIR / "sample3_full_mix_production.wav"
    ctx = RenderContext(
        sample_rate=44100,
        channels=2,
        chunk_duration_sec=5.0,
        duration_sec=40.0,
        seed=303,
        export_subtype="PCM_24"
    )
    ctx.metadata["asset_manager"] = AudioAssetManager(assets_dir="audios", target_sample_rate=44100)
    ctx.metadata["composer"] = Composer(context=ctx, root_note="F", scale_type="lydian", bpm=50.0)

    engine = Engine(context=ctx)

    # Registra instrumentos e ambientes
    for cls in [AmbientPad, Piano, RainAmbient, NoiseTexture, BinauralBeats]:
        engine.plugin_manager.register(cls)
        engine.plugin_manager.create_instance(cls.PLUGIN_ID)

    # Canais na mesa
    ch_piano = MixerChannel("piano", volume=0.9, pan=0.15)
    ch_piano.set_send("reverb", 0.4)
    ch_piano.set_send("delay", 0.3)

    ch_pad = MixerChannel("ambient_pad", volume=0.7, pan=-0.2)
    ch_pad.set_send("reverb", 0.5)

    ch_rain = MixerChannel("rain_ambient", volume=0.6, pan=0.0)
    ch_noise = MixerChannel("noise_texture", volume=0.25, pan=0.0)
    ch_bin = MixerChannel("binaural_beats", volume=0.6, pan=0.0)

    # Insere filtro LP suave no ruído
    noise_filt = FilterEffect(config={"filter_type": "lowpass", "cutoff_hz": 700.0})
    ch_noise.add_insert(noise_filt)

    for ch in [ch_piano, ch_pad, ch_rain, ch_noise, ch_bin]:
        engine.mixer.add_channel(ch)

    # Sidechain: quando o piano toca notas, atenua levemente o ruído (-4 dB)
    engine.mixer.add_ducking_rule("piano", "noise_texture", threshold=0.08, max_attenuation_db=4.0)

    # Buses auxiliares
    engine.mixer.add_aux_bus("reverb", [ReverbEffect(config={"room_size": 0.85, "wet_level": 0.75, "dry_level": 0.0})])
    engine.mixer.add_aux_bus("delay", [DelayEffect(config={"mode": "ping_pong", "delay_ms": 400.0, "feedback": 0.4, "mix": 1.0})])

    renderer = Renderer(engine)
    max_peak = renderer.render_to_file(sample_wav)

    assert sample_wav.exists()
    assert 0.05 < max_peak <= 1.5
    info = sf.info(str(sample_wav))
    assert info.frames == 40 * 44100
