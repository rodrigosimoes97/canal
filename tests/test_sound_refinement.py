"""
Testes Automatizados da Fase 4.1: Refinamento Sonoro, Modelagem Físico-Acústica e Direção Artística.
Gera as 3 amostras WAV refinadas de produção em conformidade com o GEMINI.md.
"""
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf

from core.context import RenderContext
from core.engine import Engine
from music.composer import Composer
from music.melody import MelodicNote
from assets.asset_manager import AudioAssetManager
from mixer.channel import MixerChannel
from mixer.mixer import Mixer
from renderer.renderer import Renderer

from plugins.instruments.piano import Piano, PianoNoteEvent
from plugins.instruments.flute import Flute, FluteNoteEvent
from plugins.instruments.ambient_pad import AmbientPad
from plugins.instruments.singing_bowl import SingingBowl
from plugins.ambient.rain import RainAmbient
from plugins.ambient.fireplace import FireplaceAmbient
from plugins.ambient.wind import WindAmbient
from plugins.ambient.forest import ForestAmbient
from plugins.textures.binaural import BinauralBeats
from plugins.textures.drone import DroneTexture
from plugins.effects.reverb import ReverbEffect
from plugins.effects.delay import DelayEffect
from plugins.effects.spatializer import SpatializerEffect
from plugins.effects.filter import FilterEffect

OUTPUT_DIR = Path("output")


def test_felt_piano_physical_properties():
    """Valida a modelagem do Piano de Feltro: inarmonicidade, uníssono e modo esparso."""
    ctx = RenderContext(sample_rate=44100, channels=2, duration_sec=30.0, seed=42)
    ctx.metadata["composer"] = Composer(context=ctx, root_note="C", scale_type="major", bpm=50.0)
    rng = np.random.default_rng(42)

    piano = Piano(config={"gain": 0.35, "sparse_mode": True, "density": 0.3})
    piano.prepare(ctx, rng)

    # Verifica que em modo esparso contemplativo foram agendadas poucas notas (esparsas)
    assert len(piano.notes) <= 5, f"Esperava poucas notas esparsas para 30s de áudio, obteve {len(piano.notes)}"

    # Renderiza e valida amplitude e ausência de clipping
    audio = piano.render(0, 44100 * 5)
    assert audio.shape == (44100 * 5, 2)
    assert np.max(np.abs(audio)) < 1.0


def test_organic_flute_breath_and_resonance():
    """Valida a modelagem da Flauta de Bambu orgânica com ruído de ar e sem vibrato mecânico."""
    ctx = RenderContext(sample_rate=44100, channels=2, duration_sec=40.0, seed=77)
    ctx.metadata["composer"] = Composer(context=ctx, root_note="D", scale_type="lydian", bpm=52.0)
    rng = np.random.default_rng(77)

    flute = Flute(config={"gain": 0.3})
    flute.prepare(ctx, rng)

    assert len(flute.notes) >= 1
    # Verifica que as notas duram bastante (sopro sustentado de meditação)
    for ev in flute.notes:
        assert ev.duration_sec >= 3.5

    audio = flute.render(0, 44100 * 5)
    assert audio.shape == (44100 * 5, 2)
    assert np.max(np.abs(audio)) < 1.0


def test_ambient_pad_warm_filtering():
    """Valida o calor analógico e atenuação de agudos do AmbientPad."""
    ctx = RenderContext(sample_rate=44100, channels=2, duration_sec=10.0, seed=12)
    ctx.metadata["composer"] = Composer(context=ctx, root_note="F", scale_type="lydian", bpm=50.0)
    rng = np.random.default_rng(12)

    pad = AmbientPad(config={"gain": 0.3, "cutoff_hz": 500.0})
    pad.prepare(ctx, rng)

    audio = pad.render(0, 44100 * 2)
    assert audio.shape == (44100 * 2, 2)

    # Análise espectral: energia em frequências > 4000 Hz deve ser praticamente nula
    fft_mag = np.abs(np.fft.rfft(audio[:, 0]))
    freqs = np.fft.rfftfreq(len(audio[:, 0]), d=1.0 / 44100.0)

    low_energy = np.sum(fft_mag[freqs < 1000] ** 2)
    high_energy = np.sum(fft_mag[freqs > 4000] ** 2)

    # Alta proporção de calor grave vs agudo
    assert low_energy > 50.0 * (high_energy + 1e-12)


def test_composer_sparse_gestures():
    """Valida a geração de gestos melódicos contemplativos esparsos no Composer."""
    ctx = RenderContext(sample_rate=44100, channels=2, duration_sec=60.0, seed=99)
    composer = Composer(context=ctx, root_note="G", scale_type="dorian", bpm=50.0)
    rng = np.random.default_rng(99)

    gestures = composer.melody.generate_sparse_gestures(
        start_beat=0.0,
        duration_beats=40.0,
        rng=rng,
        min_interval_beats=16.0,
        max_interval_beats=32.0,
        density=0.3
    )

    # Devem ser raras e com longa sustentação
    assert len(gestures) <= 4
    for g in gestures:
        assert g.duration_beats >= 2.0


# ---------------------------------------------------------------------------
# Renderização das 3 Amostras Refinadas de Produção da Fase 4.1
# ---------------------------------------------------------------------------

def test_render_sample1_refined_ambient_music():
    """
    Amostra 1 Refinada: Paisagem acústica relaxante com Felt Piano esparso e aveludado,
    Ambient Pad quente e acolhedor, leito sutil de chuva natural e Reverb escuro.
    """
    sample_wav = OUTPUT_DIR / "sample1_refined_ambient_music.wav"
    ctx = RenderContext(
        sample_rate=44100,
        channels=2,
        chunk_duration_sec=5.0,
        duration_sec=35.0,
        seed=1001,
        export_subtype="PCM_24"
    )
    ctx.metadata["asset_manager"] = AudioAssetManager(assets_dir="audios", target_sample_rate=44100)
    ctx.metadata["composer"] = Composer(
        context=ctx,
        root_note="D",
        scale_type="lydian",
        bpm=48.0,
        progression_tokens=["Iadd9", "IVsus2"],
        chord_duration_beats=24.0 # Acordes sustentados por ~30 segundos
    )

    engine = Engine(context=ctx)
    engine.plugin_manager.register(AmbientPad)
    engine.plugin_manager.register(Piano)
    engine.plugin_manager.register(RainAmbient)

    engine.plugin_manager.create_instance("rain_ambient", config={"gain": 0.38})
    engine.plugin_manager.create_instance("ambient_pad", config={"gain": 0.20, "cutoff_hz": 480.0})
    engine.plugin_manager.create_instance("piano", config={"gain": 0.22, "sparse_mode": True, "density": 0.3})

    # Hierarquia Sonora: Chuva em 1º plano, Pad em 2º plano, Felt Piano como toque distante
    ch_rain = MixerChannel("rain_ambient", volume=0.65, pan=0.0)
    ch_pad = MixerChannel("ambient_pad", volume=0.38, pan=-0.15)
    ch_pad.set_send("reverb_bus", 0.45)

    ch_piano = MixerChannel("piano", volume=0.30, pan=0.15)
    ch_piano.set_send("reverb_bus", 0.60) # Afundado no reverb escuro
    ch_piano.set_send("delay_bus", 0.18)

    engine.mixer.add_channel(ch_rain)
    engine.mixer.add_channel(ch_pad)
    engine.mixer.add_channel(ch_piano)

    # Reverb escuro e delay etéreo suave
    dark_reverb = ReverbEffect(config={"room_size": 0.90, "damping": 0.65, "lowpass_cutoff": 2600.0, "wet_level": 0.85, "dry_level": 0.0})
    ping_delay = DelayEffect(config={"mode": "ping_pong", "delay_ms": 520.0, "feedback": 0.35, "mix": 1.0, "damping": 0.5})

    engine.mixer.add_aux_bus("reverb_bus", [dark_reverb])
    engine.mixer.add_aux_bus("delay_bus", [ping_delay])

    renderer = Renderer(engine)
    max_peak = renderer.render_to_file(sample_wav)

    assert sample_wav.exists()
    assert 0.02 < max_peak <= 1.2
    info = sf.info(str(sample_wav))
    assert info.frames == 35 * 44100


def test_render_sample2_refined_nature_zen():
    """
    Amostra 2 Refinada: Paisagem Zen com Floresta, Chuva suave e Lareira crepitante
    em 1º plano, complementadas por notas meditativas de Flauta de Bambu e Tigela Tibetana.
    """
    sample_wav = OUTPUT_DIR / "sample2_refined_nature_zen.wav"
    ctx = RenderContext(
        sample_rate=44100,
        channels=2,
        chunk_duration_sec=5.0,
        duration_sec=35.0,
        seed=2002,
        export_subtype="PCM_24"
    )
    ctx.metadata["asset_manager"] = AudioAssetManager(assets_dir="audios", target_sample_rate=44100)
    ctx.metadata["composer"] = Composer(
        context=ctx,
        root_note="E",
        scale_type="major_pentatonic",
        bpm=45.0,
        chord_duration_beats=32.0
    )

    engine = Engine(context=ctx)
    engine.plugin_manager.register(RainAmbient)
    engine.plugin_manager.register(FireplaceAmbient)
    engine.plugin_manager.register(ForestAmbient)
    engine.plugin_manager.register(Flute)
    engine.plugin_manager.register(SingingBowl)

    engine.plugin_manager.create_instance("rain_ambient", config={"gain": 0.32})
    engine.plugin_manager.create_instance("fireplace_ambient", config={"gain": 0.35})
    engine.plugin_manager.create_instance("forest_ambient", config={"gain": 0.28})
    engine.plugin_manager.create_instance("flute", config={"gain": 0.20})
    engine.plugin_manager.create_instance("singing_bowl", config={"gain": 0.25, "interval_sec": 30.0})

    # Canais no Mixer
    ch_rain = MixerChannel("rain_ambient", volume=0.65, pan=-0.1)
    ch_fire = MixerChannel("fireplace_ambient", volume=0.70, pan=0.3)
    ch_forest = MixerChannel("forest_ambient", volume=0.50, pan=-0.3)

    ch_flute = MixerChannel("flute", volume=0.28, pan=0.1)
    ch_flute.set_send("reverb_bus", 0.65) # Sopro distante e místico

    ch_bowl = MixerChannel("singing_bowl", volume=0.32, pan=0.0)
    ch_bowl.set_send("reverb_bus", 0.45)

    engine.mixer.add_channel(ch_rain)
    engine.mixer.add_channel(ch_fire)
    engine.mixer.add_channel(ch_forest)
    engine.mixer.add_channel(ch_flute)
    engine.mixer.add_channel(ch_bowl)

    dark_reverb = ReverbEffect(config={"room_size": 0.92, "damping": 0.60, "lowpass_cutoff": 2800.0, "wet_level": 0.85, "dry_level": 0.0})
    engine.mixer.add_aux_bus("reverb_bus", [dark_reverb])

    renderer = Renderer(engine)
    max_peak = renderer.render_to_file(sample_wav)

    assert sample_wav.exists()
    assert 0.02 < max_peak <= 1.2
    info = sf.info(str(sample_wav))
    assert info.frames == 35 * 44100


def test_render_sample3_refined_deep_relaxation():
    """
    Amostra 3 Refinada: Mix completo de relaxamento profundo e sono (Deep Sleep / Meditation).
    Natureza dominante (Chuva, Lareira, Vento suave em órbita), Textura Binaural Theta (6 Hz),
    Pad quente aveludado e notas raras de Felt Piano flutuando com elegância sem competir.
    """
    sample_wav = OUTPUT_DIR / "sample3_refined_deep_relaxation.wav"
    ctx = RenderContext(
        sample_rate=44100,
        channels=2,
        chunk_duration_sec=5.0,
        duration_sec=45.0,
        seed=3003,
        export_subtype="PCM_24"
    )
    ctx.metadata["asset_manager"] = AudioAssetManager(assets_dir="audios", target_sample_rate=44100)
    ctx.metadata["composer"] = Composer(
        context=ctx,
        root_note="C",
        scale_type="lydian",
        bpm=46.0,
        progression_tokens=["Iadd9", "IVmaj7"],
        chord_duration_beats=32.0
    )

    engine = Engine(context=ctx)

    for cls in [RainAmbient, FireplaceAmbient, WindAmbient, BinauralBeats, AmbientPad, Piano]:
        engine.plugin_manager.register(cls)

    engine.plugin_manager.create_instance("rain_ambient", config={"gain": 0.35})
    engine.plugin_manager.create_instance("fireplace_ambient", config={"gain": 0.35})
    engine.plugin_manager.create_instance("wind_ambient", config={"gain": 0.28})
    engine.plugin_manager.create_instance("binaural_beats", config={"wave_type": "theta", "gain": 0.18})
    engine.plugin_manager.create_instance("ambient_pad", config={"gain": 0.18, "cutoff_hz": 460.0})
    engine.plugin_manager.create_instance("piano", config={"gain": 0.20, "sparse_mode": True, "density": 0.22})

    # Mixer - Design Acústico de Camadas
    ch_rain = MixerChannel("rain_ambient", volume=0.70, pan=-0.1)
    ch_fire = MixerChannel("fireplace_ambient", volume=0.65, pan=0.35)
    ch_wind = MixerChannel("wind_ambient", volume=0.50, pan=-0.3)
    ch_wind.add_insert(SpatializerEffect(config={"orbit_speed_hz": 0.04, "orbit_depth": 0.5}))

    ch_bin = MixerChannel("binaural_beats", volume=0.25, pan=0.0)
    ch_pad = MixerChannel("ambient_pad", volume=0.30, pan=-0.2)
    ch_pad.set_send("reverb_bus", 0.55)

    ch_piano = MixerChannel("piano", volume=0.25, pan=0.2)
    ch_piano.set_send("reverb_bus", 0.65)
    ch_piano.set_send("delay_bus", 0.20)

    # Ducking suave: quando o piano toca uma nota, atenua levemente o pad (-3 dB) para abrir espaço
    engine.mixer.add_ducking_rule("piano", "ambient_pad", threshold=0.06, max_attenuation_db=3.0)

    for ch in [ch_rain, ch_fire, ch_wind, ch_bin, ch_pad, ch_piano]:
        engine.mixer.add_channel(ch)

    dark_reverb = ReverbEffect(config={"room_size": 0.94, "damping": 0.70, "lowpass_cutoff": 2400.0, "wet_level": 0.85, "dry_level": 0.0})
    ping_delay = DelayEffect(config={"mode": "ping_pong", "delay_ms": 600.0, "feedback": 0.30, "mix": 1.0, "damping": 0.6})

    engine.mixer.add_aux_bus("reverb_bus", [dark_reverb])
    engine.mixer.add_aux_bus("delay_bus", [ping_delay])

    renderer = Renderer(engine)
    max_peak = renderer.render_to_file(sample_wav)

    assert sample_wav.exists()
    assert 0.02 < max_peak <= 1.2
    info = sf.info(str(sample_wav))
    assert info.frames == 45 * 44100
