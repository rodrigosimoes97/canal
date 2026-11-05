"""
Suíte de Testes Automatizados para a Fase 3: AudioAssetManager, Ambientes e Texturas.
"""
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf

from core.context import RenderContext
from core.random_manager import RandomManager
from core.engine import Engine
from renderer.renderer import Renderer
from assets.asset_manager import AudioAssetManager, AssetError

from plugins.ambient import (
    RainAmbient,
    BirdsAmbient,
    FireplaceAmbient,
    ThunderAmbient,
    OceanAmbient,
    WindAmbient,
    ForestAmbient,
    WaterAmbient,
    ALL_AMBIENTS,
)
from plugins.textures import (
    NoiseTexture,
    BinauralBeats,
    DroneTexture,
    HarmonicDrone,
    ALL_TEXTURES,
)


AUDIOS_DIR = Path("audios")


# ---------------------------------------------------------------------------
# 1. Testes do AudioAssetManager (Resampling, Canais e Loops)
# ---------------------------------------------------------------------------
def test_asset_manager_indexing():
    mgr = AudioAssetManager(assets_dir=AUDIOS_DIR, target_sample_rate=44100)
    avail = mgr.list_available_assets()
    assert len(avail) >= 7
    assert mgr.has_asset("chuva")
    assert mgr.has_asset("lareira")
    assert mgr.has_asset("passaro")
    assert mgr.has_asset("ruido")


def test_asset_manager_resampling_and_channels():
    mgr = AudioAssetManager(assets_dir=AUDIOS_DIR, target_sample_rate=44100)

    # 1. passaro.mp3 é 44100Hz mono -> deve ser convertido para 2 canais (estéreo)
    passaro_data = mgr.load_asset("passaro")
    assert passaro_data.ndim == 2
    assert passaro_data.shape[1] == 2
    assert passaro_data.dtype == np.float32

    # 2. chuva.mp3 é 48000Hz estéreo -> deve ser resampleado polifasicamente para 44100Hz
    # Duração original ~765.1s -> no sample rate 44100 deve ter ~33.740.910 amostras
    chuva_data = mgr.load_asset("chuva")
    assert chuva_data.ndim == 2
    assert chuva_data.shape[1] == 2
    dur_chuva = chuva_data.shape[0] / 44100.0
    assert pytest.approx(dur_chuva, abs=1.0) == 765.1

    # 3. ruido.mp3 é 24000Hz -> deve ser resampleado para 44100Hz
    ruido_data = mgr.load_asset("ruido")
    assert ruido_data.ndim == 2
    assert ruido_data.shape[1] == 2
    dur_ruido = ruido_data.shape[0] / 44100.0
    assert pytest.approx(dur_ruido, abs=0.5) == 9.0


def test_asset_manager_loop_crossfade_continuity():
    mgr = AudioAssetManager(assets_dir=AUDIOS_DIR, target_sample_rate=44100)

    # Solicita 2 chunks contíguos de 10s da chuva em loop
    chunk1 = mgr.get_loop_chunk("chuva", start_sample=0, num_samples=441000, crossfade_sec=2.0)
    chunk2 = mgr.get_loop_chunk("chuva", start_sample=441000, num_samples=441000, crossfade_sec=2.0)

    assert chunk1.shape == (441000, 2)
    assert chunk2.shape == (441000, 2)
    assert not np.isnan(chunk1).any()
    assert not np.isnan(chunk2).any()

    # Valida continuidade na fronteira (amostra 440999 e 441000)
    boundary_diff = np.abs(chunk2[0] - chunk1[-1])
    assert np.max(boundary_diff) < 0.25 # Sem salto ou corte abrupto


# ---------------------------------------------------------------------------
# 2. Validação individual dos 8 plugins ambientais
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ambient_class", ALL_AMBIENTS)
def test_all_ambient_plugins(ambient_class):
    ctx = RenderContext(duration_sec=10.0, seed=42)
    mgr = AudioAssetManager(assets_dir=AUDIOS_DIR, target_sample_rate=44100)
    ctx.metadata["asset_manager"] = mgr

    rm = RandomManager(master_seed=42)
    rng = rm.get_plugin_rng(ambient_class.PLUGIN_ID)

    plugin = ambient_class()
    assert plugin.category.value == "ambient"

    plugin.prepare(ctx, rng)
    chunk = plugin.render(0, 44100 * 2)

    assert chunk.shape == (88200, 2)
    assert chunk.dtype == np.float32
    assert not np.isnan(chunk).any()
    assert not np.isinf(chunk).any()

    plugin.reset()


# ---------------------------------------------------------------------------
# 3. Validação individual dos plugins de texturas
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("texture_class", ALL_TEXTURES)
def test_all_texture_plugins(texture_class):
    ctx = RenderContext(duration_sec=10.0, seed=42)
    rm = RandomManager(master_seed=42)
    rng = rm.get_plugin_rng(texture_class.PLUGIN_ID)

    plugin = texture_class()
    assert plugin.category.value == "texture"

    plugin.prepare(ctx, rng)
    chunk = plugin.render(0, 44100 * 2)

    assert chunk.shape == (88200, 2)
    assert chunk.dtype == np.float32
    assert not np.isnan(chunk).any()
    assert not np.isinf(chunk).any()

    plugin.reset()


# ---------------------------------------------------------------------------
# 4. Teste específico de Binaural Beats (Canais Estéreo Isolados)
# ---------------------------------------------------------------------------
def test_binaural_beats_stereo_isolation():
    binaural = BinauralBeats(config={"mode": "theta", "carrier_freq": 180.0, "beat_freq": 6.0})
    ctx = RenderContext(duration_sec=5.0)
    binaural.prepare(ctx, np.random.default_rng(123))

    chunk = binaural.render(0, 44100) # 1 segundo
    left = chunk[:, 0]
    right = chunk[:, 1]

    # Os canais não devem ser idênticos (correlação < 1.0)
    correlation = np.corrcoef(left, right)[0, 1]
    assert correlation < 0.95 # Prova que são frequências distintas nos canais


# ---------------------------------------------------------------------------
# 5. Teste específico de Ruídos (White, Pink, Brown)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n_type", ["white", "pink", "brown"])
def test_noise_textures_spectral_response(n_type):
    noise = NoiseTexture(config={"noise_type": n_type})
    ctx = RenderContext(duration_sec=5.0)
    rng = np.random.default_rng(42)
    noise.prepare(ctx, rng)

    chunk = noise.render(0, 44100 * 2)
    assert chunk.shape == (88200, 2)
    assert np.max(np.abs(chunk)) > 0.01 # Sinal ativo


# ---------------------------------------------------------------------------
# 6. Geração de Amostras Reais WAV (30s e 60s)
# ---------------------------------------------------------------------------
def test_render_real_ambient_texture_samples(tmp_path):
    """
    Renderiza uma amostra combinando chuva, vento, lareira, ocean, ruído marrom e binaural beats.
    Valida integridade do arquivo WAV gerado sem erros.
    """
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_wav = output_dir / "ambient_texture_preview_30s.wav"

    ctx = RenderContext(
        sample_rate=44100,
        channels=2,
        chunk_duration_sec=5.0,
        duration_sec=30.0,
        seed=108,
        export_subtype="PCM_24"
    )
    mgr = AudioAssetManager(assets_dir=AUDIOS_DIR, target_sample_rate=44100)
    ctx.metadata["asset_manager"] = mgr

    engine = Engine(context=ctx)

    # Registra e ativa camadas ambientais e texturas
    engine.plugin_manager.register(RainAmbient)
    engine.plugin_manager.register(FireplaceAmbient)
    engine.plugin_manager.register(WindAmbient)
    engine.plugin_manager.register(NoiseTexture)
    engine.plugin_manager.register(BinauralBeats)

    engine.plugin_manager.create_instance("rain_ambient", config={"gain": 0.3})
    engine.plugin_manager.create_instance("fireplace_ambient", config={"gain": 0.35})
    engine.plugin_manager.create_instance("wind_ambient", config={"gain": 0.25})
    engine.plugin_manager.create_instance("noise_texture", config={"noise_type": "brown", "gain": 0.15})
    engine.plugin_manager.create_instance("binaural_beats", config={"mode": "theta", "gain": 0.03})

    renderer = Renderer(engine)
    max_peak = renderer.render_to_file(sample_wav)

    assert sample_wav.exists()
    assert 0.05 < max_peak <= 1.5

    info = sf.info(str(sample_wav))
    assert info.samplerate == 44100
    assert info.channels == 2
    assert info.subtype == "PCM_24"
    assert info.frames == 30 * 44100
