"""
Testes automatizados dos componentes do Core Engine.
"""
import pytest
import numpy as np

from core.context import RenderContext
from core.config_loader import ConfigLoader, ConfigError
from core.timeline import Timeline, TimelineEvent
from core.scheduler import Scheduler
from core.plugin_manager import PluginManager, PluginManagerError
from plugins.base import BasePlugin, PluginCategory


# Dummy Plugin para teste de contrato
class DummyTonePlugin(BasePlugin):
    PLUGIN_ID = "dummy_tone"

    def __init__(self, plugin_id: str = "dummy_tone", config=None):
        super().__init__(plugin_id=plugin_id, category=PluginCategory.INSTRUMENT, config=config)
        self.freq = self.config.get("freq", 440.0)
        self.prepared = False
        self.reset_count = 0

    def prepare(self, context, rng):
        self.prepared = True

    def render(self, start_sample: int, num_samples: int) -> np.ndarray:
        t = np.arange(num_samples) / 44100.0
        wave = np.sin(2 * np.pi * self.freq * t).astype(np.float32)
        return np.column_stack((wave, wave))

    def reset(self):
        self.reset_count += 1


def test_render_context_calculations():
    ctx = RenderContext(
        sample_rate=44100,
        channels=2,
        chunk_duration_sec=10.0,
        duration_sec=25.0
    )
    assert ctx.total_samples == 25 * 44100
    assert ctx.chunk_samples == 10 * 44100
    assert ctx.total_chunks == 3

    # Bounds do chunk 0: [0, 441000)
    assert ctx.get_chunk_bounds(0) == (0, 441000)
    # Bounds do chunk 1: [441000, 882000)
    assert ctx.get_chunk_bounds(1) == (441000, 882000)
    # Bounds do chunk 2 (restante): [882000, 1102500)
    assert ctx.get_chunk_bounds(2) == (882000, 1102500)

    # Conversões
    assert ctx.sample_to_time(44100) == 1.0
    assert ctx.time_to_sample(2.5) == int(2.5 * 44100)


def test_render_context_invalid_parameters():
    with pytest.raises(ValueError):
        RenderContext(sample_rate=-1)
    with pytest.raises(ValueError):
        RenderContext(channels=5)
    with pytest.raises(ValueError):
        RenderContext(duration_sec=0)


def test_config_loader(tmp_path):
    valid_yaml = tmp_path / "engine.yaml"
    valid_yaml.write_text("""
engine:
  sample_rate: 48000
  channels: 2
  chunk_duration_sec: 5.0
""", encoding="utf-8")

    cfg = ConfigLoader.load_yaml(valid_yaml)
    validated = ConfigLoader.validate_engine_config(cfg)
    assert validated["engine"]["sample_rate"] == 48000

    invalid_yaml = tmp_path / "invalid.yaml"
    invalid_yaml.write_text("not_a_dict", encoding="utf-8")
    with pytest.raises(ConfigError):
        ConfigLoader.load_yaml(invalid_yaml)


def test_timeline_and_scheduler():
    timeline = Timeline()
    # Evento de 500 a 1500 amostras
    event = TimelineEvent(
        event_id="ev1",
        plugin_id="synth",
        start_sample=500,
        duration_samples=1000
    )
    timeline.add_event(event)

    scheduler = Scheduler(timeline)

    # Chunk antes do evento: [0, 400) -> nada
    assert len(scheduler.schedule_chunk(0, 400)) == 0

    # Chunk que pega o início do evento: [400, 800)
    # Interseção de 500 a 800 (length 300)
    # read_start = 500 - 500 = 0
    # write_start = 500 - 400 = 100
    res = scheduler.schedule_chunk(400, 800)
    assert len(res) == 1
    ev, r_start, w_start, length = res[0]
    assert ev.event_id == "ev1"
    assert r_start == 0
    assert w_start == 100
    assert length == 300

    # Chunk que pega o final do evento: [1200, 1800)
    # Interseção de 1200 a 1500 (length 300)
    # read_start = 1200 - 500 = 700
    # write_start = 1200 - 1200 = 0
    res2 = scheduler.schedule_chunk(1200, 1800)
    assert len(res2) == 1
    ev, r_start, w_start, length = res2[0]
    assert r_start == 700
    assert w_start == 0
    assert length == 300


def test_plugin_manager_lifecycle():
    pm = PluginManager()
    pm.register(DummyTonePlugin, "tone")
    assert pm.is_registered("tone")
    assert "tone" in pm.get_registered_ids()

    instance = pm.create_instance("tone", instance_id="tone_1", config={"freq": 220.0})
    assert instance.plugin_id == "tone_1"
    assert instance.category == PluginCategory.INSTRUMENT
    assert pm.get_instance("tone_1") is instance

    ctx = RenderContext(duration_sec=10.0)
    from core.random_manager import RandomManager
    rm = RandomManager(master_seed=123)

    pm.prepare_all(ctx, rm)
    assert instance.prepared is True

    chunk = instance.render(0, 1000)
    assert chunk.shape == (1000, 2)

    pm.reset_all()
    assert instance.reset_count == 1

    pm.teardown_all()
    assert len(pm.get_active_plugins()) == 0
