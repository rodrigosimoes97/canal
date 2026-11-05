"""
Testes de escrita contínua em streaming e do Renderer.
"""
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf

from core.context import RenderContext
from core.engine import Engine
from renderer.chunk_writer import ChunkWriter, ChunkWriterError
from renderer.renderer import Renderer
from plugins.base import BasePlugin, PluginCategory


class SinePlugin(BasePlugin):
    PLUGIN_ID = "sine_test"

    def __init__(self, plugin_id: str = "sine_test", config=None):
        super().__init__(plugin_id=plugin_id, category=PluginCategory.INSTRUMENT, config=config)
        self.freq = 220.0

    def prepare(self, context, rng):
        self._is_prepared = True

    def render(self, start_sample: int, num_samples: int) -> np.ndarray:
        t = (np.arange(num_samples) + start_sample) / 44100.0
        wave = (np.sin(2 * np.pi * self.freq * t) * 0.2).astype(np.float32)
        return np.column_stack((wave, wave))

    def reset(self):
        pass


def test_chunk_writer_basic(tmp_path):
    output_wav = tmp_path / "test_stream.wav"
    ctx = RenderContext(
        sample_rate=44100,
        channels=2,
        duration_sec=2.0,
        export_subtype="PCM_24"
    )

    with ChunkWriter(output_wav, ctx) as writer:
        chunk1 = np.ones((44100, 2), dtype=np.float32) * 0.1
        chunk2 = np.ones((44100, 2), dtype=np.float32) * 0.2
        writer.write_chunk(chunk1)
        writer.write_chunk(chunk2)

    assert output_wav.exists()

    info = sf.info(str(output_wav))
    assert info.samplerate == 44100
    assert info.channels == 2
    assert info.subtype == "PCM_24"
    assert info.frames == 88200
    assert pytest.approx(info.duration, 0.01) == 2.0


def test_chunk_writer_invalid_shape(tmp_path):
    output_wav = tmp_path / "test_invalid.wav"
    ctx = RenderContext(channels=2)

    with ChunkWriter(output_wav, ctx) as writer:
        invalid_mono = np.zeros((1000, 1), dtype=np.float32)
        with pytest.raises(ChunkWriterError):
            writer.write_chunk(invalid_mono)


def test_engine_renderer_pipeline(tmp_path):
    output_wav = tmp_path / "render_full.wav"
    ctx = RenderContext(
        sample_rate=44100,
        channels=2,
        chunk_duration_sec=1.0,
        duration_sec=3.0,
        export_subtype="PCM_24"
    )

    engine = Engine(context=ctx)
    engine.plugin_manager.register(SinePlugin, "sine")
    engine.plugin_manager.create_instance("sine")

    progress_calls = []

    def on_prog(chunk_idx, total_chunks, cur_sec, tot_sec):
        progress_calls.append((chunk_idx, cur_sec))

    renderer = Renderer(engine)
    max_peak = renderer.render_to_file(output_wav, on_progress=on_prog)

    assert output_wav.exists()
    assert len(progress_calls) == 3
    assert progress_calls[-1] == (3, 3.0)
    assert 0.1 < max_peak <= 0.25

    info = sf.info(str(output_wav))
    assert info.frames == 3 * 44100
    assert info.channels == 2
    assert info.subtype == "PCM_24"
