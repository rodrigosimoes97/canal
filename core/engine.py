"""
Orquestrador Principal do Relaxation Audio Engine.
"""
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np

from core.context import RenderContext
from core.random_manager import RandomManager
from core.timeline import Timeline
from core.scheduler import Scheduler
from core.plugin_manager import PluginManager
from core.config_loader import ConfigLoader
from mixer.mixer import Mixer


class Engine:
    """
    Núcleo unificado de orquestração do Relaxation Audio Engine.
    Coordena contexto, plugins, linha do tempo e geração por streaming.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        context: Optional[RenderContext] = None
    ) -> None:
        self.raw_config: Dict[str, Any] = config or {}
        
        # Se um contexto não foi passado, cria o padrão baseado no config ou defaults
        if context is None:
            eng_cfg = self.raw_config.get("engine", {})
            self.context = RenderContext(
                sample_rate=eng_cfg.get("sample_rate", 44100),
                channels=eng_cfg.get("channels", 2),
                chunk_duration_sec=eng_cfg.get("chunk_duration_sec", 10.0),
                duration_sec=eng_cfg.get("duration_sec", 120.0),
                seed=eng_cfg.get("default_seed", 42),
                internal_dtype=eng_cfg.get("internal_dtype", "float32"),
                export_format=eng_cfg.get("export_format", "WAV"),
                export_subtype=eng_cfg.get("export_subtype", "PCM_24"),
            )
        else:
            self.context = context

        self.random_manager = RandomManager(master_seed=self.context.seed)
        self.timeline = Timeline()
        self.scheduler = Scheduler(self.timeline)
        self.plugin_manager = PluginManager()
        self.mixer = Mixer()
        self._is_prepared = False

    @classmethod
    def from_config_file(cls, config_path: Path | str, **overrides: Any) -> "Engine":
        """Instancia o Engine a partir de um arquivo engine.yaml."""
        raw_config = ConfigLoader.load_yaml(config_path)
        ConfigLoader.validate_engine_config(raw_config)

        eng_cfg = raw_config.get("engine", {})
        sample_rate = overrides.get("sample_rate", eng_cfg.get("sample_rate", 44100))
        channels = overrides.get("channels", eng_cfg.get("channels", 2))
        chunk_sec = overrides.get("chunk_duration_sec", eng_cfg.get("chunk_duration_sec", 10.0))
        duration = overrides.get("duration_sec", overrides.get("duration", 120.0))
        seed = overrides.get("seed", eng_cfg.get("default_seed", 42))

        ctx = RenderContext(
            sample_rate=sample_rate,
            channels=channels,
            chunk_duration_sec=chunk_sec,
            duration_sec=duration,
            seed=seed,
            internal_dtype=eng_cfg.get("internal_dtype", "float32"),
            export_format=eng_cfg.get("export_format", "WAV"),
            export_subtype=eng_cfg.get("export_subtype", "PCM_24"),
        )

        return cls(config=raw_config, context=ctx)

    def prepare(self) -> None:
        """Prepara o motor, plugins e mixer para o início da renderização."""
        self.random_manager.reset()
        self.plugin_manager.prepare_all(self.context, self.random_manager)
        self.mixer.prepare(self.context, self.random_manager.get_plugin_rng("mixer_master"))
        self._is_prepared = True

    def render_chunk(self, chunk_idx: int) -> np.ndarray:
        """
        Renderiza um único bloco amostral pelo índice do chunk.
        Se o Mixer possuir canais cadastrados, utiliza a mesa de mixagem com DSP e Sends.
        Caso contrário, realiza a soma direta como fallback.
        """
        if not self._is_prepared:
            self.prepare()

        start_sample, end_sample = self.context.get_chunk_bounds(chunk_idx)
        num_samples = end_sample - start_sample

        active_plugins = self.plugin_manager.get_active_plugins()

        # Se houver canais configurados no Mixer
        if self.mixer.channels:
            raw_inputs: Dict[str, np.ndarray] = {}
            for plugin in active_plugins:
                if not plugin.is_active:
                    continue
                audio = plugin.render(start_sample, num_samples)
                if audio is not None:
                    raw_inputs[plugin.plugin_id] = audio.astype(np.float32)

            return self.mixer.mix_chunk(raw_inputs, start_sample, num_samples)

        # Fallback de soma direta tradicional
        output = np.zeros((num_samples, self.context.channels), dtype=np.float32)
        for plugin in active_plugins:
            if not plugin.is_active:
                continue
            chunk_audio = plugin.render(start_sample, num_samples)
            if chunk_audio is not None and chunk_audio.shape == output.shape:
                output += chunk_audio.astype(np.float32)

        return output

    def reset(self) -> None:
        """Reinicializa o estado de execução do motor."""
        self.random_manager.reset()
        self.plugin_manager.reset_all()
        self.mixer.reset()
        self._is_prepared = False
