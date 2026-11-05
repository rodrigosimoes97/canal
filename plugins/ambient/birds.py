"""
Plugin Ambiental: Canto de Pássaros (Eventos Esparsos Espacializados).
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np

from plugins.base import BasePlugin, PluginCategory
from assets.asset_manager import AudioAssetManager


@dataclass
class BirdCallEvent:
    asset_name: str
    start_sample: int
    num_samples: int
    gain: float = 0.5
    pan: float = 0.0


class BirdsAmbient(BasePlugin):
    """
    Agendador determinístico de cantos de pássaros ('passaro.mp3' e 'passaro2.mp3')
    com espacialização estéreo e densidade natural.
    """
    PLUGIN_ID = "birds_ambient"

    def __init__(self, plugin_id: str = "birds_ambient", config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(plugin_id=plugin_id, category=PluginCategory.AMBIENT, config=config)
        self.gain = float(self.config.get("gain", 0.4))
        self.sample_rate: int = 44100
        self.asset_mgr: Optional[AudioAssetManager] = None
        self.events: List[BirdCallEvent] = []

    def prepare(self, context: Any, rng: np.random.Generator) -> None:
        self.sample_rate = context.sample_rate
        self.asset_mgr = context.metadata.get("asset_manager")
        self.events.clear()

        if self.asset_mgr is None:
            self._is_prepared = True
            return

        available_birds = []
        for name in ["passaro", "passaro2"]:
            if self.asset_mgr.has_asset(name):
                available_birds.append(name)

        if not available_birds:
            self._is_prepared = True
            return

        cur_sec = 3.0
        total_sec = context.duration_sec

        while cur_sec < total_sec - 4.0:
            asset_choice = str(rng.choice(available_birds))
            raw_data = self.asset_mgr.load_asset(asset_choice)
            asset_len = raw_data.shape[0]

            s_sample = int(round(cur_sec * self.sample_rate))
            pan = float(rng.uniform(-0.8, 0.8)) # Espalhado no panorama estéreo
            call_gain = float(rng.uniform(0.6, 0.95))

            self.events.append(BirdCallEvent(
                asset_name=asset_choice,
                start_sample=s_sample,
                num_samples=asset_len,
                gain=call_gain,
                pan=pan
            ))
            # Intervalo entre cantos: 8 a 18 segundos
            cur_sec += float(rng.uniform(8.0, 18.0))

        self._is_prepared = True

    def render(self, start_sample: int, num_samples: int) -> np.ndarray:
        if not self.is_active or num_samples <= 0 or self.asset_mgr is None:
            return np.zeros((num_samples, 2), dtype=np.float32)

        chunk_end = start_sample + num_samples
        out_l = np.zeros(num_samples, dtype=np.float32)
        out_r = np.zeros(num_samples, dtype=np.float32)

        for ev in self.events:
            ev_start = ev.start_sample
            ev_end = ev_start + ev.num_samples

            if ev_end > start_sample and ev_start < chunk_end:
                read_start = max(0, start_sample - ev_start)
                write_start = max(0, ev_start - start_sample)
                length = min(ev.num_samples - read_start, num_samples - write_start)

                if length > 0:
                    raw_data = self.asset_mgr.load_asset(ev.asset_name)
                    segment = raw_data[read_start : read_start + length]

                    # Converte segment para mono base e aplica pan
                    mono_seg = np.mean(segment, axis=1) * (ev.gain * self.gain)
                    pan_l = 0.5 - (ev.pan * 0.5)
                    pan_r = 0.5 + (ev.pan * 0.5)

                    out_l[write_start : write_start + length] += mono_seg * pan_l
                    out_r[write_start : write_start + length] += mono_seg * pan_r

        return np.column_stack((out_l, out_r))

    def reset(self) -> None:
        pass
