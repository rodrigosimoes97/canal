"""
Plugin Ambiental: Trovões Distantes (Eventos Dramáticos de Baixa Frequência).
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np

from plugins.base import BasePlugin, PluginCategory
from assets.asset_manager import AudioAssetManager


@dataclass
class ThunderEvent:
    start_sample: int
    duration_samples: int
    gain: float = 0.6
    pan: float = 0.0


class ThunderAmbient(BasePlugin):
    """
    Simulador de trovões distantes e atmosféricos utilizando 'trovao.mp3' e síntese grave.
    """
    PLUGIN_ID = "thunder_ambient"

    def __init__(self, plugin_id: str = "thunder_ambient", config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(plugin_id=plugin_id, category=PluginCategory.AMBIENT, config=config)
        self.gain = float(self.config.get("gain", 0.5))
        self.sample_rate: int = 44100
        self.asset_mgr: Optional[AudioAssetManager] = None
        self.events: List[ThunderEvent] = []

    def prepare(self, context: Any, rng: np.random.Generator) -> None:
        self.sample_rate = context.sample_rate
        self.asset_mgr = context.metadata.get("asset_manager")
        self.events.clear()

        # Trovões ocorrem esparsamente a cada 45 a 85 segundos
        cur_sec = float(rng.uniform(15.0, 30.0))
        total_sec = context.duration_sec

        while cur_sec < total_sec - 10.0:
            dur_s = float(rng.uniform(8.0, 14.0))
            dur_samples = int(round(dur_s * self.sample_rate))
            s_sample = int(round(cur_sec * self.sample_rate))
            pan = float(rng.uniform(-0.4, 0.4))
            t_gain = float(rng.uniform(0.6, 0.95))

            self.events.append(ThunderEvent(
                start_sample=s_sample,
                duration_samples=dur_samples,
                gain=t_gain,
                pan=pan
            ))
            cur_sec += float(rng.uniform(45.0, 85.0))

        self._is_prepared = True

    def render(self, start_sample: int, num_samples: int) -> np.ndarray:
        if not self.is_active or num_samples <= 0:
            return np.zeros((num_samples, 2), dtype=np.float32)

        chunk_end = start_sample + num_samples
        out_l = np.zeros(num_samples, dtype=np.float32)
        out_r = np.zeros(num_samples, dtype=np.float32)

        # Se houver asset real 'trovao'
        has_real_asset = self.asset_mgr is not None and self.asset_mgr.has_asset("trovao")
        if has_real_asset:
            trovao_data = self.asset_mgr.load_asset("trovao")
            trovao_len = trovao_data.shape[0]

        for ev in self.events:
            ev_start = ev.start_sample
            ev_end = ev_start + ev.duration_samples

            if ev_end > start_sample and ev_start < chunk_end:
                read_start = max(0, start_sample - ev_start)
                write_start = max(0, ev_start - start_sample)
                length = min(ev.duration_samples - read_start, num_samples - write_start)

                if length > 0:
                    if has_real_asset:
                        # Extrai trecho do áudio real com offset circular
                        idx = (read_start + np.arange(length)) % trovao_len
                        seg = trovao_data[idx]
                        mono_seg = np.mean(seg, axis=1) * (ev.gain * self.gain)
                    else:
                        # Síntese de rumble sub-grave se o asset não estiver presente
                        t = np.arange(length) / self.sample_rate
                        decay = np.exp(-0.35 * (t + (read_start / self.sample_rate)))
                        mono_seg = np.sin(2 * np.pi * 45.0 * t) * decay * (ev.gain * self.gain * 0.4)

                    pan_l = 0.5 - (ev.pan * 0.5)
                    pan_r = 0.5 + (ev.pan * 0.5)

                    out_l[write_start : write_start + length] += mono_seg * pan_l
                    out_r[write_start : write_start + length] += mono_seg * pan_r

        return np.column_stack((out_l, out_r))

    def reset(self) -> None:
        pass
