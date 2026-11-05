"""
Evolução Temporal, Dinâmica do Arranjo e Densidade Musical.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional
import numpy as np


class ArrangementStage(str, Enum):
    """Estágios macro de uma composição musical de relaxamento."""
    INTRO = "intro"
    DEVELOPMENT = "development"
    DEEP_FLOW = "deep_flow"
    WIND_DOWN = "wind_down"
    OUTRO = "outro"


@dataclass
class ArrangementSection:
    """Seção estrutural da timeline com início, fim e nível de densidade."""
    stage: ArrangementStage
    start_sec: float
    end_sec: float
    start_density: float
    end_density: float
    active_roles: List[str] = field(default_factory=list)


class ArrangementEngine:
    """
    Controla a evolução dinâmica do arranjo e a entrada/saída de camadas sonoras.
    """

    def __init__(self, total_duration_sec: float) -> None:
        self.total_duration_sec = float(total_duration_sec)
        self.sections: List[ArrangementSection] = self._create_default_sections()

    def _create_default_sections(self) -> List[ArrangementSection]:
        """Cria uma curva de arranjo adaptada para música generativa de relaxamento."""
        tot = self.total_duration_sec
        # Intro: 10% do tempo (densidade sobe de 0.2 a 0.6)
        t_intro = tot * 0.10
        # Development: 30% do tempo (densidade sobe de 0.6 a 0.85)
        t_dev = tot * 0.40
        # Deep Flow / Sustentação: 35% do tempo (densidade constante em 0.75)
        t_flow = tot * 0.75
        # Wind Down / Desaceleração: 15% do tempo (densidade cai para 0.4)
        t_wind = tot * 0.90

        return [
            ArrangementSection(
                stage=ArrangementStage.INTRO,
                start_sec=0.0,
                end_sec=t_intro,
                start_density=0.2,
                end_density=0.6,
                active_roles=["pad", "drone"]
            ),
            ArrangementSection(
                stage=ArrangementStage.DEVELOPMENT,
                start_sec=t_intro,
                end_sec=t_dev,
                start_density=0.6,
                end_density=0.85,
                active_roles=["pad", "drone", "bass", "melody"]
            ),
            ArrangementSection(
                stage=ArrangementStage.DEEP_FLOW,
                start_sec=t_dev,
                end_sec=t_flow,
                start_density=0.85,
                end_density=0.75,
                active_roles=["pad", "drone", "bass", "melody", "arpeggio", "chimes"]
            ),
            ArrangementSection(
                stage=ArrangementStage.WIND_DOWN,
                start_sec=t_flow,
                end_sec=t_wind,
                start_density=0.75,
                end_density=0.4,
                active_roles=["pad", "drone", "melody"]
            ),
            ArrangementSection(
                stage=ArrangementStage.OUTRO,
                start_sec=t_wind,
                end_sec=tot,
                start_density=0.4,
                end_density=0.1,
                active_roles=["pad", "drone"]
            ),
        ]

    def get_density_at_time(self, time_sec: float) -> float:
        """Calcula o valor contínuo da densidade de notas (0.0 a 1.0) no tempo especificado."""
        t = np.clip(time_sec, 0.0, self.total_duration_sec)
        for sec in self.sections:
            if sec.start_sec <= t <= sec.end_sec:
                span = sec.end_sec - sec.start_sec
                if span <= 1e-6:
                    return sec.start_density
                ratio = (t - sec.start_sec) / span
                return float(sec.start_density + ratio * (sec.end_density - sec.start_density))
        return 0.5

    def is_role_active(self, role: str, time_sec: float) -> bool:
        """Determina se um papel instrumental deve tocar no instante especificado."""
        t = np.clip(time_sec, 0.0, self.total_duration_sec)
        for sec in self.sections:
            if sec.start_sec <= t <= sec.end_sec:
                return role.lower() in [r.lower() for r in sec.active_roles]
        return True
