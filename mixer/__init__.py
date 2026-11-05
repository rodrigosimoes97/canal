"""
Módulo de Mixagem e Roteamento do Relaxation Audio Engine.
"""
from mixer.channel import MixerChannel, AutomationCurve, AutomationPoint
from mixer.mixer import Mixer, DuckingRule

__all__ = [
    "MixerChannel",
    "AutomationCurve",
    "AutomationPoint",
    "Mixer",
    "DuckingRule",
]
