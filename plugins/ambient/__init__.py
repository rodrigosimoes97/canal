"""
Registro e exportação dos Plugins Ambientais do Relaxation Audio Engine.
"""
from plugins.ambient.rain import RainAmbient
from plugins.ambient.birds import BirdsAmbient
from plugins.ambient.fireplace import FireplaceAmbient
from plugins.ambient.thunder import ThunderAmbient
from plugins.ambient.ocean import OceanAmbient
from plugins.ambient.wind import WindAmbient
from plugins.ambient.forest import ForestAmbient
from plugins.ambient.water import WaterAmbient

ALL_AMBIENTS = [
    RainAmbient,
    BirdsAmbient,
    FireplaceAmbient,
    ThunderAmbient,
    OceanAmbient,
    WindAmbient,
    ForestAmbient,
    WaterAmbient,
]

__all__ = [
    "RainAmbient",
    "BirdsAmbient",
    "FireplaceAmbient",
    "ThunderAmbient",
    "OceanAmbient",
    "WindAmbient",
    "ForestAmbient",
    "WaterAmbient",
    "ALL_AMBIENTS",
]
