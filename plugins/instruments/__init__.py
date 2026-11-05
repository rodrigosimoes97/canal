"""
Registro e exportação dos Instrumentos Sintetizados do Relaxation Audio Engine.
"""
from plugins.instruments.ambient_pad import AmbientPad
from plugins.instruments.piano import Piano
from plugins.instruments.harp import Harp
from plugins.instruments.lute import Lute
from plugins.instruments.singing_bowl import SingingBowl
from plugins.instruments.bells import Bells
from plugins.instruments.bass import Bass
from plugins.instruments.strings import Strings
from plugins.instruments.flute import Flute

ALL_INSTRUMENTS = [
    AmbientPad,
    Piano,
    Harp,
    Lute,
    SingingBowl,
    Bells,
    Bass,
    Strings,
    Flute,
]

__all__ = [
    "AmbientPad",
    "Piano",
    "Harp",
    "Lute",
    "SingingBowl",
    "Bells",
    "Bass",
    "Strings",
    "Flute",
    "ALL_INSTRUMENTS",
]
