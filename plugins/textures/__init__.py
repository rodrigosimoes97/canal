"""
Registro e exportação dos Plugins de Textura do Relaxation Audio Engine.
"""
from plugins.textures.noise import NoiseTexture
from plugins.textures.binaural import BinauralBeats
from plugins.textures.drone import DroneTexture
from plugins.textures.harmonic_drone import HarmonicDrone

ALL_TEXTURES = [
    NoiseTexture,
    BinauralBeats,
    DroneTexture,
    HarmonicDrone,
]

__all__ = [
    "NoiseTexture",
    "BinauralBeats",
    "DroneTexture",
    "HarmonicDrone",
    "ALL_TEXTURES",
]
