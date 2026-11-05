"""
Registro e exportação dos Efeitos DSP do Relaxation Audio Engine.
"""
from plugins.effects.filter import FilterEffect
from plugins.effects.delay import DelayEffect
from plugins.effects.reverb import ReverbEffect
from plugins.effects.convolution import ConvolutionReverb
from plugins.effects.saturation import SaturationEffect
from plugins.effects.spatializer import SpatializerEffect

ALL_EFFECTS = [
    FilterEffect,
    DelayEffect,
    ReverbEffect,
    ConvolutionReverb,
    SaturationEffect,
    SpatializerEffect,
]

__all__ = [
    "FilterEffect",
    "DelayEffect",
    "ReverbEffect",
    "ConvolutionReverb",
    "SaturationEffect",
    "SpatializerEffect",
    "ALL_EFFECTS",
]
