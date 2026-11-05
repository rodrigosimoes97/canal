"""
Perfis de Masterização com Alvos Específicos de Loudness e True Peak.
"""
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class MasteringProfile:
    """Configuração dos parâmetros de masterização para um determinado estilo sonoro."""
    name: str
    target_lufs: float
    target_true_peak_dbtp: float
    max_gain_boost_db: float = 6.0
    fade_in_sec: float = 2.0
    fade_out_sec: float = 4.0
    lookahead_ms: float = 4.0
    release_ms: float = 80.0


MASTERING_PROFILES: Dict[str, MasteringProfile] = {
    "nature_ambient": MasteringProfile(
        name="nature_ambient",
        target_lufs=-20.0,
        target_true_peak_dbtp=-1.5,
        max_gain_boost_db=6.0,
        fade_in_sec=2.0,
        fade_out_sec=4.0
    ),
    "meditation": MasteringProfile(
        name="meditation",
        target_lufs=-18.0,
        target_true_peak_dbtp=-1.0,
        max_gain_boost_db=6.0,
        fade_in_sec=2.5,
        fade_out_sec=4.5
    ),
    "deep_relaxation": MasteringProfile(
        name="deep_relaxation",
        target_lufs=-21.0,
        target_true_peak_dbtp=-1.5,
        max_gain_boost_db=5.0,
        fade_in_sec=3.0,
        fade_out_sec=5.0
    ),
    "deep_sleep": MasteringProfile(
        name="deep_sleep",
        target_lufs=-23.0,
        target_true_peak_dbtp=-2.0,
        max_gain_boost_db=4.0,
        fade_in_sec=4.0,
        fade_out_sec=6.0
    ),
    "asmr": MasteringProfile(
        name="asmr",
        target_lufs=-19.0,
        target_true_peak_dbtp=-1.5,
        max_gain_boost_db=5.0,
        fade_in_sec=1.5,
        fade_out_sec=3.5
    ),
    "fantasy_tavern": MasteringProfile(
        name="fantasy_tavern",
        target_lufs=-16.0,
        target_true_peak_dbtp=-1.0,
        max_gain_boost_db=6.0,
        fade_in_sec=2.0,
        fade_out_sec=4.0
    ),
    "medieval_ambient": MasteringProfile(
        name="medieval_ambient",
        target_lufs=-17.0,
        target_true_peak_dbtp=-1.0,
        max_gain_boost_db=6.0,
        fade_in_sec=2.0,
        fade_out_sec=4.0
    ),
    "cinematic_ambient": MasteringProfile(
        name="cinematic_ambient",
        target_lufs=-16.0,
        target_true_peak_dbtp=-1.0,
        max_gain_boost_db=6.0,
        fade_in_sec=2.0,
        fade_out_sec=5.0
    ),
}


def get_mastering_profile(profile_name: str, **overrides) -> MasteringProfile:
    """
    Retorna o perfil de masterização solicitado com suporte a overrides pontuais.
    Fallback para 'deep_relaxation' caso não seja encontrado.
    """
    base = MASTERING_PROFILES.get(profile_name.lower(), MASTERING_PROFILES["deep_relaxation"])
    if not overrides:
        return base

    return MasteringProfile(
        name=overrides.get("name", base.name),
        target_lufs=float(overrides.get("target_lufs", base.target_lufs)),
        target_true_peak_dbtp=float(overrides.get("target_true_peak_dbtp", base.target_true_peak_dbtp)),
        max_gain_boost_db=float(overrides.get("max_gain_boost_db", base.max_gain_boost_db)),
        fade_in_sec=float(overrides.get("fade_in_sec", base.fade_in_sec)),
        fade_out_sec=float(overrides.get("fade_out_sec", base.fade_out_sec)),
        lookahead_ms=float(overrides.get("lookahead_ms", base.lookahead_ms)),
        release_ms=float(overrides.get("release_ms", base.release_ms)),
    )
