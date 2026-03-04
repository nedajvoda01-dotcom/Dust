"""VolumetricsToAudio — Stage 65 adapter: volumetrics → audio parameters.

Translates volumetric domain density state into audio modulation signals
consumed by the procedural audio system (Stages 46/36).

Effects modelled (spec §8)
--------------------------
fog_dampening :
    High-frequency attenuation proportional to mean fog/steam density.
dust_rustling :
    Low-frequency granular noise proportional to mean dust density.
wind_muffling :
    Overall wind noise reduction in dense fog (visibility proxy).

Public API
----------
VolumetricsToAudio(config=None)
  .audio_params(grid) → VolumetricAudioParams

VolumetricAudioParams (dataclass)
  .fog_dampening    float [0..1]
  .dust_rustling    float [0..1]
  .wind_muffling    float [0..1]
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.vol.DensityGrid import DensityGrid, VolumeLayerType


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


@dataclass
class VolumetricAudioParams:
    fog_dampening:  float = 0.0
    dust_rustling:  float = 0.0
    wind_muffling:  float = 0.0


class VolumetricsToAudio:
    """Map volumetric density grids to audio modulation parameters.

    Parameters
    ----------
    config :
        Optional dict; reads ``vol.audio.*`` keys.
    """

    _DEFAULT_FOG_DAMP_SCALE   = 0.9
    _DEFAULT_DUST_RUSTLE_SCALE = 0.7
    _DEFAULT_WIND_MUFFLE_SCALE = 0.6

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg  = config or {}
        vcfg = cfg.get("vol", {}).get("audio", {}) or {}
        self._fog_scale:   float = float(
            vcfg.get("fog_damp_scale",   self._DEFAULT_FOG_DAMP_SCALE)
        )
        self._dust_scale:  float = float(
            vcfg.get("dust_rustle_scale", self._DEFAULT_DUST_RUSTLE_SCALE)
        )
        self._muffle_scale: float = float(
            vcfg.get("wind_muffle_scale", self._DEFAULT_WIND_MUFFLE_SCALE)
        )

    def audio_params(self, grid: DensityGrid) -> VolumetricAudioParams:
        """Compute audio parameters from *grid* density state."""
        n = grid.width * grid.height * grid.depth
        mean_density = grid.total_density() / max(n, 1)

        ltype = grid.layer_type

        if ltype in (VolumeLayerType.FOG, VolumeLayerType.STEAM):
            fog_damp   = _clamp(mean_density * self._fog_scale)
            wind_muffle = _clamp(mean_density * self._muffle_scale)
            dust_rustle = 0.0
        elif ltype in (VolumeLayerType.DUST, VolumeLayerType.SNOW_DRIFT):
            fog_damp    = 0.0
            wind_muffle = _clamp(mean_density * self._muffle_scale * 0.5)
            dust_rustle = _clamp(mean_density * self._dust_scale)
        else:
            fog_damp = dust_rustle = wind_muffle = 0.0

        return VolumetricAudioParams(
            fog_dampening  = fog_damp,
            dust_rustling  = dust_rustle,
            wind_muffling  = wind_muffle,
        )
