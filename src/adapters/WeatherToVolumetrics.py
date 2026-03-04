"""WeatherToVolumetrics — Stage 65 adapter: weather (64) → volumetric sources.

Translates :class:`~src.atmo.AtmosphereSystem.LocalAtmoParams` from Stage 64
into density source rates that are injected into volumetric
:class:`~src.vol.DensityGrid.DensityGrid` objects managed by
:class:`~src.vol.VolumetricDomainManager.VolumetricDomainManager`.

Source rules (spec §3)
-----------------------
DustVolume :
    source rate = erosionLift = f(wind_speed, aerosol, roughness)
    liftRate = wind_speed × aerosol × lift_k

FogVolume :
    source rate = condenseRate from cold + humid + low-pressure conditions
    condenseRate = humidity × (1 − temperature) × p_condition × fog_lift_k

SteamVolume :
    source rate from local heat anomaly proxy (thermal_effect > threshold)

Only the lowest voxel layer (iz=0) of the grid receives the injection, so
that density rises naturally from the ground.

Public API
----------
WeatherToVolumetrics(config=None)
  .inject(grid, local_params, dt) → None
"""
from __future__ import annotations

from typing import Optional

from src.atmo.AtmosphereSystem     import LocalAtmoParams
from src.atmo.WeatherRegimeDetector import WeatherRegime
from src.vol.DensityGrid            import DensityGrid, VolumeLayerType


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


class WeatherToVolumetrics:
    """Inject atmospheric source terms into volumetric density grids.

    Parameters
    ----------
    config :
        Optional dict; reads ``vol.lift_k``, ``vol.fog_lift_k``,
        ``vol.steam_threshold``, ``vol.steam_lift_k``.
    """

    _DEFAULT_LIFT_K        = 0.08   # dust lift per second
    _DEFAULT_FOG_LIFT_K    = 0.06   # fog condensation per second
    _DEFAULT_STEAM_THRESH  = 0.7    # thermal_effect threshold for steam
    _DEFAULT_STEAM_LIFT_K  = 0.10   # steam rise per second

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg  = config or {}
        vcfg = cfg.get("vol", {}) or {}

        self._lift_k:       float = float(vcfg.get("lift_k",         self._DEFAULT_LIFT_K))
        self._fog_lift_k:   float = float(vcfg.get("fog_lift_k",     self._DEFAULT_FOG_LIFT_K))
        self._steam_thresh: float = float(vcfg.get("steam_threshold", self._DEFAULT_STEAM_THRESH))
        self._steam_lift_k: float = float(vcfg.get("steam_lift_k",   self._DEFAULT_STEAM_LIFT_K))

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def inject(
        self,
        grid:         DensityGrid,
        local_params: LocalAtmoParams,
        dt:           float,
    ) -> None:
        """Inject atmospheric source density into the ground layer of *grid*.

        Parameters
        ----------
        grid :
            The :class:`~src.vol.DensityGrid.DensityGrid` to inject into.
        local_params :
            Current local atmospheric parameters (from Stage 64).
        dt :
            Time step in seconds.
        """
        ltype = grid.layer_type

        if ltype == VolumeLayerType.DUST:
            self._inject_dust(grid, local_params, dt)

        elif ltype == VolumeLayerType.FOG:
            self._inject_fog(grid, local_params, dt)

        elif ltype == VolumeLayerType.STEAM:
            self._inject_steam(grid, local_params, dt)

        elif ltype == VolumeLayerType.SNOW_DRIFT:
            self._inject_snow(grid, local_params, dt)

    # ------------------------------------------------------------------
    # Private per-layer injectors
    # ------------------------------------------------------------------

    def _inject_dust(
        self,
        grid:   DensityGrid,
        params: LocalAtmoParams,
        dt:     float,
    ) -> None:
        """Lift dust from surface into the volumetric domain."""
        rate = self._lift_k * params.wind_speed * params.aerosol * dt
        if rate <= 0.0:
            return
        for iy in range(grid.height):
            for ix in range(grid.width):
                grid.add_density(ix, iy, 0, rate)

    def _inject_fog(
        self,
        grid:   DensityGrid,
        params: LocalAtmoParams,
        dt:     float,
    ) -> None:
        """Condense fog from humid cold air near the ground."""
        cold_bias    = _clamp(1.0 - params.temperature)
        p_condition  = _clamp(1.0 - abs(params.pressure - 0.4))
        rate = self._fog_lift_k * params.humidity * cold_bias * p_condition * dt
        if rate <= 0.0:
            return
        for iy in range(grid.height):
            for ix in range(grid.width):
                grid.add_density(ix, iy, 0, rate)

    def _inject_steam(
        self,
        grid:   DensityGrid,
        params: LocalAtmoParams,
        dt:     float,
    ) -> None:
        """Emit steam from localised heat anomalies (lava/water)."""
        if params.thermal_effect < self._steam_thresh:
            return
        rate = self._steam_lift_k * (params.thermal_effect - self._steam_thresh) * dt
        for iy in range(grid.height):
            for ix in range(grid.width):
                grid.add_density(ix, iy, 0, rate)

    def _inject_snow(
        self,
        grid:   DensityGrid,
        params: LocalAtmoParams,
        dt:     float,
    ) -> None:
        """Loft snow particles in blizzard conditions."""
        if params.regime != WeatherRegime.SNOW_DEPOSITION:
            return
        rate = self._lift_k * params.wind_speed * params.humidity * dt
        for iy in range(grid.height):
            for ix in range(grid.width):
                grid.add_density(ix, iy, 0, rate)
