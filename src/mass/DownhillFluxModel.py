"""DownhillFluxModel — Stage 66 granular downhill flow (slope creep).

Computes flux of loose material (dust, snow, debris) down a slope when the
angle of repose is exceeded.  Returns a signed delta for the source cell and
a corresponding positive delta for the downslope neighbour; callers apply
both via :class:`~src.material.MassExchangeAPI.MassExchangeAPI` to conserve
mass.

Downhill flow rules (spec §2.3)
--------------------------------
Only triggers when slope > slope_threshold_dust (or _snow).
flux = downhill_flux_k × (slope − threshold) × available_mass

The flux is clamped to ``max_flux_per_tick``.

Config keys (``mass.*``)
-------------------------
downhill_flux_k          : float (default 0.15)
slope_threshold_dust     : float (default 0.30)
slope_threshold_snow     : float (default 0.20)
max_flux_per_tick        : float (default 0.05)

Public API
----------
DownhillFluxModel(config=None)
  .compute_downhill_flux(surface, slope, stress=0.0) → DownhillFlux
      surface : PlanetChunkState
      slope   : float [0..1]
      stress  : float [0..1]  (from instability; boosts flux)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.material.PlanetChunkState import PlanetChunkState


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


@dataclass
class DownhillFlux:
    """Per-material downhill flux for one cell this tick.

    Positive values mean mass *leaves* the source cell.  Callers should
    add the negative of each value to the source cell and the positive
    value to the downslope neighbour.
    """
    dust_flux:   float = 0.0
    snow_flux:   float = 0.0
    debris_flux: float = 0.0


class DownhillFluxModel:
    """Compute gravity-driven granular flow rates.

    Parameters
    ----------
    config :
        Optional dict; reads ``mass.*`` keys (see module docstring).
    """

    _DEFAULT_DOWNHILL_FLUX_K      = 0.15
    _DEFAULT_SLOPE_THRESH_DUST    = 0.30
    _DEFAULT_SLOPE_THRESH_SNOW    = 0.20
    _DEFAULT_MAX_FLUX_PER_TICK    = 0.05

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = (config or {}).get("mass", {}) or {}
        self._k:             float = float(cfg.get("downhill_flux_k",      self._DEFAULT_DOWNHILL_FLUX_K))
        self._thresh_dust:   float = float(cfg.get("slope_threshold_dust", self._DEFAULT_SLOPE_THRESH_DUST))
        self._thresh_snow:   float = float(cfg.get("slope_threshold_snow", self._DEFAULT_SLOPE_THRESH_SNOW))
        self._max_flux:      float = float(cfg.get("max_flux_per_tick",    self._DEFAULT_MAX_FLUX_PER_TICK))

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def compute_downhill_flux(
        self,
        surface: PlanetChunkState,
        slope:   float,
        stress:  float = 0.0,
    ) -> DownhillFlux:
        """Return how much material flows downhill this tick.

        Parameters
        ----------
        surface :
            Current surface material state.
        slope :
            Local terrain slope [0..1].
        stress :
            External stress boost (e.g. from instability / vibration) [0..1].

        Returns
        -------
        DownhillFlux
            Positive values = mass leaving the source cell.
        """
        s = _clamp(slope)
        boost = 1.0 + _clamp(stress) * 0.5   # stress can increase flux by up to 50 %

        # Dust
        dust_flux = 0.0
        if s > self._thresh_dust:
            raw = self._k * (s - self._thresh_dust) * surface.dustThickness * boost
            dust_flux = _clamp(min(raw, self._max_flux, surface.dustThickness), 0.0, 1.0)

        # Snow
        snow_flux = 0.0
        if s > self._thresh_snow:
            snow_looseness = _clamp(1.0 - surface.snowCompaction)
            raw = self._k * (s - self._thresh_snow) * surface.snowMass * snow_looseness * boost
            snow_flux = _clamp(min(raw, self._max_flux, surface.snowMass), 0.0, 1.0)

        # Debris — same threshold as dust (heavier; lower k)
        debris_flux = 0.0
        if s > self._thresh_dust:
            raw = self._k * 0.5 * (s - self._thresh_dust) * surface.debrisMass * boost
            debris_flux = _clamp(min(raw, self._max_flux, surface.debrisMass), 0.0, 1.0)

        return DownhillFlux(
            dust_flux=dust_flux,
            snow_flux=snow_flux,
            debris_flux=debris_flux,
        )
