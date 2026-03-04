"""SettlingModel — Stage 66 air-to-surface mass settling.

Computes the rate at which airborne material (dust, snow drift) settles
back onto the surface from the volumetric domain.  Callers apply the
returned deltas via :class:`~src.material.MassExchangeAPI.MassExchangeAPI`.

Settling rules (spec §2.2)
---------------------------
dust_settle_rate  = settle_k_dust  × (1 − wind_speed) × (1 − slope) × air_dust_density
snow_settle_rate  = settle_k_snow  × (1 − wind_speed) × (1 − slope) × air_snow_density

Both rates are bounded by ``max_flux_per_tick``.

Config keys (``mass.*``)
-------------------------
settle_k_dust          : float (default 0.10)
settle_k_snow          : float (default 0.08)
max_flux_per_tick      : float (default 0.05)

Public API
----------
SettlingModel(config=None)
  .compute_settling_rate(air_dust_density, air_snow_density,
                         wind_speed, slope, shelter) → SettlingRates
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


@dataclass
class SettlingRates:
    """Amount of each material settling onto the surface this tick."""
    dust_settle: float = 0.0
    snow_settle: float = 0.0


class SettlingModel:
    """Compute air-to-surface settling rates for dust and snow drift.

    Parameters
    ----------
    config :
        Optional dict; reads ``mass.*`` keys (see module docstring).
    """

    _DEFAULT_SETTLE_K_DUST     = 0.10
    _DEFAULT_SETTLE_K_SNOW     = 0.08
    _DEFAULT_MAX_FLUX_PER_TICK = 0.05

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = (config or {}).get("mass", {}) or {}
        self._k_dust:   float = float(cfg.get("settle_k_dust",      self._DEFAULT_SETTLE_K_DUST))
        self._k_snow:   float = float(cfg.get("settle_k_snow",      self._DEFAULT_SETTLE_K_SNOW))
        self._max_flux: float = float(cfg.get("max_flux_per_tick",  self._DEFAULT_MAX_FLUX_PER_TICK))

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def compute_settling_rate(
        self,
        air_dust_density:  float,
        air_snow_density:  float,
        wind_speed:        float,
        slope:             float = 0.0,
        shelter:           float = 0.0,
    ) -> SettlingRates:
        """Return how much dust/snow settles this tick.

        Parameters
        ----------
        air_dust_density :
            Normalised dust density in the air above the cell [0..1].
        air_snow_density :
            Normalised snow-drift density in the air above the cell [0..1].
        wind_speed :
            Local wind speed [0..1]; high wind reduces settling.
        slope :
            Terrain slope [0..1]; steeper slope retains less deposit.
        shelter :
            Topographic shelter factor [0..1]; more shelter → more settling.

        Returns
        -------
        SettlingRates
            Clamped to ``max_flux_per_tick`` and available air density.
        """
        calm        = _clamp(1.0 - wind_speed)
        flat        = _clamp(1.0 - slope)
        shelter_mod = _clamp(1.0 + shelter * 0.5)   # shelter boosts settling up to 50 %
        air_dust    = _clamp(air_dust_density)
        air_snow    = _clamp(air_snow_density)

        dust_rate = self._k_dust * calm * flat * shelter_mod * air_dust
        snow_rate = self._k_snow * calm * flat * shelter_mod * air_snow

        dust_settle = _clamp(min(dust_rate, self._max_flux, air_dust), 0.0, 1.0)
        snow_settle = _clamp(min(snow_rate, self._max_flux, air_snow), 0.0, 1.0)

        return SettlingRates(dust_settle=dust_settle, snow_settle=snow_settle)
