"""LiftModel — Stage 66 surface-to-air mass lift by wind.

Computes the rate at which loose surface material (dust, snow) is lifted
into the air by wind.  The result is returned as a delta that callers apply
via :class:`~src.material.MassExchangeAPI.MassExchangeAPI`; the model does
**not** write to any state directly.

Lift rules (spec §2.1)
-----------------------
dust_lift_rate = lift_k_dust × wind_speed × (1 − crustHardness) × (1 − surfaceRoughness_clamp)
snow_lift_rate = lift_k_snow × wind_speed × (1 − snowCompaction)

Both rates are bounded by ``max_flux_per_tick`` and the available
surface mass (cannot lift more than exists).

Config keys (``mass.*``)
-------------------------
lift_k_dust            : float (default 0.12)
lift_k_snow            : float (default 0.10)
max_flux_per_tick      : float (default 0.05)

Public API
----------
LiftModel(config=None)
  .compute_lift_rate(surface, wind_speed, temperature) → LiftRates
      surface      : PlanetChunkState
      wind_speed   : float [0..1]
      temperature  : float [0..1]
      returns LiftRates (dust_lift, snow_lift)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.material.PlanetChunkState import PlanetChunkState


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


@dataclass
class LiftRates:
    """Amount of each material to lift from surface into air this tick."""
    dust_lift: float = 0.0
    snow_lift: float = 0.0


class LiftModel:
    """Compute wind-driven lift rates from a surface chunk state.

    Parameters
    ----------
    config :
        Optional dict; reads ``mass.*`` keys (see module docstring).
    """

    _DEFAULT_LIFT_K_DUST       = 0.12
    _DEFAULT_LIFT_K_SNOW       = 0.10
    _DEFAULT_MAX_FLUX_PER_TICK = 0.05

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = (config or {}).get("mass", {}) or {}
        self._k_dust:    float = float(cfg.get("lift_k_dust",       self._DEFAULT_LIFT_K_DUST))
        self._k_snow:    float = float(cfg.get("lift_k_snow",       self._DEFAULT_LIFT_K_SNOW))
        self._max_flux:  float = float(cfg.get("max_flux_per_tick", self._DEFAULT_MAX_FLUX_PER_TICK))

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def compute_lift_rate(
        self,
        surface:    PlanetChunkState,
        wind_speed: float,
        temperature: float = 0.5,
    ) -> LiftRates:
        """Return how much dust/snow to lift this tick.

        Parameters
        ----------
        surface :
            Current surface material state.
        wind_speed :
            Local wind speed normalised to [0..1].
        temperature :
            Local temperature proxy [0..1]; warmer makes dust lift easier.

        Returns
        -------
        LiftRates
            Clamped to available surface mass and ``max_flux_per_tick``.
        """
        ws = _clamp(wind_speed)

        # Dust: harder crust resists lift; rough surface slightly protects
        dust_looseness = _clamp(1.0 - surface.crustHardness)
        dust_rate = self._k_dust * ws * dust_looseness
        dust_lift = _clamp(
            min(dust_rate, self._max_flux, surface.dustThickness),
            0.0, 1.0,
        )

        # Snow: compact snow resists lofting
        snow_looseness = _clamp(1.0 - surface.snowCompaction)
        snow_rate = self._k_snow * ws * snow_looseness
        snow_lift = _clamp(
            min(snow_rate, self._max_flux, surface.snowMass),
            0.0, 1.0,
        )

        return LiftRates(dust_lift=dust_lift, snow_lift=snow_lift)
