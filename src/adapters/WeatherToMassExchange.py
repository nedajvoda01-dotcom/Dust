"""WeatherToMassExchange — Stage 66 adapter: weather (64) → mass exchange inputs.

Translates :class:`~src.atmo.AtmosphereSystem.LocalAtmoParams` from Stage 64
into the scalar parameters expected by the Stage 66 mass exchange models
(:class:`~src.mass.LiftModel.LiftModel` and
:class:`~src.mass.SettlingModel.SettlingModel`).

Mapping rules
-------------
wind_speed  ← local_params.wind_speed
temperature ← local_params.temperature
shelter     ← 1.0 − local_params.wind_speed  (proxy: low wind ≈ sheltered)

For snow-drift injection the SNOW_DEPOSITION regime is detected via the
regime field on LocalAtmoParams.

Public API
----------
WeatherToMassExchange(config=None)
  .wind_speed(local_params)  → float [0..1]
  .temperature(local_params) → float [0..1]
  .shelter(local_params)     → float [0..1]
  .air_snow_boost(local_params) → float [0..1]
"""
from __future__ import annotations

from typing import Optional

from src.atmo.AtmosphereSystem import LocalAtmoParams
from src.atmo.WeatherRegimeDetector import WeatherRegime


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


class WeatherToMassExchange:
    """Map Stage 64 atmospheric params to Stage 66 mass-exchange inputs.

    Parameters
    ----------
    config :
        Optional dict (currently unused; reserved for future coefficients).
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        pass   # no config keys at this time

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def wind_speed(self, local_params: LocalAtmoParams) -> float:
        """Return local wind speed [0..1] for lift/settling computation."""
        return _clamp(local_params.wind_speed)

    def temperature(self, local_params: LocalAtmoParams) -> float:
        """Return local temperature proxy [0..1]."""
        return _clamp(local_params.temperature)

    def shelter(self, local_params: LocalAtmoParams) -> float:
        """Return topographic shelter proxy [0..1].

        Approximated as 1 − wind_speed: calm conditions imply sheltered area.
        Downstream systems with explicit shelter maps can override this.
        """
        return _clamp(1.0 - local_params.wind_speed)

    def air_snow_boost(self, local_params: LocalAtmoParams) -> float:
        """Return extra snow-lofting factor during snow-deposition regime [0..1]."""
        if getattr(local_params, "regime", None) == WeatherRegime.SNOW_DEPOSITION:
            return _clamp(local_params.humidity)
        return 0.0
