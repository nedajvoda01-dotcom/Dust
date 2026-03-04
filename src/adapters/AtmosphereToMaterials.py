"""AtmosphereToMaterials — Stage 64 adapter: atmosphere → material mass flows.

Translates atmospheric regime and field values into mass/heat delta calls on
:class:`~src.material.MassExchangeAPI.MassExchangeAPI`.

The atmosphere **never** writes material fields directly — it only calls the
MassExchangeAPI, preserving the mass-conservation invariant of Stage 63.

Flows applied per tick
----------------------
DUST_STORM :
    AerosolDust → RegolithDust deposition (TransferMass)
    Erosion: RegolithDust → AerosolDust lift (TransferMass)
SNOW_DEPOSITION :
    Apply +snowMass delta
    Apply +/-vapor delta (condensation proxy)
FOG :
    Apply +vapor delta (fog proxy adds atmospheric moisture)
ELECTRICAL :
    No mass flow; heat delta only (lightning proxy).
All regimes :
    Apply heat delta from thermal_effect.

Public API
----------
AtmosphereToMaterials(config=None)
  .apply(chunk_api, local_params, dt) -> None
      Apply mass/heat deltas to *chunk_api* (MassExchangeAPI).
"""
from __future__ import annotations

from typing import Optional

from src.atmo.AtmosphereSystem  import LocalAtmoParams
from src.atmo.WeatherRegimeDetector import WeatherRegime
from src.material.MassExchangeAPI   import MassExchangeAPI


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


class AtmosphereToMaterials:
    """Apply atmospheric effects to a terrain chunk via MassExchangeAPI.

    Parameters
    ----------
    config :
        Optional dict; reads ``atmo64.*`` rate keys.
    """

    _DEFAULT_DEPOSITION_RATE = 0.002   # per second
    _DEFAULT_EROSION_RATE    = 0.001   # per second
    _DEFAULT_SNOW_RATE       = 0.003   # per second
    _DEFAULT_FOG_VAPOR_RATE  = 0.001   # per second
    _DEFAULT_HEAT_SCALE      = 0.005   # per second per unit thermal_effect

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg  = config or {}
        acfg = cfg.get("atmo64", {}) or {}

        self._dep_rate   = float(acfg.get("deposition_rate", self._DEFAULT_DEPOSITION_RATE))
        self._eros_rate  = float(acfg.get("erosion_rate",    self._DEFAULT_EROSION_RATE))
        self._snow_rate  = float(acfg.get("snow_rate",       self._DEFAULT_SNOW_RATE))
        self._fog_rate   = float(acfg.get("fog_vapor_rate",  self._DEFAULT_FOG_VAPOR_RATE))
        self._heat_scale = float(acfg.get("heat_scale",      self._DEFAULT_HEAT_SCALE))

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def apply(
        self,
        chunk_api:    MassExchangeAPI,
        local_params: LocalAtmoParams,
        dt:           float,
    ) -> None:
        """Apply atmospheric mass/heat deltas to *chunk_api*.

        Parameters
        ----------
        chunk_api :
            MassExchangeAPI for the terrain chunk to modify.
        local_params :
            Current LocalAtmoParams at the chunk location.
        dt :
            Time step in seconds.
        """
        regime = local_params.regime

        if regime == WeatherRegime.DUST_STORM:
            # Deposition: atmospheric aerosol settles onto surface dust layer
            dep_rate = self._dep_rate * local_params.aerosol * dt
            chunk_api.apply_mass_delta("dustThickness", dep_rate)
            # Erosion: wind lifts surface dust (debrisMass → atmosphere)
            eros_rate = self._eros_rate * local_params.wind_speed * dt
            chunk_api.apply_mass_delta("dustThickness", -eros_rate)

        elif regime == WeatherRegime.SNOW_DEPOSITION:
            snow_rate = self._snow_rate * local_params.humidity * dt
            chunk_api.apply_mass_delta("snowMass", snow_rate)
            # Some vapor condensed out
            chunk_api.apply_mass_delta("moistureProxy", -snow_rate * 0.5)

        elif regime == WeatherRegime.FOG:
            chunk_api.apply_mass_delta("moistureProxy", self._fog_rate * dt)

        # Heat delta from thermal effect (applies in all regimes)
        heat = local_params.thermal_effect * self._heat_scale * dt
        chunk_api.apply_heat_delta(heat)
