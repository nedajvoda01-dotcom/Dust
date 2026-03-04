"""WeatherRegimeDetector — Stage 64 threshold-based weather regime detection.

Detects discrete weather regimes from continuous atmospheric field values.
Regimes are **not presets** — they emerge from threshold crossings in the
field space defined by AtmosphereSystem.

Regimes
-------
CLEAR          — baseline, no special conditions
DUST_STORM     — high windSpeed + high aerosol + strong pressure gradients
SNOW_DEPOSITION— cold enough + sufficient humidity + appropriate pressure
FOG            — high fogPotential (cold + humid + low-wind lowlands)
ELECTRICAL     — high electroActivity + dust storm conditions

Public API
----------
WeatherRegime (enum-like named constants)
WeatherRegimeDetector(config=None)
  .detect(tile, fog_potential, front_intensity, storm_potential) → WeatherRegime
"""
from __future__ import annotations

import math
from typing import Optional

from src.atmo.GlobalFieldGrid import AtmoTile


# ---------------------------------------------------------------------------
# Regime constants (string flags for forward-compatibility)
# ---------------------------------------------------------------------------

class WeatherRegime:
    CLEAR            = "CLEAR"
    DUST_STORM       = "DUST_STORM"
    SNOW_DEPOSITION  = "SNOW_DEPOSITION"
    FOG              = "FOG"
    ELECTRICAL       = "ELECTRICAL"


# ---------------------------------------------------------------------------
# WeatherRegimeDetector
# ---------------------------------------------------------------------------

class WeatherRegimeDetector:
    """Classify a single tile into a :class:`WeatherRegime`.

    Parameters
    ----------
    config :
        Optional dict; reads ``atmo64.storm_thresholds`` sub-dict.
    """

    _DEFAULT_STORM_WIND        = 0.55   # minimum wind_speed for dust storm
    _DEFAULT_STORM_AEROSOL     = 0.45   # minimum aerosol for dust storm
    _DEFAULT_STORM_FRONT       = 0.15   # minimum front_intensity for dust storm
    _DEFAULT_SNOW_TEMP         = 0.35   # maximum temperature for snow deposition
    _DEFAULT_SNOW_HUMIDITY     = 0.30   # minimum humidity for snow deposition
    _DEFAULT_FOG_THRESHOLD     = 0.35   # minimum fogPotential for fog regime
    _DEFAULT_ELEC_THRESHOLD    = 0.55   # minimum electro for electrical event

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg  = config or {}
        acfg = cfg.get("atmo64", {}) or {}
        thresholds = acfg.get("storm_thresholds", {}) or {}

        self._storm_wind:    float = float(thresholds.get("wind",    self._DEFAULT_STORM_WIND))
        self._storm_aerosol: float = float(thresholds.get("aerosol", self._DEFAULT_STORM_AEROSOL))
        self._storm_front:   float = float(thresholds.get("front",   self._DEFAULT_STORM_FRONT))
        self._snow_temp:     float = float(thresholds.get("snow_temp",     self._DEFAULT_SNOW_TEMP))
        self._snow_humidity: float = float(thresholds.get("snow_humidity", self._DEFAULT_SNOW_HUMIDITY))
        self._fog_threshold: float = float(thresholds.get("fog",     self._DEFAULT_FOG_THRESHOLD))
        self._elec_threshold: float = float(thresholds.get("electro", self._DEFAULT_ELEC_THRESHOLD))

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def detect(
        self,
        tile:           AtmoTile,
        fog_potential:  float,
        front_intensity: float,
        storm_potential: float,
    ) -> str:
        """Return the dominant weather regime for *tile*.

        Parameters
        ----------
        tile :
            The atmospheric field tile.
        fog_potential :
            Pre-computed fog potential [0..1].
        front_intensity :
            Pre-computed front intensity [0..1].
        storm_potential :
            Pre-computed storm potential [0..1].

        Returns
        -------
        str
            One of :class:`WeatherRegime` constants.
        """
        ws = tile.wind_speed

        # Electrical: requires storm + high electro
        if (tile.electro >= self._elec_threshold
                and ws >= self._storm_wind
                and tile.aerosol >= self._storm_aerosol):
            return WeatherRegime.ELECTRICAL

        # Dust storm: high wind + high aerosol + strong front
        if (ws >= self._storm_wind
                and tile.aerosol >= self._storm_aerosol
                and front_intensity >= self._storm_front):
            return WeatherRegime.DUST_STORM

        # Snow deposition: cold + sufficient humidity
        if (tile.temperature <= self._snow_temp
                and tile.humidity >= self._snow_humidity):
            return WeatherRegime.SNOW_DEPOSITION

        # Fog: cold + humid + sheltered (fog_potential above threshold)
        if fog_potential >= self._fog_threshold:
            return WeatherRegime.FOG

        return WeatherRegime.CLEAR
