"""LocalClimateComposer — Stage 49 MacroClimate + Microclimate → LocalClimate.

Combines the macro-climate snapshot (wind, dust, temperature) with the
per-chunk :class:`~src.microclimate.MicroclimateState.MicroclimateState`
offsets to produce a :class:`LocalClimate` that all downstream systems
(Perception 37, Materials 45, Fatigue 44, Audio 46, Injury 48) consume.

Formulas (from design spec §5)
-------------------------------
::

    local_wind_speed =
        macro_wind_speed * (1 - windShelter) + macro_wind_speed * windChannel

    local_dust_density =
        macro_dust_density
        + dustTrap * deposition_boost
        - windChannel * dispersion_boost

    local_temp_proxy = smoothed(
        macro_temp_proxy - coldBias * cold_delta,
        thermalInertia
    )

Public API
----------
LocalClimate (dataclass)
LocalClimateComposer(config=None)
  .compose(macro, micro_state, dt) → LocalClimate
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from src.math.Vec3 import Vec3
from src.microclimate.MicroclimateState import MicroclimateState


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


# ---------------------------------------------------------------------------
# MacroClimateSnapshot — lightweight input struct
# ---------------------------------------------------------------------------

@dataclass
class MacroClimateSnapshot:
    """Macro-climate values at the player's location.

    All speeds are normalised [0..1].
    """
    wind_speed:   float = 0.0   # macro wind magnitude, normalised
    wind_dir:     Vec3  = None  # macro wind direction (unit Vec3)
    dust_density: float = 0.0   # airborne dust [0..1]
    temp_proxy:   float = 0.5   # temperature proxy [0=cold, 1=hot]

    def __post_init__(self):
        if self.wind_dir is None:
            self.wind_dir = Vec3(1.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# LocalClimate — output struct
# ---------------------------------------------------------------------------

@dataclass
class LocalClimate:
    """Locally adjusted climate values for downstream systems.

    Attributes
    ----------
    wind_speed : float
        Local wind speed [0..1] after shelter / channeling.
    wind_dir : Vec3
        Wind direction (unchanged from macro).
    dust_density : float
        Local dust concentration [0..1].
    temp_proxy : float
        Smoothed local temperature proxy [0..1].
    shelter : float
        Raw windShelter value from MicroclimateState [0..1].
    wind_channel : float
        Raw windChannel value [0..1].
    cold_bias : float
        Raw coldBias value [0..1].
    thermal_inertia : float
        Raw thermalInertia value [0..1].
    echo_potential : float
        Raw echoPotential value [0..1].
    """
    wind_speed:      float = 0.0
    wind_dir:        Vec3  = None
    dust_density:    float = 0.0
    temp_proxy:      float = 0.5
    shelter:         float = 0.0
    wind_channel:    float = 0.0
    cold_bias:       float = 0.0
    thermal_inertia: float = 0.0
    echo_potential:  float = 0.0

    def __post_init__(self):
        if self.wind_dir is None:
            self.wind_dir = Vec3(1.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# LocalClimateComposer
# ---------------------------------------------------------------------------

class LocalClimateComposer:
    """Combines macro + microclimate into LocalClimate.

    Parameters
    ----------
    config :
        Optional dict; reads ``micro.*`` keys.
    """

    _DEFAULT_DEPOSITION_BOOST  = 0.4
    _DEFAULT_DISPERSION_BOOST  = 0.3
    _DEFAULT_COLD_DELTA        = 0.3
    _DEFAULT_INERTIA_TAU       = 60.0   # seconds

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg  = config or {}
        mcfg = cfg.get("micro", {}) or {}

        dcfg = mcfg.get("dusttrap", {}) or {}
        bcfg = mcfg.get("coldbias", {}) or {}

        self._deposition_boost: float = float(
            dcfg.get("deposition_boost", self._DEFAULT_DEPOSITION_BOOST)
        )
        self._dispersion_boost: float = float(
            dcfg.get("dispersion_boost", self._DEFAULT_DISPERSION_BOOST)
        )
        self._cold_delta: float = float(
            bcfg.get("cold_delta", self._DEFAULT_COLD_DELTA)
        )
        self._inertia_tau: float = float(
            mcfg.get("thermal_inertia_tau", self._DEFAULT_INERTIA_TAU)
        )

        # Smoothed temperature state
        self._local_temp: Optional[float] = None

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def compose(
        self,
        macro:      MacroClimateSnapshot,
        micro:      MicroclimateState,
        dt:         float = 1.0,
    ) -> LocalClimate:
        """Produce a :class:`LocalClimate` from macro + micro inputs.

        Parameters
        ----------
        macro :
            Macro-climate snapshot at the player's position.
        micro :
            Per-chunk microclimate state.
        dt :
            Elapsed time [s] since last call (for temperature smoothing).

        Returns
        -------
        LocalClimate
        """
        # --- Wind ---
        # Shelter reduces wind; channeling adds a fraction of macro wind
        local_wind = (
            macro.wind_speed * (1.0 - micro.windShelter)
            + macro.wind_speed * micro.windChannel
        )
        local_wind = _clamp(local_wind)

        # --- Dust ---
        local_dust = (
            macro.dust_density
            + micro.dustTrap * self._deposition_boost
            - micro.windChannel * self._dispersion_boost
        )
        local_dust = _clamp(local_dust)

        # --- Temperature (smoothed by thermalInertia) ---
        target_temp = macro.temp_proxy - micro.coldBias * self._cold_delta
        target_temp = _clamp(target_temp)

        if self._local_temp is None:
            self._local_temp = target_temp

        # Higher thermalInertia → slower tau → slower convergence
        eff_tau = max(0.1, self._inertia_tau * (0.1 + micro.thermalInertia * 0.9))
        alpha = 1.0 - math.exp(-dt / eff_tau)
        self._local_temp += alpha * (target_temp - self._local_temp)

        return LocalClimate(
            wind_speed=local_wind,
            wind_dir=macro.wind_dir,
            dust_density=local_dust,
            temp_proxy=self._local_temp,
            shelter=micro.windShelter,
            wind_channel=micro.windChannel,
            cold_bias=micro.coldBias,
            thermal_inertia=micro.thermalInertia,
            echo_potential=micro.echoPotential,
        )

    def reset_temperature(self, value: Optional[float] = None) -> None:
        """Reset the smoothed temperature state (e.g., after teleport)."""
        self._local_temp = value
