"""ThermalFractureModel — Stage 52 thermal gradient micro-fracture model.

Fires when ``thermalGradientField`` exceeds threshold (rapid insolation
change driven by binary-star / ring-shadow cycles).

On trigger:

* crustFailurePotential is increased on the tile (weakens crust over time).
* thermalGradientField is partially discharged.

Public API
----------
ThermalFractureModel(config=None)
  .update_gradient(state, tile, insolation_delta, dt) → None
  .process(state, tile) → ThermalFractureEvent | None
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.instability.InstabilityState import InstabilityState


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


@dataclass
class ThermalFractureEvent:
    """Fired when thermalGradientField exceeds threshold.

    Attributes
    ----------
    tile             : Flat tile index.
    intensity        : Discharge energy.
    crust_potential_gain : Added to crustFailurePotential [0..1].
    """
    tile:                  int
    intensity:             float
    crust_potential_gain:  float


class ThermalFractureModel:
    """Detects and processes thermal fracture events.

    Parameters
    ----------
    config :
        Optional dict; reads ``instability.*`` keys.
    """

    _DEFAULT_THRESHOLD   = 0.55
    _DEFAULT_DISCHARGE_K = 0.7
    _DEFAULT_CRUST_GAIN  = 0.25   # fraction of discharge added to crust potential

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = (config or {}).get("instability", {}) or {}
        self._threshold:   float = float(cfg.get("thermal_threshold",    self._DEFAULT_THRESHOLD))
        self._discharge_k: float = float(cfg.get("thermal_discharge_k",  self._DEFAULT_DISCHARGE_K))
        self._crust_gain:  float = float(cfg.get("thermal_crust_gain",   self._DEFAULT_CRUST_GAIN))

    def update_gradient(
        self,
        state:             InstabilityState,
        tile:              int,
        insolation_delta:  float,
        dt:                float,
    ) -> None:
        """Accumulate thermal gradient from insolation change.

        Parameters
        ----------
        state            : InstabilityState to update.
        tile             : Flat tile index.
        insolation_delta : Absolute change in insolation this dt [0..1].
        dt               : Elapsed time in seconds.
        """
        gain = abs(insolation_delta) * dt * 0.3
        state.thermalGradientField[tile] = _clamp(
            state.thermalGradientField[tile] + gain
        )

    def process(
        self,
        state: InstabilityState,
        tile:  int,
    ) -> Optional[ThermalFractureEvent]:
        """Test tile for thermal fracture and discharge if above threshold.

        Parameters
        ----------
        state : InstabilityState (modified in-place on discharge).
        tile  : Flat tile index to test.

        Returns
        -------
        ThermalFractureEvent if fracture occurred, else None.
        """
        gradient = state.thermalGradientField[tile]
        if gradient <= self._threshold:
            return None

        excess    = gradient - self._threshold
        discharge = excess * self._discharge_k

        state.thermalGradientField[tile] = _clamp(gradient - discharge)
        crust_gain = discharge * self._crust_gain
        state.crustFailurePotential[tile] = _clamp(
            state.crustFailurePotential[tile] + crust_gain
        )

        return ThermalFractureEvent(
            tile=tile,
            intensity=discharge,
            crust_potential_gain=crust_gain,
        )
