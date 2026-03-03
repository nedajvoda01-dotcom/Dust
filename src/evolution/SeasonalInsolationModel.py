"""SeasonalInsolationModel — Stage 50 seasonal effects from dual suns + ring.

Derives a ``seasonalInsolationPhase`` from the planetary orbital state and
updates the ``iceBeltDistribution`` field based on latitude and the current
seasonal phase.

Ice belt logic
--------------
At any given season, the ice belt is most stable at high latitudes and in
the latitude band most distant from the subsolar point.  The ice belt
coefficient for each tile is::

    lat_factor  = sin²(latitude)                          # polar preference
    cold_factor = max(0, cos(seasonal_phase - lat_phase)) # annual shadow wave
    iceBelt[tile] = clamp(lat_factor * cold_factor
                          + ice_belt_k * dt * cold_factor
                          - ice_melt_k * dt * (1 - cold_factor))

Config keys (under ``evolution.*``)
------------------------------------
ice_belt_k   : float — ice accumulation coefficient  (default 0.04)
ice_melt_k   : float — ice melt coefficient          (default 0.06)
season_speed : float — orbital angular velocity [rad / planetTime unit]
                       (default 7.27e-5, ≈ one season = 86 400 planetTime)
max_tiles_per_tick : int — budget                    (default 256)

Public API
----------
SeasonalInsolationModel(config=None)
  .tick(state, dt) -> None
  .thermal_cycle_amplitude(state) -> float
"""
from __future__ import annotations

import math
from typing import List

from src.evolution.PlanetEvolutionState import PlanetEvolutionState


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


class SeasonalInsolationModel:
    """Advances ``seasonalInsolationPhase`` and ``iceBeltDistribution``.

    Parameters
    ----------
    config : dict or None
        See module docstring for keys.
    """

    _DEFAULTS = {
        "ice_belt_k":          0.04,
        "ice_melt_k":          0.06,
        "season_speed":        7.27e-5,   # rad per planetTime unit
        "max_tiles_per_tick":  256,
    }

    def __init__(self, config=None) -> None:
        cfg = dict(self._DEFAULTS)
        if isinstance(config, dict):
            src = config.get("evolution", config)
            for k in self._DEFAULTS:
                if k in src:
                    cfg[k] = src[k]

        self._ice_k:    float = float(cfg["ice_belt_k"])
        self._melt_k:   float = float(cfg["ice_melt_k"])
        self._speed:    float = float(cfg["season_speed"])
        self._budget:   int   = int(cfg["max_tiles_per_tick"])
        self._cursor:   int   = 0

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def tick(self, state: PlanetEvolutionState, dt: float) -> None:
        """Advance seasonal phase and ice belt distribution.

        Parameters
        ----------
        state : PlanetEvolutionState to update in-place.
        dt    : Evolution time step (planetTime units).
        """
        if dt <= 0.0:
            return

        # Advance the orbital phase
        state.seasonalInsolationPhase = (
            (state.seasonalInsolationPhase + self._speed * dt) % (2.0 * math.pi)
        )
        phase = state.seasonalInsolationPhase

        W = state.width
        H = state.height
        n = W * H
        ice = state.iceBeltDistribution

        count = min(self._budget, n)
        for _ in range(count):
            idx = self._cursor % n
            self._cursor = (self._cursor + 1) % n

            iy = idx // W
            # Latitude of this tile row: -π/2 (south pole) .. +π/2 (north pole)
            lat = math.pi * (iy + 0.5) / H - math.pi * 0.5

            lat_factor  = math.sin(lat) ** 2               # polar preference

            # "Cold" factor: how much this latitude is away from the subsolar point
            # The subsolar latitude oscillates as sin(phase) * axial_tilt
            # We approximate axial_tilt as π/6 (30°)
            subsolar_lat = math.pi / 6.0 * math.sin(phase)
            angular_dist = abs(lat - subsolar_lat)
            cold_factor  = _clamp(angular_dist / (math.pi * 0.5))

            net = lat_factor * cold_factor
            if cold_factor > 0.5:
                ice[idx] = _clamp(ice[idx] + self._ice_k * dt * cold_factor)
            else:
                ice[idx] = _clamp(ice[idx] - self._melt_k * dt * (1.0 - cold_factor))

            # Blend toward natural equilibrium (lat_factor * cold_factor)
            ice[idx] = _clamp(ice[idx] * 0.999 + net * 0.001)

    # ------------------------------------------------------------------

    def thermal_cycle_amplitude(self, state: PlanetEvolutionState) -> float:
        """Return a [0, 1] scalar representing global thermal cycle stress.

        Higher when the two suns produce large temperature swings
        (i.e. near equinoxes in the binary orbit).

        Uses the seasonalInsolationPhase as a proxy: amplitude is maximum
        when d(insolation)/dt is steepest (near quadrature at phase ≈ π/2).
        """
        # Derivative of sin(phase) is cos(phase); use |cos| as amplitude
        return abs(math.cos(state.seasonalInsolationPhase))
