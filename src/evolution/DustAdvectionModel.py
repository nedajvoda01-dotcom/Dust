"""DustAdvectionModel — Stage 50 global dust migration.

Models the slow transport of dust across the planetary tile grid using a
simplified advection-erosion-deposition scheme.

The update per tile is::

    dust_next = dust_current
                + advection(windField)
                - erosion(windSpeed, roughness)
                + deposition(shelter, concavity)

The wind field is supplied as a 2-D prevailing-wind direction per tile (or
a single global direction for simplified usage).

This is intentionally a *coarse-grid diffusion-like* scheme — not a
full PDE solver.  It runs at 0.01–0.2 Hz on a background worker.

Config keys (under ``evolution.*``)
------------------------------------
dust_advection_k : float — advection strength coefficient   (default 0.05)
dust_erosion_k   : float — erosion rate coefficient         (default 0.03)
deposition_k     : float — deposition rate coefficient      (default 0.04)
max_tiles_per_tick : int — budget: tiles updated per tick   (default 256)

Public API
----------
DustAdvectionModel(config=None)
  .tick(state, wind_u, wind_v, dt) -> None
"""
from __future__ import annotations

import math
from typing import List, Optional

from src.evolution.PlanetEvolutionState import PlanetEvolutionState


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


class DustAdvectionModel:
    """Advances the ``dustReservoirMap`` field of a :class:`PlanetEvolutionState`.

    Parameters
    ----------
    config : dict or None
        Flat or nested dict.  Expected keys under ``evolution`` sub-dict or
        at top level: ``dust_advection_k``, ``dust_erosion_k``,
        ``deposition_k``, ``max_tiles_per_tick``.
    """

    _DEFAULTS = {
        "dust_advection_k":  0.05,
        "dust_erosion_k":    0.03,
        "deposition_k":      0.04,
        "max_tiles_per_tick": 256,
    }

    def __init__(self, config=None) -> None:
        cfg = dict(self._DEFAULTS)
        if isinstance(config, dict):
            src = config.get("evolution", config)
            for k in self._DEFAULTS:
                if k in src:
                    cfg[k] = src[k]

        self._adv_k:   float = float(cfg["dust_advection_k"])
        self._ero_k:   float = float(cfg["dust_erosion_k"])
        self._dep_k:   float = float(cfg["deposition_k"])
        self._budget:  int   = int(cfg["max_tiles_per_tick"])
        self._cursor:  int   = 0

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def tick(
        self,
        state: PlanetEvolutionState,
        wind_u: float,
        wind_v: float,
        dt: float,
        shelter_map: Optional[List[float]] = None,
        roughness_map: Optional[List[float]] = None,
    ) -> None:
        """Advance dust reservoir by one evolution tick.

        Parameters
        ----------
        state       : PlanetEvolutionState to update in-place.
        wind_u      : Global prevailing wind X component (normalised [-1,1]).
        wind_v      : Global prevailing wind Y component (normalised [-1,1]).
        dt          : Evolution time step (planetTime units).
        shelter_map : Optional per-tile shelter values [0,1].
                      If None, uniform 0 (open terrain) is assumed.
        roughness_map : Optional per-tile roughness values [0,1].
                        If None, uniform 0.5 is assumed.
        """
        if dt <= 0.0:
            return

        W = state.width
        H = state.height
        n = W * H
        dust = state.dustReservoirMap

        wind_speed = math.sqrt(wind_u * wind_u + wind_v * wind_v)
        # Normalise direction; default east if zero
        if wind_speed > 1e-6:
            wn_u = wind_u / wind_speed
            wn_v = wind_v / wind_speed
        else:
            wn_u, wn_v = 1.0, 0.0

        # Map wind direction to upwind tile offsets (integer displacement)
        # Use nearest-neighbour: dominant axis
        if abs(wn_u) >= abs(wn_v):
            dx = 1 if wn_u > 0 else -1
            dy = 0
        else:
            dx = 0
            dy = 1 if wn_v > 0 else -1

        count = min(self._budget, n)
        for _ in range(count):
            idx = self._cursor % n
            self._cursor = (self._cursor + 1) % n

            iy = idx // W
            ix = idx % W

            # Upwind neighbour index (periodic in X, clamped in Y)
            src_x = (ix - dx) % W
            src_y = max(0, min(iy - dy, H - 1))
            src_idx = src_y * W + src_x

            shelter   = shelter_map[idx]   if shelter_map   else 0.0
            roughness = roughness_map[idx] if roughness_map else 0.5

            # Advection: carry dust from upwind tile
            adv = self._adv_k * wind_speed * dust[src_idx] * dt
            # Erosion: remove dust from current tile (wind + roughness)
            ero = self._ero_k * wind_speed * (1.0 - roughness) * dust[idx] * dt
            # Deposition: dust settles in sheltered / concave areas
            dep = self._dep_k * shelter * (1.0 - dust[idx]) * dt

            dust[idx] = _clamp(dust[idx] + adv - ero + dep)

        # Remove the advected amount from upwind sources in the same pass
        # (simplified: we only update destination tiles for budget reasons)
