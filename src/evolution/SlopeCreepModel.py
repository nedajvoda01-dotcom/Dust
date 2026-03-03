"""SlopeCreepModel — Stage 50 slow downhill mass redistribution.

When a tile has both high slope and high dust thickness, material slowly
migrates to lower-elevation neighbours.  This creates a gradual rounding
of slopes without touching the underlying SDF geometry.

The slope magnitude per tile must be provided externally (e.g. sampled from
the SDF height field at coarse resolution).

Model per tile::

    if slope > slope_threshold AND dustReservoir > dust_threshold:
        creep = slope_creep_k * slope * dustReservoir * dt
        slopeCreepMap[tile]   += creep          # accumulated mass flux
        slopeCreepMap[downhill] -= creep * 0.5  # mass moves downhill
        dustReservoir[tile]   -= creep * dust_transfer_frac
        dustReservoir[downhill] += creep * dust_transfer_frac

Config keys (under ``evolution.*``)
------------------------------------
slope_creep_k         : float — creep rate coefficient        (default 0.03)
slope_threshold       : float — minimum slope to activate creep (default 0.3)
dust_threshold        : float — minimum dust for creep         (default 0.2)
dust_transfer_frac    : float — fraction of creep that transfers dust (default 0.4)
max_tiles_per_tick    : int   — budget                        (default 256)

Public API
----------
SlopeCreepModel(config=None)
  .tick(state, slope_map, dt) -> None
"""
from __future__ import annotations

import math
from typing import List, Optional

from src.evolution.PlanetEvolutionState import PlanetEvolutionState


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


class SlopeCreepModel:
    """Advances ``slopeCreepMap`` and modifies ``dustReservoirMap`` in-place.

    Parameters
    ----------
    config : dict or None
        See module docstring for keys.
    """

    _DEFAULTS = {
        "slope_creep_k":       0.03,
        "slope_threshold":     0.3,
        "dust_threshold":      0.2,
        "dust_transfer_frac":  0.4,
        "max_tiles_per_tick":  256,
    }

    def __init__(self, config=None) -> None:
        cfg = dict(self._DEFAULTS)
        if isinstance(config, dict):
            src = config.get("evolution", config)
            for k in self._DEFAULTS:
                if k in src:
                    cfg[k] = src[k]

        self._creep_k:      float = float(cfg["slope_creep_k"])
        self._slope_thr:    float = float(cfg["slope_threshold"])
        self._dust_thr:     float = float(cfg["dust_threshold"])
        self._xfer_frac:    float = float(cfg["dust_transfer_frac"])
        self._budget:       int   = int(cfg["max_tiles_per_tick"])
        self._cursor:       int   = 0

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def tick(
        self,
        state: PlanetEvolutionState,
        slope_map: Optional[List[float]],
        dt: float,
    ) -> None:
        """Advance slope creep for a budget slice of tiles.

        Parameters
        ----------
        state     : PlanetEvolutionState updated in-place.
        slope_map : Per-tile slope magnitude [0, 1] (None → uniform 0.5).
        dt        : Evolution time step (planetTime units).
        """
        if dt <= 0.0:
            return

        W = state.width
        H = state.height
        n = W * H
        creep_map = state.slopeCreepMap
        dust      = state.dustReservoirMap

        count = min(self._budget, n)
        for _ in range(count):
            idx = self._cursor % n
            self._cursor = (self._cursor + 1) % n

            slope = slope_map[idx] if slope_map else 0.5
            if slope < self._slope_thr:
                continue
            if dust[idx] < self._dust_thr:
                continue

            creep = self._creep_k * slope * dust[idx] * dt

            # Accumulate creep at this tile
            creep_map[idx] = _clamp(creep_map[idx] + creep)

            # Downhill neighbour: use tile directly below (y+1 in grid coords)
            # In our grid y increases toward the south pole, so +1 row is
            # a neighbouring latitude.  Wrap handled by PlanetEvolutionState.tile.
            iy = idx // W
            ix = idx % W
            down_idx = state.tile(ix, iy + 1)

            # Reduce creep map at downhill target (mass moves downhill)
            creep_map[down_idx] = _clamp(creep_map[down_idx] - creep * 0.5)

            # Transfer dust fraction downhill
            xfer = creep * self._xfer_frac
            dust[idx]      = _clamp(dust[idx]      - xfer)
            dust[down_idx] = _clamp(dust[down_idx] + xfer)
