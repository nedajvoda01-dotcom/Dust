"""MagmaPressureModel — Stage 67 magma pressure field update.

Maintains per-tile magma pressure proxy and exposes helpers for other
subsurface modules.  Works on a :class:`SubsurfaceFieldGrid`.

Public API
----------
MagmaPressureModel(config=None)
  .tick(grid, dt, insolation=0.0)  → None
      Advance magma pressure on every tile in *grid*.
  .pressure(grid, tile_idx)        → float  [0, 1]
"""
from __future__ import annotations

from typing import Optional

from src.subsurface.SubsurfaceFieldGrid import SubsurfaceFieldGrid


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


class MagmaPressureModel:
    """Update magma pressure proxy fields in a :class:`SubsurfaceFieldGrid`.

    Parameters
    ----------
    config :
        Optional dict; reads ``subsurface67.*`` keys.
    """

    _DEFAULT_GAIN  = 0.003   # pressure rise per second
    _DEFAULT_DECAY = 0.001   # passive decay per second

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = (config or {}).get("subsurface67", {}) or {}
        self._gain  = float(cfg.get("magma_pressure_gain",  self._DEFAULT_GAIN))
        self._decay = float(cfg.get("magma_pressure_decay", self._DEFAULT_DECAY))

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def tick(
        self,
        grid:       SubsurfaceFieldGrid,
        dt:         float,
        insolation: float = 0.0,
    ) -> None:
        """Advance magma pressure on all tiles.

        Parameters
        ----------
        grid :
            The subsurface field grid to update.
        dt :
            Time step in seconds.
        insolation :
            Normalised insolation [0, 1] — higher insolation slightly
            raises deep thermal gradients and thus pressure.
        """
        thermal_boost = insolation * 0.001
        for i in range(grid.tile_count):
            t = grid.tile(i)
            dp = (
                self._gain * dt
                + thermal_boost * dt
                - self._decay * t.magmaPressureProxy * dt
            )
            t.magmaPressureProxy = _clamp(t.magmaPressureProxy + dp)

    def pressure(self, grid: SubsurfaceFieldGrid, tile_idx: int) -> float:
        """Return magma pressure proxy for *tile_idx*."""
        return grid.tile(tile_idx).magmaPressureProxy
