"""VolumetricsToVisibility — Stage 65 adapter: volumetrics → visibility proxy.

Translates volumetric domain density state into a visibility proxy value
consumed by the gameplay/AI visibility system (spec §8).

The visibility proxy from Stage 64 (``AtmosphereSystem.visibility_proxy``)
is a coarse 2-D value.  This adapter computes a finer-grained per-column
visibility considering the full 3-D density distribution in the domain.

Model
-----
Column visibility:
    col_vis = exp(−absorption × mean_column_density)
    visibility_proxy = mean over all columns

Absorption coefficient is taken from the renderer config (spec §7) to
keep the visual and gameplay proxies consistent.

Public API
----------
VolumetricsToVisibility(config=None)
  .visibility_proxy(grid)         → float [0..1]
  .column_visibility(grid, ix, iy) → float [0..1]
"""
from __future__ import annotations

import math
from typing import Optional

from src.vol.DensityGrid import DensityGrid


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


class VolumetricsToVisibility:
    """Compute visibility proxy from a 3-D density grid.

    Parameters
    ----------
    config :
        Optional dict; reads ``vol.absorption`` (Beer-Lambert coefficient).
    """

    _DEFAULT_ABSORPTION = 0.4

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg  = config or {}
        vcfg = cfg.get("vol", {}) or {}
        self._absorption: float = float(
            vcfg.get("absorption", self._DEFAULT_ABSORPTION)
        )

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def visibility_proxy(self, grid: DensityGrid) -> float:
        """Mean column visibility across the entire domain [0..1].

        0 = completely opaque, 1 = fully transparent.
        """
        total = 0.0
        n     = grid.width * grid.height
        for iy in range(grid.height):
            for ix in range(grid.width):
                total += self.column_visibility(grid, ix, iy)
        return _clamp(total / max(n, 1))

    def column_visibility(self, grid: DensityGrid, ix: int, iy: int) -> float:
        """Beer-Lambert visibility through column (ix, iy) [0..1]."""
        col_density = sum(
            grid.density(ix, iy, iz) for iz in range(grid.depth)
        ) / max(grid.depth, 1)
        return _clamp(math.exp(-self._absorption * col_density))
