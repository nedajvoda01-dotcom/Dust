"""AdvectionSolver — Stage 65 density advection by local wind.

Implements a cheap, stable semi-Lagrangian advection step for a
:class:`~src.vol.DensityGrid.DensityGrid`.

Method (spec §4.2)
------------------
For each voxel (ix, iy, iz) in the destination grid, back-trace
the wind vector one time step to find the source position, then
tri-linearly interpolate the source density.  The result is
numerically dissipative (unconditionally stable for any time step).

Wind input
----------
Wind is supplied as a 2-D horizontal field (from Stage 64) that is
broadcast across all vertical layers.  Near-ground layers have their
wind attenuated by a factor proportional to their height normalised
within the domain — matching the spec's "wind weaker close to ground"
rule.

Public API
----------
AdvectionSolver(config=None)
  .step(grid, wind_x, wind_y, dt)  → None
      In-place advection of *grid*.
      wind_x, wind_y : uniform horizontal wind components [−1..1].
      dt             : time step in seconds.
"""
from __future__ import annotations

from typing import Optional

from src.vol.DensityGrid import DensityGrid


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


class AdvectionSolver:
    """Semi-Lagrangian advection for a DensityGrid.

    Parameters
    ----------
    config :
        Optional dict; reads ``vol.advection_scale`` to tune the
        mapping of normalised wind [−1..1] onto voxels/second.
    """

    _DEFAULT_ADVECTION_SCALE = 4.0   # voxels per second per unit wind

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg  = config or {}
        vcfg = cfg.get("vol", {}) or {}
        self._scale: float = float(
            vcfg.get("advection_scale", self._DEFAULT_ADVECTION_SCALE)
        )

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def step(
        self,
        grid:   DensityGrid,
        wind_x: float,
        wind_y: float,
        dt:     float,
    ) -> None:
        """Advect *grid* in-place for one time step *dt*.

        Parameters
        ----------
        grid :
            The DensityGrid to advect.
        wind_x, wind_y :
            Horizontal wind components (normalised [−1..1]).
        dt :
            Time step in seconds.
        """
        w, h, d = grid.width, grid.height, grid.depth
        # Displacement in voxels
        dx = wind_x * self._scale * dt
        dy = wind_y * self._scale * dt

        new_data = [0.0] * (w * h * d)

        for iz in range(d):
            # Attenuate wind near ground (iz=0 → no wind; iz=d-1 → full wind)
            height_factor = iz / max(d - 1, 1)
            ldx = dx * height_factor
            ldy = dy * height_factor

            for iy in range(h):
                for ix in range(w):
                    # Back-trace source position
                    sx = ix - ldx
                    sy = iy - ldy

                    # Tri-linear interpolation (2-D: horizontal slice)
                    x0 = int(sx)
                    y0 = int(sy)
                    fx = sx - x0
                    fy = sy - y0

                    # Sample four corners
                    d00 = grid.density(_wrap(x0,     w), _wrap(y0,     h), iz)
                    d10 = grid.density(_wrap(x0 + 1, w), _wrap(y0,     h), iz)
                    d01 = grid.density(_wrap(x0,     w), _wrap(y0 + 1, h), iz)
                    d11 = grid.density(_wrap(x0 + 1, w), _wrap(y0 + 1, h), iz)

                    val = _lerp(_lerp(d00, d10, fx), _lerp(d01, d11, fx), fy)
                    idx = iz * (w * h) + iy * w + ix
                    new_data[idx] = _clamp(val)

        grid._data = new_data


def _wrap(v: int, size: int) -> int:
    """Clamp *v* to [0, size-1] (boundary handling: clamp, not wrap)."""
    return max(0, min(size - 1, v))
