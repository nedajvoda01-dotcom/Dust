"""VolumetricsToSettling — Stage 66 adapter: volumetrics (65) → settling inputs.

Reads the mean density of the bottom layer of a
:class:`~src.vol.DensityGrid.DensityGrid` to feed the Stage 66
:class:`~src.mass.SettlingModel.SettlingModel` with realistic air-density
values.

Only DUST and SNOW_DRIFT grids contribute to surface settling; FOG and
STEAM grids do not deliver dry mass to the surface and are ignored.

Public API
----------
VolumetricsToSettling(config=None)
  .air_dust_density(dust_grid)       → float [0..1]
  .air_snow_density(snow_drift_grid) → float [0..1]
"""
from __future__ import annotations

from typing import Optional

from src.vol.DensityGrid import DensityGrid, VolumeLayerType


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


class VolumetricsToSettling:
    """Extract settling density inputs from volumetric grids.

    Parameters
    ----------
    config :
        Optional dict (reserved for future coefficients).
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        pass

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def air_dust_density(self, dust_grid: DensityGrid) -> float:
        """Return mean dust density in the lowest voxel layer [0..1].

        Parameters
        ----------
        dust_grid :
            A :class:`~src.vol.DensityGrid.DensityGrid` of type DUST.

        Returns
        -------
        float
            Mean density of the iz=0 layer, clamped to [0..1].
        """
        return _clamp(self._mean_ground_density(dust_grid))

    def air_snow_density(self, snow_drift_grid: DensityGrid) -> float:
        """Return mean snow-drift density in the lowest voxel layer [0..1].

        Parameters
        ----------
        snow_drift_grid :
            A :class:`~src.vol.DensityGrid.DensityGrid` of type SNOW_DRIFT.

        Returns
        -------
        float
            Mean density of the iz=0 layer, clamped to [0..1].
        """
        return _clamp(self._mean_ground_density(snow_drift_grid))

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    @staticmethod
    def _mean_ground_density(grid: DensityGrid) -> float:
        """Mean density of the lowest layer (iz=0) of *grid*."""
        w, h = grid.width, grid.height
        if w == 0 or h == 0:
            return 0.0
        total = sum(grid.density(ix, iy, 0) for iy in range(h) for ix in range(w))
        return total / (w * h)
