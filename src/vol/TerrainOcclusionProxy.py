"""TerrainOcclusionProxy — Stage 65 terrain influence on volumetric fields.

Implements spec §4.3: a lightweight terrain-height-based occlusion mask
for a volumetric domain, avoiding full 3-D collision per voxel.

Model
-----
A 2-D heightfield is sampled once per domain at construction/refresh.
For each column (ix, iy), all voxels at iz ≤ height_voxel are marked as
"blocked" (inside terrain) and their density is immediately zeroed out.

Wind attenuation near terrain
------------------------------
For voxels just above the terrain surface, the effective wind is reduced
by a friction factor.  Voxels in narrow channels get a channeling boost
(spec §4.3 "channels").

Public API
----------
TerrainOcclusionProxy(width, height, depth, config=None)
  .update_heightfield(heights)   → None
      heights : list of (width * height) normalised [0..1] terrain heights.
  .apply_occlusion(grid)         → None
      Zero out density inside terrain.
  .wind_scale(ix, iy, iz)        → float [0..1]
      Effective wind attenuation at voxel (ix, iy, iz).
"""
from __future__ import annotations

from typing import List, Optional

from src.vol.DensityGrid import DensityGrid


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


class TerrainOcclusionProxy:
    """Lightweight terrain mask for a volumetric domain.

    Parameters
    ----------
    width, height, depth :
        Voxel grid dimensions (must match the DensityGrids to be processed).
    config :
        Optional dict; reads ``vol.terrain_friction`` and
        ``vol.channel_boost``.
    """

    _DEFAULT_FRICTION     = 0.6   # wind attenuation factor near ground
    _DEFAULT_CHANNEL_BOOST = 1.4  # channeling multiplier in narrow valleys

    def __init__(
        self,
        width:  int,
        height: int,
        depth:  int,
        config: Optional[dict] = None,
    ) -> None:
        self._w = width
        self._h = height
        self._d = depth

        cfg  = config or {}
        vcfg = cfg.get("vol", {}) or {}
        self._friction:      float = float(
            vcfg.get("terrain_friction", self._DEFAULT_FRICTION)
        )
        self._channel_boost: float = float(
            vcfg.get("channel_boost", self._DEFAULT_CHANNEL_BOOST)
        )

        # Heightfield in voxel units [0..depth]; 0 = flat terrain
        self._height_voxels: List[float] = [0.0] * (width * height)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def update_heightfield(self, heights: List[float]) -> None:
        """Set normalised terrain heights [0..1].

        *heights* must have length ``width × height``.  A height of 0.0
        means terrain at voxel-layer 0; 1.0 means terrain fills the full
        domain depth.
        """
        if len(heights) != self._w * self._h:
            raise ValueError(
                f"Expected {self._w * self._h} height values, got {len(heights)}"
            )
        for i, h in enumerate(heights):
            self._height_voxels[i] = _clamp(h) * self._d

    def apply_occlusion(self, grid: DensityGrid) -> None:
        """Zero density for all voxels inside terrain."""
        for iy in range(self._h):
            for ix in range(self._w):
                terrain_z = self._height_voxels[iy * self._w + ix]
                for iz in range(self._d):
                    if iz < terrain_z:
                        grid.set_density(ix, iy, iz, 0.0)

    def wind_scale(self, ix: int, iy: int, iz: int) -> float:
        """Return effective wind scale at voxel (ix, iy, iz) [0..1].

        Voxels just above terrain are attenuated; voxels high above terrain
        get full wind.  (Channeling logic is a stub — returns base scale.)
        """
        if ix < 0 or ix >= self._w or iy < 0 or iy >= self._h:
            return 1.0
        terrain_z = self._height_voxels[iy * self._w + ix]
        if iz < terrain_z:
            return 0.0   # inside terrain
        clearance = iz - terrain_z
        if clearance < 2.0:
            return _clamp(1.0 - self._friction * (1.0 - clearance / 2.0))
        return 1.0
