"""RegionIndexing — Stage 60 lat/lon → regionId mapping.

Deterministically maps a point on the spherical planet to a unique integer
region ID.  The grid is aligned to whole-degree lat/lon boundaries and the
mapping is derived only from ``region_size_deg`` and the point coordinates,
making it fully reproducible from any world seed.

Coordinate conventions
-----------------------
* latitude  ∈ [-90, 90]   — north is positive
* longitude ∈ [-180, 180) — east is positive

Region IDs
----------
The planet is sliced into a lat/lon grid of ``region_size_deg``-degree tiles.

    cols = ceil(360 / region_size_deg)
    rows = ceil(180 / region_size_deg)

    col  = floor((lon + 180) / region_size_deg)   clamped to [0, cols-1]
    row  = floor((lat +  90) / region_size_deg)   clamped to [0, rows-1]
    id   = row * cols + col

Region (0, 0) corresponds to the tile that contains (lat=-90, lon=-180).

Public API
----------
RegionIndexing(region_size_deg=10.0)
  .region_id(lat, lon) → int
  .region_bounds(region_id) → (lat_min, lon_min, lat_max, lon_max)
  .neighbours(region_id) → list[int]
  .total_regions → int
  .cols → int
  .rows → int
"""
from __future__ import annotations

import math
from typing import List, Tuple


class RegionIndexing:
    """Deterministic spherical-planet region grid."""

    def __init__(self, region_size_deg: float = 10.0) -> None:
        if region_size_deg <= 0:
            raise ValueError("region_size_deg must be > 0")
        self._size  = float(region_size_deg)
        self._cols  = max(1, math.ceil(360.0 / self._size))
        self._rows  = max(1, math.ceil(180.0 / self._size))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def cols(self) -> int:
        return self._cols

    @property
    def rows(self) -> int:
        return self._rows

    @property
    def total_regions(self) -> int:
        return self._rows * self._cols

    # ------------------------------------------------------------------
    # Core mapping
    # ------------------------------------------------------------------

    def region_id(self, lat: float, lon: float) -> int:
        """Return the region ID for *(lat, lon)*."""
        col = int((lon + 180.0) / self._size)
        row = int((lat +  90.0) / self._size)
        col = max(0, min(self._cols - 1, col))
        row = max(0, min(self._rows - 1, row))
        return row * self._cols + col

    def region_bounds(
        self, region_id: int
    ) -> Tuple[float, float, float, float]:
        """Return *(lat_min, lon_min, lat_max, lon_max)* for *region_id*."""
        if region_id < 0 or region_id >= self.total_regions:
            raise ValueError(f"region_id {region_id} out of range")
        row = region_id // self._cols
        col = region_id  % self._cols
        lat_min = -90.0 + row * self._size
        lon_min = -180.0 + col * self._size
        lat_max = min( 90.0, lat_min + self._size)
        lon_max = min(180.0, lon_min + self._size)
        return lat_min, lon_min, lat_max, lon_max

    def neighbours(self, region_id: int) -> List[int]:
        """Return IDs of the (up to 8) neighbouring regions.

        Longitude wraps around; latitude clamps at the poles.
        """
        row = region_id // self._cols
        col = region_id  % self._cols
        result: List[int] = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr = row + dr
                nc = (col + dc) % self._cols   # longitude wraps
                if 0 <= nr < self._rows:
                    nid = nr * self._cols + nc
                    result.append(nid)
        return result
