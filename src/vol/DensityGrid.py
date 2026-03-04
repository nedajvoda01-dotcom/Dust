"""DensityGrid — Stage 65 3-D voxel grid for volumetric density simulation.

Stores one floating-point density value per voxel for a volumetric domain.
A domain is axis-aligned box centred on a world-space anchor.

Supported layer types (spec §1.1)
---------------------------------
* ``DustVolume``      — airborne regolith suspension
* ``FogVolume``       — condensate / low-visibility moisture
* ``SteamVolume``     — localised steam from lava/water sources
* ``SnowDriftVolume`` — wind-driven snow suspension (optional)

Storage model
-------------
Flat 1-D array of float32-equivalent Python floats, indexed as::

    idx = iz * (width * height) + iy * width + ix

All density values are clamped to [0, 1].

Public API
----------
DensityGrid(width, height, depth)
  .density(ix, iy, iz)                       → float [0..1]
  .set_density(ix, iy, iz, value)            → None
  .add_density(ix, iy, iz, delta)            → None  (clamped)
  .total_density()                           → float
  .clear()                                   → None
  .width, .height, .depth                    → int
  .grid_hash()                               → str  (hex MD5)

VolumeLayerType — string constants for the four supported layer types.
"""
from __future__ import annotations

import hashlib
import struct
from typing import List


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


# ---------------------------------------------------------------------------
# Layer type constants
# ---------------------------------------------------------------------------

class VolumeLayerType:
    DUST        = "DustVolume"
    FOG         = "FogVolume"
    STEAM       = "SteamVolume"
    SNOW_DRIFT  = "SnowDriftVolume"

    ALL = (DUST, FOG, STEAM, SNOW_DRIFT)


# ---------------------------------------------------------------------------
# DensityGrid
# ---------------------------------------------------------------------------

class DensityGrid:
    """3-D voxel grid of normalised density values for one volumetric layer.

    Parameters
    ----------
    width, height, depth :
        Grid dimensions in voxels.
    layer_type :
        One of :class:`VolumeLayerType` constants.
    """

    def __init__(
        self,
        width:      int,
        height:     int,
        depth:      int,
        layer_type: str = VolumeLayerType.DUST,
    ) -> None:
        self._w = width
        self._h = height
        self._d = depth
        self.layer_type = layer_type
        self._data: List[float] = [0.0] * (width * height * depth)

    # ------------------------------------------------------------------
    # Dimensions
    # ------------------------------------------------------------------

    @property
    def width(self) -> int:
        return self._w

    @property
    def height(self) -> int:
        return self._h

    @property
    def depth(self) -> int:
        return self._d

    # ------------------------------------------------------------------
    # Index helpers
    # ------------------------------------------------------------------

    def _idx(self, ix: int, iy: int, iz: int) -> int:
        return iz * (self._w * self._h) + iy * self._w + ix

    def _in_bounds(self, ix: int, iy: int, iz: int) -> bool:
        return 0 <= ix < self._w and 0 <= iy < self._h and 0 <= iz < self._d

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def density(self, ix: int, iy: int, iz: int) -> float:
        """Return density at voxel (ix, iy, iz); returns 0.0 if out of bounds."""
        if not self._in_bounds(ix, iy, iz):
            return 0.0
        return self._data[self._idx(ix, iy, iz)]

    def set_density(self, ix: int, iy: int, iz: int, value: float) -> None:
        """Set density at voxel; clamped to [0, 1]."""
        if not self._in_bounds(ix, iy, iz):
            return
        self._data[self._idx(ix, iy, iz)] = _clamp(value)

    def add_density(self, ix: int, iy: int, iz: int, delta: float) -> None:
        """Add *delta* to existing density; result clamped to [0, 1]."""
        if not self._in_bounds(ix, iy, iz):
            return
        idx = self._idx(ix, iy, iz)
        self._data[idx] = _clamp(self._data[idx] + delta)

    # ------------------------------------------------------------------
    # Aggregates
    # ------------------------------------------------------------------

    def total_density(self) -> float:
        """Sum of all voxel densities."""
        return sum(self._data)

    def clear(self) -> None:
        """Set all voxels to zero."""
        for i in range(len(self._data)):
            self._data[i] = 0.0

    # ------------------------------------------------------------------
    # Serialisation / hashing
    # ------------------------------------------------------------------

    def to_bytes(self) -> bytes:
        """Serialise density grid as packed half-precision uint16 values."""
        parts = []
        for v in self._data:
            u16 = max(0, min(65535, int(round(v * 65535.0))))
            parts.append(struct.pack("!H", u16))
        header = struct.pack("!HHH", self._w, self._h, self._d)
        return header + b"".join(parts)

    def grid_hash(self) -> str:
        """Hex MD5 of packed density data."""
        return hashlib.md5(self.to_bytes()).hexdigest()
