"""SDFBase — base SDF primitive that defines the planet surface.

Convention (same as the rest of the SDF subsystem):
  d > 0  →  outside / air
  d < 0  →  inside / rock

Public API
----------
SDFBase(radius)
  .eval(x, y, z) → float   signed distance to surface
  .radius         → float   planet radius
"""
from __future__ import annotations

import math


class SDFBase:
    """Planet sphere: signed distance = distance_to_origin - radius.

    Parameters
    ----------
    radius :
        Planet radius in simulation units.
    """

    __slots__ = ("_radius",)

    def __init__(self, radius: float = 1000.0) -> None:
        self._radius = float(radius)

    @property
    def radius(self) -> float:
        return self._radius

    def eval(self, x: float, y: float, z: float) -> float:
        """Return signed distance to the sphere surface at (x, y, z)."""
        return math.sqrt(x * x + y * y + z * z) - self._radius
