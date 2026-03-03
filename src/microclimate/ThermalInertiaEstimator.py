"""ThermalInertiaEstimator — Stage 49 enclosure-based thermal inertia proxy.

Enclosed spaces (caves, deep canyons, under ledges) change temperature
more slowly than open terrain.  This proxy estimates enclosure as the
fraction of the *sky dome* that is blocked by surrounding geometry
("sky visibility" ≈ 1 - enclosure).

Algorithm
---------
Cast ``num_sky_samples`` rays uniformly over the upper hemisphere from *pos*.
Count how many are blocked (terrain height above ray elevation).  The
sky-visibility fraction drives inertia.

Public API
----------
ThermalInertiaEstimator(config=None, height_fn=None)
  .estimate(pos) → float   (thermalInertia 0..1)
"""
from __future__ import annotations

import math
from typing import Callable, Optional

from src.math.Vec3 import Vec3


HeightFn = Callable[[float, float], float]


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


class ThermalInertiaEstimator:
    """Estimates thermal inertia from sky visibility.

    Parameters
    ----------
    config :
        Optional dict; reads ``micro.*`` keys.
    height_fn :
        ``(x: float, z: float) -> float`` terrain height query.
    """

    _DEFAULT_SKY_RADIUS   = 30.0   # metres for sky-occlusion samples
    _DEFAULT_NUM_SAMPLES  = 8      # number of horizontal sky directions
    _DEFAULT_INERTIA_TAU  = 60.0   # seconds (used by LocalClimateComposer)

    def __init__(
        self,
        config:    Optional[dict] = None,
        height_fn: Optional[HeightFn] = None,
    ) -> None:
        cfg  = config or {}
        mcfg = cfg.get("micro", {}) or {}

        self._radius:  float = float(mcfg.get("sky_radius",    self._DEFAULT_SKY_RADIUS))
        self._n:       int   = int(mcfg.get("sky_num_samples", self._DEFAULT_NUM_SAMPLES))
        self._height_fn: HeightFn = height_fn or (lambda x, z: 0.0)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def estimate(self, pos: Vec3) -> float:
        """Compute thermalInertia proxy.

        High value → enclosed (cave-like, temperature changes slowly).
        Low value  → open (exposed, temperature tracks macro quickly).
        """
        centre_h = pos.y
        if self._n == 0:
            return 0.0
        step = (2.0 * math.pi) / self._n
        blocked = 0
        for i in range(self._n):
            angle = i * step
            sx = pos.x + self._radius * math.cos(angle)
            sz = pos.z + self._radius * math.sin(angle)
            sh = self._height_fn(sx, sz)
            if sh > centre_h:
                blocked += 1
        # thermalInertia = fraction of sky blocked = enclosure
        return _clamp(blocked / self._n)
