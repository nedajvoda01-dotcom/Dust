"""ValleyConcavityProxy — Stage 46 terrain channelling multiplier.

Estimates how much a terrain location "channels" sound by sampling
heights around the query point.  Concave areas (valleys, craters) retain
sound better than ridge-tops.

The caller supplies a ``height_fn(x, z) → float`` that returns terrain
elevation at a 2-D point.  The proxy samples N radial directions,
computes the mean height deviation from the centre, and maps this to a
``valley_gain`` in ``[valley_gain_min, valley_gain_max]``.

Public API
----------
ValleyConcavityProxy(config=None, height_fn=None)
  .compute(pos) → float   (valley_gain multiplier)
"""
from __future__ import annotations

import math
from typing import Callable, Optional

from src.math.Vec3 import Vec3


HeightFn = Callable[[float, float], float]


class ValleyConcavityProxy:
    """Computes a terrain-channelling gain multiplier.

    Parameters
    ----------
    config :
        Optional dict; reads ``audio_world.*`` keys.
    height_fn :
        ``(x: float, z: float) -> float`` terrain height query.
        Defaults to a flat plane at height 0.
    """

    _DEFAULT_VALLEY_GAIN_MIN = 0.8
    _DEFAULT_VALLEY_GAIN_MAX = 1.3
    _DEFAULT_SAMPLE_RADIUS   = 40.0   # metres around the query point
    _NUM_SAMPLES             = 8      # radial directions sampled

    def __init__(
        self,
        config:    Optional[dict] = None,
        height_fn: Optional[HeightFn] = None,
    ) -> None:
        awcfg = (config or {}).get("audio_world", {}) or {}
        self._gain_min: float = float(awcfg.get("valley_gain_min", self._DEFAULT_VALLEY_GAIN_MIN))
        self._gain_max: float = float(awcfg.get("valley_gain_max", self._DEFAULT_VALLEY_GAIN_MAX))
        self._radius:   float = float(awcfg.get("valley_sample_radius", self._DEFAULT_SAMPLE_RADIUS))
        self._height_fn: HeightFn = height_fn or (lambda x, z: 0.0)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def compute(self, pos: Vec3) -> float:
        """Return the valley-gain multiplier at *pos*.

        A value > 1 means the position is in a concave area that channels
        sound; < 1 means it is on a convex ridge that disperses sound.
        """
        centre_h = self._height_fn(pos.x, pos.z)
        step = (2.0 * math.pi) / self._NUM_SAMPLES
        higher_count = 0
        for i in range(self._NUM_SAMPLES):
            angle = i * step
            sx = pos.x + self._radius * math.cos(angle)
            sz = pos.z + self._radius * math.sin(angle)
            sh = self._height_fn(sx, sz)
            if sh > centre_h:
                higher_count += 1

        # Fraction of surrounding samples that are higher than centre
        # → 1.0 means fully enclosed (deep valley / crater)
        concavity = higher_count / self._NUM_SAMPLES
        # Map [0, 1] concavity to [gain_min, gain_max]
        return self._gain_min + concavity * (self._gain_max - self._gain_min)
