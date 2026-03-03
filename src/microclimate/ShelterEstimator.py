"""ShelterEstimator — Stage 49 wind-shelter proxy.

Estimates how protected a point is from macro-wind by sampling terrain
heights upwind.  No fluid simulation is performed; this is purely a
geometric proxy.

Algorithm
---------
1. Bucket the current wind direction into one of *N* discrete buckets.
2. Sample ``num_samples`` heights along the upwind direction at increasing
   distances up to ``sample_distance``.
3. If a sampled height is above the player's height, the shelter factor rises.
4. Results are cached per (chunk_key, wind_bucket) so the computation only
   runs when the wind direction bucket changes or the cache is cold.

Additionally estimates edge-gust potential: if the player is just behind a
ridge (one sample above, surrounded by open), a gust multiplier > 1 is
returned.

Public API
----------
ShelterEstimator(config=None, height_fn=None)
  .estimate(pos, wind_dir_2d) → ShelterResult
  .invalidate_cache()
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

from src.math.Vec3 import Vec3


HeightFn = Callable[[float, float], float]

_TWO_PI = 2.0 * math.pi


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


@dataclass
class ShelterResult:
    """Output of :class:`ShelterEstimator`.

    Attributes
    ----------
    shelter : float
        Shelter from wind [0 = fully exposed, 1 = fully sheltered].
    gust_factor : float
        Edge-gust multiplier [1.0 = no gust, >1 = elevated near ridge edge].
    """
    shelter:     float = 0.0
    gust_factor: float = 1.0


class ShelterEstimator:
    """Estimates wind shelter from terrain geometry.

    Parameters
    ----------
    config :
        Optional dict; reads ``micro.shelter.*`` keys.
    height_fn :
        ``(x: float, z: float) -> float`` terrain height query.
        Defaults to a flat plane at height 0.
    """

    _DEFAULT_SAMPLE_DISTANCE  = 50.0
    _DEFAULT_NUM_SAMPLES_NEAR = 8
    _DEFAULT_WINDDIR_BUCKETS  = 16

    def __init__(
        self,
        config:    Optional[dict] = None,
        height_fn: Optional[HeightFn] = None,
    ) -> None:
        cfg  = config or {}
        mcfg = cfg.get("micro", {}) or {}
        scfg = mcfg.get("shelter", {}) or {}

        self._dist:     float = float(scfg.get("sample_distance",   self._DEFAULT_SAMPLE_DISTANCE))
        self._n_near:   int   = int(scfg.get("num_samples_near",    self._DEFAULT_NUM_SAMPLES_NEAR))
        self._buckets:  int   = int(mcfg.get("winddir_buckets",     self._DEFAULT_WINDDIR_BUCKETS))

        self._height_fn: HeightFn = height_fn or (lambda x, z: 0.0)

        # Cache: {(chunk_key, wind_bucket): ShelterResult}
        self._cache: Dict[Tuple, ShelterResult] = {}

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def estimate(
        self,
        pos:          Vec3,
        wind_dir_2d:  Vec3,
        chunk_key:    Tuple = (0, 0),
    ) -> ShelterResult:
        """Estimate shelter at *pos* for the given wind direction.

        Parameters
        ----------
        pos :
            World-space position of the query point.
        wind_dir_2d :
            2-D wind direction (x, z components used; y ignored).
            Does not need to be normalised.
        chunk_key :
            Optional hashable identifier for the chunk containing *pos*,
            used to namespace the cache.
        """
        wind_bucket = self._bucket(wind_dir_2d)
        cache_key = (chunk_key, wind_bucket)
        if cache_key in self._cache:
            return self._cache[cache_key]

        result = self._compute(pos, wind_dir_2d)
        self._cache[cache_key] = result
        return result

    def invalidate_cache(self) -> None:
        """Clear all cached shelter values (call when geometry changes)."""
        self._cache.clear()

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _bucket(self, wind_dir_2d: Vec3) -> int:
        """Map wind direction to a discrete bucket index."""
        angle = math.atan2(wind_dir_2d.z, wind_dir_2d.x)
        if angle < 0.0:
            angle += _TWO_PI
        bucket = int(angle / _TWO_PI * self._buckets) % self._buckets
        return bucket

    def _compute(self, pos: Vec3, wind_dir_2d: Vec3) -> ShelterResult:
        """Perform the upwind height-field sampling."""
        # Normalise direction
        wlen = math.sqrt(wind_dir_2d.x ** 2 + wind_dir_2d.z ** 2)
        if wlen < 1e-6:
            return ShelterResult(shelter=0.0, gust_factor=1.0)
        dx = wind_dir_2d.x / wlen
        dz = wind_dir_2d.z / wlen

        player_h = pos.y
        blocked_count = 0

        for i in range(1, self._n_near + 1):
            t = (i / self._n_near) * self._dist
            sx = pos.x - dx * t  # upwind direction
            sz = pos.z - dz * t
            sh = self._height_fn(sx, sz)
            if sh > player_h:
                blocked_count += 1

        shelter = _clamp(blocked_count / max(self._n_near, 1))

        # Edge-gust: sheltered (some samples above) but close to an edge
        # → brief acceleration as wind wraps around obstacle
        if 0 < blocked_count < self._n_near and blocked_count <= 2:
            gust_factor = 1.0 + 0.4 * (1.0 - shelter)
        else:
            gust_factor = 1.0

        return ShelterResult(shelter=shelter, gust_factor=gust_factor)
