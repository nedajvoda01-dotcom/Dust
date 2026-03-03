"""ChannelEstimator — Stage 49 wind-channeling proxy.

Estimates how much a narrow passage or canyon accelerates local wind.
High channel values mean that macro wind speed should be *multiplied*
when computing local wind — the Venturi / funnelling effect.

Algorithm
---------
1. Sample terrain heights in four orthogonal directions (crosswind ± and
   downwind ±) at a short radius.
2. "Channel" is high when walls are tall on both crosswind sides and the
   passage is elongated along the wind direction (open downwind, blocked
   crosswind).

Public API
----------
ChannelEstimator(config=None, height_fn=None)
  .estimate(pos, wind_dir_2d) → float   (windChannel 0..1)
"""
from __future__ import annotations

import math
from typing import Callable, Optional

from src.math.Vec3 import Vec3


HeightFn = Callable[[float, float], float]


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


class ChannelEstimator:
    """Estimates wind-channeling in narrow passages.

    Parameters
    ----------
    config :
        Optional dict; reads ``micro.channel.*`` keys.
    height_fn :
        ``(x: float, z: float) -> float`` terrain height query.
        Defaults to a flat plane at height 0.
    """

    _DEFAULT_SAMPLE_RADIUS = 15.0   # metres for crosswind sampling
    _DEFAULT_THRESHOLD     = 0.5    # min wall fraction to count as channel

    def __init__(
        self,
        config:    Optional[dict] = None,
        height_fn: Optional[HeightFn] = None,
    ) -> None:
        cfg  = config or {}
        mcfg = cfg.get("micro", {}) or {}
        ccfg = mcfg.get("channel", {}) or {}

        self._radius:    float = float(ccfg.get("sample_radius", self._DEFAULT_SAMPLE_RADIUS))
        self._threshold: float = float(ccfg.get("threshold",     self._DEFAULT_THRESHOLD))
        self._height_fn: HeightFn = height_fn or (lambda x, z: 0.0)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def estimate(self, pos: Vec3, wind_dir_2d: Vec3) -> float:
        """Compute windChannel proxy at *pos*.

        Parameters
        ----------
        pos :
            World-space query position.
        wind_dir_2d :
            2-D wind direction vector (x, z used; y ignored).

        Returns
        -------
        float
            windChannel in [0, 1].
        """
        wlen = math.sqrt(wind_dir_2d.x ** 2 + wind_dir_2d.z ** 2)
        if wlen < 1e-6:
            return 0.0
        # Along-wind unit vector
        fx = wind_dir_2d.x / wlen
        fz = wind_dir_2d.z / wlen
        # Crosswind (perpendicular, 90° CCW)
        cx = -fz
        cz = fx

        player_h = pos.y

        # Sample heights: crosswind left, crosswind right, upwind, downwind
        def _sample_above(sx: float, sz: float) -> float:
            h = self._height_fn(sx, sz)
            return max(0.0, h - player_h)

        left_excess  = _sample_above(pos.x + cx * self._radius, pos.z + cz * self._radius)
        right_excess = _sample_above(pos.x - cx * self._radius, pos.z - cz * self._radius)
        upwind_exc   = _sample_above(pos.x - fx * self._radius, pos.z - fz * self._radius)
        dnwind_exc   = _sample_above(pos.x + fx * self._radius, pos.z + fz * self._radius)

        # Walls on both crosswind sides → channel factor
        cross_wall = min(
            _clamp(left_excess  / max(self._radius, 1.0)),
            _clamp(right_excess / max(self._radius, 1.0)),
        )
        # Passage is open downwind (no wall ahead)
        open_downwind = _clamp(1.0 - dnwind_exc / max(self._radius, 1.0))
        # Wind can enter from upwind
        open_upwind   = _clamp(1.0 - upwind_exc  / max(self._radius, 1.0))

        channel = cross_wall * open_downwind * open_upwind
        return _clamp(channel if cross_wall >= self._threshold else 0.0)
