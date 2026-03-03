"""EchoPotentialEstimator — Stage 49 acoustic echo / reverb proxy.

Highly enclosed spaces produce more reflections and reverb.  This proxy
re-uses the sky-enclosure estimate (same concept as
:class:`~src.microclimate.ThermalInertiaEstimator`) but is tuned toward
the acoustic use case and applies a gain multiplier for late-reflection
mixing.

Public API
----------
EchoPotentialEstimator(config=None, height_fn=None)
  .estimate(pos) → float   (echoPotential 0..1)
  .reverb_gain(echo_potential) → float
"""
from __future__ import annotations

import math
from typing import Callable, Optional

from src.math.Vec3 import Vec3


HeightFn = Callable[[float, float], float]


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


class EchoPotentialEstimator:
    """Estimates acoustic echo potential from terrain enclosure.

    Parameters
    ----------
    config :
        Optional dict; reads ``micro.echo.*`` and ``micro.*`` keys.
    height_fn :
        ``(x: float, z: float) -> float`` terrain height query.
    """

    _DEFAULT_SKY_RADIUS  = 30.0
    _DEFAULT_NUM_SAMPLES = 8
    _DEFAULT_REVERB_GAIN = 0.8   # max extra reverb mix at full enclosure

    def __init__(
        self,
        config:    Optional[dict] = None,
        height_fn: Optional[HeightFn] = None,
    ) -> None:
        cfg  = config or {}
        mcfg = cfg.get("micro", {}) or {}
        ecfg = mcfg.get("echo", {}) or {}

        self._radius:      float = float(mcfg.get("sky_radius",    self._DEFAULT_SKY_RADIUS))
        self._n:           int   = int(mcfg.get("sky_num_samples", self._DEFAULT_NUM_SAMPLES))
        self._reverb_gain: float = float(ecfg.get("reverb_gain",   self._DEFAULT_REVERB_GAIN))
        self._height_fn: HeightFn = height_fn or (lambda x, z: 0.0)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def estimate(self, pos: Vec3) -> float:
        """Compute echoPotential proxy at *pos*.

        Returns
        -------
        float
            echoPotential in [0, 1].  1.0 = maximally reflective enclosure.
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
            if self._height_fn(sx, sz) > centre_h:
                blocked += 1
        return _clamp(blocked / self._n)

    def reverb_gain(self, echo_potential: float) -> float:
        """Map echo potential to a reverb mix multiplier.

        Returns
        -------
        float
            Additive reverb-mix gain in [0, reverb_gain_max].
        """
        return _clamp(echo_potential) * self._reverb_gain
