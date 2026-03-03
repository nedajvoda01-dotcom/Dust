"""DustTrapEstimator — Stage 49 dust deposition proxy.

Estimates how prone a location is to accumulating wind-borne dust:

* Sheltered areas (behind obstacles) trap dust → dustTrap↑
* Concave terrain (valleys, basins) trap dust → dustTrap↑
* Strongly channelled areas disperse dust → dustTrap↓

The value is used by :class:`~src.microclimate.LocalClimateComposer.LocalClimateComposer`
to offset local dust density above the macro level.

Public API
----------
DustTrapEstimator(config=None, height_fn=None)
  .estimate(pos, shelter, wind_channel) → float   (dustTrap 0..1)
"""
from __future__ import annotations

import math
from typing import Callable, Optional

from src.math.Vec3 import Vec3


HeightFn = Callable[[float, float], float]


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


class DustTrapEstimator:
    """Computes dust-trapping tendency from shelter and terrain concavity.

    Parameters
    ----------
    config :
        Optional dict; reads ``micro.dusttrap.*`` keys.
    height_fn :
        ``(x: float, z: float) -> float`` terrain height query.
    """

    _DEFAULT_SAMPLE_RADIUS    = 20.0
    _DEFAULT_NUM_SAMPLES      = 8
    _DEFAULT_DEPOSITION_BOOST = 0.4

    def __init__(
        self,
        config:    Optional[dict] = None,
        height_fn: Optional[HeightFn] = None,
    ) -> None:
        cfg  = config or {}
        mcfg = cfg.get("micro", {}) or {}
        dcfg = mcfg.get("dusttrap", {}) or {}

        self._radius: float = float(dcfg.get("sample_radius",    self._DEFAULT_SAMPLE_RADIUS))
        self._n:      int   = int(dcfg.get("num_samples",        self._DEFAULT_NUM_SAMPLES))
        self._boost:  float = float(dcfg.get("deposition_boost", self._DEFAULT_DEPOSITION_BOOST))
        self._height_fn: HeightFn = height_fn or (lambda x, z: 0.0)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def estimate(self, pos: Vec3, shelter: float, wind_channel: float) -> float:
        """Compute dustTrap proxy.

        Parameters
        ----------
        pos :
            World-space query position.
        shelter :
            windShelter value [0..1] from :class:`~src.microclimate.ShelterEstimator`.
        wind_channel :
            windChannel value [0..1] from :class:`~src.microclimate.ChannelEstimator`.

        Returns
        -------
        float
            dustTrap in [0, 1].
        """
        concavity = self._concavity(pos)
        # Shelter and concavity both increase trapping; channeling decreases it
        raw = _clamp(shelter * 0.5 + concavity * 0.5) - wind_channel * 0.4
        return _clamp(raw)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _concavity(self, pos: Vec3) -> float:
        """Fraction of surrounding samples higher than the centre point."""
        centre_h = self._height_fn(pos.x, pos.z)
        if self._n == 0:
            return 0.0
        step = (2.0 * math.pi) / self._n
        higher = 0
        for i in range(self._n):
            angle = i * step
            sx = pos.x + self._radius * math.cos(angle)
            sz = pos.z + self._radius * math.sin(angle)
            if self._height_fn(sx, sz) > centre_h:
                higher += 1
        return higher / self._n
