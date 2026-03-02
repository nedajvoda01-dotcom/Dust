"""WindLoad — Stage 37 wind-force perception field.

Produces three scalars describing aerodynamic load on the character:

* ``windLoad``   (0..1) — normalised wind pressure on body
* ``windDir``    (Vec3, unit) — direction wind is coming from
* ``gustiness``  (0..1) — variability / unpredictability of wind

Inputs:
* ``wind_vec``         — 3-D wind velocity vector [m/s or normalised]
* ``shelter_factor``   — 0..1 local shelter (0 = fully exposed, 1 = sheltered)
* ``dust_wall_near``   — 0..1 proximity to a dust-wall front (Stage 32)

Public API
----------
WindLoadField(config=None)
  .update(wind_vec, shelter_factor, dust_wall_near, dt) → None
  .wind_load   → float
  .wind_dir    → Vec3
  .gustiness   → float
"""
from __future__ import annotations

import math
from typing import Optional

from src.math.Vec3 import Vec3


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


class WindLoadField:
    """Perception sub-field: wind force and gustiness.

    Parameters
    ----------
    config :
        Optional dict; reads ``perception.wind.*`` keys.
    """

    _DEFAULT_LOAD_K         = 1.0
    _DEFAULT_SMOOTHING_TAU  = 0.20  # seconds
    # Wind speed that produces load = 1.0 (before shelter)
    _WIND_FULL_LOAD_SPEED   = 25.0  # m/s

    def __init__(self, config: Optional[dict] = None) -> None:
        pcfg = ((config or {}).get("perception", {}) or {}).get("wind", {}) or {}
        self._load_k: float = float(pcfg.get("load_k", self._DEFAULT_LOAD_K))
        tau = float(
            ((config or {}).get("perception", {}) or {}).get(
                "smoothing_tau_sec", self._DEFAULT_SMOOTHING_TAU
            )
        )
        self._tau: float = max(1e-3, tau)

        self._load:     float = 0.0
        self._gustiness: float = 0.0
        self._dir:      Vec3  = Vec3(0.0, 0.0, 0.0)
        self._prev_speed: float = 0.0

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def update(
        self,
        wind_vec:       Vec3  = None,
        shelter_factor: float = 0.0,
        dust_wall_near: float = 0.0,
        dt:             float = 1.0 / 10.0,
    ) -> None:
        """Advance wind load perception one tick.

        Parameters
        ----------
        wind_vec :
            3-D wind velocity vector (world space).  If None, treated as zero.
        shelter_factor :
            0 = fully exposed, 1 = completely sheltered (cave / behind ridge).
        dust_wall_near :
            0..1 proximity to a dust-wall front; adds to effective load.
        dt :
            Elapsed simulation time [s].
        """
        if wind_vec is None:
            wind_vec = Vec3(0.0, 0.0, 0.0)

        shelter_factor  = _clamp(shelter_factor,  0.0, 1.0)
        dust_wall_near  = _clamp(dust_wall_near,  0.0, 1.0)

        speed = wind_vec.length()

        # Gustiness: rate of change of speed (smoothed)
        d_speed = abs(speed - self._prev_speed)
        raw_gustiness = _clamp(d_speed / max(1.0, speed + 1.0), 0.0, 1.0)
        self._prev_speed = speed

        # Wind load normalised against full-load speed
        raw_load = _clamp(
            speed / self._WIND_FULL_LOAD_SPEED * self._load_k
            + dust_wall_near * 0.3,
            0.0, 1.0,
        )
        # Shelter reduces load
        raw_load *= 1.0 - shelter_factor

        # Direction: FROM which direction wind blows
        if speed > 1e-6:
            raw_dir = wind_vec * (-1.0 / speed)  # into-wind direction
        else:
            raw_dir = Vec3(0.0, 0.0, 0.0)

        # Exponential smoothing
        alpha = 1.0 - math.exp(-dt / self._tau)
        self._load     = self._load     + alpha * (raw_load     - self._load)
        self._gustiness = self._gustiness + alpha * (raw_gustiness - self._gustiness)

        prev_len = self._dir.length()
        if prev_len < 1e-6:
            self._dir = raw_dir
        else:
            blended = self._dir * (1.0 - alpha) + raw_dir * alpha
            bl = blended.length()
            self._dir = blended * (1.0 / bl) if bl > 1e-6 else raw_dir

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------

    @property
    def wind_load(self) -> float:
        """Normalised aerodynamic load on character [0..1]."""
        return self._load

    @property
    def wind_dir(self) -> Vec3:
        """Unit vector pointing INTO the wind (from source)."""
        return self._dir

    @property
    def gustiness(self) -> float:
        """Wind variability / unpredictability [0..1]."""
        return self._gustiness
