"""PlanetTimeSystem — Stage 7 local solar time and day/night.

Computes local solar time on the planet surface and provides day/night
classification via N·L checks against the binary sun directions.

API
---
get_spin_angle(t)             -> float  (radians, unbounded)
get_lat_long(world_pos)       -> LatLong
get_local_solar_time_01(world_pos, t) -> float  (0..1, 0.5 ≈ noon reference)
is_day(world_normal)          -> bool
is_night(world_normal)        -> bool
"""
from __future__ import annotations

import math

from src.core.Config import Config
from src.math.PlanetMath import PlanetMath, LatLong
from src.math.Vec3 import Vec3
from src.systems.AstroSystem import AstroSystem

_TWO_PI = 2.0 * math.pi


class PlanetTimeSystem:
    """Local solar time and day/night state for the planet surface."""

    def __init__(self, config: Config, astro: AstroSystem) -> None:
        self._day_len_s: float = config.get("day", "length_minutes", default=90) * 60.0
        self._astro = astro

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_spin_angle(self, t: float) -> float:
        """Planet spin angle at game time *t* (radians, unbounded)."""
        return _TWO_PI * t / self._day_len_s

    def get_lat_long(self, world_pos: Vec3) -> LatLong:
        """Geodetic lat/long for a planet-frame direction / world-space position."""
        return PlanetMath.from_direction(world_pos)

    def get_local_solar_time_01(self, world_pos: Vec3, t: float) -> float:
        """Return local solar time in [0, 1) for a surface position.

        Computed geometrically from longitude + accumulated spin angle.
        The cycle length is exactly one planet day; 0.5 is the solar-noon
        reference longitude at *t* = 0 for positions facing the primary sun.

        Parameters
        ----------
        world_pos:
            Any planet-frame or world-space direction from the planet centre
            (does not need to be on the surface).
        t:
            Unscaled game time in seconds.
        """
        ll = PlanetMath.from_direction(world_pos)
        lon_rot = ll.lon_rad + self.get_spin_angle(t)
        raw = lon_rot / _TWO_PI
        return raw - math.floor(raw)   # frac → [0, 1)

    def is_day(self, world_normal: Vec3) -> bool:
        """True if either sun is above the horizon (N·L > 0)."""
        d1, d2 = self._astro.get_sun_directions()
        return world_normal.dot(d1) > 0.0 or world_normal.dot(d2) > 0.0

    def is_night(self, world_normal: Vec3) -> bool:
        """True if both suns are below the horizon (N·L ≤ 0 for each)."""
        return not self.is_day(world_normal)
