"""FieldSet — procedural global fields: temperature, wind, dust.

Fields are functions of (lat_rad, lon_rad, t_sec) — no grid storage
at this stage.  The ``sample`` method converts a world-space point to
latitude/longitude and evaluates each field analytically.

This architecture allows a real grid to be swapped in later without
changing callers.

Public API
----------
FieldSet(planet_radius, seed)
  .sample(x, y, z, t)      → FieldSample(temp, wind_x, wind_y, wind_z, dust)
  .fields_revision          → int   bumped when generator params change
  .to_snapshot_dict()       → dict  serialisable generator parameters
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict


# ---------------------------------------------------------------------------
# FieldSample
# ---------------------------------------------------------------------------

@dataclass
class FieldSample:
    """Values of all fields at one world-space point and time."""

    temp:   float   # normalised temperature [0, 1]  (0 = cold, 1 = hot)
    wind_x: float   # wind vector x (simulation units/sec)
    wind_y: float   # wind vector y
    wind_z: float   # wind vector z
    dust:   float   # dust density [0, 1]


# ---------------------------------------------------------------------------
# FieldSet
# ---------------------------------------------------------------------------

class FieldSet:
    """Procedural field generator.

    Parameters
    ----------
    planet_radius :
        Used to convert world-space coordinates to lat/lon.
    seed :
        Deterministic noise seed.
    """

    def __init__(self, planet_radius: float = 1000.0, seed: int = 42) -> None:
        self._planet_radius  = float(planet_radius)
        self._seed           = int(seed)
        self._fields_revision: int = 0

        # Tunable generator parameters (allowlist for admin tuning)
        self.temp_base:       float = 0.5    # mean temperature
        self.temp_amplitude:  float = 0.4    # lat-dependent range
        self.wind_speed:      float = 8.0    # max wind speed (sim units/sec)
        self.wind_period_sec: float = 3600.0 # temporal wind cycle period
        self.dust_base:       float = 0.1
        self.dust_amplitude:  float = 0.3

    @property
    def fields_revision(self) -> int:
        return self._fields_revision

    def bump_revision(self) -> None:
        self._fields_revision += 1

    # ------------------------------------------------------------------
    def sample(self, x: float, y: float, z: float, t: float) -> FieldSample:
        """Evaluate all fields at world-space point (x, y, z) and sim time t.

        Coordinate system: planet centre is the origin.
        """
        r = math.sqrt(x * x + y * y + z * z)
        if r < 1e-6:
            return FieldSample(
                temp=self.temp_base, wind_x=0.0, wind_y=0.0, wind_z=0.0,
                dust=self.dust_base,
            )

        # Latitude: angle from equatorial plane  (-π/2 .. +π/2)
        lat = math.asin(max(-1.0, min(1.0, y / r)))
        # Longitude: angle in xz plane
        lon = math.atan2(x, z)

        # Temperature: hot at equator, cold at poles + slow temporal drift
        lat_factor = math.cos(lat)  # 1 at equator, 0 at poles
        slow_drift = math.sin(2.0 * math.pi * t / max(1.0, self.wind_period_sec) * 0.1)
        temp = self.temp_base + self.temp_amplitude * (lat_factor - 0.5 + slow_drift * 0.05)
        temp = max(0.0, min(1.0, temp))

        # Wind: rotational flow around y-axis + temporal oscillation
        wind_phase = 2.0 * math.pi * t / max(1.0, self.wind_period_sec)
        ws = self.wind_speed * (0.5 + 0.5 * math.sin(wind_phase + lat))
        # Approximate zonal flow: perpendicular to the meridian
        wind_x = -ws * math.sin(lon + wind_phase * 0.1)
        wind_z =  ws * math.cos(lon + wind_phase * 0.1)
        wind_y =  ws * 0.1 * math.sin(lat * 3.0)

        # Dust: more in tropical band + temporal variation
        dust_factor = math.pow(lat_factor, 2.0)
        dust = self.dust_base + self.dust_amplitude * dust_factor * (
            0.5 + 0.5 * math.sin(wind_phase * 2.0 + lon)
        )
        dust = max(0.0, min(1.0, dust))

        return FieldSample(
            temp=temp,
            wind_x=wind_x,
            wind_y=wind_y,
            wind_z=wind_z,
            dust=dust,
        )

    # ------------------------------------------------------------------
    def to_snapshot_dict(self) -> Dict[str, Any]:
        """Serialise generator parameters for FIELDS_SNAPSHOT."""
        return {
            "fields_revision":  self._fields_revision,
            "planet_radius":    self._planet_radius,
            "seed":             self._seed,
            "temp_base":        self.temp_base,
            "temp_amplitude":   self.temp_amplitude,
            "wind_speed":       self.wind_speed,
            "wind_period_sec":  self.wind_period_sec,
            "dust_base":        self.dust_base,
            "dust_amplitude":   self.dust_amplitude,
        }
