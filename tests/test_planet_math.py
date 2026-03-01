"""test_planet_math — validates PlanetMath correctness."""
import math
import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.math.Vec3 import Vec3
from src.math.PlanetMath import PlanetMath, LatLong

EPS = 1e-6


class TestUpAtPosition(unittest.TestCase):
    def _check_up(self, x, y, z):
        pos = Vec3(x, y, z)
        up = PlanetMath.up_at_position(pos)
        # Must be unit length
        self.assertAlmostEqual(up.length(), 1.0, places=6, msg=f"up not unit at {pos}")
        # Must point away from center
        dot = up.dot(pos.normalized())
        self.assertAlmostEqual(dot, 1.0, places=6, msg=f"up not aligned at {pos}")

    def test_up_north_pole(self):
        self._check_up(0, 1000, 0)

    def test_up_equator_x(self):
        self._check_up(1000, 0, 0)

    def test_up_equator_z(self):
        self._check_up(0, 0, 1000)

    def test_up_south_pole(self):
        self._check_up(0, -1000, 0)

    def test_up_diagonal(self):
        self._check_up(577, 577, 577)

    def test_down_is_neg_up(self):
        pos = Vec3(300, 800, 200)
        up = PlanetMath.up_at_position(pos)
        down = PlanetMath.down_at_position(pos)
        self.assertAlmostEqual((up + down).length(), 0.0, places=6)


class TestLatLong(unittest.TestCase):
    def _round_trip(self, lat, lon):
        ll = LatLong(lat_rad=lat, lon_rad=lon)
        d = PlanetMath.direction_from_lat_long(ll)
        self.assertAlmostEqual(d.length(), 1.0, places=6, msg="direction not unit")
        ll2 = PlanetMath.from_direction(d)
        self.assertAlmostEqual(ll2.lat_rad, lat, places=5, msg=f"lat mismatch lat={lat} lon={lon}")
        self.assertAlmostEqual(ll2.lon_rad, lon, places=5, msg=f"lon mismatch lat={lat} lon={lon}")

    def test_north_pole(self):
        self._round_trip(math.pi / 2 - 1e-7, 0.0)

    def test_south_pole(self):
        self._round_trip(-math.pi / 2 + 1e-7, 0.0)

    def test_equator(self):
        self._round_trip(0.0, 0.0)

    def test_equator_east(self):
        self._round_trip(0.0, math.pi / 2)

    def test_mid_lat(self):
        self._round_trip(math.radians(45), math.radians(90))

    def test_negative_lon(self):
        self._round_trip(math.radians(-30), math.radians(-120))


class TestMoveAlongSurface(unittest.TestCase):
    def test_radius_preserved(self):
        """After movement, position stays on unit sphere."""
        unit_dir = Vec3(0.0, 1.0, 0.0)  # north pole
        tangent = Vec3(1.0, 0.0, 0.0)   # east
        # Walk a large arc (2*pi = full circle)
        step = 0.1
        pos = unit_dir
        for _ in range(63):  # ~2pi radians total
            pos = PlanetMath.move_along_surface(pos, tangent, step)
            length_err = abs(pos.length() - 1.0)
            self.assertLess(length_err, 1e-6, f"radius drift: {pos.length()}")

    def test_full_circle_returns_to_start(self):
        """Walking a full great circle should return near the start."""
        # Start on equator; use fixed rotation axis to derive tangent at each step
        # so tangent is always perpendicular to both the position and the axis.
        pos = Vec3(1.0, 0.0, 0.0)
        rotation_axis = Vec3(0.0, 1.0, 0.0)  # equatorial great circle
        n_steps = 360
        arc = 2.0 * math.pi / n_steps
        start = Vec3(pos.x, pos.y, pos.z)
        for _ in range(n_steps):
            # Tangent = axis × pos: always perpendicular to pos, stays in tangent plane
            tangent = rotation_axis.cross(pos)
            pos = PlanetMath.move_along_surface(pos, tangent, arc)
        dist = (pos - start).length()
        self.assertLess(dist, 1e-4, f"circle not closed: dist={dist}")

    def test_tangent_orthogonal_to_up(self):
        """Tangent projection must be orthogonal to up."""
        unit_dir = Vec3(0.5, 0.5, 0.707).normalized()
        desired = Vec3(0.3, 0.8, 0.1)
        tangent = PlanetMath.tangent_forward(unit_dir, desired)
        up = unit_dir
        dot = abs(tangent.dot(up))
        self.assertLess(dot, 1e-6, f"tangent not orthogonal to up: dot={dot}")

    def test_orientation_not_broken_after_long_travel(self):
        """
        After traversing a large portion of the sphere, up should still be correct.
        Checks that up/down orientation doesn't degrade.
        """
        unit_dir = Vec3(1.0, 0.0, 0.0)
        tangent = Vec3(0.0, 1.0, 0.0)
        arc = 0.05
        for _ in range(130):  # > 2pi total
            unit_dir = PlanetMath.move_along_surface(unit_dir, tangent, arc)
            up = PlanetMath.up_at_position(unit_dir * 1000.0)
            self.assertAlmostEqual(up.length(), 1.0, places=5)
            # up must still point from center outward
            dot = up.dot(unit_dir)
            self.assertGreater(dot, 0.9999)


if __name__ == "__main__":
    unittest.main()
