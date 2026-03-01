"""test_astro — Stage 6 AstroSystem tests.

Tests
-----
1. TestBinaryPeriod      — after one full binary period, sun positions repeat
2. TestEclipseResponse   — at closest approach, eclipseFactor>0 and totalDirect drops
3. TestRingShadowHit     — ray through ring returns shadow~1; ray outside returns 0
4. TestDayNightCycle     — equatorial point sees max insolation then 0 over one day
"""
from __future__ import annotations

import math
import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.core.Config import Config
from src.math.Vec3 import Vec3
from src.systems.AstroSystem import AstroSystem, InsolationSample


def _make_astro() -> AstroSystem:
    """Create AstroSystem with default config."""
    return AstroSystem(Config())


def _rotate_y(v: Vec3, angle: float) -> Vec3:
    """Rotate Vec3 around Y-axis by angle (radians)."""
    c, s = math.cos(angle), math.sin(angle)
    return Vec3(v.x * c + v.z * s, v.y, -v.x * s + v.z * c)


# ---------------------------------------------------------------------------
# TestBinaryPeriod
# ---------------------------------------------------------------------------

class TestBinaryPeriod(unittest.TestCase):
    """After exactly one binary period, sun positions must return to their start."""

    def test_sun1_position_repeats(self) -> None:
        astro = _make_astro()
        astro.update(0.0)
        p1_start = astro.sun1_world_pos

        period_s = astro._binary_period_s  # noqa: SLF001
        astro.update(period_s / astro._time_scale)  # noqa: SLF001
        p1_end = astro.sun1_world_pos

        dist = (p1_end - p1_start).length()
        self.assertAlmostEqual(dist, 0.0, delta=1e-4,
                               msg=f"sun1 position not periodic: Δ={dist:.6f}")

    def test_sun2_position_repeats(self) -> None:
        astro = _make_astro()
        astro.update(0.0)
        p2_start = astro.sun2_world_pos

        period_s = astro._binary_period_s  # noqa: SLF001
        astro.update(period_s / astro._time_scale)  # noqa: SLF001
        p2_end = astro.sun2_world_pos

        dist = (p2_end - p2_start).length()
        self.assertAlmostEqual(dist, 0.0, delta=1e-4,
                               msg=f"sun2 position not periodic: Δ={dist:.6f}")

    def test_sun_directions_repeat(self) -> None:
        astro = _make_astro()
        astro.update(0.0)
        d1_start, d2_start = astro.get_sun_directions()

        period_s = astro._binary_period_s  # noqa: SLF001
        astro.update(period_s / astro._time_scale)  # noqa: SLF001
        d1_end, d2_end = astro.get_sun_directions()

        self.assertAlmostEqual((d1_end - d1_start).length(), 0.0, delta=1e-5)
        self.assertAlmostEqual((d2_end - d2_start).length(), 0.0, delta=1e-5)


# ---------------------------------------------------------------------------
# TestEclipseResponse
# ---------------------------------------------------------------------------

class TestEclipseResponse(unittest.TestCase):
    """Eclipse geometry: find minimum angular separation; eclipseFactor>0; insolation drops."""

    def _find_min_theta_time(self, astro: AstroSystem) -> tuple[float, float]:
        """Scan one binary period to find the time of minimum sun-sun angle."""
        period_s = astro._binary_period_s  # noqa: SLF001
        steps = 2000
        best_theta = math.pi
        best_t = 0.0
        for i in range(steps + 1):
            t = period_s * i / steps / astro._time_scale  # noqa: SLF001
            astro.update(t)
            d1, d2 = astro.get_sun_directions()
            cos_t = max(-1.0, min(1.0, d1.dot(d2)))
            theta = math.acos(cos_t)
            if theta < best_theta:
                best_theta = theta
                best_t = t
        return best_t, best_theta

    def test_eclipse_factor_positive_at_conjunction(self) -> None:
        astro = _make_astro()
        best_t, _ = self._find_min_theta_time(astro)
        astro.update(best_t)
        ef = astro.get_eclipse_factor()
        self.assertGreater(ef, 0.0,
                           msg="eclipseFactor should be > 0 at closest angular approach")

    def test_insolation_drops_during_eclipse(self) -> None:
        """totalDirect at eclipse should be less than the peak (uneclipsed) value."""
        astro = _make_astro()

        # At t=0 both suns are aligned along +Z (eclipse).
        # Use a surface normal facing +Z to maximise the contrast.
        ref_normal = Vec3(0.0, 0.0, 1.0)
        ref_pos = ref_normal * 500.0  # small offset from centre, well inside ring

        # Sample away from eclipse (T/4 → maximum angular separation, no eclipse)
        period_s = astro._binary_period_s  # noqa: SLF001
        astro.update(period_s * 0.25 / astro._time_scale)  # noqa: SLF001
        sample_away = astro.sample_insolation(ref_pos, ref_normal)

        # Eclipse: t=0, both suns in +Z, eclipse_factor ≈ 1
        astro.update(0.0)
        sample_eclipse = astro.sample_insolation(ref_pos, ref_normal)

        self.assertGreater(sample_away.total_direct, 0.0,
                           msg="Away from eclipse, totalDirect should be positive")
        self.assertLess(
            sample_eclipse.total_direct,
            sample_away.total_direct,
            msg=(
                f"Eclipse should reduce totalDirect. "
                f"eclipse={sample_eclipse.total_direct:.4f}  "
                f"no_eclipse={sample_away.total_direct:.4f}"
            ),
        )

    def test_eclipse_factor_below_threshold_away_from_conjunction(self) -> None:
        """Far from conjunction (T/4) the eclipse factor should be zero."""
        astro = _make_astro()
        period_s = astro._binary_period_s  # noqa: SLF001
        # At quarter-period the two suns are at maximum angular separation
        astro.update(period_s * 0.25 / astro._time_scale)  # noqa: SLF001
        ef = astro.get_eclipse_factor()
        self.assertAlmostEqual(ef, 0.0, delta=0.05,
                               msg=f"eclipseFactor should be ~0 at quarter-period; got {ef:.4f}")


# ---------------------------------------------------------------------------
# TestRingShadowHit
# ---------------------------------------------------------------------------

class TestRingShadowHit(unittest.TestCase):
    """Ring shadow geometry: point under ring strip → shadow; outside → clear."""

    def setUp(self) -> None:
        self.astro = _make_astro()
        self.astro.update(0.0)

    def _surface_point(self, normal: Vec3) -> Vec3:
        """Return a surface point just outside the planet along normal."""
        planet_r: float = 1000.0  # config default
        return normal.normalized() * (planet_r * 1.01)

    def test_point_in_ring_shadow(self) -> None:
        """A surface point below the ring plane should be shadowed when the ring
        is between it and the sun.

        Geometry (analytically verified):
          Ring normal N = (0, cos14°, sin14°).
          At t=0 sun1 is at (0,0,large_D+r1), so sun1_dir ≈ (0, 0, 1).
          Surface point P = (0, -sin22°, -cos22°) * planet_r is below the ring
          plane (P·N < 0).  The ray P + t*(0,0,1) intersects the ring plane at
          r ≈ 1549, which lies in [ring_inner=1400, ring_outer=2100].
        """
        astro = self.astro
        planet_r = 1000.0
        # Surface point: below the ring plane
        angle_below = math.radians(22.0)
        pos_shadow = Vec3(0.0,
                          -math.sin(angle_below) * planet_r,
                          -math.cos(angle_below) * planet_r)
        # At t=0 sun1 direction is +Z (suns aligned along planet→barycenter axis)
        sun_dir = astro._sun1_dir  # noqa: SLF001
        shadow = astro.get_ring_shadow_for_sun(pos_shadow, sun_dir)
        self.assertGreater(shadow, 0.5,
                           msg=f"Point below ring plane should be in ring shadow; got {shadow:.4f}")

    def test_point_above_ring_plane_not_shadowed(self) -> None:
        """A surface point on the same side as the sun is never in ring shadow."""
        astro = self.astro
        planet_r = 1000.0
        # Surface point: above the ring plane (mirrored from shadow test)
        angle_above = math.radians(22.0)
        pos_no_shadow = Vec3(0.0,
                             math.sin(angle_above) * planet_r,
                             math.cos(angle_above) * planet_r)
        sun_dir = astro._sun1_dir  # noqa: SLF001
        # The point is on the same side as the sun (P·N > 0, sun·N > 0),
        # so t_hit < 0 and there is no ring between point and sun.
        shadow = astro.get_ring_shadow_for_sun(pos_no_shadow, sun_dir)
        self.assertAlmostEqual(shadow, 0.0, delta=0.05,
                               msg=f"Point above ring plane should not be shadowed; got {shadow:.4f}")

    def test_ring_shadow_factor_interface(self) -> None:
        """get_ring_shadow_factor uses sun1 direction and returns a float in [0,1]."""
        astro = self.astro
        pos = Vec3(500.0, 0.0, 0.0)
        shadow = astro.get_ring_shadow_factor(pos)
        self.assertGreaterEqual(shadow, 0.0)
        self.assertLessEqual(shadow, 1.0)


# ---------------------------------------------------------------------------
# TestDayNightCycle
# ---------------------------------------------------------------------------

class TestDayNightCycle(unittest.TestCase):
    """Over one full day, an equatorial point must pass through both day and night."""

    def test_day_and_night_occur(self) -> None:
        astro = _make_astro()
        day_len_s = astro._day_len_s   # noqa: SLF001
        planet_r = 1000.0

        # Fixed local surface normal on the equator (local +X)
        local_normal = Vec3(1.0, 0.0, 0.0)
        local_pos_scale = planet_r * 1.01

        steps = 360
        max_total = 0.0
        min_total = float("inf")

        for i in range(steps):
            t = day_len_s * i / steps / astro._time_scale  # noqa: SLF001
            astro.update(t)
            spin = astro.spin_angle
            # Rotate local normal and pos into world space by spin angle around Y
            world_normal = _rotate_y(local_normal, spin)
            world_pos = _rotate_y(local_normal * local_pos_scale, spin)
            s = astro.sample_insolation(world_pos, world_normal)
            if s.total_direct > max_total:
                max_total = s.total_direct
            if s.total_direct < min_total:
                min_total = s.total_direct

        self.assertGreater(max_total, 0.1,
                           msg=f"No daytime found; max totalDirect={max_total:.4f}")
        self.assertAlmostEqual(min_total, 0.0, delta=0.05,
                               msg=f"No nighttime found; min totalDirect={min_total:.4f}")

    def test_insolation_returns_insolation_sample(self) -> None:
        """sample_insolation must return an InsolationSample with expected fields."""
        astro = _make_astro()
        astro.update(0.0)
        s = astro.sample_insolation(Vec3(0.0, 1000.0, 0.0), Vec3(0.0, 1.0, 0.0))
        self.assertIsInstance(s, InsolationSample)
        self.assertGreaterEqual(s.total_direct, 0.0)
        self.assertGreaterEqual(s.ring_shadow, 0.0)
        self.assertLessEqual(s.ring_shadow, 1.0)
        self.assertGreaterEqual(s.eclipse_factor, 0.0)
        self.assertLessEqual(s.eclipse_factor, 1.0)
        self.assertIsInstance(s.dir1, Vec3)
        self.assertIsInstance(s.dir2, Vec3)


if __name__ == "__main__":
    unittest.main()
