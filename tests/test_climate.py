"""test_climate — validates Stage 8 ClimateSystem.

Tests:
  1. TestClimateDeterminism     — same seed → bit-identical fields after N steps
  2. TestTemperatureResponse    — insolation drives temperature; night cools
  3. TestDustAdvectionStability — no NaN/inf; global mass within tolerance
  4. TestStormBirthConditions   — storms appear under high dust+wind; not under calm
  5. TestClimateAPIBounds       — sample_wind, sample_dust, get_visibility, etc. bounded
  6. TestWindLatLongToWorld     — static utility produces correct tangent vectors
"""
from __future__ import annotations

import math
import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.core.Config import Config
from src.math.Vec3 import Vec3
from src.systems.ClimateSystem import ClimateSystem, StormCell


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_climate(seed: int = 42, w: int = 32, h: int = 16) -> ClimateSystem:
    """Build a small ClimateSystem for fast unit tests."""
    cfg = Config.__new__(Config)
    cfg._data = {
        "planet":  {"radius_units": 1000.0},
        "climate": {
            "grid_w": w,
            "grid_h": h,
            "temp_base_equator":    290.0,
            "temp_lat_k":            60.0,
            "heat_gain":             30.0,
            "heat_loss":              0.1,
            "pressure_from_temp":     0.1,
            "wind_base_strength":     4.5,
            "wind_gradP_strength":    0.5,
            "wind_max":              40.0,
            "wind_damping":           0.02,
            "dust_deposit_rate":      0.05,
            "dust_lift_rate":         0.02,
            "dust_lift_threshold":    8.0,
            "storm_dust_threshold":   0.6,
            "storm_wind_threshold":  15.0,
            "visibility_k":           5.0,
            "freeze_threshold":     270.0,
            "melt_threshold":       280.0,
        },
    }
    return ClimateSystem(cfg, seed=seed, width=w, height=h)


class _ConstantInsolation:
    """Fake InsolationField that returns a fixed direct_total everywhere."""

    def __init__(self, value: float) -> None:
        self._value = value

    def sample_at(self, world_pos: Vec3):
        from src.systems.InsolationField import InsolCell
        return InsolCell(
            direct1=self._value,
            direct2=0.0,
            direct_total=self._value,
            ring_shadow_eff=0.0,
            eclipse_eff=0.0,
        )


# ---------------------------------------------------------------------------
# 1. TestClimateDeterminism
# ---------------------------------------------------------------------------

class TestClimateDeterminism(unittest.TestCase):
    """Same seed, same steps → bit-identical fields."""

    _STEPS = 20

    def _run(self, seed: int) -> ClimateSystem:
        c = _make_climate(seed=seed)
        for _ in range(self._STEPS):
            c.update(1.0)
        return c

    def test_same_seed_same_temp(self):
        c1 = self._run(42)
        c2 = self._run(42)
        for a, b in zip(c1._temp, c2._temp):
            self.assertAlmostEqual(a, b, places=12)

    def test_same_seed_same_dust(self):
        c1 = self._run(42)
        c2 = self._run(42)
        for a, b in zip(c1._dust, c2._dust):
            self.assertAlmostEqual(a, b, places=12)

    def test_same_seed_same_wind(self):
        c1 = self._run(42)
        c2 = self._run(42)
        for a, b in zip(c1._wind_u, c2._wind_u):
            self.assertAlmostEqual(a, b, places=12)

    def test_different_seed_different_dust(self):
        c1 = self._run(42)
        c2 = self._run(999)
        # Initial dust is seeded; at least one cell must differ
        any_diff = any(abs(a - b) > 1e-9 for a, b in zip(c1._dust, c2._dust))
        self.assertTrue(any_diff, "Different seeds should produce different dust fields")


# ---------------------------------------------------------------------------
# 2. TestTemperatureResponse
# ---------------------------------------------------------------------------

class TestTemperatureResponse(unittest.TestCase):
    """Temperature responds to insolation changes."""

    def test_high_insolation_raises_temperature(self):
        """With strong insolation, equatorial temperature should increase."""
        c = _make_climate()
        eq_dir = Vec3(0.0, 0.0, 1.0)   # lat=0, lon=0

        t_before = c.sample_temperature(eq_dir)
        high_sol  = _ConstantInsolation(1.0)
        for _ in range(50):
            c.update(1.0, insolation=high_sol)
        t_after = c.sample_temperature(eq_dir)
        self.assertGreater(t_after, t_before,
            msg=f"Temperature should rise under high insolation: {t_before:.2f} → {t_after:.2f}")

    def test_zero_insolation_cools_below_equatorial_base(self):
        """Without insolation, cells that started above T_base should cool."""
        c = _make_climate()
        # Force a high initial temperature in all cells
        for idx in range(c._n):
            c._temp[idx] = 400.0

        zero_sol = _ConstantInsolation(0.0)
        for _ in range(100):
            c.update(1.0, insolation=zero_sol)

        # Every cell should have dropped from 400 K
        max_temp = max(c._temp)
        self.assertLess(max_temp, 400.0,
            msg=f"Temperature should drop with zero insolation; max={max_temp:.2f}")

    def test_temperature_clamped(self):
        """Temperature must stay within [-150, 500] K regardless of input."""
        c = _make_climate()
        extreme_sol = _ConstantInsolation(100.0)   # absurdly high insolation
        for _ in range(500):
            c.update(1.0, insolation=extreme_sol)
        for t in c._temp:
            self.assertGreaterEqual(t, -150.0)
            self.assertLessEqual(t, 500.0)

    def test_polar_cooler_than_equator(self):
        """After equilibration, poles should be colder than equator."""
        c = _make_climate()
        for _ in range(100):
            c.update(1.0)
        eq_t  = c.sample_temperature(Vec3(0.0, 0.0, 1.0))     # lat=0
        pole_t = c.sample_temperature(Vec3(0.0, 1.0, 0.0))     # lat=+90
        self.assertGreater(eq_t, pole_t,
            msg=f"Equator ({eq_t:.2f} K) should be warmer than pole ({pole_t:.2f} K)")


# ---------------------------------------------------------------------------
# 3. TestDustAdvectionStability
# ---------------------------------------------------------------------------

class TestDustAdvectionStability(unittest.TestCase):
    """Dust advection must be stable: no NaN/inf and near-conserved mass."""

    _STEPS = 100
    _MASS_TOLERANCE = 0.50   # allow up to 50% global mass change (lifting/deposition)

    def _total_dust(self, c: ClimateSystem) -> float:
        return sum(c._dust)

    def test_no_nan_inf(self):
        c = _make_climate()
        for step in range(self._STEPS):
            c.update(1.0)
        for idx, d in enumerate(c._dust):
            self.assertFalse(math.isnan(d), f"NaN in dust at cell {idx}")
            self.assertFalse(math.isinf(d), f"Inf in dust at cell {idx}")

    def test_dust_within_unit_range(self):
        c = _make_climate()
        for _ in range(self._STEPS):
            c.update(1.0)
        for d in c._dust:
            self.assertGreaterEqual(d, 0.0, "Dust must be non-negative")
            self.assertLessEqual(d,   1.0, "Dust must not exceed 1.0")

    def test_global_mass_within_tolerance(self):
        """Global dust sum must not increase by more than the tolerance fraction.

        Semi-Lagrangian advection with deposition allows dust to settle into
        the surface (mass decrease), but unchecked lifting could cause runaway
        growth.  This test verifies the upper bound only.
        """
        c = _make_climate()
        mass_before = self._total_dust(c)
        for _ in range(self._STEPS):
            c.update(1.0)
        mass_after = self._total_dust(c)
        if mass_before > 0.0:
            growth = (mass_after - mass_before) / mass_before
            self.assertLessEqual(growth, self._MASS_TOLERANCE,
                msg=f"Global dust mass grew by {growth*100:.1f}% (tolerance={self._MASS_TOLERANCE*100:.0f}%)")

    def test_wind_values_finite(self):
        c = _make_climate()
        for _ in range(self._STEPS):
            c.update(1.0)
        for u, v in zip(c._wind_u, c._wind_v):
            self.assertFalse(math.isnan(u) or math.isinf(u), "Wind U must be finite")
            self.assertFalse(math.isnan(v) or math.isinf(v), "Wind V must be finite")

    def test_wind_speed_bounded(self):
        c = _make_climate()
        for _ in range(self._STEPS):
            c.update(1.0)
        for u, v in zip(c._wind_u, c._wind_v):
            spd = math.sqrt(u * u + v * v)
            self.assertLessEqual(spd, c._wind_max + 1e-6,
                msg=f"Wind speed {spd:.2f} exceeds wind_max={c._wind_max}")


# ---------------------------------------------------------------------------
# 4. TestStormBirthConditions
# ---------------------------------------------------------------------------

class TestStormBirthConditions(unittest.TestCase):
    """Storm cells spawn when dust+wind exceed thresholds; not otherwise."""

    def _climate_with_conditions(
        self,
        dust_val: float,
        wind_val: float,
        seed: int = 42,
    ) -> ClimateSystem:
        c = _make_climate(seed=seed)
        # Force uniform conditions across all cells
        for idx in range(c._n):
            c._dust[idx]   = dust_val
            c._wind_u[idx] = wind_val
            c._wind_v[idx] = 0.0
        return c

    def test_storm_spawns_under_high_conditions(self):
        """High dust + high wind must produce at least one storm within N checks."""
        c = _make_climate(seed=42)
        # Set conditions well above thresholds
        dust_val = c._storm_dust_threshold + 0.3
        wind_val = c._storm_wind_threshold + 10.0
        for idx in range(c._n):
            c._dust[idx]   = dust_val
            c._wind_u[idx] = wind_val
            c._wind_v[idx] = 0.0
        spawned_any = False
        for _ in range(c._storm_check_interval * 30):
            c.update(0.1)   # tiny dt to avoid rapid decay
            if c._storms:
                spawned_any = True
                break
        self.assertTrue(spawned_any,
            "Expected at least one storm to spawn under high dust+wind conditions")

    def test_no_storm_under_calm_conditions(self):
        """Below-threshold conditions must never spawn a storm."""
        c = _make_climate(seed=42)
        # Conditions clearly below both thresholds
        low_dust = c._storm_dust_threshold * 0.3
        low_wind = c._storm_wind_threshold * 0.3
        for idx in range(c._n):
            c._dust[idx]   = low_dust
            c._wind_u[idx] = low_wind
            c._wind_v[idx] = 0.0
        for _ in range(500):
            c.update(1.0)
        self.assertEqual(len(c._storms), 0,
            "Storm should not spawn under clearly below-threshold conditions")

    def test_storm_properties_valid(self):
        """Spawned storms must have valid property ranges."""
        c = _make_climate(seed=42)
        dust_val = c._storm_dust_threshold + 0.3
        wind_val = c._storm_wind_threshold + 10.0
        for idx in range(c._n):
            c._dust[idx]   = dust_val
            c._wind_u[idx] = wind_val
        for _ in range(c._storm_check_interval * 50):
            c.update(0.1)
            if c._storms:
                break
        for storm in c._storms:
            self.assertGreaterEqual(storm.intensity, 0.0)
            self.assertLessEqual(storm.intensity,    1.0)
            self.assertGreaterEqual(storm.lifetime,   0.0)
            self.assertGreaterEqual(storm.radius,     0.0)
            self.assertGreaterEqual(storm.center_lat, -math.pi / 2.0)
            self.assertLessEqual(storm.center_lat,    math.pi / 2.0)

    def test_storm_decays_over_time(self):
        """A manually inserted storm should lose intensity over time."""
        c = _make_climate()
        # Insert a storm directly
        c._storms.append(StormCell(
            center_lat=0.0, center_lon=0.0,
            radius=0.3, intensity=0.8,
            vel_u=0.0, vel_v=0.0,
            lifetime=1000.0,
        ))
        initial_intensity = c._storms[0].intensity
        for _ in range(100):
            c.update(1.0)
        # If storm survived, intensity should have dropped
        if c._storms:
            self.assertLess(c._storms[0].intensity, initial_intensity,
                "Storm intensity should decrease over time")


# ---------------------------------------------------------------------------
# 5. TestClimateAPIBounds
# ---------------------------------------------------------------------------

class TestClimateAPIBounds(unittest.TestCase):
    """Public API must return bounded values for arbitrary surface positions."""

    def setUp(self):
        self.climate = _make_climate()
        for _ in range(20):
            self.climate.update(1.0)

    def _sample_positions(self):
        import random
        rng = random.Random(7)
        positions = []
        for _ in range(100):
            v = Vec3(rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1))
            if v.length() > 1e-9:
                positions.append(v.normalized())
        return positions

    def test_sample_dust_bounded(self):
        for pos in self._sample_positions():
            d = self.climate.sample_dust(pos)
            self.assertGreaterEqual(d, 0.0)
            self.assertLessEqual(d,   1.0)

    def test_sample_temperature_finite(self):
        for pos in self._sample_positions():
            t = self.climate.sample_temperature(pos)
            self.assertFalse(math.isnan(t))
            self.assertFalse(math.isinf(t))

    def test_get_visibility_bounded(self):
        for pos in self._sample_positions():
            v = self.climate.get_visibility(pos)
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v,   1.0)

    def test_get_wind_force_factor_bounded(self):
        for pos in self._sample_positions():
            f = self.climate.get_wind_force_factor(pos)
            self.assertGreaterEqual(f, 0.0)
            self.assertLessEqual(f,   1.0)

    def test_get_wetness_bounded(self):
        for pos in self._sample_positions():
            w = self.climate.get_wetness(pos)
            self.assertGreaterEqual(w, 0.0)
            self.assertLessEqual(w,   1.0)

    def test_get_erosion_factor_bounded(self):
        for pos in self._sample_positions():
            e = self.climate.get_erosion_factor(pos)
            self.assertGreaterEqual(e, 0.0)
            self.assertLessEqual(e,   1.0)

    def test_get_freeze_thaw_factor_bounded(self):
        for pos in self._sample_positions():
            f = self.climate.get_freeze_thaw_factor(pos)
            self.assertGreaterEqual(f, 0.0)
            self.assertLessEqual(f,   1.0)

    def test_sample_wind_finite(self):
        for pos in self._sample_positions():
            w = self.climate.sample_wind(pos)
            self.assertFalse(math.isnan(w.x) or math.isinf(w.x))
            self.assertFalse(math.isnan(w.y) or math.isinf(w.y))
            self.assertFalse(math.isnan(w.z) or math.isinf(w.z))


# ---------------------------------------------------------------------------
# 6. TestWindLatLongToWorld
# ---------------------------------------------------------------------------

class TestWindLatLongToWorld(unittest.TestCase):
    """Static utility produces correct tangent vectors at known points."""

    def test_eastward_at_prime_meridian_equator(self):
        """At lat=0, lon=0, east tangent = +X."""
        w = ClimateSystem.wind_lat_long_to_world(0.0, 0.0, 1.0, 0.0)
        self.assertAlmostEqual(w.x, 1.0, places=6, msg="East component should be +X")
        self.assertAlmostEqual(w.y, 0.0, places=6)
        self.assertAlmostEqual(w.z, 0.0, places=6)

    def test_northward_at_prime_meridian_equator(self):
        """At lat=0, lon=0, north tangent = +Y."""
        w = ClimateSystem.wind_lat_long_to_world(0.0, 0.0, 0.0, 1.0)
        self.assertAlmostEqual(w.x, 0.0, places=6)
        self.assertAlmostEqual(w.y, 1.0, places=6, msg="North component should be +Y at equator/meridian")
        self.assertAlmostEqual(w.z, 0.0, places=6)

    def test_east_tangent_orthogonal_to_surface_normal(self):
        """East wind vector must be perpendicular to the surface normal at any point."""
        import random
        rng = random.Random(5)
        for _ in range(50):
            lat = rng.uniform(-math.pi / 2 + 0.1, math.pi / 2 - 0.1)
            lon = rng.uniform(-math.pi, math.pi)
            # Surface normal at (lat, lon)
            from src.math.PlanetMath import PlanetMath, LatLong
            normal = PlanetMath.direction_from_lat_long(LatLong(lat, lon))
            east_w = ClimateSystem.wind_lat_long_to_world(lat, lon, 1.0, 0.0)
            dot = abs(normal.dot(east_w))
            self.assertLess(dot, 1e-6,
                msg=f"East wind not tangent at lat={math.degrees(lat):.1f}°, lon={math.degrees(lon):.1f}°, dot={dot}")

    def test_north_tangent_orthogonal_to_surface_normal(self):
        """North wind vector must be perpendicular to the surface normal at any point."""
        import random
        rng = random.Random(6)
        for _ in range(50):
            lat = rng.uniform(-math.pi / 2 + 0.1, math.pi / 2 - 0.1)
            lon = rng.uniform(-math.pi, math.pi)
            from src.math.PlanetMath import PlanetMath, LatLong
            normal = PlanetMath.direction_from_lat_long(LatLong(lat, lon))
            north_w = ClimateSystem.wind_lat_long_to_world(lat, lon, 0.0, 1.0)
            dot = abs(normal.dot(north_w))
            self.assertLess(dot, 1e-6,
                msg=f"North wind not tangent at lat={math.degrees(lat):.1f}°, lon={math.degrees(lon):.1f}°, dot={dot}")

    def test_magnitude_preserved(self):
        """Output vector magnitude should equal input speed magnitude."""
        import random
        rng = random.Random(3)
        for _ in range(30):
            lat = rng.uniform(-math.pi / 2 + 0.1, math.pi / 2 - 0.1)
            lon = rng.uniform(-math.pi, math.pi)
            u   = rng.uniform(-10.0, 10.0)
            v   = rng.uniform(-10.0, 10.0)
            w   = ClimateSystem.wind_lat_long_to_world(lat, lon, u, v)
            expected_mag = math.sqrt(u * u + v * v)
            actual_mag   = w.length()
            self.assertAlmostEqual(actual_mag, expected_mag, places=5,
                msg=f"Wind magnitude {actual_mag:.4f} ≠ expected {expected_mag:.4f}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
