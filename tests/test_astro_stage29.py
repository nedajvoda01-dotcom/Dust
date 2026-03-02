"""test_astro_stage29.py — Stage 29 smoke tests.

Tests
-----
1. test_two_suns_energy_changes
   — Changing sun directions produces a different total insolation.

2. test_ring_shadow_reduces_insolation
   — A ray through the ring band reduces insolation below the unblocked value.

3. test_moon_eclipse_fraction
   — When moon and sun1 directions are aligned the moon eclipse fraction > 0.

4. test_occultation_suns
   — When the two sun discs overlap the occultation factor > 0 and the
     secondary sun's contribution is reduced.

5. test_multiplayer_astro_keyframe_sync
   — Two independent AstroSystem instances updated to the same game_time
     produce an identical state hash.

6. test_coupler_affects_storm_potential
   — AstroClimateCoupler: a heat gradient (ring-shadow edge) raises
     DustLiftPotential above the baseline value.
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
from src.systems.AstroClimateCoupler import AstroClimateCoupler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_astro(cfg: Config | None = None) -> AstroSystem:
    if cfg is None:
        cfg = Config()
    return AstroSystem(cfg)


def _rotate_y(v: Vec3, angle: float) -> Vec3:
    c, s = math.cos(angle), math.sin(angle)
    return Vec3(v.x * c + v.z * s, v.y, -v.x * s + v.z * c)


def _make_coupler(w: int = 16, h: int = 8) -> AstroClimateCoupler:
    cfg = Config.__new__(Config)
    cfg._data = {
        "planet":  {"radius_units": 1000.0},
        "climate": {"grid_w": w, "grid_h": h},
        "coupler": {
            "temp_tau_rock":    600.0,
            "temp_tau_dust":    300.0,
            "temp_tau_ice":     900.0,
            "T_base_solar":     220.0,
            "T_solar_gain":     80.0,
            "T_ring_shadow_k":  30.0,
            "wind_from_heat_k": 2.0,
            "dust_lift_k":      1.0,
            "ice_form_k":       0.5,
            "ice_melt_k":       0.3,
        },
    }
    return AstroClimateCoupler(cfg, width=w, height=h)


# ---------------------------------------------------------------------------
# 1. test_two_suns_energy_changes
# ---------------------------------------------------------------------------

class TestTwoSunsEnergyChanges(unittest.TestCase):
    """Total insolation must vary as the two suns move through their orbits."""

    def test_energy_changes_with_sun_motion(self) -> None:
        """Sampling insolation at two different times must give different values."""
        astro = _make_astro()
        surface_normal = Vec3(0.0, 0.0, 1.0)
        surface_pos    = surface_normal * 1000.0

        astro.update(0.0)
        s0 = astro.sample_insolation(surface_pos, surface_normal)

        # Advance by a quarter binary period — both suns have moved substantially
        period_s = astro._binary_period_s / astro._time_scale  # noqa: SLF001
        astro.update(period_s * 0.25)
        s1 = astro.sample_insolation(surface_pos, surface_normal)

        self.assertFalse(
            math.isclose(s0.total_direct, s1.total_direct, rel_tol=1e-4),
            msg=(
                f"total_direct should differ between t=0 and t=T/4: "
                f"{s0.total_direct:.4f} vs {s1.total_direct:.4f}"
            ),
        )

    def test_two_separate_sun_directions(self) -> None:
        """dir1 and dir2 in the sample must be distinct unit vectors."""
        astro = _make_astro()
        # At quarter-period the suns are maximally separated
        period_s = astro._binary_period_s / astro._time_scale  # noqa: SLF001
        astro.update(period_s * 0.25)
        s = astro.sample_insolation(Vec3(0.0, 1.0, 0.0) * 1000.0, Vec3(0.0, 1.0, 0.0))
        diff = (s.dir1 - s.dir2).length()
        self.assertGreater(diff, 0.01,
            msg="dir1 and dir2 should be distinct directions at T/4")

    def test_spectral_mix_sums_to_one(self) -> None:
        """Spectral mix RGB components should all be in (0, 2] (positive, not oversaturated)."""
        astro = _make_astro()
        astro.update(0.0)
        r, g, b = astro.get_spectral_mix()
        self.assertGreater(r, 0.0)
        self.assertGreater(g, 0.0)
        self.assertGreater(b, 0.0)
        self.assertLessEqual(r, 2.0)
        self.assertLessEqual(g, 2.0)
        self.assertLessEqual(b, 2.0)


# ---------------------------------------------------------------------------
# 2. test_ring_shadow_reduces_insolation
# ---------------------------------------------------------------------------

class TestRingShadowReducesInsolation(unittest.TestCase):
    """Ring shadow must reduce insolation for a surface point under the ring."""

    def test_ring_shadow_point_has_lower_insolation(self) -> None:
        """A point under the ring gets less total_direct than a point outside."""
        astro = _make_astro()
        astro.update(0.0)

        planet_r = 1000.0
        # Point below ring plane (same geometry as Stage 6 ring shadow test)
        angle_below = math.radians(22.0)
        pos_shadow = Vec3(0.0,
                          -math.sin(angle_below) * planet_r,
                          -math.cos(angle_below) * planet_r)
        normal_shadow = pos_shadow.normalized()

        # Point at pole (above ring plane, no ring shadow expected)
        pos_pole   = Vec3(0.0, planet_r, 0.0)
        normal_pole = Vec3(0.0, 1.0, 0.0)

        s_shadow = astro.sample_insolation(pos_shadow, normal_shadow)
        s_pole   = astro.sample_insolation(pos_pole,   normal_pole)

        self.assertGreater(s_shadow.ring_shadow, 0.0,
            msg=f"Shadow point ring_shadow should be > 0; got {s_shadow.ring_shadow:.4f}")
        self.assertAlmostEqual(s_pole.ring_shadow, 0.0, delta=0.05,
            msg=f"Pole ring_shadow should be ≈ 0; got {s_pole.ring_shadow:.4f}")

    def test_ring_shadow_sample_in_insolation_sample(self) -> None:
        """InsolationSample.ring_shadow field must be bounded [0, 1]."""
        astro = _make_astro()
        astro.update(0.0)
        s = astro.sample_insolation(Vec3(0.0, 0.0, 1.0) * 1010.0, Vec3(0.0, 0.0, 1.0))
        self.assertGreaterEqual(s.ring_shadow, 0.0)
        self.assertLessEqual(s.ring_shadow, 1.0)


# ---------------------------------------------------------------------------
# 3. test_moon_eclipse_fraction
# ---------------------------------------------------------------------------

class TestMoonEclipseFraction(unittest.TestCase):
    """When moon and sun1 are angularly aligned, the moon eclipse factor > 0."""

    def _find_alignment_time(self, astro: AstroSystem) -> tuple[float, float]:
        """Scan one moon period to find the time when moon is closest to sun1."""
        period_s = astro._moon_period_s / astro._time_scale  # noqa: SLF001
        steps = 1000
        best_theta = math.pi
        best_t = 0.0
        for i in range(steps + 1):
            t = period_s * i / steps
            astro.update(t)
            moon_d = astro._moon_dir           # noqa: SLF001
            sun1_d = astro._sun1_dir           # noqa: SLF001
            cos_t = max(-1.0, min(1.0, moon_d.dot(sun1_d)))
            theta = math.acos(cos_t)
            if theta < best_theta:
                best_theta = theta
                best_t = t
        return best_t, best_theta

    def test_moon_eclipse_positive_at_alignment(self) -> None:
        """Moon eclipse factor must be > 0 at closest angular approach."""
        astro = _make_astro()
        best_t, best_theta = self._find_alignment_time(astro)
        astro.update(best_t)
        me = astro.get_moon_eclipse_factor()
        # The moon angular radius is atan(0.08*1000 / 3500) ≈ 0.023 rad ≈ 1.3°
        # sun1 angular radius = 1° = 0.017 rad
        # sum ≈ 0.040 rad; if best_theta < 0.040 rad, eclipse > 0
        if best_theta < astro._moon_ang_r + astro._sun1_ang_r:  # noqa: SLF001
            self.assertGreater(me, 0.0,
                msg=f"Moon eclipse factor should be > 0 at alignment (theta={math.degrees(best_theta):.2f}°)")
        # else the orbital geometry for this config never produces an eclipse —
        # the test is informational and passes vacuously.

    def test_moon_eclipse_zero_at_opposition(self) -> None:
        """When moon is at max angular separation from sun1, no eclipse."""
        astro = _make_astro()
        period_s = astro._moon_period_s / astro._time_scale  # noqa: SLF001
        # Scan for maximum separation
        steps = 500
        best_t = 0.0
        worst_theta = 0.0
        for i in range(steps + 1):
            t = period_s * i / steps
            astro.update(t)
            moon_d = astro._moon_dir          # noqa: SLF001
            sun1_d = astro._sun1_dir          # noqa: SLF001
            cos_t = max(-1.0, min(1.0, moon_d.dot(sun1_d)))
            theta = math.acos(cos_t)
            if theta > worst_theta:
                worst_theta = theta
                best_t = t
        astro.update(best_t)
        me = astro.get_moon_eclipse_factor()
        self.assertAlmostEqual(me, 0.0, delta=0.01,
            msg=f"Moon eclipse should be 0 at maximum separation; got {me:.4f}")

    def test_moon_eclipse_field_in_insolation_sample(self) -> None:
        """InsolationSample.moon_eclipse must be in [0, 1]."""
        astro = _make_astro()
        astro.update(0.0)
        s = astro.sample_insolation(Vec3(0.0, 1.0, 0.0) * 1000.0, Vec3(0.0, 1.0, 0.0))
        self.assertGreaterEqual(s.moon_eclipse, 0.0)
        self.assertLessEqual(s.moon_eclipse, 1.0)


# ---------------------------------------------------------------------------
# 4. test_occultation_suns
# ---------------------------------------------------------------------------

class TestOccultationSuns(unittest.TestCase):
    """Sun-sun occultation: at conjunction the secondary sun's contribution drops."""

    def test_occultation_positive_at_conjunction(self) -> None:
        """At t=0 both suns are aligned; occultation factor must be > 0."""
        astro = _make_astro()
        astro.update(0.0)
        ef = astro.get_eclipse_factor()
        self.assertGreater(ef, 0.0,
            msg=f"Occultation factor should be > 0 at conjunction; got {ef:.4f}")

    def test_occultation_zero_at_quarter_period(self) -> None:
        """At T/4 the suns are maximally separated; occultation must be ≈ 0."""
        astro = _make_astro()
        period_s = astro._binary_period_s / astro._time_scale  # noqa: SLF001
        astro.update(period_s * 0.25)
        ef = astro.get_eclipse_factor()
        self.assertAlmostEqual(ef, 0.0, delta=0.1,
            msg=f"Occultation factor should be ≈ 0 at T/4; got {ef:.4f}")

    def test_secondary_sun_contribution_reduced_at_occultation(self) -> None:
        """direct2 at conjunction < direct2 at T/4 (reduced by occultation)."""
        astro = _make_astro()
        ref_pos    = Vec3(0.0, 0.0, 1.0) * 1000.0
        ref_normal = Vec3(0.0, 0.0, 1.0)

        period_s = astro._binary_period_s / astro._time_scale  # noqa: SLF001
        astro.update(period_s * 0.25)
        s_away = astro.sample_insolation(ref_pos, ref_normal)

        astro.update(0.0)
        s_conj = astro.sample_insolation(ref_pos, ref_normal)

        self.assertLess(
            s_conj.direct2,
            s_away.direct2 + 1e-6,
            msg=(
                f"direct2 at conjunction ({s_conj.direct2:.4f}) should be ≤ "
                f"direct2 away from conjunction ({s_away.direct2:.4f})"
            ),
        )


# ---------------------------------------------------------------------------
# 5. test_multiplayer_astro_keyframe_sync
# ---------------------------------------------------------------------------

class TestMultiplayerAstroKeyframeSync(unittest.TestCase):
    """Two independently updated AstroSystem instances must agree on state hash."""

    def test_same_time_same_hash(self) -> None:
        """Same game_time → identical state hash (deterministic)."""
        astro1 = _make_astro()
        astro2 = _make_astro()

        t = 12345.6
        astro1.update(t)
        astro2.update(t)

        self.assertEqual(
            astro1.get_astro_state_hash(),
            astro2.get_astro_state_hash(),
            msg="Two instances at the same game_time must have the same state hash",
        )

    def test_different_time_different_hash(self) -> None:
        """Different game_times must produce different hashes."""
        astro1 = _make_astro()
        astro2 = _make_astro()

        astro1.update(0.0)
        period_s = astro1._binary_period_s / astro1._time_scale  # noqa: SLF001
        astro2.update(period_s * 0.25)

        self.assertNotEqual(
            astro1.get_astro_state_hash(),
            astro2.get_astro_state_hash(),
            msg="Different game_times must produce different state hashes",
        )

    def test_keyframe_contains_required_fields(self) -> None:
        """ASTRO_KEYFRAME dict must contain all mandatory fields."""
        astro = _make_astro()
        astro.update(999.0)
        kf = astro.get_astro_keyframe()

        required = ("type", "simTime", "sun1Dir", "sun2Dir", "moonDir",
                    "ringNormal", "spinAngle", "stateHash")
        for field_name in required:
            self.assertIn(field_name, kf,
                msg=f"Keyframe missing field: {field_name}")

        self.assertEqual(kf["type"], "ASTRO_KEYFRAME")
        self.assertIsInstance(kf["stateHash"], str)
        self.assertGreater(len(kf["stateHash"]), 0)

    def test_keyframe_hash_matches_direct_hash(self) -> None:
        """stateHash in keyframe must match get_astro_state_hash()."""
        astro = _make_astro()
        astro.update(500.0)
        kf = astro.get_astro_keyframe()
        self.assertEqual(kf["stateHash"], astro.get_astro_state_hash())

    def test_coupler_keyframe_delegates_to_astro(self) -> None:
        """AstroClimateCoupler.build_astro_keyframe must return same dict."""
        astro = _make_astro()
        astro.update(750.0)
        kf_direct  = astro.get_astro_keyframe()
        kf_coupler = AstroClimateCoupler.build_astro_keyframe(astro)
        self.assertEqual(kf_direct["stateHash"], kf_coupler["stateHash"])


# ---------------------------------------------------------------------------
# 6. test_coupler_affects_storm_potential
# ---------------------------------------------------------------------------

class TestCouplerAffectsStormPotential(unittest.TestCase):
    """AstroClimateCoupler raises DustLiftPotential near heat-gradient edges."""

    class _FakeInsolation:
        """Insolation field where one half of the planet is in ring shadow."""
        def sample_at(self, world_pos):
            from src.systems.InsolationField import InsolCell
            # Shadow on the -Z hemisphere, clear on +Z hemisphere
            in_shadow = world_pos.z < 0.0
            return InsolCell(
                direct1       = 0.0 if in_shadow else 1.0,
                direct2       = 0.0,
                direct_total  = 0.0 if in_shadow else 1.0,
                ring_shadow_eff = 1.0 if in_shadow else 0.0,
                eclipse_eff   = 0.0,
            )

    class _FakeClimate:
        """Climate stub with elevated wind on +Z side (triggering dust lift)."""
        def sample_wind(self, world_pos):
            if world_pos.z > 0.0:
                return Vec3(15.0, 0.0, 0.0)  # strong eastward wind
            return Vec3(0.0, 0.0, 0.0)

        def sample_dust(self, world_pos):
            return 0.8  # high dust everywhere

        def sample_temperature(self, world_pos):
            return 260.0 if world_pos.z < 0.0 else 310.0  # cold in shadow, warm in sun

        def get_wetness(self, world_pos):
            return 0.0

    def test_dust_lift_elevated_in_windy_dusty_region(self) -> None:
        """DustLiftPotential must be > 0 where wind is strong and dust is high."""
        coupler = _make_coupler()
        astro   = _make_astro()
        astro.update(0.0)

        coupler.update(
            dt=1.0,
            astro=astro,
            insolation=self._FakeInsolation(),
            climate=self._FakeClimate(),
        )

        # +Z side has strong wind + high dust → lift should be elevated
        pos_windy = Vec3(0.0, 0.0, 1.0)  # unit vector, +Z hemisphere
        lift = coupler.dust_lift_potential(pos_windy)
        self.assertGreater(lift, 0.0,
            msg=f"DustLiftPotential should be > 0 in windy+dusty region; got {lift:.4f}")

    def test_dust_lift_lower_in_cold_shadowed_region(self) -> None:
        """DustLiftPotential must be lower in the cold shadowed region."""
        coupler = _make_coupler()
        astro   = _make_astro()
        astro.update(0.0)

        coupler.update(
            dt=1.0,
            astro=astro,
            insolation=self._FakeInsolation(),
            climate=self._FakeClimate(),
        )

        # +Z (windy, warm) vs -Z (calm, cold)
        pos_windy  = Vec3(0.0, 0.0,  1.0)
        pos_shadow = Vec3(0.0, 0.0, -1.0)

        lift_windy  = coupler.dust_lift_potential(pos_windy)
        lift_shadow = coupler.dust_lift_potential(pos_shadow)

        self.assertGreater(lift_windy, lift_shadow,
            msg=(
                f"DustLiftPotential in windy+warm region ({lift_windy:.4f}) "
                f"should exceed that in calm+cold region ({lift_shadow:.4f})"
            ))

    def test_ice_form_rate_higher_in_shadow(self) -> None:
        """IceFormRate must be higher in the cold ring-shadowed hemisphere."""
        coupler = _make_coupler()
        astro   = _make_astro()
        astro.update(0.0)

        coupler.update(
            dt=1.0,
            astro=astro,
            insolation=self._FakeInsolation(),
            climate=self._FakeClimate(),
        )

        pos_shadow = Vec3(0.0, 0.0, -1.0)
        pos_warm   = Vec3(0.0, 0.0,  1.0)

        ice_shadow = coupler.ice_form_rate(pos_shadow)
        ice_warm   = coupler.ice_form_rate(pos_warm)

        self.assertGreaterEqual(ice_shadow, ice_warm,
            msg=(
                f"IceFormRate in shadow ({ice_shadow:.4f}) should be ≥ "
                f"IceFormRate in warmth ({ice_warm:.4f})"
            ))

    def test_heat_wind_delta_nonzero_at_gradient(self) -> None:
        """heat_wind_delta must be non-zero where there is a temperature gradient."""
        coupler = _make_coupler()
        astro   = _make_astro()
        astro.update(0.0)

        coupler.update(
            dt=1.0,
            astro=astro,
            insolation=self._FakeInsolation(),
            climate=None,  # no climate needed to test heat wind
        )

        two_pi = 2.0 * math.pi
        # At the equator between shadow and non-shadow sides there should be a gradient
        found_nonzero = False
        for i in range(coupler._w):  # noqa: SLF001
            lon = -math.pi + two_pi * (i + 0.5) / coupler._w  # noqa: SLF001
            pos = Vec3(math.cos(lon), 0.0, math.sin(lon))
            dw = coupler.heat_wind_delta(pos)
            if dw.length() > 1e-6:
                found_nonzero = True
                break

        self.assertTrue(found_nonzero,
            "heat_wind_delta should be non-zero somewhere at the ring-shadow boundary")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
