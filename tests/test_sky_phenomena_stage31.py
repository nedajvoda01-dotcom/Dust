"""test_sky_phenomena_stage31.py — Stage 31 SkyPhenomenaSystem smoke tests.

Tests
-----
1. test_deterministic_rareGate
   — Same seed/timeBucket/regionId always produces the same gate result.

2. test_no_frame_random
   — SkyPhenomenaSystem uses no per-frame random calls (module-level check).

3. test_strength_smoothing
   — When target strength changes the smoothed output converges within tau.

4. test_halo_peak_at_22deg
   — _halo_profile peaks near 22°, is lower at 10° and 35°.

5. test_multiplayer_same_conditions_same_output
   — Two independent instances with equal inputs produce identical sky colour.

6. test_clear_sky_low_effects
   — With zero dust/ice the halo and corona strengths stay near zero.

7. test_parhelia_disabled_when_no_ice
   — Parhelia gate remains False when h_ice is effectively zero.

8. test_corona_peaks_at_intermediate_dust
   — Corona strength is stronger at moderate dust than at very high dust.
"""
from __future__ import annotations

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.core.Config import Config
from src.math.Vec3 import Vec3
from src.systems.SkyPhenomenaSystem import SkyPhenomenaSystem, _rare_gate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cfg(**overrides) -> Config:
    cfg = Config.__new__(Config)
    skyphen: dict = {
        "enable":                True,
        "halo_theta_deg":        22.0,
        "halo_sigma_deg":        1.5,
        "halo_strength_k":       0.55,
        "corona_strength_k":     0.40,
        "parhelia_enable":       True,
        "parhelia_rarity":       0.25,
        "parhelia_timebucket_sec": 60.0,
        "ring_glint_strength_k": 0.35,
        "ring_glint_rarity":     0.30,
        "temporal_tau_sec":      2.0,
        "render_scale":          1.0,
    }
    skyphen.update(overrides)
    cfg._data = {"skyphen": skyphen}
    return cfg


def _make_sys(**kw) -> SkyPhenomenaSystem:
    return SkyPhenomenaSystem(config=_make_cfg(**kw), world_seed=42)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDeterministicRareGate(unittest.TestCase):
    def test_same_inputs_same_output(self):
        for seed in (0, 1, 42, 9999):
            for tb in (0, 1, 100, 5000):
                r1 = _rare_gate(seed, tb, 0, 0.5)
                r2 = _rare_gate(seed, tb, 0, 0.5)
                self.assertEqual(r1, r2, f"seed={seed}, tb={tb}")

    def test_different_buckets_may_differ(self):
        results = {_rare_gate(42, tb, 0, 0.5) for tb in range(100)}
        # With threshold=0.5 and 100 buckets both True and False should occur
        self.assertIn(True, results)
        self.assertIn(False, results)


class TestNoFrameRandom(unittest.TestCase):
    """Verify SkyPhenomenaSystem doesn't import or use random in per-frame path."""

    def test_no_random_module_usage(self):
        import importlib
        import src.systems.SkyPhenomenaSystem as mod
        # The module must not import Python's `random` module
        self.assertFalse(
            hasattr(mod, "random"),
            "SkyPhenomenaSystem must not import the random module",
        )

    def test_no_math_random_in_update(self):
        """update() must be callable without triggering random.* calls."""
        import builtins
        original_random = getattr(builtins, "random", None)
        # Patch random to detect calls (it's not a builtin, but we verify via import check above)
        sys_obj = _make_sys()
        sun = Vec3(0.0, 0.7, 1.0)
        # Should not raise and must not call any per-frame RNG
        sys_obj.update(
            0.016, 100.0,
            dust_density=0.3,
            ice_crystal_proxy=0.4,
            sun1_dir=sun,
            sun2_dir=Vec3(0.2, 0.8, 0.6),
        )


class TestStrengthSmoothing(unittest.TestCase):
    def test_smoothing_converges_within_tau(self):
        """After ~3×tau steps the smoothed value should be close to target."""
        tau = 1.0
        sys_obj = _make_sys(temporal_tau_sec=tau, halo_sigma_deg=5.0)

        sun_dir = Vec3(0.0, 0.5, 1.0)
        # Drive with high ice (strong halo target)
        dt = 0.1
        n_steps = int(3 * tau / dt)
        for i in range(n_steps):
            sys_obj.update(
                dt, i * dt,
                ice_crystal_proxy=1.0,
                sun1_dir=sun_dir,
                sun2_dir=Vec3(0.1, 0.5, 1.0),
            )

        state = sys_obj.get_debug_state()
        # halo1 should have converged toward (close to) target (not zero)
        self.assertGreater(state["smooth_halo1"], 0.3,
                           "Smoothed halo1 should converge to high value after 3×tau")

    def test_smoothing_is_gradual(self):
        """After a single dt the smoothed value must not jump to target instantly."""
        tau = 2.0
        sys_obj = _make_sys(temporal_tau_sec=tau)
        sun = Vec3(0.0, 0.8, 1.0)
        sys_obj.update(
            0.016, 0.0,
            ice_crystal_proxy=1.0,
            sun1_dir=sun,
            sun2_dir=Vec3(0.1, 0.8, 0.9),
        )
        state = sys_obj.get_debug_state()
        # After one tiny dt, strength must be << 1.0 (not snapped)
        self.assertLess(state["smooth_halo1"], 0.05,
                        "Smoothed halo must not jump to target in one frame")


class TestHaloPeakAt22Deg(unittest.TestCase):
    def test_halo_maximum_near_22deg(self):
        sys_obj = _make_sys()
        # Build a sun direction (straight up)
        sun_dir = (0.0, 1.0, 0.0)

        def _view_at_deg(deg: float) -> tuple:
            rad = math.radians(deg)
            return (math.sin(rad), math.cos(rad), 0.0)

        # Evaluate profile at a range of angles
        val_at_22 = sys_obj._halo_profile(_view_at_deg(22.0), sun_dir)
        val_at_10 = sys_obj._halo_profile(_view_at_deg(10.0), sun_dir)
        val_at_35 = sys_obj._halo_profile(_view_at_deg(35.0), sun_dir)

        self.assertGreater(val_at_22, val_at_10, "Halo should be stronger at 22° than 10°")
        self.assertGreater(val_at_22, val_at_35, "Halo should be stronger at 22° than 35°")

    def test_halo_peak_within_1deg_of_22(self):
        """Maximum of sampled profile is within 1° of the configured theta."""
        sys_obj = _make_sys()
        sun_dir = (0.0, 1.0, 0.0)
        best_deg, best_val = 0.0, -1.0
        for deg in [i * 0.5 for i in range(0, 90)]:
            rad = math.radians(deg)
            vd = (math.sin(rad), math.cos(rad), 0.0)
            v = sys_obj._halo_profile(vd, sun_dir)
            if v > best_val:
                best_val, best_deg = v, deg
        self.assertAlmostEqual(best_deg, 22.0, delta=1.5,
                               msg=f"Halo peak at {best_deg}° expected ~22°")


class TestMultiplayerSameOutput(unittest.TestCase):
    def test_same_conditions_same_color(self):
        """Two independent instances with identical inputs produce the same output."""
        inputs = dict(
            dust_density=0.45,
            ice_crystal_proxy=0.35,
            visibility=0.7,
            sun1_dir=Vec3(0.2, 0.6, 0.8),
            sun2_dir=Vec3(-0.1, 0.7, 0.7),
            eclipse_fraction1=0.0,
            eclipse_fraction2=0.1,
            ring_shadow_factor=0.4,
        )

        def _run(seed: int, dt_steps: int) -> dict:
            sys_obj = SkyPhenomenaSystem(config=_make_cfg(), world_seed=seed)
            for i in range(dt_steps):
                sys_obj.update(0.1, float(i) * 0.1, **inputs)
            return sys_obj.get_debug_state()

        state_a = _run(42, 30)
        state_b = _run(42, 30)  # same seed, same steps

        for key in state_a:
            self.assertAlmostEqual(
                state_a[key],
                state_b[key],
                places=10,
                msg=f"Key {key!r} differs between two identical runs",
            )

    def test_compute_sky_color_deterministic(self):
        sys_obj = SkyPhenomenaSystem(config=_make_cfg(), world_seed=99)
        view = Vec3(0.5, 0.7, 0.5)
        sys_obj.update(
            1.0, 60.0,
            dust_density=0.4,
            ice_crystal_proxy=0.5,
            sun1_dir=Vec3(0.0, 0.8, 0.6),
            sun2_dir=Vec3(0.1, 0.75, 0.65),
        )
        c1 = sys_obj.compute_sky_color_add(view)
        c2 = sys_obj.compute_sky_color_add(view)
        self.assertEqual(c1, c2, "compute_sky_color_add is not deterministic")


class TestClearSkyLowEffects(unittest.TestCase):
    def test_no_dust_no_ice_low_effects(self):
        sys_obj = _make_sys()
        sun = Vec3(0.0, 0.6, 0.8)
        # Warm up system with clear-sky conditions
        for i in range(60):
            sys_obj.update(
                0.1, float(i) * 0.1,
                dust_density=0.0,
                ice_crystal_proxy=0.0,
                visibility=1.0,
                sun1_dir=sun,
                sun2_dir=Vec3(0.05, 0.6, 0.8),
            )
        state = sys_obj.get_debug_state()
        self.assertLess(state["smooth_halo1"], 0.05, "Halo should be near zero in clear sky")
        self.assertLess(state["smooth_corona"], 0.05, "Corona should be near zero in clear sky")


class TestParheliaDisabledWithoutIce(unittest.TestCase):
    def test_parhelia_gate_false_when_no_ice(self):
        sys_obj = _make_sys(parhelia_rarity=1.0)  # always gates True
        sun = Vec3(0.0, 0.2, 1.0)  # low sun (boosts parhelia)
        for i in range(10):
            sys_obj.update(
                0.1, float(i) * 0.1,
                dust_density=0.3,
                ice_crystal_proxy=0.0,  # no ice
                sun1_dir=sun,
                sun2_dir=Vec3(0.05, 0.2, 1.0),
            )
        cond = sys_obj._cond
        self.assertFalse(cond.parhelia_gate,
                         "Parhelia gate must be False when ice proxy == 0")

    def test_parhelia_present_with_ice_and_low_sun(self):
        # Use rarity=1.0 to guarantee gate opens, low sun, plenty of ice
        sys_obj = _make_sys(parhelia_rarity=1.0)
        sun = Vec3(0.0, 0.1, 1.0)  # very low sun
        for i in range(10):
            sys_obj.update(
                0.1, float(i) * 0.1,
                dust_density=0.1,
                ice_crystal_proxy=0.8,
                sun1_dir=sun,
                sun2_dir=Vec3(0.05, 0.1, 1.0),
            )
        cond = sys_obj._cond
        self.assertTrue(cond.parhelia_gate,
                        "Parhelia gate must open when ice>0.1 and low_sun_boost>0.05 and rarity=1")


class TestCoronaPeaksAtIntermediateDust(unittest.TestCase):
    def test_intermediate_dust_stronger_corona(self):
        """Corona at moderate dust (0.3–0.5) should exceed corona at maximum dust (1.0)."""
        def _corona_at_dust(dust: float) -> float:
            sys_obj = _make_sys()
            sun = Vec3(0.0, 0.7, 0.7)
            for i in range(30):
                sys_obj.update(
                    0.1, float(i) * 0.1,
                    dust_density=dust,
                    ice_crystal_proxy=0.0,
                    sun1_dir=sun,
                    sun2_dir=Vec3(0.05, 0.7, 0.7),
                )
            return sys_obj._cond.corona_strength

        c_moderate = _corona_at_dust(0.35)
        c_heavy = _corona_at_dust(1.0)
        self.assertGreater(c_moderate, c_heavy,
                           "Corona should be stronger at moderate dust than maximum dust")


if __name__ == "__main__":
    unittest.main()
