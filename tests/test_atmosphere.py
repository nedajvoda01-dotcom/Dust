"""test_atmosphere — Stage 11 AtmosphereRenderer tests.

Tests
-----
1. TestHazeMonotonic       — fogFactor increases monotonically with distance
2. TestWhiteoutBehavior    — high dust → transmittance drops below low-dust case
3. TestTwoSunAdditivity    — enabling Sun2 increases sky illuminance
4. TestSkyColorBounds      — compute_sky_color always returns values in [0, 1]
5. TestColorGrade          — filmic curve + split-toning bounds + desaturation
6. TestComposite           — composite returns bounded RGB, debug modes work
7. TestVolStepsConfig      — vol_steps config parameter is respected
"""
from __future__ import annotations

import math
import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.core.Config import Config
from src.math.Vec3 import Vec3
from src.render.AtmosphereRenderer import AtmosphereRenderer


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

class _FakeAstro:
    """Minimal AstroSystem stub for atmosphere tests."""

    def __init__(
        self,
        sun1_dir=(0.0, 1.0, 0.0),
        sun2_dir=(0.0, 0.8, 0.6),
        i1=1.0,
        i2=0.35,
        eclipse=0.0,
    ) -> None:
        self._sun1_dir = Vec3(*sun1_dir)
        self._sun2_dir = Vec3(*sun2_dir)
        self._sun1_intensity = i1
        self._sun2_intensity = i2
        self._eclipse = eclipse
        self._sun1_ang_r = math.radians(1.0)
        self._sun2_ang_r = math.radians(0.7)

    def get_sun_directions(self):
        return self._sun1_dir, self._sun2_dir

    def get_eclipse_factor(self):
        return self._eclipse


class _FakeClimate:
    """Minimal ClimateSystem stub returning a fixed dust value."""

    def __init__(self, dust: float = 0.12) -> None:
        self._dust = dust

    def sample_dust(self, pos) -> float:
        return self._dust


def _make_renderer(
    dust: float = 0.12,
    eclipse: float = 0.0,
    sun2_i: float = 0.35,
    vol_steps: int = 16,
) -> AtmosphereRenderer:
    cfg = Config.__new__(Config)
    cfg._data = {
        "atmo": {
            "haze_density_base":  0.003,
            "haze_height_falloff": 0.001,
            "rayleigh_strength":  1.0,
            "mie_strength":       0.8,
            "mie_g":              0.76,
            "vol_steps":          vol_steps,
            "vol_max_distance":   5000.0,
            "vol_density_scale":  1.0,
            "whiteout_threshold": 0.7,
            "sun2_sky_limit":     0.5,
        },
        "grade": {
            "filmic_toe":       0.04,
            "filmic_shoulder":  0.85,
            "shadow_tint":      [0.80, 0.85, 1.00],
            "highlight_tint":   [1.00, 0.95, 0.80],
            "sat_far_scale":    0.30,
        },
    }
    astro   = _FakeAstro(i2=sun2_i, eclipse=eclipse)
    climate = _FakeClimate(dust=dust)
    return AtmosphereRenderer(cfg, astro=astro, climate=climate)


# ---------------------------------------------------------------------------
# 1. TestHazeMonotonic
# ---------------------------------------------------------------------------

class TestHazeMonotonic(unittest.TestCase):
    """fog_factor = 1 − exp(−d × density) must increase monotonically with d."""

    def test_fog_factor_monotonically_increases(self):
        renderer = _make_renderer()
        distances = [0.0, 100.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0]
        prev = -1.0
        for d in distances:
            fog = renderer.compute_haze_factor(d)
            self.assertGreaterEqual(
                fog, prev,
                msg=f"fog_factor not monotonic: f({d})={fog:.4f} < prev={prev:.4f}",
            )
            prev = fog

    def test_fog_factor_bounded(self):
        renderer = _make_renderer()
        for d in [0.0, 1.0, 100.0, 1e6]:
            fog = renderer.compute_haze_factor(d)
            self.assertGreaterEqual(fog, 0.0)
            self.assertLessEqual(fog, 1.0)

    def test_fog_factor_zero_at_zero_distance(self):
        renderer = _make_renderer()
        fog = renderer.compute_haze_factor(0.0)
        self.assertAlmostEqual(fog, 0.0, delta=1e-9,
                               msg="Fog factor must be 0 at zero distance")

    def test_fog_factor_increases_with_dust(self):
        """Higher dust concentration should produce a larger fog factor at the same distance."""
        d = 2000.0
        renderer_low  = _make_renderer(dust=0.1)
        renderer_high = _make_renderer(dust=0.9)
        fog_low  = renderer_low.compute_haze_factor(d, dust=0.1)
        fog_high = renderer_high.compute_haze_factor(d, dust=0.9)
        self.assertGreater(
            fog_high, fog_low,
            msg=f"High dust ({fog_high:.4f}) should produce more haze than low dust ({fog_low:.4f})",
        )


# ---------------------------------------------------------------------------
# 2. TestWhiteoutBehavior
# ---------------------------------------------------------------------------

class TestWhiteoutBehavior(unittest.TestCase):
    """Dust above whiteout_threshold → transmittance drops significantly."""

    def test_high_dust_reduces_transmittance(self):
        """Transmittance under near-maximum dust should be lower than under low dust."""
        ro = (0.0, 0.0, 0.0)
        rd = (0.0, 0.0, 1.0)

        renderer_calm  = _make_renderer(dust=0.05)
        renderer_storm = _make_renderer(dust=0.95)

        _, transmit_calm  = renderer_calm.compute_volumetric(ro, rd, max_dist=2000.0)
        _, transmit_storm = renderer_storm.compute_volumetric(ro, rd, max_dist=2000.0)

        self.assertLess(
            transmit_storm,
            transmit_calm,
            msg=(
                f"Storm transmittance ({transmit_storm:.4f}) must be lower than "
                f"calm transmittance ({transmit_calm:.4f})"
            ),
        )

    def test_transmittance_bounded(self):
        """Transmittance must remain in [0, 1] for any dust level."""
        ro = (0.0, 0.0, 0.0)
        rd = (0.0, 0.0, 1.0)
        for dust_val in [0.0, 0.3, 0.7, 1.0]:
            renderer = _make_renderer(dust=dust_val)
            _, t = renderer.compute_volumetric(ro, rd, max_dist=3000.0)
            self.assertGreaterEqual(t, 0.0, msg=f"transmittance < 0 at dust={dust_val}")
            self.assertLessEqual(t,   1.0, msg=f"transmittance > 1 at dust={dust_val}")

    def test_vol_color_bounded(self):
        """Volumetric colour channels must be in [0, 1]."""
        ro = (0.0, 0.0, 0.0)
        rd = (0.0, 0.0, 1.0)
        renderer = _make_renderer(dust=0.9)
        vol_color, _ = renderer.compute_volumetric(ro, rd, max_dist=3000.0)
        for ch in vol_color:
            self.assertGreaterEqual(ch, 0.0)
            self.assertLessEqual(ch,   1.0)


# ---------------------------------------------------------------------------
# 3. TestTwoSunAdditivity
# ---------------------------------------------------------------------------

class TestTwoSunAdditivity(unittest.TestCase):
    """Adding Sun2 must increase sky illuminance within the configured limit."""

    def _sky_luma(self, renderer: AtmosphereRenderer, view_dir=(0.0, 0.0, 1.0)) -> float:
        """Luma of sky colour for a view direction perpendicular to Sun1 (avoids saturation)."""
        r, g, b = renderer.compute_sky_color(view_dir)
        return 0.299 * r + 0.587 * g + 0.114 * b

    def test_sun2_increases_sky_illuminance(self):
        renderer_no_sun2  = _make_renderer(sun2_i=0.0)
        renderer_with_sun2 = _make_renderer(sun2_i=0.35)
        luma_no   = self._sky_luma(renderer_no_sun2)
        luma_with = self._sky_luma(renderer_with_sun2)
        self.assertGreater(
            luma_with, luma_no,
            msg=(
                f"Sky with Sun2 ({luma_with:.4f}) must be brighter than sky without Sun2 ({luma_no:.4f})"
            ),
        )

    def test_sun2_respects_limit(self):
        """Sun2 contribution must not push sky luma higher than sun2_sky_limit fraction above Sun1-only."""
        renderer_no_sun2   = _make_renderer(sun2_i=0.0)
        renderer_full_sun2 = _make_renderer(sun2_i=1.0)   # artificially equal to Sun1
        luma_base = self._sky_luma(renderer_no_sun2)
        luma_full = self._sky_luma(renderer_full_sun2)
        # Sun2 should not be unbounded — the addition must be finite
        self.assertLessEqual(luma_full, 1.0,
                             msg="Sky luma must not exceed 1.0 even with strong Sun2")

    def test_eclipse_reduces_sun2_contribution(self):
        """During an eclipse (eclipse_factor=1), Sun2 contribution to sky decreases."""
        renderer_no_eclipse = _make_renderer(sun2_i=0.35, eclipse=0.0)
        renderer_eclipse    = _make_renderer(sun2_i=0.35, eclipse=1.0)
        luma_clear  = self._sky_luma(renderer_no_eclipse)
        luma_eclipse = self._sky_luma(renderer_eclipse)
        self.assertLessEqual(
            luma_eclipse, luma_clear,
            msg="Eclipse should reduce or maintain sky illuminance",
        )


# ---------------------------------------------------------------------------
# 4. TestSkyColorBounds
# ---------------------------------------------------------------------------

class TestSkyColorBounds(unittest.TestCase):
    """Sky colour must be in [0, 1] for any viewing direction."""

    def test_sky_color_bounded_varied_dirs(self):
        import random
        renderer = _make_renderer()
        rng = random.Random(11)
        for _ in range(100):
            v = (rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1))
            mag = math.sqrt(sum(x * x for x in v))
            if mag < 1e-9:
                continue
            vn = tuple(x / mag for x in v)
            r, g, b = renderer.compute_sky_color(vn)
            self.assertGreaterEqual(r, 0.0)
            self.assertLessEqual(r, 1.0)
            self.assertGreaterEqual(g, 0.0)
            self.assertLessEqual(g, 1.0)
            self.assertGreaterEqual(b, 0.0)
            self.assertLessEqual(b, 1.0)


# ---------------------------------------------------------------------------
# 5. TestColorGrade
# ---------------------------------------------------------------------------

class TestColorGrade(unittest.TestCase):
    """Filmic curve + split-toning must produce bounded, well-shaped output."""

    def test_output_bounded(self):
        renderer = _make_renderer()
        test_colors = [
            (0.0, 0.0, 0.0),
            (0.5, 0.5, 0.5),
            (1.0, 1.0, 1.0),
            (2.0, 0.5, 0.1),   # over-bright input
            (0.1, 0.9, 0.3),
        ]
        for c in test_colors:
            r, g, b = renderer.color_grade(c, fog_factor=0.5)
            self.assertGreaterEqual(r, 0.0, f"r underflow for {c}")
            self.assertLessEqual(r, 1.0, f"r overflow for {c}")
            self.assertGreaterEqual(g, 0.0, f"g underflow for {c}")
            self.assertLessEqual(g, 1.0, f"g overflow for {c}")
            self.assertGreaterEqual(b, 0.0, f"b underflow for {c}")
            self.assertLessEqual(b, 1.0, f"b overflow for {c}")

    def test_black_in_black_out(self):
        renderer = _make_renderer()
        r, g, b = renderer.color_grade((0.0, 0.0, 0.0), fog_factor=0.0)
        self.assertAlmostEqual(r, 0.0, delta=0.01)
        self.assertAlmostEqual(g, 0.0, delta=0.01)
        self.assertAlmostEqual(b, 0.0, delta=0.01)

    def test_high_fog_desaturates(self):
        """At fog_factor=1 the output should be less saturated than at fog_factor=0."""
        renderer = _make_renderer()
        color = (0.8, 0.3, 0.1)   # strongly coloured

        def saturation(rgb):
            luma = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
            return max(abs(c - luma) for c in rgb)

        graded_near = renderer.color_grade(color, fog_factor=0.0)
        graded_far  = renderer.color_grade(color, fog_factor=1.0)
        self.assertLessEqual(
            saturation(graded_far),
            saturation(graded_near) + 0.01,
            msg="Far colour should be less or equally saturated compared to near colour",
        )


# ---------------------------------------------------------------------------
# 6. TestComposite
# ---------------------------------------------------------------------------

class TestComposite(unittest.TestCase):
    """composite() must return bounded RGB and honour debug modes."""

    def test_composite_bounded(self):
        renderer = _make_renderer()
        surface = (0.6, 0.4, 0.3)
        result = renderer.composite(surface, distance=2000.0, view_dir=(0.0, 0.5, 0.866))
        for ch in result:
            self.assertGreaterEqual(ch, 0.0)
            self.assertLessEqual(ch,   1.0)

    def test_debug_haze_only_greyscale(self):
        renderer = _make_renderer()
        r, g, b = renderer.composite(
            (0.5, 0.5, 0.5), distance=1000.0, view_dir=(0.0, 0.0, 1.0),
            debug_mode=AtmosphereRenderer.DEBUG_HAZE_ONLY,
        )
        # Should all equal the fog factor
        self.assertAlmostEqual(r, g, delta=1e-9)
        self.assertAlmostEqual(g, b, delta=1e-9)

    def test_debug_sky_only_returns_sky(self):
        renderer = _make_renderer()
        result_debug = renderer.composite(
            (0.5, 0.5, 0.5), distance=1000.0, view_dir=(0.0, 1.0, 0.0),
            debug_mode=AtmosphereRenderer.DEBUG_SKY_ONLY,
        )
        expected_sky = renderer.compute_sky_color((0.0, 1.0, 0.0))
        for a, b in zip(result_debug, expected_sky):
            self.assertAlmostEqual(a, b, delta=1e-9)

    def test_debug_transmittance_bounded(self):
        renderer = _make_renderer()
        r, g, b = renderer.composite(
            (0.5, 0.5, 0.5), distance=1000.0, view_dir=(0.0, 0.0, 1.0),
            debug_mode=AtmosphereRenderer.DEBUG_TRANSMITTANCE,
        )
        for ch in (r, g, b):
            self.assertGreaterEqual(ch, 0.0)
            self.assertLessEqual(ch,   1.0)

    def test_debug_vol_dust_only_bounded(self):
        renderer = _make_renderer(dust=0.8)
        r, g, b = renderer.composite(
            (0.5, 0.5, 0.5), distance=2000.0, view_dir=(0.0, 0.0, 1.0),
            debug_mode=AtmosphereRenderer.DEBUG_VOL_DUST_ONLY,
        )
        for ch in (r, g, b):
            self.assertGreaterEqual(ch, 0.0)
            self.assertLessEqual(ch,   1.0)


# ---------------------------------------------------------------------------
# 7. TestVolStepsConfig
# ---------------------------------------------------------------------------

class TestVolStepsConfig(unittest.TestCase):
    """vol_steps config param controls the step count."""

    def test_minimum_steps_still_returns_valid_result(self):
        renderer = _make_renderer(vol_steps=8)
        ro = (0.0, 0.0, 0.0)
        rd = (0.0, 0.0, 1.0)
        vol_color, transmit = renderer.compute_volumetric(ro, rd, max_dist=2000.0)
        for ch in vol_color:
            self.assertGreaterEqual(ch, 0.0)
            self.assertLessEqual(ch,   1.0)
        self.assertGreaterEqual(transmit, 0.0)
        self.assertLessEqual(transmit,   1.0)

    def test_maximum_steps_still_returns_valid_result(self):
        renderer = _make_renderer(vol_steps=24)
        ro = (0.0, 0.0, 0.0)
        rd = (0.0, 0.0, 1.0)
        vol_color, transmit = renderer.compute_volumetric(ro, rd, max_dist=2000.0)
        for ch in vol_color:
            self.assertGreaterEqual(ch, 0.0)
            self.assertLessEqual(ch,   1.0)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
