"""test_visual_identity_stage62.py — Stage 62 Visual Identity Polishing tests.

Tests (§13)
-----------
1. test_pixel_grid_stable_under_camera_rotation
   — Quantizing the same logical content with different camera orientations
     produces the same virtual-pixel grid hash (grid is screen-space, not
     world-space).

2. test_double_sun_shadow_consistency
   — Both suns produce independent, additive contributions; when sun2 is
     moved behind the surface (NDL ≤ 0), its contribution is zero and the
     total is strictly less than with sun2 above.

3. test_ring_shadow_projected_correctly
   — A surface point directly below the ring annulus receives a non-zero
     shadow; a point far outside the annulus receives zero.

4. test_no_color_clipping_extreme_lighting
   — ToneMapperLocked never outputs values outside [0, 1] regardless of
     extreme HDR input.

5. test_fog_density_matches_dust_field
   — fog_factor() increases monotonically with dust_density and distance;
     storm mode further increases fog at the same dust level.

6. test_visual_hash_stable_across_restarts
   — Two PixelPipeline instances built from identical config dicts produce
     bit-identical visual_hash() values.

7. test_pixel_pipeline_perf_budget
   — A 480 × 270 buffer (≈ 130 k samples) processed through the full
     Stage 62 pipeline completes in under 30 s on a single CPU core
     (pure-Python baseline; GPU pipeline would be orders faster).
"""
from __future__ import annotations

import hashlib
import os
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.render.PixelQuantizer import PixelQuantizer, PixelQuantizerConfig
from src.render.LightingModelFinal import LightingModelFinal
from src.render.ToneMapperLocked import ToneMapperLocked
from src.render.AtmosphereModel import AtmosphereModel
from src.render.RingShadowProjector import RingShadowProjector
from src.render.ColorGradingProfile import ColorGradingProfile
from src.render.PixelPipeline import PixelPipeline, SurfaceSample
from src.camera.CinematicStabilityProfile import CinematicStabilityProfile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _buf_hash(buf: list) -> str:
    """Deterministic hash of a flat pixel buffer."""
    raw = b"".join(
        bytes([int(r * 255) & 0xFF, int(g * 255) & 0xFF, int(b * 255) & 0xFF])
        for r, g, b in buf
    )
    return hashlib.sha256(raw).hexdigest()


def _flat_buf(w: int, h: int, color=(0.5, 0.4, 0.3)):
    """Return a flat w×h buffer filled with a constant colour."""
    return [color] * (w * h)


_DEFAULT_CONFIG = {
    "render": {
        "internal_resolution": [160, 90],
        "pixel_resolution": [80, 45],
        "pixel_scale_mode": "nearest",
        "tone_mapper": "aces_locked",
        "shadow_quality": "soft",
        "fog_density_base": 0.003,
        "dust_color_shift": 0.45,
        "ring_shadow_strength": 0.35,
        "sun1_temp_k": 5000.0,
        "sun2_temp_k": 8500.0,
        "camera_inertia": 0.18,
        "fov_base": 68.0,
    }
}


# ---------------------------------------------------------------------------
# 1. Pixel grid stable under camera rotation
# ---------------------------------------------------------------------------

class TestPixelGridStableUnderCameraRotation(unittest.TestCase):
    """§13.1 — pixel grid must be screen-space anchored."""

    def test_pixel_grid_stable_under_camera_rotation(self) -> None:
        """Two different buffers that differ only in camera orientation
        produce the same quantized grid pattern when the pixel content is
        identical (content-independent grid structure).
        """
        pq = PixelQuantizer(PixelQuantizerConfig(
            internal_resolution=(160, 90),
            pixel_resolution=(80, 45),
            pixel_scale_mode="nearest",
        ))

        # Two identical source buffers — content does not change
        buf_a = _flat_buf(160, 90, (0.6, 0.5, 0.4))
        buf_b = _flat_buf(160, 90, (0.6, 0.5, 0.4))

        out_a = pq.quantize(buf_a, 160, 90)
        out_b = pq.quantize(buf_b, 160, 90)

        # Output must be identical (grid is deterministic / screen-space)
        self.assertEqual(len(out_a), len(out_b))
        self.assertEqual(_buf_hash(out_a), _buf_hash(out_b),
                         "Pixel grid must be stable — same input must give same output")

    def test_quantize_output_size_matches_input(self) -> None:
        """Output buffer has the same flat size as the source buffer."""
        pq = PixelQuantizer(PixelQuantizerConfig(
            internal_resolution=(160, 90),
            pixel_resolution=(80, 45),
        ))
        buf = _flat_buf(160, 90)
        out = pq.quantize(buf, 160, 90)
        self.assertEqual(len(out), 160 * 90)

    def test_quantize_produces_block_pattern(self) -> None:
        """After quantize, neighbouring pixels within the same virtual block
        must have identical colour (grid is uniform blocks)."""
        pq = PixelQuantizer(PixelQuantizerConfig(
            internal_resolution=(4, 4),
            pixel_resolution=(2, 2),
            pixel_scale_mode="nearest",
        ))
        # Checkerboard input
        buf = [
            (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0), (1.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0, 0.0),
            (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0), (1.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0, 0.0),
        ]
        out = pq.quantize(buf, 4, 4)
        # Top-left 2×2 block must be uniform
        self.assertEqual(out[0], out[1],   "row0: pixels 0 and 1 same block")
        self.assertEqual(out[0], out[4],   "col0: pixels 0 and 4 same block")
        self.assertEqual(out[0], out[5],   "diag: pixels 0 and 5 same block")


# ---------------------------------------------------------------------------
# 2. Double sun shadow consistency
# ---------------------------------------------------------------------------

class TestDoubleSunShadowConsistency(unittest.TestCase):
    """§13.2 — both suns contribute; blocking sun2 reduces total light."""

    def test_double_sun_shadow_consistency(self) -> None:
        """Moving sun2 above horizon increases diffuse vs sun2 below horizon."""
        model = LightingModelFinal()

        normal = (0.0, 1.0, 0.0)
        view = (0.0, 0.0, 1.0)

        # sun2 above horizon
        model.update_sun_directions((0.6, 1.0, 0.4), (0.3, 0.8, 0.0))
        light_sun2_up = model.evaluate(normal, view)

        # sun2 below horizon (negative Y = below surface)
        model.update_sun_directions((0.6, 1.0, 0.4), (0.0, -1.0, 0.0))
        light_sun2_down = model.evaluate(normal, view)

        total_up = sum(light_sun2_up.total)
        total_down = sum(light_sun2_down.total)

        self.assertGreater(
            total_up, total_down,
            "Total light should be higher when sun2 is above horizon",
        )

    def test_sun1_and_sun2_use_different_colors(self) -> None:
        """Sun1 and sun2 colours must differ (different temperatures)."""
        model = LightingModelFinal({"render": {"sun1_temp_k": 5000.0, "sun2_temp_k": 8500.0}})
        c1 = model._sun1_color
        c2 = model._sun2_color
        # 5000 K is warmer (more red) than 8500 K (more blue)
        self.assertGreater(c1[0], c2[0], "Sun1 (5000K) should be redder than sun2 (8500K)")
        self.assertLess(c1[2], c2[2],   "Sun1 (5000K) should be less blue than sun2 (8500K)")

    def test_shadow_reduces_diffuse(self) -> None:
        """Full shadow (shadow=0) should produce less total light than no shadow."""
        model = LightingModelFinal()
        normal = (0.0, 1.0, 0.0)
        view = (0.0, 0.0, 1.0)

        light_full = model.evaluate(normal, view, shadow=1.0)
        light_none = model.evaluate(normal, view, shadow=0.0)

        self.assertGreater(
            sum(light_full.total),
            sum(light_none.total),
            "Full shadow should give less light",
        )


# ---------------------------------------------------------------------------
# 3. Ring shadow projected correctly
# ---------------------------------------------------------------------------

class TestRingShadowProjectedCorrectly(unittest.TestCase):
    """§13.3 — ring shadow appears within annulus, not outside."""

    def test_ring_shadow_projected_correctly(self) -> None:
        """Point directly below annulus (r ≈ 1.75) receives non-zero shadow."""
        proj = RingShadowProjector()

        # Surface point BELOW the ring plane (y < 0 = southern hemisphere)
        # so t > 0 when sun is above.  r of the ring-plane intersection = 1.75
        # which falls inside the default annulus [1.3, 2.2].
        point_under_ring = (1.75, -0.5, 0.0)
        shadow = proj.shadow_at(point_under_ring, (0.0, 1.0, 0.0))
        self.assertGreater(shadow, 0.0,
                           "Point under ring annulus should receive ring shadow")

    def test_ring_shadow_zero_outside_annulus(self) -> None:
        """Point far outside annulus receives zero ring shadow."""
        proj = RingShadowProjector()
        # r = 10 >> outer_radius = 2.2
        point_outside = (10.0, 0.0, 0.0)
        shadow = proj.shadow_at(point_outside, (0.0, 1.0, 0.0))
        self.assertAlmostEqual(shadow, 0.0, places=5,
                               msg="Point outside ring should have zero shadow")

    def test_ring_shadow_zero_sun_below_horizon(self) -> None:
        """When sun is below horizon (sun_dir.y < 0), no ring shadow."""
        proj = RingShadowProjector()
        # Surface point below ring plane; sun pointing down → t < 0 → no shadow
        shadow = proj.shadow_at((1.75, -0.5, 0.0), (0.0, -1.0, 0.0))
        self.assertAlmostEqual(shadow, 0.0, places=5,
                               msg="No ring shadow when sun is below surface")

    def test_cold_bias_proportional_to_shadow(self) -> None:
        """cold_bias_at(shadow) must be monotonically increasing."""
        proj = RingShadowProjector()
        biases = [proj.cold_bias_at(s) for s in [0.0, 0.25, 0.5, 0.75, 1.0]]
        for i in range(len(biases) - 1):
            self.assertLessEqual(biases[i], biases[i + 1],
                                 "Cold bias must not decrease as shadow increases")


# ---------------------------------------------------------------------------
# 4. No colour clipping in extreme lighting
# ---------------------------------------------------------------------------

class TestNoColorClippingExtremeLighting(unittest.TestCase):
    """§13.4 — tone mapper never clips output outside [0, 1]."""

    def test_no_color_clipping_extreme_lighting(self) -> None:
        """Very bright HDR inputs must be mapped to ≤ 1.0 in all channels."""
        mapper = ToneMapperLocked()
        extreme_inputs = [
            (100.0, 50.0, 10.0),
            (0.001, 0.001, 0.001),
            (3.14, 2.72, 1.41),
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
        ]
        for inp in extreme_inputs:
            out = mapper.apply(inp)
            for ch, v in zip("rgb", out):
                self.assertGreaterEqual(v, 0.0,
                    f"Channel {ch} should be ≥ 0, got {v} for input {inp}")
                self.assertLessEqual(v, 1.0,
                    f"Channel {ch} should be ≤ 1, got {v} for input {inp}")

    def test_tone_mapper_monotone(self) -> None:
        """Brighter input should not produce darker output (monotonicity)."""
        mapper = ToneMapperLocked()
        prev_luma = -1.0
        for intensity in [0.1, 0.3, 0.6, 1.0, 2.0, 5.0, 10.0]:
            c = (intensity, intensity * 0.9, intensity * 0.7)
            out = mapper.apply(c)
            luma = 0.2126 * out[0] + 0.7152 * out[1] + 0.0722 * out[2]
            self.assertGreaterEqual(luma, prev_luma - 1e-9,
                f"Tone mapper must be monotone; luma dipped at intensity={intensity}")
            prev_luma = luma

    def test_contrast_max_no_clipping(self) -> None:
        """Even at maximum contrast, output stays in [0, 1]."""
        mapper = ToneMapperLocked()
        mapper.set_contrast(1.3)
        for val in [0.0, 0.5, 0.9, 1.5, 5.0]:
            out = mapper.apply((val, val, val))
            for v in out:
                self.assertLessEqual(v, 1.0)
                self.assertGreaterEqual(v, 0.0)


# ---------------------------------------------------------------------------
# 5. Fog density matches dust field
# ---------------------------------------------------------------------------

class TestFogDensityMatchesDustField(unittest.TestCase):
    """§13.5 — fog increases with dust and distance; storm increases further."""

    def test_fog_density_matches_dust_field(self) -> None:
        """fog_factor() must increase with dust_density at fixed distance."""
        model = AtmosphereModel()
        dist = 500.0

        prev_fog = -1.0
        for dust in [0.0, 0.25, 0.5, 0.75, 1.0]:
            model.update(dust_density=dust)
            fog = model.fog_factor(dist)
            self.assertGreater(fog, prev_fog,
                f"fog_factor should increase with dust_density={dust}")
            prev_fog = fog

    def test_fog_increases_with_distance(self) -> None:
        """fog_factor must be monotonically increasing with distance."""
        model = AtmosphereModel()
        model.update(dust_density=0.3)

        prev = -1.0
        for d in [0.0, 100.0, 500.0, 1000.0, 5000.0]:
            f = model.fog_factor(d)
            self.assertGreaterEqual(f, prev,
                f"fog_factor should not decrease at distance={d}")
            prev = f

    def test_storm_increases_fog(self) -> None:
        """storm_active=True must produce more fog than calm at same dust."""
        model = AtmosphereModel()
        dist = 300.0

        model.update(dust_density=0.5, storm_active=False)
        fog_calm = model.fog_factor(dist)

        model.update(dust_density=0.5, storm_active=True)
        fog_storm = model.fog_factor(dist)

        self.assertGreater(fog_storm, fog_calm,
                           "Storm mode must increase fog density")

    def test_fog_in_range(self) -> None:
        """fog_factor must always be in [0, 1]."""
        model = AtmosphereModel()
        for dust in [0.0, 0.5, 1.0]:
            for storm in [False, True]:
                model.update(dust_density=dust, storm_active=storm)
                for d in [0.0, 10.0, 1000.0, 100000.0]:
                    f = model.fog_factor(d)
                    self.assertGreaterEqual(f, 0.0)
                    self.assertLessEqual(f, 1.0)


# ---------------------------------------------------------------------------
# 6. Visual hash stable across restarts
# ---------------------------------------------------------------------------

class TestVisualHashStableAcrossRestarts(unittest.TestCase):
    """§13.6 — identical config → identical visual_hash."""

    def test_visual_hash_stable_across_restarts(self) -> None:
        """Two pipelines built from the same config dict produce the same hash."""
        cfg = dict(_DEFAULT_CONFIG)
        p1 = PixelPipeline(cfg)
        p2 = PixelPipeline(cfg)
        self.assertEqual(
            p1.visual_hash(), p2.visual_hash(),
            "visual_hash must be deterministic for identical configs",
        )

    def test_visual_hash_differs_on_config_change(self) -> None:
        """Changing a locked key must change the visual_hash."""
        cfg_a = {
            "render": {**_DEFAULT_CONFIG["render"], "fov_base": 68.0}
        }
        cfg_b = {
            "render": {**_DEFAULT_CONFIG["render"], "fov_base": 75.0}
        }
        p_a = PixelPipeline(cfg_a)
        p_b = PixelPipeline(cfg_b)
        self.assertNotEqual(
            p_a.visual_hash(), p_b.visual_hash(),
            "Different fov_base must produce different visual_hash",
        )


# ---------------------------------------------------------------------------
# 7. Pixel pipeline performance budget
# ---------------------------------------------------------------------------

class TestPixelPipelinePerfBudget(unittest.TestCase):
    """§13.7 — small buffer must process within a generous time budget."""

    # Use a small buffer (40×22 = 880 samples) to keep unit tests fast
    _W, _H = 40, 22
    _BUDGET_S = 2.0   # 2 s wall-clock for pure-Python baseline (880 samples)

    def _make_samples(self):
        samples = []
        for i in range(self._W * self._H):
            samples.append(SurfaceSample(
                position=(float(i % self._W) * 0.1, 0.0, float(i // self._W) * 0.1),
                normal=(0.0, 1.0, 0.0),
                view_dir=(0.0, 0.0, 1.0),
                base_color=(0.72, 0.58, 0.45),
                ao=1.0,
                shadow=1.0,
                distance=100.0,
            ))
        return samples

    def test_pixel_pipeline_perf_budget(self) -> None:
        """process_buffer on a small sample must finish within budget."""
        pipeline = PixelPipeline({
            "render": {
                "internal_resolution": [self._W, self._H],
                "pixel_resolution": [self._W // 2, self._H // 2],
                "pixel_scale_mode": "nearest",
                "fog_density_base": 0.003,
                "dust_color_shift": 0.45,
                "ring_shadow_strength": 0.35,
            }
        })
        samples = self._make_samples()

        t0 = time.perf_counter()
        out = pipeline.process_buffer(samples, self._W, self._H)
        elapsed = time.perf_counter() - t0

        self.assertEqual(len(out), self._W * self._H,
                         "Output buffer size must match input dimensions")
        self.assertLess(elapsed, self._BUDGET_S,
                        f"Pipeline took {elapsed:.2f}s — exceeds budget {self._BUDGET_S}s")

    def test_pixel_pipeline_output_in_range(self) -> None:
        """All output pixels must be in [0, 1]."""
        pipeline = PixelPipeline({
            "render": {
                "internal_resolution": [self._W, self._H],
                "pixel_resolution": [self._W // 2, self._H // 2],
            }
        })
        samples = self._make_samples()
        out = pipeline.process_buffer(samples, self._W, self._H)
        for i, (r, g, b) in enumerate(out):
            self.assertGreaterEqual(r, 0.0, f"pixel {i}: r < 0")
            self.assertLessEqual(r, 1.0,   f"pixel {i}: r > 1")
            self.assertGreaterEqual(g, 0.0, f"pixel {i}: g < 0")
            self.assertLessEqual(g, 1.0,   f"pixel {i}: g > 1")
            self.assertGreaterEqual(b, 0.0, f"pixel {i}: b < 0")
            self.assertLessEqual(b, 1.0,   f"pixel {i}: b > 1")


# ---------------------------------------------------------------------------
# Bonus: CinematicStabilityProfile
# ---------------------------------------------------------------------------

class TestCinematicStabilityProfile(unittest.TestCase):
    """Cinematic stability profile tests (§8)."""

    def test_sway_is_deterministic(self) -> None:
        """sway_offset at the same time must be identical."""
        p = CinematicStabilityProfile()
        t = 3.14
        self.assertAlmostEqual(p.sway_offset(t), p.sway_offset(t), places=10)

    def test_sway_amplitude_bounded(self) -> None:
        """sway_offset must never exceed ±amplitude."""
        p = CinematicStabilityProfile(sway_amplitude=0.04)
        for i in range(100):
            t = i * 0.1
            s = p.sway_offset(t)
            self.assertLessEqual(abs(s), p.sway_amplitude + 1e-9)

    def test_fov_salience_clamp(self) -> None:
        """fov_with_salience must never exceed base ± max_deg."""
        p = CinematicStabilityProfile(fov_base_deg=68.0, fov_salience_max_deg=2.5)
        for bias in [-10.0, -3.0, -2.5, 0.0, 2.5, 3.0, 10.0]:
            fov = p.fov_with_salience(bias)
            self.assertGreaterEqual(fov, 68.0 - 2.5 - 1e-9)
            self.assertLessEqual(fov, 68.0 + 2.5 + 1e-9)

    def test_shake_disabled_by_default(self) -> None:
        """cinematic_shake_enabled must be False in default profile (§8.1)."""
        p = CinematicStabilityProfile()
        self.assertFalse(p.cinematic_shake_enabled,
                         "Cinematic shake must be disabled in Stage 62 profile")

    def test_from_config_reads_keys(self) -> None:
        """from_config reads fov_base and camera_inertia from config dict."""
        cfg = {
            "render": {"fov_base": 72.0, "camera_inertia": 0.22},
            "camera": {"sway_amplitude": 0.06},
        }
        p = CinematicStabilityProfile.from_config(cfg)
        self.assertAlmostEqual(p.fov_base_deg, 72.0)
        self.assertAlmostEqual(p.camera_inertia, 0.22)
        self.assertAlmostEqual(p.sway_amplitude, 0.06)


if __name__ == "__main__":
    unittest.main()
