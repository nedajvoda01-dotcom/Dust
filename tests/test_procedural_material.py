"""test_procedural_material — Stage 10 ProceduralMaterialSystem tests.

Tests
-----
1. TestNoTexturesLoaded      — no texture-sampler API calls anywhere in project
2. TestMaterialDeterminism   — fixed worldPos → stable colour/roughness
3. TestDustVisualResponse    — higher climate.dust → higher brightness & roughness
4. TestBoundaryAccent        — boundary-zone material differs from interior zones
"""
from __future__ import annotations

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.math.Vec3 import Vec3
from src.planet.TectonicPlatesSystem import BoundaryType
from src.render.ProceduralMaterialSystem import (
    DebugMode,
    MaterialInput,
    ProceduralMaterialSystem,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_input(
    world_pos: Vec3 | None = None,
    world_normal: Vec3 | None = None,
    height: float = 100.0,
    slope: float = 0.1,
    curvature: float = 0.0,
    hardness: float = 0.5,
    fracture: float = 0.1,
    boundary_type: BoundaryType = BoundaryType.NONE,
    dust: float = 0.1,
    wetness: float = 0.0,
    ice: float = 0.0,
    temperature: float = 290.0,
) -> MaterialInput:
    if world_pos is None:
        world_pos = Vec3(1.0, 2.0, 3.0)
    if world_normal is None:
        world_normal = Vec3(0.0, 1.0, 0.0)
    return MaterialInput(
        world_pos     = world_pos,
        world_normal  = world_normal,
        height        = height,
        slope         = slope,
        curvature     = curvature,
        hardness      = hardness,
        fracture      = fracture,
        boundary_type = boundary_type,
        dust          = dust,
        wetness       = wetness,
        ice           = ice,
        temperature   = temperature,
    )


_SYSTEM = ProceduralMaterialSystem()


# ---------------------------------------------------------------------------
# 1. TestNoTexturesLoaded
# ---------------------------------------------------------------------------

class TestNoTexturesLoaded(unittest.TestCase):
    """Verify that no bitmap-texture API calls exist in the project sources."""

    # Patterns that would indicate texture file usage.
    _FORBIDDEN_PATTERNS = [
        "glTexImage2D",
        "glBindTexture",
        "sampler2D",
        "sampler3D",
        "texture(",
        "Image.open(",
        "PIL.Image",
        "cv2.imread",
        "pygame.image.load",
        "load_texture",
        "load_image",
    ]

    # Directories to scan (relative to project root)
    _SCAN_DIRS = ["src", "tests"]

    def _project_root(self) -> str:
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def _collect_py_files(self) -> list[str]:
        root = self._project_root()
        # Exclude this test file itself (its _FORBIDDEN_PATTERNS list would
        # otherwise trigger false positives).
        self_path = os.path.abspath(__file__)
        result: list[str] = []
        for d in self._SCAN_DIRS:
            base = os.path.join(root, d)
            for dirpath, _dirs, filenames in os.walk(base):
                for fn in filenames:
                    if fn.endswith(".py"):
                        full = os.path.join(dirpath, fn)
                        if os.path.abspath(full) != self_path:
                            result.append(full)
        return result

    def test_no_texture_sampler_calls(self):
        """None of the forbidden texture-API strings appear in .py source files."""
        violations: list[str] = []
        for path in self._collect_py_files():
            with open(path, encoding="utf-8", errors="replace") as fh:
                for lineno, line in enumerate(fh, start=1):
                    for pattern in self._FORBIDDEN_PATTERNS:
                        if pattern in line:
                            violations.append(
                                f"{path}:{lineno}: found '{pattern}'"
                            )
        self.assertEqual(
            violations,
            [],
            "Texture-sampler API calls found in project:\n" +
            "\n".join(violations),
        )


# ---------------------------------------------------------------------------
# 2. TestMaterialDeterminism
# ---------------------------------------------------------------------------

class TestMaterialDeterminism(unittest.TestCase):
    """Same MaterialInput → bit-identical MaterialOutput on repeated calls."""

    def _eval(self) -> tuple:
        inp = _make_input(world_pos=Vec3(12.34, 56.78, 90.12), slope=0.3, fracture=0.4)
        out = _SYSTEM.evaluate(inp)
        return (out.color.x, out.color.y, out.color.z, out.roughness)

    def test_color_stable(self):
        r1 = self._eval()
        r2 = self._eval()
        self.assertEqual(r1, r2, "Material output must be deterministic")

    def test_output_in_valid_range(self):
        inp = _make_input()
        out = _SYSTEM.evaluate(inp)
        for comp in (out.color.x, out.color.y, out.color.z):
            self.assertGreaterEqual(comp, 0.0)
            self.assertLessEqual(comp, 1.0)
        self.assertGreaterEqual(out.roughness, 0.0)
        self.assertLessEqual(out.roughness, 1.0)

    def test_micro_normal_is_unit(self):
        """Micro-normal must be (approximately) a unit vector."""
        inp = _make_input(fracture=0.7)
        out = _SYSTEM.evaluate(inp)
        length = out.micro_normal.length()
        self.assertAlmostEqual(length, 1.0, places=5,
            msg=f"micro_normal length={length:.6f} is not unit")

    def test_no_nan_or_inf(self):
        for pos in [Vec3(0, 0, 0), Vec3(1e6, 1e6, 1e6), Vec3(-500, 200, 300)]:
            inp = _make_input(world_pos=pos)
            out = _SYSTEM.evaluate(inp)
            for val in (out.color.x, out.color.y, out.color.z, out.roughness,
                        out.micro_normal.x, out.micro_normal.y, out.micro_normal.z):
                self.assertFalse(math.isnan(val), f"NaN in output at pos={pos}")
                self.assertFalse(math.isinf(val), f"Inf in output at pos={pos}")


# ---------------------------------------------------------------------------
# 3. TestDustVisualResponse
# ---------------------------------------------------------------------------

class TestDustVisualResponse(unittest.TestCase):
    """Higher climate.dust → brighter colour and higher roughness."""

    def _brightness(self, out) -> float:
        return (out.color.x + out.color.y + out.color.z) / 3.0

    def test_dust_increases_brightness(self):
        """Dusty surface (dust=0.8) must be brighter than clean (dust=0.0)."""
        low_dust  = _make_input(slope=0.05, curvature=-0.2, dust=0.0)
        high_dust = _make_input(slope=0.05, curvature=-0.2, dust=0.8)

        out_low  = _SYSTEM.evaluate(low_dust)
        out_high = _SYSTEM.evaluate(high_dust)

        b_low  = self._brightness(out_low)
        b_high = self._brightness(out_high)
        self.assertGreater(b_high, b_low,
            f"High dust brightness {b_high:.4f} should exceed low dust {b_low:.4f}")

    def test_dust_increases_roughness(self):
        """Dusty surface must have higher roughness than clean surface."""
        low_dust  = _make_input(slope=0.05, curvature=-0.1, dust=0.0, hardness=0.8)
        high_dust = _make_input(slope=0.05, curvature=-0.1, dust=0.9, hardness=0.8)

        out_low  = _SYSTEM.evaluate(low_dust)
        out_high = _SYSTEM.evaluate(high_dust)

        self.assertGreaterEqual(out_high.roughness, out_low.roughness,
            f"Roughness should be ≥ under dust: {out_high.roughness:.4f} vs {out_low.roughness:.4f}")

    def test_dust_accumulates_more_on_flat(self):
        """Flat areas (low slope) must have higher effective dust colour than steep."""
        flat   = _make_input(slope=0.02, curvature=-0.1, dust=0.5)
        steep  = _make_input(slope=0.90, curvature= 0.1, dust=0.5)

        out_flat  = _SYSTEM.evaluate(flat)
        out_steep = _SYSTEM.evaluate(steep)

        # Dust colour is light tan; flatter surface should trend lighter
        self.assertGreaterEqual(
            self._brightness(out_flat),
            self._brightness(out_steep),
            "Flat surfaces should accumulate more dust and be lighter",
        )


# ---------------------------------------------------------------------------
# 4. TestBoundaryAccent
# ---------------------------------------------------------------------------

class TestBoundaryAccent(unittest.TestCase):
    """Plate-boundary zones produce visually distinct material from interior."""

    def _mean_brightness(self, inputs) -> float:
        total = 0.0
        for inp in inputs:
            out = _SYSTEM.evaluate(inp)
            total += (out.color.x + out.color.y + out.color.z) / 3.0
        return total / len(inputs)

    def test_convergent_boundary_differs_from_interior(self):
        """CONVERGENT boundary material differs statistically from NONE interior."""
        interior_samples = [
            _make_input(
                world_pos = Vec3(float(i), 0.0, float(i * 2)),
                fracture  = 0.15,
                hardness  = 0.6,
                boundary_type = BoundaryType.NONE,
                dust = 0.1,
            )
            for i in range(1, 21)
        ]
        boundary_samples = [
            _make_input(
                world_pos = Vec3(float(i), 0.0, float(i * 2)),
                fracture  = 0.5,
                hardness  = 0.9,
                boundary_type = BoundaryType.CONVERGENT,
                dust = 0.1,
                slope = 0.6,
            )
            for i in range(1, 21)
        ]
        b_interior = self._mean_brightness(interior_samples)
        b_boundary = self._mean_brightness(boundary_samples)
        self.assertNotAlmostEqual(
            b_interior, b_boundary, places=2,
            msg=(
                f"CONVERGENT boundary brightness ({b_boundary:.4f}) should differ "
                f"from interior ({b_interior:.4f})"
            ),
        )

    def test_debug_boundary_mode_shows_distinct_colors(self):
        """DEBUG_BOUNDARY mode must return a different colour per boundary type."""
        colors: dict[BoundaryType, Vec3] = {}
        pos = Vec3(1.0, 2.0, 3.0)
        for bt in BoundaryType:
            inp = _make_input(world_pos=pos, boundary_type=bt)
            out = _SYSTEM.evaluate(inp, debug_mode=DebugMode.DEBUG_BOUNDARY)
            colors[bt] = out.color

        unique_colors = set(
            (round(c.x, 3), round(c.y, 3), round(c.z, 3))
            for c in colors.values()
        )
        self.assertEqual(
            len(unique_colors), len(BoundaryType),
            "Each BoundaryType must produce a unique debug colour",
        )

    def test_fracture_accent_darkens_boundary(self):
        """High fracture at a boundary should darken the surface relative to low fracture."""
        low_frac = _make_input(fracture=0.05, boundary_type=BoundaryType.CONVERGENT)
        hi_frac  = _make_input(fracture=0.90, boundary_type=BoundaryType.CONVERGENT)

        out_low = _SYSTEM.evaluate(low_frac)
        out_hi  = _SYSTEM.evaluate(hi_frac)

        b_low = (out_low.color.x + out_low.color.y + out_low.color.z) / 3.0
        b_hi  = (out_hi.color.x  + out_hi.color.y  + out_hi.color.z)  / 3.0
        self.assertLessEqual(b_hi, b_low,
            f"High fracture boundary ({b_hi:.4f}) should be ≤ low fracture ({b_low:.4f})")

    def test_all_debug_modes_return_valid_output(self):
        """All DebugMode values must produce a valid MaterialOutput."""
        inp = _make_input(slope=0.5, fracture=0.4, dust=0.3,
                          boundary_type=BoundaryType.TRANSFORM)
        for mode in DebugMode:
            out = _SYSTEM.evaluate(inp, debug_mode=mode)
            for comp in (out.color.x, out.color.y, out.color.z):
                self.assertGreaterEqual(comp, 0.0)
                self.assertLessEqual(comp, 1.0)
                self.assertFalse(math.isnan(comp))


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
