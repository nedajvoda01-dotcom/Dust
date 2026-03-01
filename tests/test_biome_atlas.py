"""test_biome_atlas — Stage 28 BiomeAtlasSystem smoke tests.

Tests
-----
1. test_biome_determinism
   Same seed + same coordinates → same BiomeId on repeated calls.

2. test_overlay_changes_with_climate
   Higher storm_intensity → higher dust overlay.

3. test_no_shimmer_static_camera
   Material output is bit-identical on successive evaluations with the
   same input (no time-varying noise, no shimmering).

4. test_fracture_bands_correlate_geo
   High-fracture geological inputs produce FRACTURE_BANDS more often.
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
from src.systems.BiomeAtlasSystem import (
    BiomeAtlas,
    BiomeId,
    ClimateInput,
    GeoInput,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_ATLAS = BiomeAtlas(seed=42)
_ATLAS_SAME_SEED = BiomeAtlas(seed=42)   # independent instance, same seed
_MAT = ProceduralMaterialSystem()


def _geo(
    fracture: float = 0.1,
    hardness: float = 0.5,
    slope: float = 0.1,
    height: float = 50.0,
    convergent: bool = False,
    divergent: bool = False,
    transform: bool = False,
    cave: bool = False,
) -> GeoInput:
    return GeoInput(
        fracture=fracture,
        hardness=hardness,
        slope=slope,
        height=height,
        is_convergent=convergent,
        is_divergent=divergent,
        is_transform=transform,
        is_cave=cave,
    )


def _climate(
    temperature: float = 290.0,
    storm: float = 0.0,
    dust: float = 0.0,
    ring_shadow: float = 0.0,
    wetness: float = 0.0,
) -> ClimateInput:
    return ClimateInput(
        temperature=temperature,
        storm_intensity=storm,
        dust_suspension=dust,
        ring_shadow=ring_shadow,
        wetness=wetness,
    )


# ---------------------------------------------------------------------------
# 1. test_biome_determinism
# ---------------------------------------------------------------------------

class TestBiomeDeterminism(unittest.TestCase):
    """Same seed + coordinates → identical BiomeId regardless of call order."""

    _COORDS = [
        (0.0,   0.0),
        (0.5,   1.2),
        (-0.8,  2.5),
        (1.1,  -1.0),
        (-0.2,  3.1),
        (0.0,  -3.1),
        (1.5,   0.0),
        (-1.5,  0.0),
    ]

    def test_same_seed_same_result(self):
        """Two BiomeAtlas instances with the same seed return the same biomes."""
        geo = _geo()
        for lat, lon in self._COORDS:
            b1 = _ATLAS.get_biome(lat, lon, geo)
            b2 = _ATLAS_SAME_SEED.get_biome(lat, lon, geo)
            self.assertEqual(
                b1, b2,
                f"BiomeId mismatch at lat={lat} lon={lon}: {b1} vs {b2}",
            )

    def test_repeated_call_same_result(self):
        """Calling get_biome twice with the same args returns the same BiomeId."""
        geo = _geo(fracture=0.4, slope=0.3)
        for lat, lon in self._COORDS:
            b1 = _ATLAS.get_biome(lat, lon, geo)
            b2 = _ATLAS.get_biome(lat, lon, geo)
            self.assertEqual(b1, b2, f"Non-deterministic biome at ({lat},{lon})")

    def test_different_seed_may_differ(self):
        """Different seeds should produce at least one different biome."""
        atlas_other = BiomeAtlas(seed=999)
        # Use a wider, more varied coordinate set to expose seed-dependent variation
        varied_coords = [
            (lat * 0.7, lon * 1.2)
            for lat in range(-3, 4)
            for lon in range(-3, 4)
        ]
        geo = _geo(fracture=0.3, hardness=0.5, slope=0.2)
        results_42    = [_ATLAS.get_biome(lat, lon, geo) for lat, lon in varied_coords]
        results_other = [atlas_other.get_biome(lat, lon, geo) for lat, lon in varied_coords]
        self.assertNotEqual(
            results_42, results_other,
            "Seeds 42 and 999 should not produce identical biome maps",
        )

    def test_all_biomes_are_valid(self):
        """Every returned BiomeId must be a member of the BiomeId enum."""
        geo = _geo()
        valid = set(BiomeId)
        for lat, lon in self._COORDS:
            b = _ATLAS.get_biome(lat, lon, geo)
            self.assertIn(b, valid, f"Invalid BiomeId {b!r} returned at ({lat},{lon})")


# ---------------------------------------------------------------------------
# 2. test_overlay_changes_with_climate
# ---------------------------------------------------------------------------

class TestOverlayChangesWithClimate(unittest.TestCase):
    """Overlay layers respond correctly to climate inputs."""

    def test_storm_increases_dust_overlay(self):
        """Higher storm_intensity → higher dust_thickness overlay."""
        geo = _geo(slope=0.1)   # flat surface → dust accumulates
        calm   = _climate(storm=0.0, dust=0.1)
        stormy = _climate(storm=1.0, dust=0.9)

        biome = BiomeId.IRON_SAND
        ov_calm   = _ATLAS.get_overlays(biome, geo, calm)
        ov_stormy = _ATLAS.get_overlays(biome, geo, stormy)

        self.assertGreater(
            ov_stormy.dust_thickness,
            ov_calm.dust_thickness,
            f"Storm should increase dust: {ov_stormy.dust_thickness:.4f} vs {ov_calm.dust_thickness:.4f}",
        )

    def test_cold_temperature_increases_ice(self):
        """Below ice threshold temperature → higher ice_film overlay."""
        geo = _geo()
        warm = _climate(temperature=290.0)
        cold = _climate(temperature=150.0)

        biome = BiomeId.ICE_CRUST
        ov_warm = _ATLAS.get_overlays(biome, geo, warm)
        ov_cold = _ATLAS.get_overlays(biome, geo, cold)

        self.assertGreater(
            ov_cold.ice_film,
            ov_warm.ice_film,
            f"Cold should increase ice: {ov_cold.ice_film:.4f} vs {ov_warm.ice_film:.4f}",
        )

    def test_ring_shadow_increases_ice(self):
        """Full ring shadow should add to ice overlay (cold + shadow)."""
        geo = _geo()
        no_shadow  = _climate(temperature=240.0, ring_shadow=0.0)
        full_shadow = _climate(temperature=240.0, ring_shadow=1.0)

        biome = BiomeId.ICE_CRUST
        ov_none = _ATLAS.get_overlays(biome, geo, no_shadow)
        ov_full = _ATLAS.get_overlays(biome, geo, full_shadow)

        self.assertGreaterEqual(
            ov_full.ice_film,
            ov_none.ice_film,
            "Ring shadow should increase ice overlay",
        )

    def test_steep_slope_reduces_dust_retention(self):
        """Steep slopes should retain less dust than flat ones."""
        biome   = BiomeId.IRON_SAND
        climate = _climate(storm=0.5, dust=0.7)
        flat    = _geo(slope=0.05)
        steep   = _geo(slope=0.90)

        ov_flat  = _ATLAS.get_overlays(biome, flat,  climate)
        ov_steep = _ATLAS.get_overlays(biome, steep, climate)

        self.assertGreaterEqual(
            ov_flat.dust_thickness,
            ov_steep.dust_thickness,
            "Flat terrain should retain more dust than steep slopes",
        )

    def test_overlay_values_in_range(self):
        """All overlay fields must be in [0, 1]."""
        climates = [
            _climate(storm=0.0, dust=0.0, temperature=300.0),
            _climate(storm=1.0, dust=1.0, temperature=100.0, ring_shadow=1.0, wetness=1.0),
        ]
        for biome in BiomeId:
            for clim in climates:
                geo = _geo()
                ov = _ATLAS.get_overlays(biome, geo, clim)
                for name, val in [
                    ("dust_thickness", ov.dust_thickness),
                    ("ice_film", ov.ice_film),
                    ("debris_thickness", ov.debris_thickness),
                    ("wetness", ov.wetness),
                ]:
                    self.assertGreaterEqual(val, 0.0, f"{name} < 0 for {biome}")
                    self.assertLessEqual(val, 1.0, f"{name} > 1 for {biome}")


# ---------------------------------------------------------------------------
# 3. test_no_shimmer_static_camera
# ---------------------------------------------------------------------------

class TestNoShimmerStaticCamera(unittest.TestCase):
    """Material output is bit-identical across evaluations — no shimmer."""

    def _make_mat_input(self, biome: BiomeId) -> MaterialInput:
        return MaterialInput(
            world_pos     = Vec3(123.4, 567.8, 901.2),
            world_normal  = Vec3(0.0, 1.0, 0.0),
            height        = 200.0,
            slope         = 0.2,
            curvature     = -0.1,
            hardness      = 0.6,
            fracture      = 0.3,
            boundary_type = BoundaryType.NONE,
            dust          = 0.2,
            wetness       = 0.0,
            ice           = 0.0,
            temperature   = 280.0,
            biome_id      = biome,
        )

    def test_identical_output_same_input(self):
        """Successive evaluate() calls with identical input → identical output."""
        for biome in BiomeId:
            inp = self._make_mat_input(biome)
            out1 = _MAT.evaluate(inp)
            out2 = _MAT.evaluate(inp)
            self.assertEqual(
                (out1.color.x, out1.color.y, out1.color.z, out1.roughness),
                (out2.color.x, out2.color.y, out2.color.z, out2.roughness),
                f"Non-deterministic output for biome {biome}",
            )

    def test_biome_changes_color(self):
        """Different biomes produce distinguishably different mean colours."""
        colours: dict[BiomeId, tuple] = {}
        for biome in BiomeId:
            inp = self._make_mat_input(biome)
            out = _MAT.evaluate(inp)
            colours[biome] = (round(out.color.x, 2), round(out.color.y, 2), round(out.color.z, 2))

        # At least two distinct rounded colours must exist
        unique = set(colours.values())
        self.assertGreater(
            len(unique), 1,
            "Different biomes should produce different colours; all identical",
        )

    def test_no_nan_or_inf_with_biome(self):
        """No NaN or Inf produced for any BiomeId."""
        for biome in BiomeId:
            inp = self._make_mat_input(biome)
            out = _MAT.evaluate(inp)
            for val in (out.color.x, out.color.y, out.color.z, out.roughness,
                        out.micro_normal.x, out.micro_normal.y, out.micro_normal.z):
                self.assertFalse(math.isnan(val), f"NaN for biome={biome}")
                self.assertFalse(math.isinf(val), f"Inf for biome={biome}")

    def test_debug_biome_mode_returns_valid_colors(self):
        """DEBUG_BIOME mode returns a valid [0,1] colour for every BiomeId."""
        for biome in BiomeId:
            inp = self._make_mat_input(biome)
            out = _MAT.evaluate(inp, debug_mode=DebugMode.DEBUG_BIOME)
            for comp in (out.color.x, out.color.y, out.color.z):
                self.assertGreaterEqual(comp, 0.0)
                self.assertLessEqual(comp, 1.0)
                self.assertFalse(math.isnan(comp))


# ---------------------------------------------------------------------------
# 4. test_fracture_bands_correlate_geo
# ---------------------------------------------------------------------------

class TestFractureBandsCorrelateGeo(unittest.TestCase):
    """FRACTURE_BANDS biome should be more common at high fracture intensity."""

    _SAMPLE_COORDS = [
        (lat * 0.3, lon * 0.5)
        for lat in range(-5, 6)
        for lon in range(-5, 6)
    ]

    def _count_biome(self, geo: GeoInput) -> int:
        count = 0
        for lat, lon in self._SAMPLE_COORDS:
            if _ATLAS.get_biome(lat, lon, geo) == BiomeId.FRACTURE_BANDS:
                count += 1
        return count

    def test_high_fracture_yields_more_fracture_bands(self):
        """High-fracture terrain must produce more FRACTURE_BANDS than low-fracture."""
        low_frac  = _geo(fracture=0.05, hardness=0.5, slope=0.2)
        high_frac = _geo(fracture=0.90, hardness=0.5, slope=0.2,
                         convergent=True, transform=False)

        n_low  = self._count_biome(low_frac)
        n_high = self._count_biome(high_frac)

        self.assertGreater(
            n_high, n_low,
            f"High fracture should yield more FRACTURE_BANDS: {n_high} vs {n_low}",
        )

    def test_convergent_boundary_increases_fracture_bands(self):
        """Convergent boundary increases FRACTURE_BANDS frequency."""
        interior   = _geo(fracture=0.60, convergent=False, transform=False)
        convergent = _geo(fracture=0.60, convergent=True,  transform=False)

        n_int  = self._count_biome(interior)
        n_conv = self._count_biome(convergent)

        self.assertGreaterEqual(
            n_conv, n_int,
            f"Convergent boundary should produce ≥ FRACTURE_BANDS than interior: {n_conv} vs {n_int}",
        )

    def test_cave_biomes_are_subsurface_types(self):
        """Cave regions should only return subsurface-appropriate biomes."""
        subsurface_biomes = {
            BiomeId.SUBSURFACE_BEDROCK,
            BiomeId.ASH_FIELDS,
            BiomeId.GLASSY_IMPACT,
        }
        geo = _geo(cave=True)
        for lat, lon in self._SAMPLE_COORDS[:25]:
            b = _ATLAS.get_biome(lat, lon, geo)
            self.assertIn(
                b, subsurface_biomes,
                f"Cave returned non-cave biome {b} at ({lat},{lon})",
            )


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
