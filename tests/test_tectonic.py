"""test_tectonic — validates Stage 5 geological / tectonic model.

Tests:
  1. TestPlateDeterminism   — same seed → same plates and velocities
  2. TestPlateAssignment    — stable plateId for fixed directions
  3. TestBoundaryClassifier — correct convergent/divergent/transform on synthetic plates
  4. TestHeightTectonicInfluence — tectonic height modifies surface measurably and within limits
  5. TestGeoFieldSampler    — GeoFieldSampler returns well-formed GeoSample
  6. TestGeoFeatures        — generate_geo_features returns deterministic non-empty list
  7. TestSDFMaterialChannels — generated chunk carries hardness/fracture/porosity arrays
"""
from __future__ import annotations

import math
import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.math.Vec3 import Vec3
from src.planet.TectonicPlatesSystem import (
    TectonicPlatesSystem,
    SphericalVoronoi,
    PlateBoundaryClassifier,
    BoundaryType,
    Plate,
    CrustType,
)
from src.planet.GeoFieldSampler import GeoFieldSampler
from src.planet.GeoFeatures import generate_geo_features, GeoFeatureKind
from src.planet.PlanetHeightProvider import PlanetHeightProvider
from src.planet.SDFGenerator import generate_chunk
from src.planet.SDFChunk import SDFChunkCoord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEED = 42


def _build_system(seed: int = SEED, count: int = 18) -> TectonicPlatesSystem:
    sys_ = TectonicPlatesSystem(seed=seed, plate_count=count)
    sys_.build()
    return sys_


# ---------------------------------------------------------------------------
# 1. TestPlateDeterminism
# ---------------------------------------------------------------------------

class TestPlateDeterminism(unittest.TestCase):
    """Same seed must produce bit-identical plates and velocities."""

    def test_same_seed_same_centers(self):
        """Two systems built with the same seed share identical plate centers."""
        s1 = _build_system()
        s2 = _build_system()
        self.assertEqual(len(s1.plates), len(s2.plates))
        for p1, p2 in zip(s1.plates, s2.plates):
            self.assertAlmostEqual(p1.center_dir.x, p2.center_dir.x, places=10)
            self.assertAlmostEqual(p1.center_dir.y, p2.center_dir.y, places=10)
            self.assertAlmostEqual(p1.center_dir.z, p2.center_dir.z, places=10)

    def test_same_seed_same_velocity(self):
        """Two systems built with the same seed share identical velocity tangents."""
        s1 = _build_system()
        s2 = _build_system()
        for p1, p2 in zip(s1.plates, s2.plates):
            self.assertAlmostEqual(
                p1.velocity_tangent.x, p2.velocity_tangent.x, places=10)
            self.assertAlmostEqual(
                p1.velocity_tangent.y, p2.velocity_tangent.y, places=10)

    def test_different_seed_different_centers(self):
        """Different seeds produce different plate configurations."""
        s1 = _build_system(seed=1)
        s2 = _build_system(seed=2)
        # At least one center should differ
        any_diff = any(
            abs(p1.center_dir.x - p2.center_dir.x) > 1e-6
            for p1, p2 in zip(s1.plates, s2.plates)
        )
        self.assertTrue(any_diff)

    def test_plate_count_from_config(self):
        """Plate count matches the value passed to TectonicPlatesSystem."""
        for n in (12, 18, 24, 30):
            s = _build_system(count=n)
            self.assertEqual(len(s.plates), n)

    def test_center_dirs_are_unit_vectors(self):
        """All plate center directions must be unit vectors."""
        s = _build_system()
        for p in s.plates:
            length = p.center_dir.length()
            self.assertAlmostEqual(length, 1.0, places=6)

    def test_velocity_tangent_is_tangent(self):
        """Velocity tangent must be perpendicular to the center direction."""
        s = _build_system()
        for p in s.plates:
            dot = abs(p.center_dir.dot(p.velocity_tangent))
            self.assertLess(dot, 1e-6,
                msg=f"Plate {p.id}: velocity not tangent to center (dot={dot})")


# ---------------------------------------------------------------------------
# 2. TestPlateAssignment
# ---------------------------------------------------------------------------

class TestPlateAssignment(unittest.TestCase):
    """Voronoi plate assignment must be stable and exhaustive."""

    def setUp(self):
        self.system = _build_system()

    def test_repeated_query_same_result(self):
        """Querying the same direction twice returns the same plate id."""
        dirs = [
            Vec3(1.0, 0.0, 0.0),
            Vec3(0.0, 1.0, 0.0),
            Vec3(0.0, 0.0, 1.0),
            Vec3(0.577, 0.577, 0.577).normalized(),
        ]
        for d in dirs:
            with self.subTest(d=d):
                id1 = self.system.sample_plate_id(d)
                id2 = self.system.sample_plate_id(d)
                self.assertEqual(id1, id2)

    def test_plate_id_in_valid_range(self):
        """All sampled plate ids must be in [0, plate_count)."""
        import random
        rng = random.Random(99)
        for _ in range(200):
            v = Vec3(rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1))
            v = v.normalized()
            pid = self.system.sample_plate_id(v)
            self.assertGreaterEqual(pid, 0)
            self.assertLess(pid, len(self.system.plates))

    def test_all_plates_assigned_at_least_once(self):
        """For a sufficient number of samples, every plate should own at least one."""
        assigned = set()
        import random
        rng = random.Random(7)
        for _ in range(10_000):
            v = Vec3(rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1))
            if v.length() < 1e-9:
                continue
            assigned.add(self.system.sample_plate_id(v.normalized()))
        # Allow at most 1 unassigned plate (tiny plates can be missed at low sample counts)
        missed = len(self.system.plates) - len(assigned)
        self.assertLessEqual(missed, 1,
            msg=f"Plates not assigned: {missed} / {len(self.system.plates)}")


# ---------------------------------------------------------------------------
# 3. TestBoundaryClassifier
# ---------------------------------------------------------------------------

class TestBoundaryClassifier(unittest.TestCase):
    """
    Synthetic two-plate tests so the expected result is analytically known.

    Plate A: centered at +X, moves in +Y direction.
    Plate B: centered at -X, moves in -Y direction.

    At the midpoint (0, 1, 0) normalised, the plates are moving away from
    each other along X → divergent.

    At (0, 0, 1) the plates slide parallel to the boundary → transform.
    """

    def _make_two_plate_system(self, vel_a: Vec3, vel_b: Vec3):
        """Build a minimal 2-plate system with controlled velocities."""
        center_a = Vec3(1.0, 0.0, 0.0)
        center_b = Vec3(-1.0, 0.0, 0.0)
        plates = [
            Plate(id=0, center_dir=center_a, velocity_tangent=vel_a,
                  crust_type=CrustType.CONTINENTAL, strength=1.0),
            Plate(id=1, center_dir=center_b, velocity_tangent=vel_b,
                  crust_type=CrustType.CONTINENTAL, strength=1.0),
        ]
        voronoi = SphericalVoronoi([center_a, center_b])
        classifier = PlateBoundaryClassifier(plates, voronoi)
        return classifier

    # Query point on the boundary between +X and -X plates.
    # (0, 0, 1) is equidistant from both centres and has no degenerate projections
    # for velocities along X or Y.
    _BOUNDARY_MIDPOINT = Vec3(0.0, 0.0, 1.0)

    def test_convergent_classification(self):
        """
        Plates moving toward each other → CONVERGENT at the boundary midpoint.

        Centers: A at +X, B at -X.  Boundary midpoint: (0, 0, 1).
        Boundary normal (A→B projected at midpoint) ≈ (-1, 0, 0).
        A moves in −X, B moves in +X → relative velocity points in +X
        → dot with (−1,0,0) is negative → CONVERGENT.
        """
        vel_a = Vec3(-1.0, 0.0, 0.0)   # plate A moves toward plate B
        vel_b = Vec3(1.0, 0.0, 0.0)    # plate B moves toward plate A
        classifier = self._make_two_plate_system(vel_a, vel_b)

        btype, strength, _, _ = classifier.classify(self._BOUNDARY_MIDPOINT)
        self.assertEqual(btype, BoundaryType.CONVERGENT,
            msg=f"Expected CONVERGENT, got {btype}")
        self.assertGreater(strength, 0.0)

    def test_divergent_classification(self):
        """Plates moving away from the boundary → DIVERGENT."""
        vel_a = Vec3(1.0, 0.0, 0.0)    # plate A moves away from plate B
        vel_b = Vec3(-1.0, 0.0, 0.0)   # plate B moves away from plate A
        classifier = self._make_two_plate_system(vel_a, vel_b)

        btype, strength, _, _ = classifier.classify(self._BOUNDARY_MIDPOINT)
        self.assertEqual(btype, BoundaryType.DIVERGENT,
            msg=f"Expected DIVERGENT, got {btype}")
        self.assertGreater(strength, 0.0)

    def test_transform_classification(self):
        """
        Plates sliding tangentially → TRANSFORM.
        A moves +Y, B moves −Y at the boundary midpoint (0,0,1).
        """
        vel_a = Vec3(0.0, 1.0, 0.0)
        vel_b = Vec3(0.0, -1.0, 0.0)
        classifier = self._make_two_plate_system(vel_a, vel_b)

        btype, strength, _, _ = classifier.classify(self._BOUNDARY_MIDPOINT)
        self.assertEqual(btype, BoundaryType.TRANSFORM,
            msg=f"Expected TRANSFORM, got {btype}")
        self.assertGreater(strength, 0.0)

    def test_interior_point_is_none(self):
        """A point far from the boundary returns NONE with zero strength."""
        vel_a = Vec3(-1.0, 0.0, 0.0)
        vel_b = Vec3(1.0, 0.0, 0.0)
        classifier = self._make_two_plate_system(vel_a, vel_b)

        # +X is the centre of plate A — well inside plate A
        interior = Vec3(1.0, 0.0, 0.0)
        btype, strength, _, _ = classifier.classify(interior)
        self.assertEqual(btype, BoundaryType.NONE)
        self.assertAlmostEqual(strength, 0.0, places=5)

    def test_boundary_strength_range(self):
        """Boundary strength must always be in [0, 1]."""
        system = _build_system()
        import random
        rng = random.Random(13)
        for _ in range(500):
            v = Vec3(rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1))
            if v.length() < 1e-9:
                continue
            _, strength = system.sample_boundary(v.normalized())
            self.assertGreaterEqual(strength, 0.0)
            self.assertLessEqual(strength, 1.0)


# ---------------------------------------------------------------------------
# 4. TestHeightTectonicInfluence
# ---------------------------------------------------------------------------

class TestHeightTectonicInfluence(unittest.TestCase):
    """Tectonic system must measurably and plausibly alter surface height."""

    def _boundary_dirs(self, system: TectonicPlatesSystem, btype: BoundaryType,
                       count: int = 40):
        """Sample the field grid and return directions with the given boundary type."""
        result = []
        for direction, col, row in system.field._iter_directions():
            cell = system.field._cells[system.field._idx(col, row)]
            if cell.boundary_type == btype and cell.boundary_strength > 0.1:
                result.append(direction)
                if len(result) >= count:
                    break
        return result

    def test_convergent_adds_positive_height(self):
        """Convergent boundary zones should have a positive tectonic contribution."""
        system = _build_system()
        hp_plain = PlanetHeightProvider(seed=SEED)
        hp_tect  = PlanetHeightProvider(seed=SEED, tectonic_system=system)

        dirs = self._boundary_dirs(system, BoundaryType.CONVERGENT)
        if not dirs:
            self.skipTest("No convergent boundary cells found for this seed")

        gains = [
            hp_tect.sample_height(d) - hp_plain.sample_height(d)
            for d in dirs
        ]
        mean_gain = sum(gains) / len(gains)
        self.assertGreater(mean_gain, 0.0,
            msg=f"Expected positive mean gain at convergent zones, got {mean_gain:.3f}")

    def test_divergent_subtracts_height(self):
        """Divergent boundary zones should have a negative tectonic contribution."""
        system = _build_system()
        hp_plain = PlanetHeightProvider(seed=SEED)
        hp_tect  = PlanetHeightProvider(seed=SEED, tectonic_system=system)

        dirs = self._boundary_dirs(system, BoundaryType.DIVERGENT)
        if not dirs:
            self.skipTest("No divergent boundary cells found for this seed")

        gains = [
            hp_tect.sample_height(d) - hp_plain.sample_height(d)
            for d in dirs
        ]
        mean_gain = sum(gains) / len(gains)
        self.assertLess(mean_gain, 0.0,
            msg=f"Expected negative mean gain at divergent zones, got {mean_gain:.3f}")

    def test_height_within_scale_limits(self):
        """
        Tectonic height must stay within ±2 × HEIGHT_SCALE to avoid
        runaway terrain.
        """
        from src.planet.PlanetHeightProvider import HEIGHT_SCALE
        system = _build_system()
        hp = PlanetHeightProvider(seed=SEED, tectonic_system=system)

        import random
        rng = random.Random(321)
        heights = []
        for _ in range(200):
            v = Vec3(rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1))
            if v.length() < 1e-9:
                continue
            heights.append(hp.sample_height(v.normalized()))

        max_h = max(abs(h) for h in heights)
        self.assertLessEqual(max_h, HEIGHT_SCALE * 2.0,
            msg=f"Height {max_h:.2f} exceeds 2×HEIGHT_SCALE={HEIGHT_SCALE * 2:.2f}")

    def test_interior_not_affected(self):
        """Interior plate points (away from boundaries) should be unchanged."""
        system = _build_system()
        hp_plain = PlanetHeightProvider(seed=SEED)
        hp_tect  = PlanetHeightProvider(seed=SEED, tectonic_system=system)

        interior_dirs = []
        for direction, col, row in system.field._iter_directions():
            cell = system.field._cells[system.field._idx(col, row)]
            if cell.boundary_type == BoundaryType.NONE:
                interior_dirs.append(direction)
                if len(interior_dirs) >= 100:
                    break

        if not interior_dirs:
            self.skipTest("No interior cells found")

        diffs = [
            abs(hp_tect.sample_height(d) - hp_plain.sample_height(d))
            for d in interior_dirs
        ]
        max_diff = max(diffs)
        self.assertAlmostEqual(max_diff, 0.0, places=6,
            msg="Interior heights must be identical with and without tectonics")


# ---------------------------------------------------------------------------
# 5. TestGeoFieldSampler
# ---------------------------------------------------------------------------

class TestGeoFieldSampler(unittest.TestCase):
    """GeoFieldSampler must return valid, bounded GeoSamples."""

    def setUp(self):
        self.system  = _build_system()
        self.sampler = GeoFieldSampler(self.system)

    def test_sample_returns_geoSample(self):
        from src.planet.GeoFieldSampler import GeoSample
        s = self.sampler.sample(Vec3(1.0, 0.0, 0.0))
        self.assertIsInstance(s, GeoSample)

    def test_plate_id_valid(self):
        s = self.sampler.sample(Vec3(0.0, 1.0, 0.0))
        self.assertGreaterEqual(s.plate_id, 0)
        self.assertLess(s.plate_id, len(self.system.plates))

    def test_fields_bounded(self):
        import random
        rng = random.Random(55)
        for _ in range(200):
            v = Vec3(rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1))
            if v.length() < 1e-9:
                continue
            s = self.sampler.sample(v.normalized())
            self.assertGreaterEqual(s.stress,    0.0)
            self.assertLessEqual(s.stress,       1.0)
            self.assertGreaterEqual(s.fracture,  0.0)
            self.assertLessEqual(s.fracture,     1.0)
            self.assertGreaterEqual(s.stability, 0.0)
            self.assertLessEqual(s.stability,    1.0)
            self.assertGreaterEqual(s.hardness,  0.0)
            self.assertLessEqual(s.hardness,     1.0)

    def test_convenience_accessors_consistent(self):
        d = Vec3(0.5, 0.5, 0.707).normalized()
        s = self.sampler.sample(d)
        self.assertEqual(self.sampler.plate_id(d),  s.plate_id)
        self.assertEqual(self.sampler.boundary_type(d), s.boundary_type)
        self.assertAlmostEqual(self.sampler.stress(d),   s.stress,   places=10)
        self.assertAlmostEqual(self.sampler.fracture(d), s.fracture, places=10)
        self.assertAlmostEqual(self.sampler.stability(d), s.stability, places=10)
        self.assertAlmostEqual(self.sampler.hardness(d),  s.hardness,  places=10)

    def test_stress_accumulates_after_update(self):
        """Stress in boundary cells should increase after calling system.update()."""
        system = _build_system()
        sampler = GeoFieldSampler(system)

        # Find a boundary cell
        boundary_dir = None
        for direction, col, row in system.field._iter_directions():
            cell = system.field._cells[system.field._idx(col, row)]
            if cell.boundary_type != BoundaryType.NONE and cell.boundary_strength > 0.1:
                boundary_dir = direction
                break

        if boundary_dir is None:
            self.skipTest("No boundary cells available")

        stress_before = sampler.sample(boundary_dir).stress
        system.update(dt=100.0)   # large dt to ensure measurable change
        stress_after = sampler.sample(boundary_dir).stress
        self.assertGreater(stress_after, stress_before,
            msg="Stress should accumulate after update in boundary zones")


# ---------------------------------------------------------------------------
# 6. TestGeoFeatures
# ---------------------------------------------------------------------------

class TestGeoFeatures(unittest.TestCase):
    """generate_geo_features must return deterministic, non-empty results."""

    def test_features_are_nonempty(self):
        system = _build_system()
        features = generate_geo_features(system, seed=SEED)
        self.assertGreater(len(features), 0,
            msg="Expected at least one geo feature")

    def test_features_are_deterministic(self):
        system1 = _build_system()
        system2 = _build_system()
        f1 = generate_geo_features(system1, seed=SEED)
        f2 = generate_geo_features(system2, seed=SEED)
        self.assertEqual(len(f1), len(f2))
        for a, b in zip(f1, f2):
            self.assertEqual(a.kind, b.kind)
            self.assertAlmostEqual(a.intensity, b.intensity, places=10)

    def test_fault_and_weakness_features_present(self):
        system = _build_system()
        features = generate_geo_features(system, seed=SEED)
        kinds = {f.kind for f in features}
        # At minimum we should see FAULT_LINE or WEAKNESS_ZONE features
        self.assertTrue(
            GeoFeatureKind.FAULT_LINE in kinds or GeoFeatureKind.WEAKNESS_ZONE in kinds,
            msg=f"Expected FAULT_LINE or WEAKNESS_ZONE, got kinds={kinds}")

    def test_feature_anchors_are_unit_vectors(self):
        system = _build_system()
        features = generate_geo_features(system, seed=SEED)
        for f in features:
            self.assertAlmostEqual(f.anchor_dir.length(), 1.0, places=5)

    def test_feature_intensity_bounded(self):
        system = _build_system()
        features = generate_geo_features(system, seed=SEED)
        for f in features:
            self.assertGreaterEqual(f.intensity, 0.0)
            self.assertLessEqual(f.intensity,    1.0)


# ---------------------------------------------------------------------------
# 7. TestSDFMaterialChannels
# ---------------------------------------------------------------------------

class TestSDFMaterialChannels(unittest.TestCase):
    """SDFChunk must carry hardness/fracture/porosity arrays of the correct size."""

    def _make_chunk(self, with_geo=False):
        coord = SDFChunkCoord(face_id=0, lod=3, tile_x=2, tile_y=2, depth_index=0)
        hp = PlanetHeightProvider(seed=SEED)
        geo_sampler = None
        if with_geo:
            system = _build_system()
            geo_sampler = GeoFieldSampler(system)
        return generate_chunk(
            coord, resolution=8, voxel_depth=2.0,
            planet_radius=1000.0, height_provider=hp,
            geo_sampler=geo_sampler,
        )

    def test_material_channel_lengths(self):
        chunk = self._make_chunk()
        n = chunk.resolution ** 3
        self.assertEqual(len(chunk.hardness_field), n)
        self.assertEqual(len(chunk.fracture_field), n)
        self.assertEqual(len(chunk.porosity_field), n)

    def test_default_hardness_is_one(self):
        """Without geo_sampler, hardness defaults to 1.0."""
        chunk = self._make_chunk(with_geo=False)
        self.assertTrue(all(h == 1.0 for h in chunk.hardness_field))

    def test_default_fracture_and_porosity_are_zero(self):
        chunk = self._make_chunk(with_geo=False)
        self.assertTrue(all(f == 0.0 for f in chunk.fracture_field))
        self.assertTrue(all(p == 0.0 for p in chunk.porosity_field))

    def test_geo_sampler_populates_channels(self):
        """With geo_sampler the channels must be within [0, 1]."""
        chunk = self._make_chunk(with_geo=True)
        for val in chunk.hardness_field:
            self.assertGreaterEqual(val, 0.0)
            self.assertLessEqual(val,   1.0)
        for val in chunk.fracture_field:
            self.assertGreaterEqual(val, 0.0)
            self.assertLessEqual(val,   1.0)
        for val in chunk.porosity_field:
            self.assertGreaterEqual(val, 0.0)
            self.assertLessEqual(val,   1.0)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
