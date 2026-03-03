"""test_world3d — 3D Field Core unit tests.

Tests
-----
1. TestSDFBase         — sphere SDF sign convention
2. TestSDFPatch        — delta computation, serialisation round-trip
3. TestSDFVolume       — eval, apply_patch, patches_since, revision tracking
4. TestFieldSet        — sample returns FieldSample, procedural variation
5. TestMaterialDB      — family lookup, default states, round-trip serialisation
6. TestBodyConstraint  — default skeleton structure, mass update, serialisation
7. TestWorld3D         — init, tick generates patches, player lifecycle
8. TestWorldState3D    — append_sdf_patch persistence, load_sdf_patches,
                         baseline3d round-trip
"""
from __future__ import annotations

import math
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.sim.sdf.SDFBase    import SDFBase
from src.sim.sdf.SDFPatch   import SDFPatch, KIND_SPHERE_DENT, KIND_SPHERE_DEPOSIT
from src.sim.sdf.SDFVolume  import SDFVolume
from src.sim.fields.FieldSet import FieldSet, FieldSample
from src.sim.materials.Material    import FAMILIES
from src.sim.materials.MaterialDB  import MaterialDB
from src.sim.materials.MaterialState import MaterialState
from src.sim.body.BodyConstraintGraph import BodyConstraintGraph
from src.sim.world.World3D  import World3D
from src.net.WorldState     import WorldState


# ---------------------------------------------------------------------------
# 1. SDFBase
# ---------------------------------------------------------------------------

class TestSDFBase(unittest.TestCase):

    def test_outside_positive(self):
        base = SDFBase(radius=100.0)
        d = base.eval(0, 0, 110)  # 10 units above surface
        self.assertAlmostEqual(d, 10.0, places=6)

    def test_on_surface_zero(self):
        base = SDFBase(radius=100.0)
        d = base.eval(100, 0, 0)
        self.assertAlmostEqual(d, 0.0, places=6)

    def test_inside_negative(self):
        base = SDFBase(radius=100.0)
        d = base.eval(0, 0, 90)  # 10 units below surface
        self.assertAlmostEqual(d, -10.0, places=6)

    def test_radius_property(self):
        base = SDFBase(radius=500.0)
        self.assertEqual(base.radius, 500.0)


# ---------------------------------------------------------------------------
# 2. SDFPatch
# ---------------------------------------------------------------------------

class TestSDFPatch(unittest.TestCase):

    def _make(self, **kw):
        defaults = dict(patch_id=1, revision=1,
                        cx=0.0, cy=0.0, cz=0.0, radius=2.0, strength=0.5)
        defaults.update(kw)
        return SDFPatch(**defaults)

    def test_delta_zero_outside_radius(self):
        p = self._make(cx=0.0, cy=0.0, cz=0.0, radius=2.0)
        self.assertEqual(p.delta(3.0, 0.0, 0.0), 0.0)

    def test_delta_positive_at_centre_dent(self):
        p = self._make(cx=0.0, cy=0.0, cz=0.0, radius=2.0, strength=1.0,
                       kind=KIND_SPHERE_DENT)
        d = p.delta(0.0, 0.0, 0.0)
        self.assertGreater(d, 0.0)
        self.assertAlmostEqual(d, 1.0, places=6)

    def test_delta_negative_at_centre_deposit(self):
        p = self._make(cx=0.0, cy=0.0, cz=0.0, radius=2.0, strength=1.0,
                       kind=KIND_SPHERE_DEPOSIT)
        d = p.delta(0.0, 0.0, 0.0)
        self.assertLess(d, 0.0)

    def test_affects_within_radius(self):
        p = self._make(cx=0.0, cy=0.0, cz=0.0, radius=2.0)
        self.assertTrue(p.affects(1.0, 0.0, 0.0))

    def test_affects_outside_radius(self):
        p = self._make(cx=0.0, cy=0.0, cz=0.0, radius=2.0)
        self.assertFalse(p.affects(3.0, 0.0, 0.0))

    def test_serialise_round_trip(self):
        p = self._make(patch_id=7, revision=3, cx=1.5, cy=-2.0, cz=0.5,
                       radius=3.0, strength=0.2, kind=KIND_SPHERE_DENT)
        d = p.to_dict()
        p2 = SDFPatch.from_dict(d)
        self.assertEqual(p.patch_id, p2.patch_id)
        self.assertAlmostEqual(p.cx, p2.cx)
        self.assertAlmostEqual(p.strength, p2.strength)
        self.assertEqual(p.kind, p2.kind)


# ---------------------------------------------------------------------------
# 3. SDFVolume
# ---------------------------------------------------------------------------

class TestSDFVolume(unittest.TestCase):

    def test_initial_revision_zero(self):
        vol = SDFVolume(radius=100.0)
        self.assertEqual(vol.sdf_revision, 0)

    def test_apply_patch_bumps_revision(self):
        vol = SDFVolume(radius=100.0)
        p = SDFPatch(patch_id=1, revision=1, cx=0, cy=0, cz=0,
                     radius=1.0, strength=0.1)
        vol.apply_patch(p)
        self.assertEqual(vol.sdf_revision, 1)

    def test_eval_without_patches_equals_base(self):
        vol  = SDFVolume(radius=100.0)
        base = SDFBase(radius=100.0)
        self.assertAlmostEqual(vol.eval(0, 0, 110), base.eval(0, 0, 110), places=6)

    def test_eval_with_dent_increases_distance(self):
        """A sphere-dent patch on the surface should push the SDF upwards
        (make the surface distance larger) near the patch centre."""
        r   = 100.0
        vol = SDFVolume(radius=r)
        # Patch exactly on the north pole surface
        p   = SDFPatch(patch_id=1, revision=1,
                       cx=0.0, cy=r, cz=0.0,
                       radius=5.0, strength=2.0)
        vol.apply_patch(p)
        # Evaluate just above the surface at the patch location
        d_with = vol.eval(0.0, r + 0.1, 0.0)
        # Without patch the base SDF gives 0.1
        self.assertGreater(d_with, 0.1)

    def test_patches_since(self):
        vol = SDFVolume(radius=100.0)
        for i in range(1, 6):
            vol.apply_patch(SDFPatch(patch_id=i, revision=i,
                                     cx=0, cy=0, cz=0, radius=1, strength=0.1))
        new = vol.patches_since(3)
        self.assertEqual(len(new), 2)
        self.assertEqual({p.revision for p in new}, {4, 5})

    def test_baseline_dict(self):
        vol = SDFVolume(radius=500.0)
        d   = vol.to_baseline_dict()
        self.assertEqual(d["planet_radius"], 500.0)
        self.assertIn("sdf_revision", d)


# ---------------------------------------------------------------------------
# 4. FieldSet
# ---------------------------------------------------------------------------

class TestFieldSet(unittest.TestCase):

    def test_sample_returns_field_sample(self):
        fs  = FieldSet(planet_radius=1000.0, seed=42)
        out = fs.sample(0.0, 1000.0, 0.0, 0.0)
        self.assertIsInstance(out, FieldSample)

    def test_temp_in_range(self):
        fs = FieldSet(planet_radius=1000.0, seed=1)
        for t in [0, 100, 3600]:
            s = fs.sample(0, 1000, 0, t)
            self.assertGreaterEqual(s.temp, 0.0)
            self.assertLessEqual(s.temp, 1.0)

    def test_dust_in_range(self):
        fs = FieldSet(planet_radius=1000.0, seed=1)
        s  = fs.sample(500.0, 800.0, 200.0, 0.0)
        self.assertGreaterEqual(s.dust, 0.0)
        self.assertLessEqual(s.dust, 1.0)

    def test_wind_varies_with_time(self):
        fs = FieldSet(planet_radius=1000.0, seed=7)
        s1 = fs.sample(0, 1000, 0, 0.0)
        s2 = fs.sample(0, 1000, 0, 3600.0)
        # Wind should differ across a full period
        diff = abs(s1.wind_x - s2.wind_x) + abs(s1.wind_z - s2.wind_z)
        self.assertGreater(diff, 0.0)

    def test_snapshot_dict(self):
        fs = FieldSet(planet_radius=1000.0, seed=99)
        d  = fs.to_snapshot_dict()
        self.assertIn("fields_revision", d)
        self.assertIn("temp_base", d)

    def test_bump_revision(self):
        fs = FieldSet(planet_radius=1000.0, seed=0)
        self.assertEqual(fs.fields_revision, 0)
        fs.bump_revision()
        self.assertEqual(fs.fields_revision, 1)


# ---------------------------------------------------------------------------
# 5. MaterialDB
# ---------------------------------------------------------------------------

class TestMaterialDB(unittest.TestCase):

    def test_families_present(self):
        db = MaterialDB()
        self.assertIn("rock",     db.families)
        self.assertIn("regolith", db.families)
        self.assertIn("ice",      db.families)

    def test_default_state_returns_material_state(self):
        db = MaterialDB()
        s  = db.default_state("rock")
        self.assertIsInstance(s, MaterialState)
        self.assertEqual(s.family_name, "rock")

    def test_default_state_fallback(self):
        db = MaterialDB()
        s  = db.default_state("unknown_material")
        self.assertEqual(s.family_name, "rock")

    def test_rock_is_dense(self):
        db = MaterialDB()
        s  = db.default_state("rock")
        self.assertLess(s.porosity, 0.1)
        self.assertGreater(s.compaction, 0.8)

    def test_ice_is_cold(self):
        db = MaterialDB()
        s  = db.default_state("ice")
        self.assertLess(s.temp, 0.2)

    def test_round_trip(self):
        db  = MaterialDB(seed=77)
        d   = db.to_dict()
        db2 = MaterialDB.from_dict(d)
        s   = db2.default_state("regolith")
        self.assertEqual(s.family_name, "regolith")


# ---------------------------------------------------------------------------
# 6. BodyConstraintGraph
# ---------------------------------------------------------------------------

class TestBodyConstraint(unittest.TestCase):

    def test_default_skeleton_has_nodes_and_edges(self):
        g = BodyConstraintGraph("player1")
        self.assertGreater(len(g.nodes), 0)
        self.assertGreater(len(g.edges), 0)

    def test_has_core_node(self):
        g     = BodyConstraintGraph("p")
        names = {n.name for n in g.nodes}
        self.assertIn("core", names)

    def test_initial_mass_zero(self):
        g = BodyConstraintGraph("p")
        self.assertAlmostEqual(g.mass_kg, 0.0)

    def test_update_mass_bumps_revision(self):
        g = BodyConstraintGraph("p")
        self.assertEqual(g.body_revision, 0)
        g.update_mass({"torso": 3.5})
        self.assertEqual(g.body_revision, 1)

    def test_mass_after_update(self):
        g = BodyConstraintGraph("p")
        g.update_mass({"torso": 2.0, "head": 1.0})
        self.assertAlmostEqual(g.mass_kg, 3.0, places=5)

    def test_round_trip(self):
        g  = BodyConstraintGraph("player_x", seed=5)
        g.update_mass({"core": 1.0})
        d  = g.to_dict()
        g2 = BodyConstraintGraph.from_dict(d)
        self.assertEqual(g2._player_id, "player_x")
        self.assertEqual(len(g2.nodes), len(g.nodes))


# ---------------------------------------------------------------------------
# 7. World3D
# ---------------------------------------------------------------------------

class TestWorld3D(unittest.TestCase):

    def test_init(self):
        w = World3D(seed=42, planet_radius=100.0)
        self.assertEqual(w.seed, 42)
        self.assertAlmostEqual(w.sim_time, 0.0)
        self.assertIsNotNone(w.sdf_volume)
        self.assertIsNotNone(w.field_set)
        self.assertIsNotNone(w.material_db)

    def test_add_remove_player(self):
        w = World3D(seed=1, planet_radius=100.0)
        w.add_player("p1")
        self.assertIsNotNone(w.get_player_state("p1"))
        w.remove_player("p1")
        self.assertIsNone(w.get_player_state("p1"))

    def test_tick_advances_time(self):
        w = World3D(seed=2, planet_radius=100.0)
        w.tick(0.05)
        self.assertAlmostEqual(w.sim_time, 0.05, places=6)

    def test_tick_generates_patch_after_interval(self):
        """After enough ticks to accumulate >= _FOOTPRINT_INTERVAL seconds,
        at least one SDF patch should be generated for a registered player."""
        w = World3D(seed=3, planet_radius=100.0)
        w.add_player("ptest")
        # Tick enough to exceed the footprint interval (2.0 s)
        total_patches = []
        for _ in range(300):  # 300 × 0.016 ≈ 4.8 s
            total_patches += w.tick(0.016)
        self.assertGreater(len(total_patches), 0)

    def test_tick_with_no_players_generates_no_patches(self):
        w = World3D(seed=4, planet_radius=100.0)
        patches = w.tick(5.0)
        self.assertEqual(len(patches), 0)

    def test_player_stays_on_surface(self):
        """After a tick the player should remain at approximately planetRadius + hover."""
        w = World3D(seed=5, planet_radius=100.0)
        w.add_player("ps")
        w.tick(0.1)
        state = w.get_player_state("ps")
        pos   = state["pos"]
        r     = math.sqrt(sum(p*p for p in pos))
        self.assertAlmostEqual(r, 100.0 + 1.8, delta=0.5)

    def test_baseline_dict(self):
        w = World3D(seed=10, planet_radius=200.0)
        d = w.to_baseline_dict()
        self.assertEqual(d["seed"], 10)
        self.assertEqual(d["planet_radius"], 200.0)
        self.assertIn("sdf_revision", d)
        self.assertIn("fields_revision", d)

    def test_set_intent_accepted(self):
        w = World3D(seed=6, planet_radius=100.0)
        w.add_player("pi")
        # Should not raise
        w.set_intent("pi", {"move_x": 1.0, "move_z": 0.0, "look_yaw": 0.0})
        w.tick(0.1)

    def test_all_player_states(self):
        w = World3D(seed=7, planet_radius=100.0)
        w.add_player("a")
        w.add_player("b")
        states = w.all_player_states()
        self.assertEqual(len(states), 2)
        ids = {s["id"] for s in states}
        self.assertIn("a", ids)
        self.assertIn("b", ids)


# ---------------------------------------------------------------------------
# 8. WorldState SDF patch persistence
# ---------------------------------------------------------------------------

class TestWorldState3D(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._ws = WorldState(state_dir=self._tmpdir)
        self._ws.load_or_create(default_seed=1)

    def test_append_and_load_sdf_patch(self):
        patch = {
            "patch_id": 1, "revision": 1,
            "cx": 0.0, "cy": 100.0, "cz": 0.0,
            "radius": 2.0, "strength": 0.1, "kind": "sphere_dent",
        }
        self._ws.append_sdf_patch(patch)
        loaded = self._ws.load_sdf_patches()
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["patch_id"], 1)

    def test_multiple_patches_sorted_by_id(self):
        for i in [3, 1, 2]:
            self._ws.append_sdf_patch({"patch_id": i, "revision": i,
                                        "cx": 0, "cy": 0, "cz": 0,
                                        "radius": 1, "strength": 0.1})
        loaded = self._ws.load_sdf_patches()
        self.assertEqual(len(loaded), 3)
        ids = [p["patch_id"] for p in loaded]
        self.assertEqual(ids, sorted(ids))

    def test_save_and_load_baseline3d(self):
        baseline = {
            "seed": 42, "planet_radius": 1000.0,
            "sdf_revision": 5, "fields_revision": 2,
        }
        self._ws.save_baseline3d(baseline)
        loaded = self._ws.load_baseline3d()
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["seed"], 42)
        self.assertEqual(loaded["sdf_revision"], 5)

    def test_load_baseline3d_missing_returns_none(self):
        result = self._ws.load_baseline3d()
        self.assertIsNone(result)

    def test_load_sdf_patches_empty_initially(self):
        patches = self._ws.load_sdf_patches()
        self.assertEqual(patches, [])


if __name__ == "__main__":
    unittest.main()
