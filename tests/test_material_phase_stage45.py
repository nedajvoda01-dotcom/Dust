"""test_material_phase_stage45.py — Stage 45 Material Phase Changes & Wear.

Tests
-----
1. test_icefilm_forms_in_shadow_and_melts_in_sun
   — IceFilm field grows when insolation < 0.3 and shrinks when
     insolation > 0.5 (ring shadow / day-night cycle).

2. test_wind_polishes_roughness
   — Sustained high-wind + dust input reduces roughness over time.

3. test_compaction_reduces_deformation
   — After applying contact load, snowCompaction rises and the
     MaterialToDeformAdapter yields a lower indent_k.

4. test_crust_break_emits_audio_impulses
   — When crustHardness exceeds brittle_threshold and a large contact
     is applied, a BrittleEvent is returned and
     MaterialToAudioAdapter.brittle_impulses() produces ≥1 ContactImpulse.

5. test_effective_mu_changes_with_fields
   — effectiveMu decreases when iceFilm is high and increases when
     roughness is high (relative to baseline).

6. test_snapshot_restore_material_state
   — A grid serialised by MaterialStateSnapshot then restored produces
     identical grid_hash values (round-trip determinism).
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.material.SurfaceMaterialState import (
    SurfaceMaterialState,
    SurfaceMaterialStateGrid,
)
from src.material.PhaseChangeSystem import (
    BrittleEvent,
    ClimateSample,
    PhaseChangeSystem,
)
from src.material.MaterialToFrictionAdapter import MaterialToFrictionAdapter
from src.material.MaterialToDeformAdapter import MaterialToDeformAdapter
from src.material.MaterialToAudioAdapter import MaterialToAudioAdapter
from src.net.MaterialChunkReplicator import (
    MaterialChunkReplicator,
    rle_encode,
    rle_decode,
)
from src.save.MaterialStateSnapshot import MaterialStateSnapshot
from src.physics.MaterialYieldModel import MaterialClass, MaterialYieldModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_grid(w: int = 4, h: int = 4) -> SurfaceMaterialStateGrid:
    return SurfaceMaterialStateGrid(chunk_id=(0, 0), w=w, h=h)


def _night_sample() -> ClimateSample:
    """Full shadow / night — ice should form."""
    return ClimateSample(
        wind_speed=0.1,
        dust_density=0.1,
        insolation=0.0,
        temperature=0.1,
        moisture=0.0,
        storm_active=False,
    )


def _sun_sample() -> ClimateSample:
    """Full sunlight — ice should melt."""
    return ClimateSample(
        wind_speed=0.1,
        dust_density=0.1,
        insolation=1.0,
        temperature=0.9,
        moisture=0.0,
        storm_active=False,
    )


def _wind_sample() -> ClimateSample:
    """Strong dusty wind — roughness should decrease."""
    return ClimateSample(
        wind_speed=0.9,
        dust_density=0.8,
        insolation=0.5,
        temperature=0.5,
        moisture=0.0,
        storm_active=False,
    )


# ---------------------------------------------------------------------------
# 1. IceFilm forms in shadow and melts in sun
# ---------------------------------------------------------------------------

class TestIceFilm(unittest.TestCase):

    def test_icefilm_forms_in_shadow_and_melts_in_sun(self) -> None:
        system = PhaseChangeSystem()
        grid   = _make_grid()

        # Verify initial ice = 0
        cell = grid.cell(0, 0)
        self.assertAlmostEqual(cell.ice_film, 0.0, places=2)

        # Run many ticks in shadow
        for _ in range(200):
            system.tick(grid, _night_sample(), dt=2.0)

        cell = grid.cell(0, 0)
        self.assertGreater(cell.ice_film, 0.05,
                           "IceFilm should grow in shadow/night")

        # Now melt under sunlight
        ice_before = cell.ice_film
        for _ in range(200):
            system.tick(grid, _sun_sample(), dt=2.0)

        cell = grid.cell(0, 0)
        self.assertLess(cell.ice_film, ice_before,
                        "IceFilm should shrink in sunlight")

    def test_icefilm_values_stay_in_01(self) -> None:
        system = PhaseChangeSystem()
        grid   = _make_grid()

        for _ in range(500):
            system.tick(grid, _night_sample(), dt=5.0)
        for _ in range(500):
            system.tick(grid, _sun_sample(), dt=5.0)

        for iy in range(grid.h):
            for ix in range(grid.w):
                cell = grid.cell(ix, iy)
                self.assertGreaterEqual(cell.ice_film, 0.0)
                self.assertLessEqual(cell.ice_film, 1.0)


# ---------------------------------------------------------------------------
# 2. Wind polishes roughness
# ---------------------------------------------------------------------------

class TestRoughnessPolish(unittest.TestCase):

    def test_wind_polishes_roughness(self) -> None:
        system = PhaseChangeSystem()
        grid   = _make_grid()

        # Verify initial roughness ≈ 0.5
        cell = grid.cell(0, 0)
        self.assertAlmostEqual(cell.roughness, 0.5, delta=0.01)

        rough_before = cell.roughness
        # Run many wind ticks
        for _ in range(300):
            system.tick(grid, _wind_sample(), dt=2.0)

        cell = grid.cell(0, 0)
        self.assertLess(cell.roughness, rough_before,
                        "Roughness should decrease under sustained wind+dust")

    def test_roughness_stays_nonnegative(self) -> None:
        system = PhaseChangeSystem()
        grid   = _make_grid()

        for _ in range(2000):
            system.tick(grid, _wind_sample(), dt=5.0)

        for iy in range(grid.h):
            for ix in range(grid.w):
                self.assertGreaterEqual(grid.cell(ix, iy).roughness, 0.0)


# ---------------------------------------------------------------------------
# 3. Compaction reduces deformation
# ---------------------------------------------------------------------------

class TestCompactionDeformation(unittest.TestCase):

    def test_compaction_reduces_deformation(self) -> None:
        system = PhaseChangeSystem()
        deform = MaterialToDeformAdapter()
        model  = MaterialYieldModel()
        base_params = model.get(MaterialClass.SNOW)
        grid = _make_grid()

        # Initial indent_k at zero compaction
        state_before = grid.cell(0, 0).copy()
        params_before = deform.apply(base_params, state_before)

        # Apply contacts to compact the snow (use larger dt to exceed uint8 quantisation step)
        for _ in range(50):
            system.apply_contact(grid, 0, 0,
                                 normal_force=3000.0,
                                 area=0.05,
                                 dt=1.0)

        state_after = grid.cell(0, 0)
        self.assertGreater(state_after.snow_compaction, 0.0,
                           "Snow compaction should increase after contacts")

        params_after = deform.apply(base_params, state_after)
        self.assertLess(params_after.indent_k, params_before.indent_k,
                        "indent_k should decrease as snow compacts")
        self.assertGreater(params_after.yield_strength,
                           params_before.yield_strength,
                           "yield_strength should rise as snow compacts")


# ---------------------------------------------------------------------------
# 4. Crust break emits audio impulses
# ---------------------------------------------------------------------------

class TestCrustBreak(unittest.TestCase):

    def test_crust_break_emits_audio_impulses(self) -> None:
        system = PhaseChangeSystem({"brittle_threshold": 0.50,
                                    "crust_break_k": 0.40})
        audio  = MaterialToAudioAdapter()
        grid   = _make_grid()

        # Force crust hardness above threshold
        cell = grid.cell(0, 0)
        cell.crust_hardness = 0.85

        # Apply a heavy contact — should trigger brittle fracture
        event = system.apply_contact(grid, 0, 0,
                                     normal_force=10000.0,
                                     area=0.03,
                                     dt=0.016)
        self.assertIsNotNone(event, "A BrittleEvent should be emitted")
        self.assertIsInstance(event, BrittleEvent)
        self.assertGreater(event.impulse_count, 0)

        impulses = audio.brittle_impulses(event)
        self.assertGreater(len(impulses), 0,
                           "BrittleEvent should produce audio impulses")
        for imp in impulses:
            self.assertGreater(imp.impulse_magnitude, 0.0)

    def test_no_break_below_threshold(self) -> None:
        system = PhaseChangeSystem({"brittle_threshold": 0.70})
        grid   = _make_grid()
        cell   = grid.cell(0, 0)
        cell.crust_hardness = 0.30  # below threshold

        event = system.apply_contact(grid, 0, 0,
                                     normal_force=10000.0,
                                     area=0.03,
                                     dt=0.016)
        self.assertIsNone(event, "No event below brittle threshold")


# ---------------------------------------------------------------------------
# 5. Effective mu changes with fields
# ---------------------------------------------------------------------------

class TestEffectiveMu(unittest.TestCase):

    def test_effective_mu_decreases_with_ice_film(self) -> None:
        adapter = MaterialToFrictionAdapter()
        base_mu = 0.5

        no_ice = SurfaceMaterialState(ice_film=0.0, roughness=0.5)
        full_ice = SurfaceMaterialState(ice_film=1.0, roughness=0.5)

        mu_no_ice  = adapter.effective_mu(no_ice,   base_mu)
        mu_ice     = adapter.effective_mu(full_ice, base_mu)
        self.assertLess(mu_ice, mu_no_ice,
                        "Ice film should reduce effective friction")

    def test_effective_mu_higher_with_rough_surface(self) -> None:
        adapter = MaterialToFrictionAdapter()
        base_mu = 0.5

        smooth = SurfaceMaterialState(roughness=0.0, ice_film=0.0)
        rough  = SurfaceMaterialState(roughness=1.0, ice_film=0.0)

        mu_smooth = adapter.effective_mu(smooth, base_mu)
        mu_rough  = adapter.effective_mu(rough,  base_mu)
        self.assertGreater(mu_rough, mu_smooth,
                           "Rough surface should have higher friction")

    def test_effective_mu_in_valid_range(self) -> None:
        adapter = MaterialToFrictionAdapter()
        for dust in (0.0, 0.5, 1.0):
            for ice in (0.0, 0.5, 1.0):
                for rough in (0.0, 0.5, 1.0):
                    state = SurfaceMaterialState(
                        dust_thickness=dust,
                        ice_film=ice,
                        roughness=rough,
                    )
                    mu = adapter.effective_mu(state)
                    self.assertGreaterEqual(mu, 0.01)
                    self.assertLessEqual(mu, 1.0)


# ---------------------------------------------------------------------------
# 6. Snapshot round-trip
# ---------------------------------------------------------------------------

class TestSnapshotRestore(unittest.TestCase):

    def test_snapshot_restore_material_state(self) -> None:
        """Saving then restoring grids produces identical hashes."""
        # Build two different grids
        g1 = SurfaceMaterialStateGrid((0, 0), 4, 4)
        g2 = SurfaceMaterialStateGrid((1, 0), 4, 4)

        # Mutate g1
        g1.cell(0, 0).ice_film = 0.7
        g1.cell(1, 2).roughness = 0.1
        g1.cell(3, 3).crust_hardness = 0.9

        # Mutate g2
        g2.cell(0, 0).snow_compaction = 0.6
        g2.cell(2, 2).dust_thickness = 0.8

        snap = MaterialStateSnapshot()
        grids = {(0, 0): g1, (1, 0): g2}
        blob = snap.save(grids)
        restored = snap.load(blob)

        # Find restored grids by hash of original chunk_id
        from src.net.MaterialChunkReplicator import _hash32
        h1 = _hash32((0, 0))
        h2 = _hash32((1, 0))

        self.assertIn(h1, restored)
        self.assertIn(h2, restored)
        self.assertEqual(restored[h1].grid_hash(), g1.grid_hash())
        self.assertEqual(restored[h2].grid_hash(), g2.grid_hash())

    def test_rle_roundtrip(self) -> None:
        """RLE encode → decode is lossless."""
        for _ in range(10):
            data = bytes([b % 256 for b in range(100)])
            self.assertEqual(rle_decode(rle_encode(data)), data)

        # Uniform data (best-case compression)
        uniform = bytes([42] * 255)
        self.assertEqual(rle_decode(rle_encode(uniform)), uniform)


# ---------------------------------------------------------------------------
# 7. MaterialChunkReplicator encode/decode
# ---------------------------------------------------------------------------

class TestMaterialChunkReplicator(unittest.TestCase):

    def test_encode_decode_roundtrip(self) -> None:
        grid = SurfaceMaterialStateGrid((5, 3), 4, 4)
        grid.cell(0, 0).ice_film     = 0.8
        grid.cell(2, 1).roughness    = 0.1
        grid.cell(3, 3).crust_hardness = 0.6

        replicator = MaterialChunkReplicator()
        packet = replicator.encode_chunk((5, 3), grid)
        _, restored = replicator.decode_chunk(packet)

        self.assertEqual(restored.grid_hash(), grid.grid_hash())

    def test_should_send_only_on_change(self) -> None:
        grid = SurfaceMaterialStateGrid((0, 0), 2, 2)
        rep  = MaterialChunkReplicator()

        h1 = grid.grid_hash()
        self.assertTrue(rep.should_send((0, 0), h1),
                        "First send should be True")
        self.assertFalse(rep.should_send((0, 0), h1),
                         "No change → should not send again")

        grid.cell(0, 0).roughness = 0.0
        h2 = grid.grid_hash()
        self.assertTrue(rep.should_send((0, 0), h2),
                        "State changed → should send")


# ---------------------------------------------------------------------------
# 8. Audio profile blending
# ---------------------------------------------------------------------------

class TestMaterialToAudioAdapter(unittest.TestCase):

    def test_dust_heavy_increases_graininess(self) -> None:
        adapter = MaterialToAudioAdapter()
        from src.audio.MaterialAcousticDB import MAT_BASALT
        base = SurfaceMaterialState(dust_thickness=0.0)
        dusty = SurfaceMaterialState(dust_thickness=1.0)
        p_base  = adapter.profile_for(base,  MAT_BASALT)
        p_dusty = adapter.profile_for(dusty, MAT_BASALT)
        self.assertGreater(p_dusty.graininess, p_base.graininess,
                           "Heavy dust → higher graininess in profile")

    def test_ice_heavy_increases_profile_toward_ice(self) -> None:
        adapter = MaterialToAudioAdapter()
        from src.audio.MaterialAcousticDB import MAT_BASALT, MAT_ICE
        db = adapter._db
        no_ice  = SurfaceMaterialState(ice_film=0.0)
        icy     = SurfaceMaterialState(ice_film=1.0)
        ice_ref = db.get(MAT_ICE)
        p_no_ice = adapter.profile_for(no_ice, MAT_BASALT)
        p_icy    = adapter.profile_for(icy,    MAT_BASALT)
        # roughness should move toward ice roughness (0.15 < basalt 0.45)
        self.assertLess(p_icy.roughness, p_no_ice.roughness)


if __name__ == "__main__":
    unittest.main()
