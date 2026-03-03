"""test_instability_stage52.py — Stage 52 Planetary Self-Organizing Instability tests.

Tests
-----
1. test_shear_stress_triggers_dust_avalanche
   — After shearStressField and dustLoadField are set above threshold on a
     tile, DustAvalancheModel.process() returns a DustAvalancheEvent and the
     tile's shearStressField and dustLoadField decrease.

2. test_crust_failure_reduces_crust_hardness
   — When crustFailurePotential exceeds threshold, CrustFailureModel fires a
     CrustFailureEvent and InstabilityToMaterialAdapter reduces crust_hardness
     and raises roughness on a SurfaceMaterialState cell.

3. test_cascade_propagates_deterministically
   — Two independent CascadeProcessor runs with the same seed tiles and
     field values produce identical processed-tile lists.

4. test_instability_reduces_stress_field
   — After InstabilitySystem.tick() fires on a tile with high
     crustFailurePotential the field on that tile is lower than before.

5. test_audio_emitter_created_on_event
   — InstabilityToAudioAdapter.emitter_from_event() returns a STRUCTURAL
     AcousticEmitterRecord with non-zero infra and audible energies.

6. test_budget_limits_tiles_processed
   — With max_tiles_per_tick=4, InstabilitySystem processes no more than 4
     tiles even if the full grid has many eligible tiles.

7. test_determinism_same_seed_same_instability
   — Two independent runs with the same field initialisation and tick
     sequence produce identical state_hash values.
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.instability.InstabilityState          import InstabilityState
from src.instability.ShearStressEstimator      import ShearStressEstimator
from src.instability.CrustFailureModel         import CrustFailureModel, CrustFailureEvent
from src.instability.DustAvalancheModel        import DustAvalancheModel, DustAvalancheEvent
from src.instability.ThermalFractureModel      import ThermalFractureModel, ThermalFractureEvent
from src.instability.CascadeProcessor          import CascadeProcessor
from src.instability.InstabilitySystem         import InstabilitySystem
from src.instability.InstabilityToMaterialAdapter import InstabilityToMaterialAdapter
from src.instability.InstabilityToAudioAdapter import InstabilityToAudioAdapter
from src.audio.audio_world.EmitterAggregator   import EmitterType
from src.material.SurfaceMaterialState         import SurfaceMaterialState
from src.math.Vec3                             import Vec3
from src.net.InstabilityEventReplicator        import InstabilityEventReplicator
from src.save.InstabilitySnapshot              import InstabilitySnapshot


# ---------------------------------------------------------------------------
# Shared config
# ---------------------------------------------------------------------------

_W, _H = 8, 4

_CFG = {
    "instability": {
        "enable":              True,
        "shear_threshold":     0.6,
        "crust_threshold":     0.65,
        "overhang_threshold":  0.6,
        "thermal_threshold":   0.55,
        "cascade_radius_max":  3,
        "max_tiles_per_tick":  _W * _H,
        "energy_to_material_k": 0.4,
        "energy_to_audio_k":   0.6,
        "cascade_stress_incr": 0.15,
        "dust_threshold":      0.4,
    }
}


def _make_state() -> InstabilityState:
    return InstabilityState(width=_W, height=_H)


# ---------------------------------------------------------------------------
# 1. test_shear_stress_triggers_dust_avalanche
# ---------------------------------------------------------------------------

class TestDustAvalanche(unittest.TestCase):

    def test_shear_stress_triggers_dust_avalanche(self):
        state = _make_state()
        model = DustAvalancheModel(_CFG)
        tile  = state.tile(2, 1)

        # Set fields above threshold
        state.shearStressField[tile] = 0.85
        state.dustLoadField[tile]    = 0.75

        shear_before = state.shearStressField[tile]
        dust_before  = state.dustLoadField[tile]

        event = model.process(state, tile)

        self.assertIsNotNone(event, "DustAvalancheEvent should fire above threshold")
        self.assertIsInstance(event, DustAvalancheEvent)
        self.assertGreater(event.intensity, 0.0)
        self.assertLess(state.shearStressField[tile], shear_before,
                        "shearStressField should decrease after avalanche")
        self.assertLess(state.dustLoadField[tile], dust_before,
                        "dustLoadField should decrease after avalanche")

    def test_no_avalanche_below_threshold(self):
        state = _make_state()
        model = DustAvalancheModel(_CFG)
        tile  = state.tile(0, 0)

        state.shearStressField[tile] = 0.3   # below threshold
        state.dustLoadField[tile]    = 0.8

        event = model.process(state, tile)
        self.assertIsNone(event, "No event below shear threshold")


# ---------------------------------------------------------------------------
# 2. test_crust_failure_reduces_crust_hardness
# ---------------------------------------------------------------------------

class TestCrustFailure(unittest.TestCase):

    def test_crust_failure_reduces_crust_hardness(self):
        state   = _make_state()
        model   = CrustFailureModel(_CFG)
        adapter = InstabilityToMaterialAdapter(_CFG)
        tile    = state.tile(1, 2)

        state.crustFailurePotential[tile] = 0.90   # above threshold (0.65)

        event = model.process(state, tile)
        self.assertIsNotNone(event, "CrustFailureEvent should fire above threshold")
        self.assertIsInstance(event, CrustFailureEvent)

        cell = SurfaceMaterialState(crust_hardness=0.8, roughness=0.3)
        hard_before  = cell.crust_hardness
        rough_before = cell.roughness

        adapter.apply_crust_failure(event, cell)

        self.assertLess(cell.crust_hardness, hard_before,
                        "crust_hardness should decrease after crust failure")
        self.assertGreater(cell.roughness, rough_before,
                           "roughness should increase after crust failure")

    def test_no_failure_below_threshold(self):
        state = _make_state()
        model = CrustFailureModel(_CFG)
        tile  = state.tile(0, 0)

        state.crustFailurePotential[tile] = 0.40   # below threshold

        event = model.process(state, tile)
        self.assertIsNone(event)


# ---------------------------------------------------------------------------
# 3. test_cascade_propagates_deterministically
# ---------------------------------------------------------------------------

class TestCascadeDeterminism(unittest.TestCase):

    def _run_cascade(self) -> list:
        state = _make_state()
        # Set some tiles near threshold so cascade can propagate
        for i in range(state.size()):
            state.shearStressField[i] = 0.50

        processor = CascadeProcessor(_CFG)
        processed = processor.run(
            state,
            seed_tiles=[state.tile(3, 1)],
            field_name="shearStressField",
            increment=0.15,
            threshold=0.60,
        )
        return processed

    def test_cascade_propagates_deterministically(self):
        run_a = self._run_cascade()
        run_b = self._run_cascade()
        self.assertEqual(run_a, run_b,
                         "Cascade must produce identical tile lists across runs")

    def test_cascade_does_not_propagate_beyond_threshold(self):
        """Neighbours receive the increment but do not propagate further
        because their resulting value stays below the threshold."""
        state = _make_state()
        # Seed tile is above threshold; all others start at 0
        state.shearStressField[0] = 0.80

        processor = CascadeProcessor(_CFG)
        processed = processor.run(
            state,
            seed_tiles=[0],
            field_name="shearStressField",
            increment=0.05,   # neighbours end up at 0.05, well below 0.70
            threshold=0.70,
        )
        # Seed (0) and its immediate neighbours are processed (depth 1),
        # but neighbours-of-neighbours (depth 2) should NOT appear.
        depth2_candidates = set()
        for nb in state.neighbors(0):
            for nb2 in state.neighbors(nb):
                if nb2 != 0 and nb2 not in state.neighbors(0):
                    depth2_candidates.add(nb2)

        for t in processed:
            self.assertNotIn(t, depth2_candidates,
                             f"Tile {t} is a depth-2 tile and should not be in cascade")


# ---------------------------------------------------------------------------
# 4. test_instability_reduces_stress_field
# ---------------------------------------------------------------------------

class TestInstabilitySystemReducesField(unittest.TestCase):

    def test_instability_reduces_crust_failure_potential(self):
        state  = _make_state()
        system = InstabilitySystem(_CFG)
        tile   = state.tile(2, 1)

        state.crustFailurePotential[tile] = 0.90

        potential_before = state.crustFailurePotential[tile]
        events = system.tick(state, slope_map=None, sim_tick=0, dt=1.0)

        self.assertLess(state.crustFailurePotential[tile], potential_before,
                        "crustFailurePotential must decrease after event discharge")
        crust_events = [e for e in events if isinstance(e, CrustFailureEvent)]
        self.assertGreater(len(crust_events), 0,
                           "At least one CrustFailureEvent should be returned")

    def test_no_events_when_all_below_threshold(self):
        state  = _make_state()
        system = InstabilitySystem(_CFG)
        # All fields at 0
        events = system.tick(state, slope_map=None, sim_tick=0, dt=1.0)
        self.assertEqual(len(events), 0)


# ---------------------------------------------------------------------------
# 5. test_audio_emitter_created_on_event
# ---------------------------------------------------------------------------

class TestAudioEmitterCreated(unittest.TestCase):

    def test_audio_emitter_created_on_event(self):
        adapter = InstabilityToAudioAdapter(_CFG)
        pos     = Vec3(100.0, 0.0, 200.0)
        record  = adapter.emitter_from_event(pos, intensity=0.75, sim_tick=42)

        self.assertEqual(record.emitter_type, EmitterType.STRUCTURAL,
                         "Emitter must be STRUCTURAL type")
        self.assertGreater(record.band_energy_infra,   0.0,
                           "Infra energy must be non-zero")
        self.assertGreater(record.band_energy_audible, 0.0,
                           "Audible energy must be non-zero")
        self.assertEqual(record.created_tick, 42)
        self.assertGreater(record.ttl, 0)

    def test_zero_intensity_gives_zero_energy(self):
        adapter = InstabilityToAudioAdapter(_CFG)
        record  = adapter.emitter_from_event(Vec3.zero(), intensity=0.0, sim_tick=0)
        self.assertAlmostEqual(record.band_energy_infra,   0.0, places=3)
        self.assertAlmostEqual(record.band_energy_audible, 0.0, places=3)


# ---------------------------------------------------------------------------
# 6. test_budget_limits_tiles_processed
# ---------------------------------------------------------------------------

class TestBudgetLimit(unittest.TestCase):

    def test_budget_limits_tiles_processed(self):
        # All tiles at high crustFailurePotential — every tile would fire
        budget_cfg = {
            "instability": {
                **_CFG["instability"],
                "max_tiles_per_tick": 4,
            }
        }
        state  = _make_state()
        system = InstabilitySystem(budget_cfg)

        for i in range(state.size()):
            state.crustFailurePotential[i] = 0.95

        events = system.tick(state, slope_map=None, sim_tick=0, dt=1.0)

        # At most 4 tiles processed → at most 4 crust events
        crust_events = [e for e in events if isinstance(e, CrustFailureEvent)]
        self.assertLessEqual(len(crust_events), 4,
                             "Budget must cap crust failure events to max_tiles_per_tick")

    def test_round_robin_continues_across_ticks(self):
        """Two successive ticks with budget=1 should process different tiles."""
        budget_cfg = {
            "instability": {
                **_CFG["instability"],
                "max_tiles_per_tick": 1,
            }
        }
        state  = _make_state()
        system = InstabilitySystem(budget_cfg)

        for i in range(state.size()):
            state.crustFailurePotential[i] = 0.95

        offset_before = system._tile_offset
        system.tick(state, sim_tick=0, dt=1.0)
        offset_after = system._tile_offset

        self.assertNotEqual(offset_before, offset_after,
                            "Round-robin cursor must advance each tick")


# ---------------------------------------------------------------------------
# 7. test_determinism_same_seed_same_instability
# ---------------------------------------------------------------------------

class TestDeterminism(unittest.TestCase):

    def _run_simulation(self) -> str:
        state    = InstabilityState(width=_W, height=_H)
        shear    = ShearStressEstimator(_CFG)
        system   = InstabilitySystem(_CFG)

        slope_map = [float(i % 5) / 5.0  for i in range(state.size())]
        dust_map  = [float(i % 3) / 3.0  for i in range(state.size())]
        stress_map= [float(i % 7) / 7.0  for i in range(state.size())]

        for step in range(30):
            shear.tick(state, slope_map, dust_map, stress_map, dt=0.1)
            system.tick(state, slope_map=slope_map, sim_tick=step, dt=0.1)

        return state.state_hash()

    def test_determinism_same_seed_same_instability(self):
        hash_a = self._run_simulation()
        hash_b = self._run_simulation()
        self.assertEqual(hash_a, hash_b,
                         "Determinism violated: same inputs produced different hashes")


# ---------------------------------------------------------------------------
# Snapshot round-trip
# ---------------------------------------------------------------------------

class TestInstabilitySnapshot(unittest.TestCase):

    def test_snapshot_round_trip(self):
        state = _make_state()
        n     = state.size()
        for i in range(n):
            state.shearStressField[i]      = (i % 10) / 10.0
            state.crustFailurePotential[i] = (i % 7)  / 7.0
            state.dustLoadField[i]         = (i % 5)  / 5.0
            state.thermalGradientField[i]  = (i % 3)  / 3.0
            state.massOverhangField[i]     = (i % 9)  / 9.0

        snap      = InstabilitySnapshot()
        blob      = snap.save(state, sim_time=77.7)
        restored, meta = snap.load(blob)

        self.assertAlmostEqual(meta["sim_time"], 77.7, places=5)
        tolerance = 1 / 255.0 + 1e-9

        for i in range(n):
            self.assertAlmostEqual(
                restored.shearStressField[i], state.shearStressField[i], delta=tolerance
            )
            self.assertAlmostEqual(
                restored.crustFailurePotential[i], state.crustFailurePotential[i], delta=tolerance
            )

    def test_bad_magic_raises(self):
        snap = InstabilitySnapshot()
        with self.assertRaises(ValueError):
            snap.load(b"XXXX" + b"\x00" * 100)


# ---------------------------------------------------------------------------
# Replicator
# ---------------------------------------------------------------------------

class TestInstabilityEventReplicator(unittest.TestCase):

    def test_crust_event_round_trip(self):
        rep   = InstabilityEventReplicator(_CFG)
        event = CrustFailureEvent(tile=5, intensity=0.7, crust_delta=0.3, roughness_gain=0.15)
        wire  = rep.serialise_event(event)
        back  = rep.deserialise_event(wire)

        self.assertIsInstance(back, CrustFailureEvent)
        self.assertEqual(back.tile, 5)
        self.assertAlmostEqual(back.intensity, 0.7, delta=1/255.0+1e-9)

    def test_dust_event_round_trip(self):
        rep   = InstabilityEventReplicator(_CFG)
        event = DustAvalancheEvent(tile=3, intensity=0.5, dust_delta=0.2)
        wire  = rep.serialise_event(event)
        back  = rep.deserialise_event(wire)

        self.assertIsInstance(back, DustAvalancheEvent)
        self.assertEqual(back.tile, 3)

    def test_snapshot_build_and_apply(self):
        rep   = InstabilityEventReplicator(_CFG)
        state = _make_state()
        state.shearStressField[0] = 0.42

        msg = rep.build_snapshot(state)
        self.assertEqual(msg["type"], "INSTABILITY_STATE_52")

        restored = _make_state()
        ok = rep.apply_snapshot(restored, msg)
        self.assertTrue(ok)
        self.assertAlmostEqual(
            restored.shearStressField[0], 0.42, delta=1/255.0+1e-9
        )

    def test_apply_bad_snapshot_returns_false(self):
        rep   = InstabilityEventReplicator(_CFG)
        state = _make_state()
        ok    = rep.apply_snapshot(state, {"type": "WRONG"})
        self.assertFalse(ok)

    def test_broadcast_scheduling(self):
        rep = InstabilityEventReplicator(_CFG)
        self.assertTrue(rep.should_broadcast(0.0))
        rep.record_broadcast(sim_time=0.0)
        self.assertFalse(rep.should_broadcast(sim_time=30.0))
        self.assertTrue(rep.should_broadcast(sim_time=61.0))


if __name__ == "__main__":
    unittest.main()
