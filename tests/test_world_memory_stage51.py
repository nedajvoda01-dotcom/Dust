"""test_world_memory_stage51.py — Stage 51 Emergent Memory of the World tests.

Tests
-----
1. test_repeated_steps_increase_compaction
   — Repeated apply_pressure calls accumulate compactionHistoryField above
     zero; a tile that receives more pressure has higher compaction than one
     that receives less.

2. test_stress_reduces_crust_stability
   — After StressAccumulator.apply_contact builds up stress on a tile,
     MemoryToEvolutionAdapter.evolution_deltas returns a negative
     crust_stability_delta (crust weakens under stress).

3. test_memory_fields_decay_over_time
   — After setting all memory fields to non-zero values, repeated
     StressAccumulator.tick / CompactionHistorySystem.tick /
     ErosionBiasSystem.tick calls with no new input reduce the fields
     toward zero (memory fades).

4. test_erosion_bias_affects_slope_creep
   — ErosionBiasSystem.tick raises erosionBiasField on a high-slope +
     high-wind tile; MemoryToEvolutionAdapter.evolution_deltas for that
     tile returns a positive slope_creep_delta.

5. test_snapshot_restore_memory
   — WorldMemorySnapshot.save() + .load() round-trips WorldMemoryState
     with lossiness bounded by 8-bit quantisation (≤ 1/255 ≈ 0.004).

6. test_determinism_same_inputs_same_memory
   — Two independent runs with the same initial contacts and tick sequence
     produce identical state_hash values.
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.memory.WorldMemoryState          import WorldMemoryState
from src.memory.StressAccumulator         import StressAccumulator
from src.memory.CompactionHistorySystem   import CompactionHistorySystem
from src.memory.ErosionBiasSystem         import ErosionBiasSystem
from src.memory.MemoryToMaterialAdapter   import MemoryToMaterialAdapter
from src.memory.MemoryToEvolutionAdapter  import MemoryToEvolutionAdapter
from src.net.WorldMemoryReplicator        import WorldMemoryReplicator
from src.save.WorldMemorySnapshot         import WorldMemorySnapshot


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_W, _H = 8, 4   # small grid for fast tests

_MEM_CFG = {
    "memory": {
        "k_stress":           0.5,   # fast gain for tests
        "k_stress_relax":     0.05,
        "k_compact_gain":     0.5,
        "k_compact_decay":    0.05,
        "k_erosion_gain":     0.5,
        "k_erosion_decay":    0.05,
        "stress_threshold":   0.6,
        "max_influence":      0.3,
        "lod_tile_limit":     _W * _H,
        "broadcast_interval_s": 30.0,
        "slope_threshold":    0.2,
        "wind_threshold":     0.2,
        "max_tiles_per_tick": _W * _H,
    }
}


def _make_state() -> WorldMemoryState:
    return WorldMemoryState(width=_W, height=_H)


# ---------------------------------------------------------------------------
# 1. test_repeated_steps_increase_compaction
# ---------------------------------------------------------------------------

class TestRepeatedStepsIncreaseCompaction(unittest.TestCase):
    """Repeated pressure at a tile must raise compactionHistoryField."""

    def test_heavy_tile_has_more_compaction_than_light_tile(self):
        state   = _make_state()
        system  = CompactionHistorySystem(_MEM_CFG)
        dt      = 1.0
        heavy   = state.tile(2, 1)
        light   = state.tile(5, 3)

        # Apply many contacts to the heavy tile, fewer to the light tile
        for _ in range(20):
            system.apply_pressure(state, heavy, contact_force=0.8, dt=dt)
        for _ in range(5):
            system.apply_pressure(state, light, contact_force=0.2, dt=dt)

        self.assertGreater(
            state.compactionHistoryField[heavy],
            state.compactionHistoryField[light],
            "Heavy-traffic tile must accumulate more compaction history",
        )
        self.assertGreater(state.compactionHistoryField[heavy], 0.0)

    def test_zero_force_does_not_change_state(self):
        state  = _make_state()
        system = CompactionHistorySystem(_MEM_CFG)
        tile   = state.tile(0, 0)
        system.apply_pressure(state, tile, contact_force=0.0, dt=1.0)
        self.assertEqual(state.compactionHistoryField[tile], 0.0)


# ---------------------------------------------------------------------------
# 2. test_stress_reduces_crust_stability
# ---------------------------------------------------------------------------

class TestStressReducesCrustStability(unittest.TestCase):
    """stressAccumulationField must translate to negative crust_stability_delta."""

    def test_stressed_tile_yields_negative_crust_delta(self):
        state    = _make_state()
        accum    = StressAccumulator(_MEM_CFG)
        adapter  = MemoryToEvolutionAdapter(_MEM_CFG)
        tile     = state.tile(1, 1)

        # Build up stress
        for _ in range(15):
            accum.apply_contact(state, tile, contact_force=0.9, dt=1.0)

        deltas = adapter.evolution_deltas(state, tile, dt=1.0)
        self.assertLess(
            deltas["crust_stability_delta"], 0.0,
            "Stressed tile must produce negative crust_stability_delta",
        )

    def test_unstressed_tile_has_zero_crust_delta(self):
        state   = _make_state()
        adapter = MemoryToEvolutionAdapter(_MEM_CFG)
        tile    = state.tile(3, 2)
        deltas  = adapter.evolution_deltas(state, tile, dt=1.0)
        self.assertEqual(deltas["crust_stability_delta"], 0.0)


# ---------------------------------------------------------------------------
# 3. test_memory_fields_decay_over_time
# ---------------------------------------------------------------------------

class TestMemoryFieldsDecayOverTime(unittest.TestCase):
    """Memory fields must decay toward zero when no new input is provided."""

    def _set_all_fields(self, state: WorldMemoryState, value: float) -> None:
        n = state.size()
        for i in range(n):
            state.stressAccumulationField[i]  = value
            state.compactionHistoryField[i]   = value
            state.erosionBiasField[i]         = value

    def test_stress_decays(self):
        state  = _make_state()
        accum  = StressAccumulator(_MEM_CFG)
        self._set_all_fields(state, 0.8)
        initial = list(state.stressAccumulationField)

        for _ in range(20):
            accum.tick(state, wind_map=None, slope_map=None, dt=1.0)

        self.assertLess(
            sum(state.stressAccumulationField),
            sum(initial),
            "Stress must decay over time without new contacts",
        )

    def test_compaction_decays(self):
        state  = _make_state()
        system = CompactionHistorySystem(_MEM_CFG)
        self._set_all_fields(state, 0.8)
        initial = sum(state.compactionHistoryField)

        for _ in range(20):
            system.tick(state, wind_map=None, dt=1.0)

        self.assertLess(
            sum(state.compactionHistoryField), initial,
            "Compaction history must decay over time",
        )

    def test_erosion_bias_decays(self):
        state  = _make_state()
        system = ErosionBiasSystem(_MEM_CFG)
        self._set_all_fields(state, 0.8)
        initial = sum(state.erosionBiasField)

        for _ in range(20):
            system.tick(state,
                        slope_map=None, wind_map=None,
                        movement_map=None, dt=1.0)

        self.assertLess(
            sum(state.erosionBiasField), initial,
            "Erosion bias must decay over time",
        )


# ---------------------------------------------------------------------------
# 4. test_erosion_bias_affects_slope_creep
# ---------------------------------------------------------------------------

class TestErosionBiasAffectsSlopeCreep(unittest.TestCase):
    """High erosionBiasField must translate to positive slope_creep_delta."""

    def test_high_erosion_tile_has_positive_creep_delta(self):
        state   = _make_state()
        system  = ErosionBiasSystem(_MEM_CFG)
        adapter = MemoryToEvolutionAdapter(_MEM_CFG)
        tile    = state.tile(3, 2)

        # Steep slope, strong wind → erosion bias builds
        slope_map    = [0.0] * state.size()
        wind_map     = [0.0] * state.size()
        movement_map = [0.0] * state.size()
        slope_map[tile]    = 0.9
        wind_map[tile]     = 0.9
        movement_map[tile] = 0.9

        for _ in range(30):
            system.tick(state, slope_map, wind_map, movement_map, dt=1.0)

        deltas = adapter.evolution_deltas(state, tile, dt=1.0)
        self.assertGreater(
            deltas["slope_creep_delta"], 0.0,
            "High erosion bias tile must yield positive slope_creep_delta",
        )

    def test_flat_calm_tile_has_zero_creep_delta(self):
        state   = _make_state()
        adapter = MemoryToEvolutionAdapter(_MEM_CFG)
        tile    = state.tile(0, 0)
        deltas  = adapter.evolution_deltas(state, tile, dt=1.0)
        self.assertEqual(deltas["slope_creep_delta"], 0.0)


# ---------------------------------------------------------------------------
# 5. test_snapshot_restore_memory
# ---------------------------------------------------------------------------

class TestSnapshotRestoreMemory(unittest.TestCase):
    """WorldMemorySnapshot must round-trip WorldMemoryState within 8-bit error."""

    def test_round_trip_preserves_fields(self):
        state = _make_state()
        n     = state.size()
        # Populate with varied values
        for i in range(n):
            state.stressAccumulationField[i]  = (i % 10) / 10.0
            state.compactionHistoryField[i]   = (i % 7)  / 7.0
            state.erosionBiasField[i]         = (i % 5)  / 5.0
            state.acousticImprintField[i]     = (i % 3)  / 3.0

        snap      = WorldMemorySnapshot()
        blob      = snap.save(state, sim_time=123.45)
        restored, meta = snap.load(blob)

        self.assertAlmostEqual(meta["sim_time"], 123.45, places=6)
        tolerance = 1 / 255.0 + 1e-9

        for i in range(n):
            self.assertAlmostEqual(
                restored.stressAccumulationField[i],
                state.stressAccumulationField[i],
                delta=tolerance,
            )
            self.assertAlmostEqual(
                restored.compactionHistoryField[i],
                state.compactionHistoryField[i],
                delta=tolerance,
            )
            self.assertAlmostEqual(
                restored.erosionBiasField[i],
                state.erosionBiasField[i],
                delta=tolerance,
            )
            self.assertAlmostEqual(
                restored.acousticImprintField[i],
                state.acousticImprintField[i],
                delta=tolerance,
            )

    def test_bad_magic_raises(self):
        snap = WorldMemorySnapshot()
        with self.assertRaises(ValueError):
            snap.load(b"XXXX" + b"\x00" * 100)


# ---------------------------------------------------------------------------
# 6. test_determinism_same_inputs_same_memory
# ---------------------------------------------------------------------------

class TestDeterminismSameInputsSameMemory(unittest.TestCase):
    """Two independent runs with the same input sequence must produce the
    same state_hash (no random anywhere in the pipeline)."""

    def _run_simulation(self) -> str:
        state   = WorldMemoryState(width=_W, height=_H)
        accum   = StressAccumulator(_MEM_CFG)
        compact = CompactionHistorySystem(_MEM_CFG)
        erosion = ErosionBiasSystem(_MEM_CFG)

        slope_map    = [float(i % 5) / 5.0 for i in range(state.size())]
        wind_map     = [float(i % 3) / 3.0 for i in range(state.size())]
        movement_map = [float(i % 4) / 4.0 for i in range(state.size())]

        for step in range(50):
            tile = step % state.size()
            accum.apply_contact(state, tile, contact_force=0.5, dt=0.1)
            compact.apply_pressure(state, tile, contact_force=0.4, dt=0.1)

            accum.tick(state, wind_map, slope_map, dt=0.1)
            compact.tick(state, wind_map, dt=0.1)
            erosion.tick(state, slope_map, wind_map, movement_map, dt=0.1)

        return state.state_hash()

    def test_two_runs_produce_same_hash(self):
        hash_a = self._run_simulation()
        hash_b = self._run_simulation()
        self.assertEqual(hash_a, hash_b,
                         "Determinism violated: same inputs produced different hashes")


# ---------------------------------------------------------------------------
# Replicator smoke test
# ---------------------------------------------------------------------------

class TestWorldMemoryReplicator(unittest.TestCase):
    """WorldMemoryReplicator should build and apply snapshots correctly."""

    def test_build_and_apply_snapshot(self):
        state = _make_state()
        state.stressAccumulationField[0] = 0.42
        state.compactionHistoryField[1]  = 0.77

        rep  = WorldMemoryReplicator(_MEM_CFG)
        msg  = rep.build_snapshot(state)
        rep.record_broadcast(sim_time=10.0)

        self.assertEqual(msg["type"], "WORLD_MEMORY_STATE_51")
        self.assertEqual(rep.last_broadcast_time, 10.0)
        self.assertFalse(rep.should_broadcast(sim_time=20.0))
        self.assertTrue(rep.should_broadcast(sim_time=40.1))

        restored = _make_state()
        ok = rep.apply_snapshot(restored, msg)
        self.assertTrue(ok)
        self.assertAlmostEqual(
            restored.stressAccumulationField[0], 0.42, delta=1 / 255.0 + 1e-9
        )

    def test_apply_bad_snapshot_returns_false(self):
        state = _make_state()
        rep   = WorldMemoryReplicator(_MEM_CFG)
        ok    = rep.apply_snapshot(state, {"type": "WRONG"})
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
