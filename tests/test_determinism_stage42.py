"""test_determinism_stage42.py — Stage 42 Determinism + Budgeted Runtime tests.

Tests
-----
1. TestDeterminismContract
   — Tick rates are positive.
   — Quantisation helpers round-trip (pos, dir, force).
   — event_sort_key orders by (tick_index, entity_id).

2. TestDetRng
   — Identical constructor args produce identical sequences.
   — Different seeds produce different sequences.
   — fork() produces independent but deterministic child streams.

3. TestFixedTickScheduler
   — Callbacks fired at the correct frequency.
   — Spiral-of-death cap limits catch-up steps.
   — tick_counts advance monotonically.

4. TestStateHashSync
   — Identical inputs produce identical hashes.
   — Individual hash functions change on differing inputs.
   — check_drift returns None on match; escalates on mismatches.

5. TestBudgetManager
   — Under-budget usage → FULL tier.
   — Over-budget usage → DISABLED tier.
   — reset_frame() clears accumulators and tier.
   — usage_summary() returns correct values.

6. TestTelemetry
   — Disabled telemetry is a no-op.
   — Enabled telemetry stores frames in ring buffer.
   — Ring buffer wraps when full.
   — check_alarms does not raise on extreme values.

7. TestFallbackLadders
   — AudioLadder / DeformLadder / IKLadder tiers match spec.
   — all_tiers() length is 4 for each ladder.

8. TestDeterminismReplayHarness
   — Two replays of same inputs → identical hashes.
   — Late-joining client reaches same hash as server.

9. TestNoBudgetGrowth
   — usage_summary() size is stable across many reset cycles.
   — Telemetry ring buffer does not exceed its maximum.
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.core.DeterminismContract import DeterminismContract
from src.core.DetRng import DetRng
from src.core.FixedTickScheduler import FixedTickScheduler
from src.net.StateHashSync import StateHashSync, DriftLevel
from src.runtime.BudgetManager import BudgetManager, FallbackTier
from src.runtime.Telemetry import Telemetry
from src.runtime.FallbackLadders import AudioLadder, DeformLadder, IKLadder
from src.ci.DeterminismReplayHarness import DeterminismReplayHarness


# ---------------------------------------------------------------------------
# 1. DeterminismContract
# ---------------------------------------------------------------------------

class TestDeterminismContract(unittest.TestCase):

    def test_tick_rates_positive(self) -> None:
        self.assertGreater(DeterminismContract.SIM_TICK_HZ, 0)
        self.assertGreater(DeterminismContract.PERCEPTION_TICK_HZ, 0)
        self.assertGreater(DeterminismContract.INTENT_TICK_HZ, 0)
        self.assertGreater(DeterminismContract.SOCIAL_TICK_HZ, 0)
        self.assertGreater(DeterminismContract.EVOLUTION_TICK_HZ, 0)

    def test_pos_quantise_roundtrip(self) -> None:
        for v in (0.0, 1.0, -3.14159, 1234.567):
            mm = DeterminismContract.quantise_pos_mm(v)
            self.assertAlmostEqual(DeterminismContract.dequantise_pos_mm(mm), v, places=3)

    def test_dir_clamping(self) -> None:
        self.assertEqual(DeterminismContract.quantise_dir_i16(1.5),
                         DeterminismContract._DIR_SCALE)
        self.assertEqual(DeterminismContract.quantise_dir_i16(-2.0),
                         -DeterminismContract._DIR_SCALE)

    def test_dir_roundtrip(self) -> None:
        tol = 1.0 / DeterminismContract._DIR_SCALE + 1e-9
        for v in (0.0, 0.5, -0.99, 1.0, -1.0):
            i16 = DeterminismContract.quantise_dir_i16(v)
            self.assertAlmostEqual(DeterminismContract.dequantise_dir_i16(i16), v, delta=tol)

    def test_force_roundtrip(self) -> None:
        for n in (0.0, 10.0, -500.0):
            dn = DeterminismContract.quantise_force_dn(n)
            self.assertAlmostEqual(DeterminismContract.dequantise_force_dn(dn), n, delta=0.1)

    def test_event_sort_key_dict(self) -> None:
        events = [
            {"tick_index": 5, "entity_id": 2},
            {"tick_index": 3, "entity_id": 1},
            {"tick_index": 3, "entity_id": 0},
        ]
        sorted_ev = DeterminismContract.sort_events(events)
        keys = [(e["tick_index"], e["entity_id"]) for e in sorted_ev]
        self.assertEqual(keys, sorted(keys))

    def test_event_sort_key_object(self) -> None:
        class Ev:
            def __init__(self, t, e):
                self.tick_index = t
                self.entity_id = e
        events = [Ev(10, 3), Ev(1, 9), Ev(1, 0)]
        sorted_ev = DeterminismContract.sort_events(events)
        keys = [(e.tick_index, e.entity_id) for e in sorted_ev]
        self.assertEqual(keys, sorted(keys))


# ---------------------------------------------------------------------------
# 2. DetRng
# ---------------------------------------------------------------------------

class TestDetRng(unittest.TestCase):

    def _make(self, **kw) -> DetRng:
        defaults = dict(world_seed=42, player_id=1, system_id="test",
                        tick_index=0, region_id=0)
        defaults.update(kw)
        return DetRng(**defaults)

    def test_identical_args_identical_sequence(self) -> None:
        r1, r2 = self._make(), self._make()
        self.assertEqual([r1.rand_float01() for _ in range(20)],
                         [r2.rand_float01() for _ in range(20)])

    def test_different_seed_different_sequence(self) -> None:
        r1, r2 = self._make(world_seed=42), self._make(world_seed=99)
        self.assertNotEqual([r1.rand_float01() for _ in range(10)],
                             [r2.rand_float01() for _ in range(10)])

    def test_different_tick_different_seed(self) -> None:
        self.assertNotEqual(self._make(tick_index=0).seed,
                            self._make(tick_index=100).seed)

    def test_rand_int_range(self) -> None:
        r = self._make()
        for _ in range(50):
            v = r.rand_int(0, 9)
            self.assertGreaterEqual(v, 0)
            self.assertLessEqual(v, 9)

    def test_rand_range_bounds(self) -> None:
        r = self._make()
        for _ in range(50):
            v = r.rand_range(2.0, 5.0)
            self.assertGreaterEqual(v, 2.0)
            self.assertLess(v, 5.0)

    def test_rand_unit_vec3_unit_length(self) -> None:
        import math
        r = self._make()
        for _ in range(20):
            x, y, z = r.rand_unit_vec3()
            self.assertAlmostEqual(math.sqrt(x*x + y*y + z*z), 1.0, places=6)

    def test_fork_independent(self) -> None:
        r = self._make()
        ca, cb = r.fork("a"), r.fork("b")
        self.assertNotEqual([ca.rand_float01() for _ in range(10)],
                             [cb.rand_float01() for _ in range(10)])

    def test_fork_deterministic(self) -> None:
        r1, r2 = self._make(), self._make()
        self.assertEqual(r1.fork("sub").seed, r2.fork("sub").seed)


# ---------------------------------------------------------------------------
# 3. FixedTickScheduler
# ---------------------------------------------------------------------------

class TestFixedTickScheduler(unittest.TestCase):

    def test_sim_fires_60_ticks_per_second(self) -> None:
        sched = FixedTickScheduler(sim_hz=60)
        fired: list = []
        sched.register("sim", lambda t: fired.append(t))
        for _ in range(60):
            sched.advance(1.0 / 60.0)
        self.assertEqual(len(fired), 60)
        self.assertEqual(fired[-1], 59)

    def test_perception_fires_less_often(self) -> None:
        sched = FixedTickScheduler(sim_hz=60, perception_hz=20)
        sim_ticks: list = []
        perc_ticks: list = []
        sched.register("sim", lambda t: sim_ticks.append(t))
        sched.register("perception", lambda t: perc_ticks.append(t))
        for _ in range(60):
            sched.advance(1.0 / 60.0)
        self.assertEqual(len(sim_ticks), 60)
        self.assertEqual(len(perc_ticks), 20)

    def test_spiral_of_death_cap(self) -> None:
        sched = FixedTickScheduler(sim_hz=60)
        fired: list = []
        sched.register("sim", lambda t: fired.append(t))
        sched.advance(1000.0)
        self.assertLessEqual(len(fired), sched.MAX_STEPS_PER_FRAME)

    def test_tick_counts_monotonic(self) -> None:
        sched = FixedTickScheduler()
        prev = dict(sched.tick_counts)
        for _ in range(10):
            sched.advance(1.0 / 60.0)
            curr = sched.tick_counts
            for name in curr:
                self.assertGreaterEqual(curr[name], prev[name])
            prev = dict(curr)

    def test_unknown_rate_raises(self) -> None:
        sched = FixedTickScheduler()
        with self.assertRaises(ValueError):
            sched.register("nonexistent", lambda t: None)

    def test_stats_returned(self) -> None:
        sched = FixedTickScheduler(sim_hz=60)
        stats = sched.advance(1.0 / 60.0)
        self.assertEqual(stats.sim, 1)


# ---------------------------------------------------------------------------
# 4. StateHashSync
# ---------------------------------------------------------------------------

class TestStateHashSync(unittest.TestCase):

    def test_hash_motor_core_deterministic(self) -> None:
        h = StateHashSync.hash_motor_core([1000, 2000, 3000], 4, 1)
        self.assertEqual(h, StateHashSync.hash_motor_core([1000, 2000, 3000], 4, 1))

    def test_hash_motor_core_changes(self) -> None:
        h1 = StateHashSync.hash_motor_core([1000, 2000, 3000], 4, 1)
        h2 = StateHashSync.hash_motor_core([1001, 2000, 3000], 4, 1)
        self.assertNotEqual(h1, h2)

    def test_hash_deform_nearby_deterministic(self) -> None:
        h = StateHashSync.hash_deform_nearby([10, 20, 30])
        self.assertEqual(h, StateHashSync.hash_deform_nearby([10, 20, 30]))

    def test_hash_grasp_deterministic(self) -> None:
        h = StateHashSync.hash_grasp([1, 2], [[100, 200, 300]])
        self.assertEqual(h, StateHashSync.hash_grasp([1, 2], [[100, 200, 300]]))

    def test_hash_astro_climate_deterministic(self) -> None:
        coeff = {"insolation": 1.23, "temperature": -5.0, "dust": 0.5}
        self.assertEqual(StateHashSync.hash_astro_climate(coeff),
                         StateHashSync.hash_astro_climate(coeff))

    def test_hash_astro_climate_order_independent(self) -> None:
        self.assertEqual(
            StateHashSync.hash_astro_climate({"a": 1.0, "b": 2.0}),
            StateHashSync.hash_astro_climate({"b": 2.0, "a": 1.0}),
        )

    def test_compute_full_hash_deterministic(self) -> None:
        h = StateHashSync.compute_full_hash(1, 2, 3, 4)
        self.assertEqual(h, StateHashSync.compute_full_hash(1, 2, 3, 4))

    def test_check_drift_no_mismatch(self) -> None:
        sync = StateHashSync()
        self.assertIsNone(sync.check_drift(0xABCD, 0xABCD))
        self.assertEqual(sync.mismatch_count, 0)

    def test_check_drift_level1_first_mismatch(self) -> None:
        sync = StateHashSync({"determinism": {"drift_thresholds": {
            "level1": 1, "level2": 3, "level3": 6, "level4": 10
        }}})
        self.assertEqual(sync.check_drift(1, 2), DriftLevel.LEVEL_1_TIME_OFFSET)

    def test_check_drift_escalates(self) -> None:
        sync = StateHashSync({"determinism": {"drift_thresholds": {
            "level1": 1, "level2": 2, "level3": 3, "level4": 4
        }}})
        level = None
        for _ in range(4):
            level = sync.check_drift(0, 1)
        self.assertEqual(level, DriftLevel.LEVEL_4_SOFT_RESET)

    def test_reset_mismatch_counter(self) -> None:
        sync = StateHashSync()
        sync.check_drift(0, 1)
        sync.reset_mismatch_counter()
        self.assertEqual(sync.mismatch_count, 0)


# ---------------------------------------------------------------------------
# 5. BudgetManager
# ---------------------------------------------------------------------------

class TestBudgetManager(unittest.TestCase):

    def _make(self) -> BudgetManager:
        return BudgetManager({"budget": {"fallback_enable": True}})

    def test_under_budget_full_tier(self) -> None:
        bm = self._make()
        self.assertEqual(bm.record("audio", "active_resonators", 5.0),
                         FallbackTier.FULL)

    def test_over_budget_disabled_tier(self) -> None:
        bm = self._make()
        self.assertEqual(bm.record("audio", "active_resonators", 100.0),
                         FallbackTier.DISABLED)

    def test_mid_budget_reduced_tier(self) -> None:
        bm = BudgetManager({"budget": {
            "fallback_enable": True,
            "audio": {"active_resonators": 100.0},
        }})
        # 60% of limit → REDUCED tier (≥50% threshold)
        self.assertEqual(bm.record("audio", "active_resonators", 60.0),
                         FallbackTier.REDUCED)

    def test_reset_frame_clears_usage(self) -> None:
        bm = self._make()
        bm.record("audio", "active_resonators", 100.0)
        bm.reset_frame()
        self.assertEqual(bm.usage_summary()["audio"]["active_resonators"], 0.0)

    def test_reset_frame_clears_tier(self) -> None:
        bm = self._make()
        bm.record("audio", "active_resonators", 100.0)
        bm.reset_frame()
        self.assertEqual(bm.tier("audio"), FallbackTier.FULL)

    def test_usage_summary_reflects_recorded(self) -> None:
        bm = self._make()
        bm.record("motor", "ik_iters", 30.0)
        self.assertAlmostEqual(bm.usage_summary()["motor"]["ik_iters"], 30.0)

    def test_unknown_subsystem_returns_full(self) -> None:
        bm = self._make()
        self.assertEqual(bm.record("nonexistent", "foo", 999.0), FallbackTier.FULL)

    def test_limits_property(self) -> None:
        bm = self._make()
        self.assertIn("audio", bm.limits)
        self.assertIn("active_resonators", bm.limits["audio"])


# ---------------------------------------------------------------------------
# 6. Telemetry
# ---------------------------------------------------------------------------

class TestTelemetry(unittest.TestCase):

    def test_disabled_noop(self) -> None:
        tel = Telemetry({"telemetry": {"enable_dev": False}})
        self.assertFalse(tel.enabled)
        tel.record_frame(0, {"ik_iters": 100.0})
        self.assertEqual(tel.get_recent(), [])
        self.assertEqual(tel.frame_count(), 0)

    def test_enabled_stores_frames(self) -> None:
        tel = Telemetry({"telemetry": {"enable_dev": True, "ringbuffer_sec": 1}})
        self.assertTrue(tel.enabled)
        for i in range(10):
            tel.record_frame(i, {"ik_iters": float(i)})
        self.assertEqual(tel.frame_count(), 10)

    def test_ring_buffer_wraps(self) -> None:
        tel = Telemetry({"telemetry": {"enable_dev": True, "ringbuffer_sec": 1}})
        for i in range(120):
            tel.record_frame(i, {"ik_iters": 1.0})
        self.assertLessEqual(tel.frame_count(), 60)

    def test_get_recent_returns_n(self) -> None:
        tel = Telemetry({"telemetry": {"enable_dev": True, "ringbuffer_sec": 60}})
        for i in range(50):
            tel.record_frame(i, {"ik_iters": 1.0})
        recent = tel.get_recent(10)
        self.assertEqual(len(recent), 10)
        self.assertEqual(recent[-1].frame_no, 49)

    def test_check_alarms_does_not_raise(self) -> None:
        tel = Telemetry({"telemetry": {"enable_dev": True}})
        tel.check_alarms({"ik_iters": 9999.0, "resonators_active": 0.0})


# ---------------------------------------------------------------------------
# 7. FallbackLadders
# ---------------------------------------------------------------------------

class TestFallbackLadders(unittest.TestCase):

    def test_audio_tiers(self) -> None:
        ladder = AudioLadder()
        self.assertEqual(ladder.quality_for(FallbackTier.FULL),     "full_modal")
        self.assertEqual(ladder.quality_for(FallbackTier.REDUCED),  "reduced_modes")
        self.assertEqual(ladder.quality_for(FallbackTier.PROXY),    "noise_proxy")
        self.assertEqual(ladder.quality_for(FallbackTier.DISABLED), "drop_quiet")

    def test_deform_tiers(self) -> None:
        ladder = DeformLadder()
        self.assertEqual(ladder.quality_for(FallbackTier.FULL),     "h_and_m_field")
        self.assertEqual(ladder.quality_for(FallbackTier.REDUCED),  "h_field_only")
        self.assertEqual(ladder.quality_for(FallbackTier.PROXY),    "material_overlay")
        self.assertEqual(ladder.quality_for(FallbackTier.DISABLED), "no_deform")

    def test_ik_tiers(self) -> None:
        ladder = IKLadder()
        self.assertEqual(ladder.quality_for(FallbackTier.FULL),     "full_body")
        self.assertEqual(ladder.quality_for(FallbackTier.REDUCED),  "reduced_constraints")
        self.assertEqual(ladder.quality_for(FallbackTier.PROXY),    "legs_only")
        self.assertEqual(ladder.quality_for(FallbackTier.DISABLED), "stabilise_only")

    def test_all_tiers_length(self) -> None:
        for Ladder in (AudioLadder, DeformLadder, IKLadder):
            ladder = Ladder()
            tiers = ladder.all_tiers()
            self.assertEqual(len(tiers), 4)
            self.assertEqual(tiers[0][0], FallbackTier.FULL)
            self.assertEqual(tiers[-1][0], FallbackTier.DISABLED)


# ---------------------------------------------------------------------------
# 8. DeterminismReplayHarness
# ---------------------------------------------------------------------------

class TestDeterminismReplayHarness(unittest.TestCase):

    def _make(self, seed: int = 42) -> DeterminismReplayHarness:
        return DeterminismReplayHarness(world_seed=seed)

    def _populate(self, h: DeterminismReplayHarness, n: int = 60) -> None:
        for i in range(n):
            h.record_event(i * 10, i % 5, "move", {"x": i})

    def test_replay_same_hash(self) -> None:
        h = self._make()
        self._populate(h, 60)
        h1, h2 = h.run_replay(n_ticks=600)
        self.assertEqual(h1, h2)

    def test_assert_deterministic_passes(self) -> None:
        h = self._make()
        self._populate(h)
        h.assert_deterministic(n_ticks=600)

    def test_join_sync_matches(self) -> None:
        h = self._make()
        self._populate(h, 100)
        sh, ch = h.run_join_sync_test(join_tick=60, total_ticks=200)
        self.assertEqual(sh, ch)

    def test_assert_join_sync_passes(self) -> None:
        h = self._make()
        self._populate(h, 50)
        h.assert_join_sync(join_tick=30, total_ticks=100)

    def test_different_seeds_different_hash(self) -> None:
        h1, h2 = self._make(seed=1), self._make(seed=2)
        self._populate(h1)
        self._populate(h2)
        r1, _ = h1.run_replay(n_ticks=600)
        r2, _ = h2.run_replay(n_ticks=600)
        self.assertNotEqual(r1, r2)


# ---------------------------------------------------------------------------
# 9. TestNoBudgetGrowth
# ---------------------------------------------------------------------------

class TestNoBudgetGrowth(unittest.TestCase):

    def test_usage_dict_stable(self) -> None:
        bm = BudgetManager({"budget": {"fallback_enable": True}})
        bm.record("audio", "active_resonators", 10.0)
        size_before = sum(len(v) for v in bm.usage_summary().values())
        for _ in range(1000):
            bm.reset_frame()
            bm.record("audio", "active_resonators", 10.0)
        size_after = sum(len(v) for v in bm.usage_summary().values())
        self.assertEqual(size_before, size_after)

    def test_telemetry_ring_bounded(self) -> None:
        tel = Telemetry({"telemetry": {"enable_dev": True, "ringbuffer_sec": 1}})
        for i in range(10000):
            tel.record_frame(i, {"ik_iters": 1.0})
        self.assertLessEqual(tel.frame_count(), 60)


if __name__ == "__main__":
    unittest.main()
