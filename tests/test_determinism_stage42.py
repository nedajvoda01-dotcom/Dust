"""test_determinism_stage42.py — Stage 42 Determinism + Budgeted Runtime tests.

Tests
-----
1. test_det_rng_same_seed_same_sequence
   — Two DetRng instances with the same seed produce identical sequences.

2. test_det_rng_for_domain_reproducible
   — DetRng.for_domain with the same five-tuple yields the same values.

3. test_det_rng_different_domains_differ
   — Different system_id values produce different sequences.

4. test_contract_quantise_round_trip
   — position and direction quantise/dequantise round-trips within tolerance.

5. test_contract_sort_events_ordering
   — sort_events produces (tick_index, entity_id)-sorted output.

6. test_contract_assert_fixed_tick_violation
   — assert_fixed_tick logs/raises when dt deviates.

7. test_fixed_tick_scheduler_fires_correct_counts
   — FixedTickScheduler fires callbacks at the right fixed rates.

8. test_fixed_tick_scheduler_catchup_cap
   — Spiral-of-death protection: large game_dt is capped at 8 steps.

9. test_state_hash_sync_no_action_when_match
   — StateHashSync.check_client returns None when hashes match.

10. test_state_hash_sync_correction_level_on_drift
    — Drift in motor_core returns a correction action.

11. test_budget_manager_fallback_engages
    — Reporting over-budget usage degrades the fallback ladder.

12. test_budget_manager_recovery
    — Reporting back-below-threshold usage recovers the ladder.

13. test_fallback_ladders_degrade_recover_reset
    — FallbackLadder degrade/recover/reset step logic is correct.

14. test_telemetry_ring_buffer_bounded
    — Telemetry ring-buffer does not grow beyond ringbuffer_sec.

15. test_telemetry_disabled_no_records
    — Telemetry with enable_dev=False stores nothing.

16. test_replay_same_inputs_same_hash
    — Two independent ReplayRunner runs produce identical hash sequences.

17. test_replay_different_inputs_different_hash
    — Different inputs yield different hashes.

18. test_no_unbounded_growth
    — Telemetry ring-buffer length stays bounded during a long simulated run.
"""
from __future__ import annotations

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.core.DetRng import DetRng
from src.core.DeterminismContract import (
    TICK_DT, EPSILON,
    quantise_position_mm, dequantise_position_mm,
    quantise_direction_int16, dequantise_direction_int16,
    sort_events, assert_fixed_tick, report_violation,
    DeterminismViolation,
    clamp,
)
import src.core.DeterminismContract as _dc
from src.core.FixedTickScheduler import FixedTickScheduler
from src.net.StateHashSync import (
    StateHashSync, StateHashes,
    hash_motor_core, hash_deform_nearby, hash_grasp, hash_astro_climate,
    CorrectionLevel,
)
from src.runtime.FallbackLadders import (
    FallbackLadder, AudioFallbackLadder, DeformFallbackLadder, IKFallbackLadder,
    FallbackLadders, AudioTier, DeformTier, IKTier,
)
from src.runtime.BudgetManager import BudgetManager
from src.runtime.Telemetry import Telemetry
from src.ci.DeterminismReplayHarness import InputRecorder, ReplayRunner


# ---------------------------------------------------------------------------
# 1 & 2 — DetRng determinism
# ---------------------------------------------------------------------------

class TestDetRng(unittest.TestCase):

    def test_same_seed_same_sequence(self):
        a = DetRng(12345)
        b = DetRng(12345)
        for _ in range(20):
            self.assertAlmostEqual(a.next_float01(), b.next_float01(), places=15)

    def test_for_domain_reproducible(self):
        r1 = DetRng.for_domain(42, 1, "audio", 100, 0)
        r2 = DetRng.for_domain(42, 1, "audio", 100, 0)
        for _ in range(10):
            self.assertEqual(r1.next_int(0, 100), r2.next_int(0, 100))

    def test_different_domains_differ(self):
        r_audio  = DetRng.for_domain(42, 1, "audio",  0, 0)
        r_deform = DetRng.for_domain(42, 1, "deform", 0, 0)
        vals_audio  = [r_audio.next_float01()  for _ in range(5)]
        vals_deform = [r_deform.next_float01() for _ in range(5)]
        self.assertNotEqual(vals_audio, vals_deform)

    def test_float_in_range(self):
        rng = DetRng(999)
        for _ in range(100):
            v = rng.next_float01()
            self.assertGreaterEqual(v, 0.0)
            self.assertLess(v, 1.0)

    def test_range_bounds(self):
        rng = DetRng(7)
        for _ in range(50):
            v = rng.next_range(-10.0, 10.0)
            self.assertGreaterEqual(v, -10.0)
            self.assertLess(v, 10.0)

    def test_int_bounds(self):
        rng = DetRng(3)
        for _ in range(50):
            v = rng.next_int(5, 10)
            self.assertGreaterEqual(v, 5)
            self.assertLessEqual(v, 10)


# ---------------------------------------------------------------------------
# 3 — DeterminismContract helpers
# ---------------------------------------------------------------------------

class TestDeterminismContract(unittest.TestCase):

    def test_quantise_position_round_trip(self):
        for m in [0.0, 1.234, -99.5, 1000.0, 0.001]:
            mm = quantise_position_mm(m)
            restored = dequantise_position_mm(mm)
            self.assertAlmostEqual(m, restored, places=2)

    def test_quantise_direction_round_trip(self):
        for d in [0.0, 1.0, -1.0, 0.5, -0.7071]:
            raw = quantise_direction_int16(d)
            restored = dequantise_direction_int16(raw)
            self.assertAlmostEqual(d, restored, delta=3e-4)

    def test_sort_events_ordering(self):
        events = [
            {"tick_index": 5, "entity_id": 2},
            {"tick_index": 3, "entity_id": 1},
            {"tick_index": 3, "entity_id": 0},
            {"tick_index": 1, "entity_id": 5},
        ]
        sorted_e = sort_events(events)
        ticks = [e["tick_index"] for e in sorted_e]
        self.assertEqual(ticks, sorted(ticks))
        # Tie-break by entity_id
        tie = [e for e in sorted_e if e["tick_index"] == 3]
        self.assertLess(tie[0]["entity_id"], tie[1]["entity_id"])

    def test_assert_fixed_tick_ok(self):
        # Should not raise or warn for exact match
        dt = TICK_DT["sim"]
        assert_fixed_tick("sim", dt)  # no exception

    def test_assert_fixed_tick_violation_strict(self):
        _dc.strict_mode = True
        try:
            with self.assertRaises(DeterminismViolation):
                assert_fixed_tick("sim", 0.1)
        finally:
            _dc.strict_mode = False

    def test_clamp(self):
        self.assertEqual(clamp(-5.0, 0.0, 1.0), 0.0)
        self.assertEqual(clamp(5.0, 0.0, 1.0), 1.0)
        self.assertAlmostEqual(clamp(0.5, 0.0, 1.0), 0.5)


# ---------------------------------------------------------------------------
# 4 — FixedTickScheduler
# ---------------------------------------------------------------------------

class TestFixedTickScheduler(unittest.TestCase):

    def test_fires_correct_counts(self):
        sched = FixedTickScheduler()
        counts = {"sim": 0, "intent": 0}

        def sim_cb(tick_idx, dt):
            counts["sim"] += 1

        def intent_cb(tick_idx, dt):
            counts["intent"] += 1

        sched.register("sim", sim_cb)
        sched.register("intent", intent_cb)

        # Advance 1 second at 60 Hz game_dt
        for _ in range(60):
            sched.tick(1.0 / 60.0)

        self.assertEqual(counts["sim"], 60)         # 60 Hz × 1 s
        self.assertIn(counts["intent"], (9, 10))    # 10 Hz × 1 s (fp tolerance)

    def test_catchup_cap(self):
        sched = FixedTickScheduler()
        call_count = {"n": 0}

        def cb(tick_idx, dt):
            call_count["n"] += 1

        sched.register("sim", cb)
        # Huge dt should be capped at 8 steps
        sched.tick(100.0)
        self.assertLessEqual(call_count["n"], 8)

    def test_tick_count_increments(self):
        sched = FixedTickScheduler()
        sched.register("sim", lambda t, dt: None)
        for _ in range(10):
            sched.tick(1.0 / 60.0)
        self.assertEqual(sched.tick_count("sim"), 10)

    def test_reset_clears_counts(self):
        sched = FixedTickScheduler()
        sched.register("social", lambda t, dt: None)
        sched.tick(1.0)
        self.assertGreater(sched.tick_count("social"), 0)
        sched.reset()
        self.assertEqual(sched.tick_count("social"), 0)


# ---------------------------------------------------------------------------
# 5 — StateHashSync
# ---------------------------------------------------------------------------

class TestStateHashSync(unittest.TestCase):

    def _make_hashes(self) -> StateHashes:
        return StateHashes(
            sim_time=10.0,
            motor_core=hash_motor_core((1.0, 0.0, 0.0), "grounded", 2),
            deform_nearby=hash_deform_nearby(4, 2, 0xDEAD),
            grasp=hash_grasp([1, 2], [(1.0, 0.0, 0.0), (2.0, 0.0, 0.0)]),
            astro_climate=hash_astro_climate(0.8, 5.0, 0.1, 280.0),
        )

    def test_no_action_when_match(self):
        sync = StateHashSync(hash_interval_sec=0.0)
        h = self._make_hashes()
        sync.record_server_snapshot(10.0, h)
        action = sync.check_client("c1", 10.0, h)
        self.assertIsNone(action)

    def test_correction_on_motor_drift(self):
        sync = StateHashSync(hash_interval_sec=0.0)
        server_h = self._make_hashes()
        sync.record_server_snapshot(10.0, server_h)

        client_h = StateHashes(
            sim_time=10.0,
            motor_core=0xBADBEEF,
            deform_nearby=server_h.deform_nearby,
            grasp=server_h.grasp,
            astro_climate=server_h.astro_climate,
        )
        action = sync.check_client("c2", 10.0, client_h)
        self.assertIsNotNone(action)
        self.assertEqual(action.client_id, "c2")

    def test_throttle_respects_interval(self):
        sync = StateHashSync(hash_interval_sec=5.0)
        h = self._make_hashes()
        sync.record_server_snapshot(0.0, h)

        # First check is done at t=0
        different_h = StateHashes(motor_core=0xBAD)
        action = sync.check_client("c3", 0.0, different_h)
        self.assertIsNotNone(action)
        # Subsequent check within interval should be throttled
        action2 = sync.check_client("c3", 1.0, different_h)
        self.assertIsNone(action2)

    def test_hash_motor_core_deterministic(self):
        h1 = hash_motor_core((1.5, 0.0, -2.3), "grounded", 3)
        h2 = hash_motor_core((1.5, 0.0, -2.3), "grounded", 3)
        self.assertEqual(h1, h2)

    def test_hash_astro_climate_deterministic(self):
        h1 = hash_astro_climate(0.9, 10.0, 0.2, 290.0)
        h2 = hash_astro_climate(0.9, 10.0, 0.2, 290.0)
        self.assertEqual(h1, h2)


# ---------------------------------------------------------------------------
# 6 — FallbackLadders
# ---------------------------------------------------------------------------

class TestFallbackLadders(unittest.TestCase):

    def test_audio_degrade_recover(self):
        ladder = AudioFallbackLadder()
        self.assertEqual(ladder.audio_tier, AudioTier.FULL_MODAL)
        ladder.degrade()
        self.assertEqual(ladder.audio_tier, AudioTier.REDUCED_MODES)
        ladder.recover()
        self.assertEqual(ladder.audio_tier, AudioTier.FULL_MODAL)

    def test_degrade_at_worst_returns_false(self):
        ladder = IKFallbackLadder()
        for _ in range(10):
            ladder.degrade()
        self.assertTrue(ladder.is_worst)
        self.assertFalse(ladder.degrade())

    def test_recover_at_best_returns_false(self):
        ladder = DeformFallbackLadder()
        self.assertFalse(ladder.recover())

    def test_reset_returns_to_best(self):
        ladder = IKFallbackLadder()
        ladder.degrade()
        ladder.degrade()
        ladder.reset()
        self.assertEqual(ladder.current, 0)
        self.assertFalse(ladder.is_degraded)

    def test_registry_get(self):
        self.assertIsNotNone(FallbackLadders.get("audio"))
        self.assertIsNotNone(FallbackLadders.get("deform"))
        self.assertIsNotNone(FallbackLadders.get("ik"))

    def test_reset_all(self):
        FallbackLadders.get("audio").degrade()
        FallbackLadders.get("ik").degrade()
        FallbackLadders.reset_all()
        self.assertEqual(FallbackLadders.get("audio").current, 0)
        self.assertEqual(FallbackLadders.get("ik").current, 0)


# ---------------------------------------------------------------------------
# 7 — BudgetManager
# ---------------------------------------------------------------------------

class TestBudgetManager(unittest.TestCase):

    def setUp(self):
        FallbackLadders.reset_all()

    def test_fallback_engages_on_overbudget(self):
        mgr = BudgetManager()
        # Default audio resonator limit is 32
        mgr.begin_frame()
        mgr.report("audio", "active_resonators", 50.0)  # > 100% of 32
        tier = mgr.fallback_tier("audio")
        self.assertGreater(tier, 0)

    def test_recovery_on_under_budget(self):
        mgr = BudgetManager()
        # First over-budget → degrade
        mgr.begin_frame()
        mgr.report("audio", "active_resonators", 50.0)
        self.assertGreater(mgr.fallback_tier("audio"), 0)
        # Then well under budget → recover
        mgr.begin_frame()
        mgr.report("audio", "active_resonators", 5.0)
        # Should have recovered (tier back down toward 0)
        self.assertEqual(mgr.fallback_tier("audio"), 0)

    def test_usage_accessor(self):
        mgr = BudgetManager()
        mgr.begin_frame()
        mgr.report("motor", "ik_iters", 42.0)
        self.assertAlmostEqual(mgr.usage("motor", "ik_iters"), 42.0)

    def test_limit_accessor(self):
        mgr = BudgetManager()
        self.assertGreater(mgr.limit("audio", "active_resonators"), 0)

    def test_unknown_category_no_crash(self):
        mgr = BudgetManager()
        mgr.begin_frame()
        mgr.report("nonexistent", "metric", 1.0)  # should not raise


# ---------------------------------------------------------------------------
# 8 — Telemetry
# ---------------------------------------------------------------------------

class TestTelemetry(unittest.TestCase):

    def test_ring_buffer_bounded(self):
        tel = Telemetry()
        # Default ringbuffer_sec is 60; simulate 120 seconds at 60 fps
        for i in range(7200):
            tel.begin_frame()
            tel.record("ik_iters", 5.0)
            tel.end_frame(sim_time=float(i) / 60.0)
        # Should not have all 7200 frames — only ~60 seconds worth
        self.assertLessEqual(tel.log_size(), 3700)  # ≤ 61 s × 60 fps + margin

    def test_disabled_no_records(self):
        tel = Telemetry()
        tel._enabled = False
        for i in range(100):
            tel.begin_frame()
            tel.record("ik_iters", 10.0)
            tel.end_frame(sim_time=float(i))
        self.assertEqual(tel.log_size(), 0)

    def test_latest_counter(self):
        tel = Telemetry()
        tel.begin_frame()
        tel.record("resonators_active", 7.0)
        tel.end_frame(sim_time=1.0)
        self.assertAlmostEqual(tel.latest("resonators_active"), 7.0)

    def test_get_log_filters_by_time(self):
        tel = Telemetry()
        for i in range(60):
            tel.begin_frame()
            tel.record("ik_iters", float(i))
            tel.end_frame(sim_time=float(i))
        log_30 = tel.get_log(last_n_seconds=30.0)
        self.assertLessEqual(len(log_30), 31)
        self.assertGreater(len(log_30), 0)


# ---------------------------------------------------------------------------
# 9 — Replay harness: same inputs → same hashes
# ---------------------------------------------------------------------------

class TestDeterminismReplay(unittest.TestCase):

    def _build_recorder(self, num_ticks: int = 300) -> InputRecorder:
        """Build a deterministic recorder with simple movement inputs."""
        recorder = InputRecorder()
        rng = DetRng(77)
        for t in range(num_ticks):
            recorder.record(t, {
                "move_x": rng.next_range(-1.0, 1.0),
                "move_z": rng.next_range(-1.0, 1.0),
                "jump": rng.next_int(0, 20) == 0,
            })
        return recorder

    def test_replay_same_inputs_same_hash(self):
        recorder = self._build_recorder(360)
        runner = ReplayRunner(world_seed=42, tick_hz=60.0)

        hashes_a = runner.run(recorder.stream())
        hashes_b = runner.run(recorder.stream())

        self.assertEqual(len(hashes_a), len(hashes_b))
        self.assertEqual(hashes_a, hashes_b, "Determinism broken: hashes differ!")

    def test_replay_different_inputs_different_hash(self):
        rec_a = self._build_recorder(60)
        rec_b = InputRecorder()
        rng = DetRng(99)  # different seed → different inputs
        for t in range(60):
            rec_b.record(t, {
                "move_x": rng.next_range(-1.0, 1.0),
                "move_z": rng.next_range(-1.0, 1.0),
                "jump": False,
            })

        runner = ReplayRunner(world_seed=42)
        hashes_a = runner.run(rec_a.stream())
        hashes_b = runner.run(rec_b.stream())
        self.assertNotEqual(hashes_a, hashes_b)

    def test_telemetry_no_unbounded_growth(self):
        """Telemetry ring-buffer must not grow linearly over a long run."""
        tel = Telemetry()
        tel._ringbuffer_sec = 60.0

        # Simulate 2 minutes at 60 fps (7200 frames) — enough to verify bounding
        for i in range(7_200):
            tel.begin_frame()
            tel.record("ik_iters", 5.0)
            tel.end_frame(sim_time=float(i) / 60.0)

        # At most 61 s worth of frames in the buffer
        max_expected = int(tel._ringbuffer_sec * 60) + 65  # 60fps + small slack
        self.assertLessEqual(
            tel.log_size(), max_expected,
            f"Unbounded growth: {tel.log_size()} records > {max_expected} expected",
        )

    def test_replay_with_hash_hook(self):
        """A custom hash hook is consistently applied across two runs."""
        recorder = InputRecorder()
        for t in range(30):
            recorder.record(t, {"move_x": 0.1})

        hook_calls = {"n": 0}

        def constant_hook(state, tick_idx):
            hook_calls["n"] += 1
            return 0xCAFEBABE

        runner = ReplayRunner(world_seed=1)
        runner.register_hash_hook(constant_hook)

        h_a = runner.run(recorder.stream())
        hook_calls["n"] = 0
        h_b = runner.run(recorder.stream())

        self.assertEqual(h_a, h_b)
        self.assertGreater(hook_calls["n"], 0)


if __name__ == "__main__":
    unittest.main()
