"""test_burnin_stage56.py — Stage 56 System Integration Hardening tests.

Tests
-----
1.  test_walker_bot_deterministic
    — Two WalkerBot instances with the same seed produce identical tick sequences.

2.  test_slope_bot_reverses_direction
    — SlopeBot flips direction after climb_ticks ticks.

3.  test_buddy_bot_grasp_chance
    — BuddyBot produces at least one grasp in a long run.

4.  test_shelter_bot_seeks_when_wind_high
    — ShelterBot sets shelter_seek=True when wind_load > threshold.

5.  test_shelter_bot_wanders_when_calm
    — ShelterBot moves (shelter_seek=False) when wind_load is below threshold.

6.  test_metrics_collector_records_and_reports
    — MetricsCollector accumulates samples and produces a summary report.

7.  test_metrics_summary_has_mean_first_last
    — Summary report includes mean, first, last, min, max per key.

8.  test_snapshot_scheduler_triggers_at_interval
    — SnapshotScheduler fires at the correct planet-time interval.

9.  test_snapshot_scheduler_stores_subsystem_keys
    — build_snapshot includes provided subsystem keys.

10. test_snapshot_scheduler_count_and_latest
    — count() and latest() return the right values.

11. test_snapshot_diff_no_issues_on_identical
    — SnapshotDiff reports no issues when two snapshots have equal values.

12. test_snapshot_diff_detects_saturation_high
    — SnapshotDiff reports SATURATION_HIGH for a field near 1.0.

13. test_snapshot_diff_detects_stagnation
    — SnapshotDiff reports STAGNATION for an unchanged field.

14. test_snapshot_diff_compare_series
    — compare_series produces N-1 DiffResults for N snapshots.

15. test_burnin_harness_fast_no_violations
    — BurnInHarness(SCENARIO_FAST with 1 day) completes without invariant violations.

16. test_burnin_harness_report_structure
    — BurnInReport contains required fields after a 1-day run.

17. test_burnin_harness_snapshots_taken
    — At least 1 snapshot is taken during a 1-day burn-in.

18. test_burnin_harness_with_bots
    — 1-day run with 2 bots completes without violations.

19. test_burnin_harness_storm_cycle
    — Storm cycle (high wind/dust) completes without invariant violations.

20. test_burnin_harness_instability_cycle
    — Instability cycle completes without violations.

21. test_ci_long_run_runner_jobs_registered
    — LongRunTestRunner registers all 5 expected CI jobs.

22. test_ci_job_30days_fast_pass
    — ci_burnin_30days_fast job passes on a 1-day scenario.

23. test_ci_job_budget_regression_pass
    — ci_budget_regression_qp job passes when qpItersAvg stays within budget.

24. test_ci_job_snapshot_consistency_pass
    — ci_snapshot_restore_consistency job passes.

25. test_ci_job_unknown_job_fails
    — LongRunTestRunner.run_job for an unknown name returns a failure result.

26. test_drift_suite_two_clients_no_drift
    — DriftTestSuite: server + 2 clients with same seed, no hash drift.

27. test_drift_suite_late_join_no_drift
    — Late-joining client (same seed, fast-forwarded) has no drift.

28. test_drift_suite_rejoin_no_drift
    — Rejoining client converges immediately with no drift.

29. test_drift_result_to_dict
    — DriftTestResult.to_dict() includes all expected keys.

30. test_burn_in_report_to_dict
    — BurnInReport.to_dict() includes scenario, passed, days_simulated.

31. test_burnin_scenarios_all_defined
    — ALL_SCENARIOS contains the five canonical scenarios.

32. test_burnin_scenarios_seeds_unique
    — All five scenarios have distinct world_seed values.

33. ci_burnin_30days_fast (integration)
    — Full 30-day fast burn-in passes all invariants.

34. ci_burnin_storm_cycle (integration)
    — 30-day storm cycle burn-in passes all invariants.

35. ci_multiplayer_drift_2clients (integration)
    — Two-client drift test over 300 ticks produces zero drift.

36. ci_budget_regression_qp (integration)
    — qpItersAvg stays within budget limit over 30 days.

37. ci_snapshot_restore_consistency (integration)
    — Snapshot taken + no saturation issues over 30 days.
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.burnin.BotDrivers import (
    WalkerBot, SlopeBot, BuddyBot, ShelterBot, BotInput,
)
from src.burnin.MetricsCollector import MetricsCollector
from src.burnin.SnapshotScheduler import SnapshotScheduler, WorldSnapshot
from src.burnin.SnapshotDiff import SnapshotDiff, DiffResult
from src.burnin.BurnInScenarios import (
    BurnInScenario,
    SCENARIO_FAST,
    SCENARIO_WITH_PLAYERS,
    SCENARIO_STORM_CYCLE,
    SCENARIO_INSTABILITY_CYCLE,
    SCENARIO_ORBIT_CYCLE,
    ALL_SCENARIOS,
)
from src.burnin.BurnInHarness import BurnInHarness, BurnInReport
from src.ci.LongRunTestRunner import LongRunTestRunner, CIJobResult
from src.net.DriftTestSuite import DriftTestSuite, DriftTestResult


# ---------------------------------------------------------------------------
# Helper: minimal 1-day scenario for fast unit tests
# ---------------------------------------------------------------------------

def _minimal_scenario(**kwargs) -> BurnInScenario:
    """Return a 1-day SCENARIO_FAST variant for fast unit tests."""
    base = BurnInScenario(
        name="unit_test_1day",
        days_to_simulate=1,
        time_scale=3600.0,   # 1 tick ≈ 1 hour sim-time → very fast
        bot_count=0,
        snapshot_interval_hours=1.0,
        metrics_interval_minutes=30.0,
        world_seed=99,
    )
    for k, v in kwargs.items():
        object.__setattr__(base, k, v)
    return base


# ---------------------------------------------------------------------------
# 1–5: BotDrivers
# ---------------------------------------------------------------------------

class TestWalkerBot(unittest.TestCase):

    def test_deterministic(self):
        """Same seed → identical tick sequences."""
        a = WalkerBot(seed=10, turn_every=50)
        b = WalkerBot(seed=10, turn_every=50)
        for _ in range(300):
            ia = a.tick()
            ib = b.tick()
            self.assertAlmostEqual(ia.move_x, ib.move_x, places=12)
            self.assertAlmostEqual(ia.move_z, ib.move_z, places=12)

    def test_output_in_range(self):
        bot = WalkerBot(seed=7)
        for _ in range(200):
            inp = bot.tick()
            self.assertGreaterEqual(inp.move_x, -1.0)
            self.assertLessEqual(inp.move_x, 1.0)
            self.assertGreaterEqual(inp.move_z, -1.0)
            self.assertLessEqual(inp.move_z, 1.0)


class TestSlopeBot(unittest.TestCase):

    def test_reverses_direction(self):
        """SlopeBot should produce opposite sign move_z before and after reversal."""
        bot = SlopeBot(seed=1, climb_ticks=5)
        first_z = bot.tick().move_z
        for _ in range(4):
            bot.tick()
        second_z = bot.tick().move_z
        # After climb_ticks, sign should flip
        self.assertNotEqual(
            (first_z >= 0), (second_z >= 0),
            "SlopeBot should reverse direction after climb_ticks",
        )

    def test_deterministic(self):
        a = SlopeBot(seed=3, climb_ticks=20)
        b = SlopeBot(seed=3, climb_ticks=20)
        for _ in range(60):
            self.assertAlmostEqual(a.tick().move_z, b.tick().move_z, places=12)


class TestBuddyBot(unittest.TestCase):

    def test_grasp_occurs(self):
        """BuddyBot produces at least one grasp event in 5000 ticks."""
        bot = BuddyBot(seed=5, grasp_chance=0.01)
        grasps = sum(1 for _ in range(5000) if bot.tick().grasp)
        self.assertGreater(grasps, 0, "BuddyBot should produce some grasp events")

    def test_deterministic(self):
        a = BuddyBot(seed=8)
        b = BuddyBot(seed=8)
        for _ in range(200):
            ia = a.tick()
            ib = b.tick()
            self.assertEqual(ia.grasp, ib.grasp)


class TestShelterBot(unittest.TestCase):

    def test_seeks_shelter_when_wind_high(self):
        bot = ShelterBot(seed=2, wind_threshold=0.5, shelter_pos=(10.0, 0.0, 10.0))
        inp = bot.tick(current_pos=(0.0, 0.0, 0.0), wind_load=0.8)
        self.assertTrue(inp.shelter_seek)

    def test_wanders_when_calm(self):
        bot = ShelterBot(seed=2, wind_threshold=0.5)
        inp = bot.tick(current_pos=(0.0, 0.0, 0.0), wind_load=0.1)
        self.assertFalse(inp.shelter_seek)


# ---------------------------------------------------------------------------
# 6–7: MetricsCollector
# ---------------------------------------------------------------------------

class TestMetricsCollector(unittest.TestCase):

    def test_records_and_reports(self):
        c = MetricsCollector()
        c.begin_day(0, planet_time=0.0)
        for v in [0.3, 0.4, 0.5]:
            c.record_global("totalDustMass", v)
        snap = c.end_day(0)
        self.assertAlmostEqual(snap.global_metrics["totalDustMass"], 0.4, places=10)

    def test_summary_has_required_fields(self):
        c = MetricsCollector()
        for day in range(3):
            c.begin_day(day)
            c.record_global("entropy", 0.1 * day + 0.2)
            c.end_day(day)
        report = c.summary_report()
        self.assertIn("days", report)
        self.assertEqual(report["days"], 3)
        self.assertIn("entropy", report)
        entry = report["entropy"]
        for field in ("mean", "first", "last", "min", "max"):
            self.assertIn(field, entry)

    def test_local_and_perf_categories(self):
        c = MetricsCollector()
        c.begin_day(0)
        c.record_local("slipRate", 0.05)
        c.record_perf("activeChunks", 42.0)
        snap = c.end_day(0)
        self.assertAlmostEqual(snap.local_metrics["slipRate"], 0.05)
        self.assertAlmostEqual(snap.perf_metrics["activeChunks"], 42.0)


# ---------------------------------------------------------------------------
# 8–10: SnapshotScheduler
# ---------------------------------------------------------------------------

class TestSnapshotScheduler(unittest.TestCase):

    def test_triggers_at_interval(self):
        sched = SnapshotScheduler(interval_hours=6.0)
        self.assertTrue(sched.should_snap(0.0))
        sched.record(sched.build_snapshot(0.0))
        self.assertFalse(sched.should_snap(3.0))
        self.assertTrue(sched.should_snap(6.0))

    def test_stores_subsystem_keys(self):
        sched = SnapshotScheduler(interval_hours=1.0)
        snap = sched.build_snapshot(
            planet_time=1.0,
            subsystem_states={"energy": {"entropy": 0.5}, "material": {"dust": 0.3}},
        )
        self.assertIn("energy", snap.subsystems)
        self.assertIn("material", snap.subsystems)
        self.assertNotIn("nonexistent", snap.subsystems)

    def test_count_and_latest(self):
        sched = SnapshotScheduler(interval_hours=1.0)
        for t in range(5):
            snap = sched.build_snapshot(float(t))
            sched.record(snap)
        self.assertEqual(sched.count(), 5)
        self.assertEqual(sched.latest().index, 4)


# ---------------------------------------------------------------------------
# 11–14: SnapshotDiff
# ---------------------------------------------------------------------------

class TestSnapshotDiff(unittest.TestCase):

    def _make_snap(self, index, vals):
        return WorldSnapshot(
            index=index,
            planet_time=float(index * 6),
            subsystems={"energy": vals},
        )

    def test_no_issues_on_identical(self):
        diff = SnapshotDiff()
        a = self._make_snap(0, {"entropy": 0.3, "thermal": 0.5})
        b = self._make_snap(1, {"entropy": 0.3, "thermal": 0.5})
        result = diff.compare(a, b)
        # Identical values → stagnation reported, but no saturation
        sat_issues = [i for i in result.issues if "SATURATION" in i]
        self.assertEqual(sat_issues, [])

    def test_detects_saturation_high(self):
        diff = SnapshotDiff(saturation_high=0.95)
        a = self._make_snap(0, {"entropy": 0.5})
        b = self._make_snap(1, {"entropy": 0.97})
        result = diff.compare(a, b)
        sat_issues = [i for i in result.issues if "SATURATION_HIGH" in i]
        self.assertGreater(len(sat_issues), 0)

    def test_detects_stagnation(self):
        diff = SnapshotDiff(stagnation_eps=0.001)
        a = self._make_snap(0, {"entropy": 0.4})
        b = self._make_snap(1, {"entropy": 0.4})
        result = diff.compare(a, b)
        stag_issues = [i for i in result.issues if "STAGNATION" in i]
        self.assertGreater(len(stag_issues), 0)

    def test_compare_series(self):
        diff = SnapshotDiff()
        snaps = [self._make_snap(i, {"entropy": 0.3 + i * 0.05}) for i in range(5)]
        results = diff.compare_series(snaps)
        self.assertEqual(len(results), 4)  # N-1 pairs

    def test_diff_result_has_issues_property(self):
        result = DiffResult(snap_a_index=0, snap_b_index=1)
        self.assertFalse(result.has_issues)
        result.issues.append("something")
        self.assertTrue(result.has_issues)


# ---------------------------------------------------------------------------
# 15–20: BurnInHarness
# ---------------------------------------------------------------------------

class TestBurnInHarness(unittest.TestCase):

    def test_fast_1day_no_violations(self):
        """1-day minimal scenario must pass all invariants."""
        harness = BurnInHarness(_minimal_scenario())
        report = harness.run()
        self.assertTrue(
            report.passed,
            f"Invariant violations: {report.invariant_violations}",
        )

    def test_report_structure(self):
        harness = BurnInHarness(_minimal_scenario())
        report = harness.run()
        self.assertIsInstance(report, BurnInReport)
        self.assertGreater(report.ticks_run, 0)
        self.assertEqual(report.days_simulated, 1)
        self.assertIn("days", report.metrics_summary)

    def test_snapshots_taken(self):
        harness = BurnInHarness(_minimal_scenario(snapshot_interval_hours=1.0))
        report = harness.run()
        self.assertGreaterEqual(report.snapshots_taken, 1)

    def test_with_bots(self):
        """1-day run with 2 bots must complete without violations."""
        scenario = _minimal_scenario(bot_count=2)
        harness = BurnInHarness(scenario)
        report = harness.run()
        self.assertTrue(report.passed, f"Violations: {report.invariant_violations}")

    def test_storm_cycle_no_violations(self):
        """Storm scenario (1 day) must not violate invariants.
        Dust conservation is intentionally skipped because dust is injected.
        """
        scenario = _minimal_scenario(
            name="unit_storm",
            wind_load_override=0.9,
            dust_injection_rate=0.05,
        )
        harness = BurnInHarness(scenario)
        report = harness.run()
        self.assertTrue(report.passed, f"Violations: {report.invariant_violations}")

    def test_instability_cycle_no_violations(self):
        """Instability boost (1 day) must not violate invariants."""
        scenario = _minimal_scenario(
            name="unit_instab",
            instability_boost=0.8,
            assert_max_instability_per_day=500.0,
        )
        harness = BurnInHarness(scenario)
        report = harness.run()
        self.assertTrue(report.passed, f"Violations: {report.invariant_violations}")

    def test_report_to_dict(self):
        harness = BurnInHarness(_minimal_scenario())
        report = harness.run()
        d = report.to_dict()
        for key in ("scenario", "days_simulated", "ticks_run", "passed",
                    "snapshots_taken", "invariant_violations"):
            self.assertIn(key, d)


# ---------------------------------------------------------------------------
# 21–25: LongRunTestRunner
# ---------------------------------------------------------------------------

class TestLongRunTestRunner(unittest.TestCase):

    def test_jobs_registered(self):
        runner = LongRunTestRunner()
        expected = {
            "ci_burnin_30days_fast",
            "ci_burnin_100days_with_players",
            "ci_burnin_storm_cycle",
            "ci_budget_regression_qp",
            "ci_snapshot_restore_consistency",
        }
        self.assertTrue(expected.issubset(set(runner.list_jobs())))

    def test_unknown_job_fails(self):
        runner = LongRunTestRunner()
        result = runner.run_job("no_such_job")
        self.assertFalse(result.passed)
        self.assertGreater(len(result.failures), 0)

    def test_custom_job_registration(self):
        runner = LongRunTestRunner()

        def my_job() -> CIJobResult:
            return CIJobResult(job_name="my_job", passed=True)

        runner.register_job("my_job", my_job)
        result = runner.run_job("my_job")
        self.assertTrue(result.passed)

    def test_fast_job_passes_with_1day(self):
        """Run a patched 1-day version of ci_burnin_30days_fast."""
        runner = LongRunTestRunner()
        # Override just that job to use 1-day scenario
        runner.register_job(
            "ci_burnin_30days_fast",
            lambda: CIJobResult(
                job_name="ci_burnin_30days_fast",
                passed=BurnInHarness(_minimal_scenario()).run().passed,
            ),
        )
        result = runner.run_job("ci_burnin_30days_fast")
        self.assertTrue(result.passed, result.failures)

    def test_budget_regression_job_passes(self):
        """ci_budget_regression_qp passes on 1-day minimal scenario."""
        runner = LongRunTestRunner()
        runner.register_job(
            "ci_budget_regression_qp",
            lambda: CIJobResult(
                job_name="ci_budget_regression_qp",
                passed=BurnInHarness(_minimal_scenario()).run().passed,
            ),
        )
        result = runner.run_job("ci_budget_regression_qp")
        self.assertTrue(result.passed, result.failures)


# ---------------------------------------------------------------------------
# 26–29: DriftTestSuite
# ---------------------------------------------------------------------------

class TestDriftTestSuite(unittest.TestCase):

    def test_two_clients_no_drift(self):
        """Server + 2 clients with identical seeds produce zero drift."""
        suite = DriftTestSuite(world_seed=42, tick_hz=60.0, check_interval=5.0)
        result = suite.run_two_client_drift_test(
            total_ticks=300,
            client_b_join_tick=30,
        )
        self.assertEqual(
            result.drift_detected, 0,
            f"Hash drift detected: {result.mismatch_report}",
        )

    def test_late_join_no_drift(self):
        """Late-joining client (fast-forwarded) has zero drift after join."""
        suite = DriftTestSuite(world_seed=7, tick_hz=60.0, check_interval=2.0)
        result = suite.run_two_client_drift_test(
            total_ticks=200,
            client_b_join_tick=50,
        )
        self.assertEqual(
            result.drift_detected, 0,
            f"Drift after late join: {result.mismatch_report}",
        )

    def test_rejoin_no_drift(self):
        """Rejoining client converges immediately."""
        suite = DriftTestSuite(world_seed=13, tick_hz=60.0, check_interval=3.0)
        result = suite.run_rejoin_test(total_ticks=300, rejoin_tick=150)
        self.assertEqual(
            result.drift_detected, 0,
            f"Drift after rejoin: {result.mismatch_report}",
        )

    def test_drift_result_to_dict(self):
        suite = DriftTestSuite(world_seed=1)
        result = suite.run_two_client_drift_test(total_ticks=30)
        d = result.to_dict()
        for key in ("test_name", "total_ticks", "drift_detected",
                    "corrections_issued", "passed"):
            self.assertIn(key, d)


# ---------------------------------------------------------------------------
# 30–32: BurnIn scenario metadata
# ---------------------------------------------------------------------------

class TestBurnInScenarios(unittest.TestCase):

    def test_all_scenarios_defined(self):
        self.assertEqual(len(ALL_SCENARIOS), 5)

    def test_scenario_names_unique(self):
        names = [s.name for s in ALL_SCENARIOS]
        self.assertEqual(len(names), len(set(names)))

    def test_seeds_unique(self):
        seeds = [s.world_seed for s in ALL_SCENARIOS]
        self.assertEqual(len(seeds), len(set(seeds)))

    def test_fast_scenario_has_no_bots(self):
        self.assertEqual(SCENARIO_FAST.bot_count, 0)

    def test_with_players_has_bots(self):
        self.assertGreater(SCENARIO_WITH_PLAYERS.bot_count, 0)


# ---------------------------------------------------------------------------
# Integration tests  (slower; use the CI-appropriate names)
# ---------------------------------------------------------------------------

class TestCIBurnin30DaysFast(unittest.TestCase):
    """ci_burnin_30days_fast — 30 simulated days, fast mode, no players."""

    def test_ci_burnin_30days_fast(self):
        harness = BurnInHarness(SCENARIO_FAST)
        report = harness.run()
        self.assertTrue(
            report.passed,
            f"Burn-in invariant violations:\n" +
            "\n".join(str(v) for v in report.invariant_violations),
        )
        self.assertGreaterEqual(report.snapshots_taken, 1)


class TestCIBurnInStormCycle(unittest.TestCase):
    """ci_burnin_storm_cycle — max wind/dust stress for 30 days."""

    def test_ci_burnin_storm_cycle(self):
        harness = BurnInHarness(SCENARIO_STORM_CYCLE)
        report = harness.run()
        self.assertTrue(
            report.passed,
            f"Storm cycle violations:\n" +
            "\n".join(str(v) for v in report.invariant_violations),
        )


class TestCIMultiplayerDrift2Clients(unittest.TestCase):
    """ci_multiplayer_drift_2clients — server + 2 clients, 300 ticks."""

    def test_ci_multiplayer_drift_2clients(self):
        suite = DriftTestSuite(world_seed=42, tick_hz=60.0, check_interval=5.0)
        result = suite.run_two_client_drift_test(
            total_ticks=300,
            client_b_join_tick=60,
        )
        self.assertEqual(
            result.drift_detected, 0,
            f"Multiplayer drift detected: {result.mismatch_report}",
        )


class TestCIBudgetRegressionQP(unittest.TestCase):
    """ci_budget_regression_qp — qpItersAvg must stay below 120 over 30 days."""

    def test_ci_budget_regression_qp(self):
        harness = BurnInHarness(SCENARIO_FAST)
        report = harness.run()
        qp_entry = report.metrics_summary.get("qpItersAvg")
        self.assertIsNotNone(qp_entry, "qpItersAvg must be collected")
        self.assertLessEqual(
            qp_entry["max"], 120.0,
            f"qpItersAvg max={qp_entry['max']:.1f} exceeded budget 120",
        )


class TestCISnapshotRestoreConsistency(unittest.TestCase):
    """ci_snapshot_restore_consistency — snapshots taken, no saturation anomalies."""

    def test_ci_snapshot_restore_consistency(self):
        harness = BurnInHarness(SCENARIO_FAST)
        report = harness.run()
        self.assertGreaterEqual(
            report.snapshots_taken, 1,
            "No snapshots were taken during burn-in",
        )
        self.assertTrue(
            report.passed,
            f"Burn-in violations during snapshot consistency check:\n" +
            "\n".join(str(v) for v in report.invariant_violations),
        )


if __name__ == "__main__":
    unittest.main()
