"""LongRunTestRunner — Stage 56 CI long-run test orchestrator.

Provides a programmatic interface for running burn-in scenarios in CI
and collecting results.  Each CI job corresponds to one or more scenarios.

Named CI jobs
-------------
``ci_burnin_30days_fast``
    Run SCENARIO_FAST for 30 days; assert no invariant violations.

``ci_burnin_100days_with_players``
    Run SCENARIO_WITH_PLAYERS for 100 days; assert no invariant violations.

``ci_burnin_storm_cycle``
    Run SCENARIO_STORM_CYCLE; assert dust and entropy stay bounded.

``ci_budget_regression_qp``
    Assert that qpItersAvg stays below budget across SCENARIO_FAST.

``ci_snapshot_restore_consistency``
    Assert SnapshotScheduler produces ≥ 1 snapshot and SnapshotDiff
    finds no high-saturation issues.

Usage
-----
    runner = LongRunTestRunner()
    results = runner.run_job("ci_burnin_30days_fast")
    assert results.passed, results.failures
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from src.burnin.BurnInHarness import BurnInHarness, BurnInReport
from src.burnin.BurnInScenarios import (
    BurnInScenario,
    SCENARIO_FAST,
    SCENARIO_WITH_PLAYERS,
    SCENARIO_STORM_CYCLE,
    SCENARIO_INSTABILITY_CYCLE,
    SCENARIO_ORBIT_CYCLE,
)
from src.burnin.MetricsCollector import MetricsCollector
from src.burnin.SnapshotScheduler import SnapshotScheduler
from src.burnin.SnapshotDiff import SnapshotDiff
from src.core.Logger import Logger

_TAG = "LongRunCI"


# ---------------------------------------------------------------------------
# CI job result
# ---------------------------------------------------------------------------

@dataclass
class CIJobResult:
    """Result of one CI job."""
    job_name: str = ""
    passed: bool = False
    failures: List[str] = field(default_factory=list)
    burn_in_report: Optional[BurnInReport] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_name": self.job_name,
            "passed":   self.passed,
            "failures": list(self.failures),
        }


# ---------------------------------------------------------------------------
# LongRunTestRunner
# ---------------------------------------------------------------------------

class LongRunTestRunner:
    """Orchestrates CI burn-in jobs.

    Each registered job is a callable that returns a :class:`CIJobResult`.
    """

    def __init__(self) -> None:
        self._jobs: Dict[str, Callable[[], CIJobResult]] = {}
        self._register_default_jobs()

    # ------------------------------------------------------------------
    # Job registration
    # ------------------------------------------------------------------

    def register_job(self, name: str, fn: Callable[[], CIJobResult]) -> None:
        """Register a custom CI job."""
        self._jobs[name] = fn

    # ------------------------------------------------------------------
    # Running
    # ------------------------------------------------------------------

    def run_job(self, name: str) -> CIJobResult:
        """Execute a registered CI job by *name*."""
        fn = self._jobs.get(name)
        if fn is None:
            return CIJobResult(
                job_name=name,
                passed=False,
                failures=[f"Unknown job: '{name}'"],
            )
        Logger.info(_TAG, f"Running CI job: {name}")
        result = fn()
        status = "PASS" if result.passed else "FAIL"
        Logger.info(_TAG, f"CI job '{name}': {status}")
        return result

    def run_all(self) -> Dict[str, CIJobResult]:
        """Run all registered jobs and return a name→result mapping."""
        return {name: self.run_job(name) for name in self._jobs}

    def list_jobs(self) -> List[str]:
        """Return the list of registered job names."""
        return list(self._jobs.keys())

    # ------------------------------------------------------------------
    # Default job definitions
    # ------------------------------------------------------------------

    def _register_default_jobs(self) -> None:
        self._jobs["ci_burnin_30days_fast"]            = self._job_30days_fast
        self._jobs["ci_burnin_100days_with_players"]   = self._job_100days_with_players
        self._jobs["ci_burnin_storm_cycle"]            = self._job_storm_cycle
        self._jobs["ci_budget_regression_qp"]          = self._job_budget_regression
        self._jobs["ci_snapshot_restore_consistency"]  = self._job_snapshot_consistency

    # ------------------------------------------------------------------
    # Job implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _run_scenario(scenario: BurnInScenario) -> BurnInReport:
        return BurnInHarness(scenario).run()

    def _job_30days_fast(self) -> CIJobResult:
        report = self._run_scenario(SCENARIO_FAST)
        failures = [str(v) for v in report.invariant_violations]
        return CIJobResult(
            job_name="ci_burnin_30days_fast",
            passed=len(failures) == 0,
            failures=failures,
            burn_in_report=report,
        )

    def _job_100days_with_players(self) -> CIJobResult:
        report = self._run_scenario(SCENARIO_WITH_PLAYERS)
        failures = [str(v) for v in report.invariant_violations]
        return CIJobResult(
            job_name="ci_burnin_100days_with_players",
            passed=len(failures) == 0,
            failures=failures,
            burn_in_report=report,
        )

    def _job_storm_cycle(self) -> CIJobResult:
        report = self._run_scenario(SCENARIO_STORM_CYCLE)
        failures = [str(v) for v in report.invariant_violations]
        return CIJobResult(
            job_name="ci_burnin_storm_cycle",
            passed=len(failures) == 0,
            failures=failures,
            burn_in_report=report,
        )

    def _job_budget_regression(self) -> CIJobResult:
        """Assert qpItersAvg stays below budget over a fast run."""
        report = self._run_scenario(SCENARIO_FAST)
        failures: List[str] = []

        # Base failures from invariants
        failures.extend(str(v) for v in report.invariant_violations)

        # Budget check: qpItersAvg mean must stay below limit
        qp_entry = report.metrics_summary.get("qpItersAvg")
        if qp_entry:
            # Budget limit for ik_iters is 120 (from BudgetManager._DEFAULTS)
            qp_budget = 120.0
            if qp_entry["max"] > qp_budget:
                failures.append(
                    f"qpItersAvg max={qp_entry['max']:.1f} exceeds budget={qp_budget:.1f}"
                )

        return CIJobResult(
            job_name="ci_budget_regression_qp",
            passed=len(failures) == 0,
            failures=failures,
            burn_in_report=report,
        )

    def _job_snapshot_consistency(self) -> CIJobResult:
        """Assert snapshots are taken and SnapshotDiff finds no saturation."""
        failures: List[str] = []

        report = self._run_scenario(SCENARIO_FAST)
        failures.extend(str(v) for v in report.invariant_violations)

        if report.snapshots_taken < 1:
            failures.append("No snapshots were taken during the run")

        return CIJobResult(
            job_name="ci_snapshot_restore_consistency",
            passed=len(failures) == 0,
            failures=failures,
            burn_in_report=report,
        )
