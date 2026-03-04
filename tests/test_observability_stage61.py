"""test_observability_stage61.py — Stage 61 Observability & Live Tuning tests.

Tests
-----
1.  test_metrics_have_required_dimensions
    — MetricsRegistry.export() includes worldId, regionId, serverTick.

2.  test_health_score_changes_with_entropy
    — WorldHealthScorer returns a lower score when entropy is out of bounds.

3.  test_tuning_applies_only_on_tick_boundary
    — A proposed change is applied only at a tick that is a multiple of the
      configured boundary; earlier ticks leave it pending.

4.  test_tuning_persisted_in_snapshot
    — snapshot_meta() includes tuningEpoch and tuningConfigHash after a change
      is applied.

5.  test_tuning_rejects_out_of_range
    — TuningValidator rejects values outside the declared range, and accepts
      values within range.

6.  test_autobudget_only_changes_lod_params
    — AutoBudgetController never proposes changes to non-LOD parameters even
      under simulated overload.

7.  test_ops_auth_rejects_unauthorized
    — OpsServer in 'token' auth mode returns 401 for requests without a valid
      Bearer token, and 200 for requests with the correct token.
"""
from __future__ import annotations

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.obs.MetricsRegistry import MetricsRegistry, REQUIRED_DIMENSIONS
from src.obs.WorldHealthScorer import WorldHealthScorer, HealthInputs
from src.obs.StructuredLogger import StructuredLogger, CANONICAL_EVENTS
from src.obs.TraceProvider import TraceProvider, CANONICAL_SPANS
from src.ops.TuningValidator import TuningValidator
from src.ops.TuningManager import TuningManager
from src.ops.AutoBudgetController import AutoBudgetController
from src.ops.OpsServer import OpsServer


class TestMetricsRequiredDimensions(unittest.TestCase):
    """1. MetricsRegistry export includes all required dimensions."""

    def test_metrics_have_required_dimensions(self):
        reg = MetricsRegistry(world_id="world-1", region_id="region-A")
        reg.advance_tick(42)
        reg.record("tick_ms_avg", 5.0)

        export = reg.export()

        for dim in REQUIRED_DIMENSIONS:
            self.assertIn(dim, export, f"Required dimension '{dim}' missing from export")

        self.assertEqual(export["worldId"],    "world-1")
        self.assertEqual(export["regionId"],   "region-A")
        self.assertEqual(export["serverTick"], 42)


class TestHealthScoreChangesWithEntropy(unittest.TestCase):
    """2. WorldHealthScorer returns lower score when entropy is out of bounds."""

    def test_health_score_changes_with_entropy(self):
        scorer = WorldHealthScorer()

        nominal = HealthInputs(entropy=0.5, entropy_lower=0.2, entropy_upper=0.8)
        out_of_bounds = HealthInputs(entropy=0.05, entropy_lower=0.2, entropy_upper=0.8)

        score_nominal = scorer.score(nominal)
        score_low     = scorer.score(out_of_bounds)

        self.assertGreater(score_nominal.score, score_low.score,
                           "Score should be lower when entropy is outside bounds")
        self.assertIn("entropy_out_of_bounds", score_low.alerts)


class TestTuningAppliesOnTickBoundary(unittest.TestCase):
    """3. Tuning change applies only at tick % boundary == 0."""

    def test_tuning_applies_only_on_tick_boundary(self):
        tm = TuningManager(apply_tick_boundary=64)

        accepted, errors = tm.propose({"net_update_hz": 20})
        self.assertTrue(accepted, errors)
        self.assertTrue(tm.has_pending)

        # Non-boundary ticks: change stays pending
        for tick in [1, 32, 63]:
            result = tm.on_tick(tick)
            self.assertIsNone(result, f"Should not apply at tick {tick}")
        self.assertTrue(tm.has_pending)

        # Exact boundary: change applied
        result = tm.on_tick(64)
        self.assertIsNotNone(result, "Should apply at tick 64 (boundary)")
        self.assertFalse(tm.has_pending)
        self.assertEqual(tm.current_config.get("net_update_hz"), 20)
        self.assertEqual(result.applied_at, 64)


class TestTuningPersistedInSnapshot(unittest.TestCase):
    """4. snapshot_meta() includes tuningEpoch and tuningConfigHash."""

    def test_tuning_persisted_in_snapshot(self):
        tm = TuningManager(apply_tick_boundary=8)

        # Baseline state
        meta0 = tm.snapshot_meta()
        self.assertIn("tuningEpoch",      meta0)
        self.assertIn("tuningConfigHash", meta0)
        self.assertEqual(meta0["tuningEpoch"], 0)

        # Propose + apply
        tm.propose({"net_update_hz": 15})
        tm.on_tick(8)

        meta1 = tm.snapshot_meta()
        self.assertEqual(meta1["tuningEpoch"], 1)
        self.assertNotEqual(meta1["tuningConfigHash"], meta0["tuningConfigHash"],
                            "Config hash must change when config changes")


class TestTuningRejectsOutOfRange(unittest.TestCase):
    """5. TuningValidator rejects out-of-range values and accepts in-range values."""

    def test_tuning_rejects_out_of_range(self):
        v = TuningValidator()

        # Out-of-range: net_update_hz must be in [1, 60]
        errors = v.validate({"net_update_hz": 1000})
        self.assertTrue(len(errors) > 0, "Should reject value above max")

        errors = v.validate({"net_update_hz": 0})
        self.assertTrue(len(errors) > 0, "Should reject value below min")

        # Not in allowlist
        errors = v.validate({"worldSeed": 42})
        self.assertTrue(len(errors) > 0, "Should reject disallowed parameter")

        # Valid value
        errors = v.validate({"net_update_hz": 30})
        self.assertEqual(errors, [], f"Should accept valid value, got: {errors}")

        # Multiple valid values
        errors = v.validate({"net_update_hz": 20, "qp_max_iters": 100})
        self.assertEqual(errors, [])


class TestAutobudgetOnlyChangesLodParams(unittest.TestCase):
    """6. AutoBudgetController only proposes LOD/budget parameter changes."""

    def test_autobudget_only_changes_lod_params(self):
        tm  = TuningManager(apply_tick_boundary=1)
        abc = AutoBudgetController(
            tuning_manager    = tm,
            tick_ms_threshold = 1.0,   # very low → always triggers
            net_out_threshold = 1.0,   # very low → always triggers
            health_threshold  = 1.0,   # very high → always triggers
        )

        reg = MetricsRegistry()
        reg.record("tick_ms_p99", 100.0)  # way over threshold
        reg.record("net_out_bps",  10e6)  # way over threshold

        delta = abc.update(metrics=reg, health_score=0.1, server_tick=1)

        self.assertTrue(
            abc.only_changes_lod_params(delta),
            f"AutoBudgetController proposed non-LOD params: {list(delta.keys())}"
        )

        # Verify none of the forbidden physics params appear
        forbidden = {"worldSeed", "entropy", "instability_threshold",
                     "dust_decay_rate", "ice_decay_rate"}
        self.assertTrue(
            forbidden.isdisjoint(delta.keys()),
            f"Forbidden physics params in delta: {forbidden & delta.keys()}"
        )


class TestOpsAuthRejectsUnauthorized(unittest.TestCase):
    """7. OpsServer with token auth rejects missing/wrong tokens."""

    def test_ops_auth_rejects_unauthorized(self):
        server = OpsServer(auth_mode="token", ops_token="secret-token-xyz")

        # No token
        status, body = server.handle("GET", "/ops/health", headers={})
        self.assertEqual(status, 401, f"Expected 401, got {status}: {body}")

        # Wrong token
        status, body = server.handle(
            "GET", "/ops/health",
            headers={"Authorization": "Bearer wrong-token"}
        )
        self.assertEqual(status, 401, f"Expected 401 for wrong token, got {status}: {body}")

        # Correct token
        status, body = server.handle(
            "GET", "/ops/health",
            headers={"Authorization": "Bearer secret-token-xyz"}
        )
        self.assertEqual(status, 200, f"Expected 200 for valid token, got {status}: {body}")
        self.assertIn("worldHealthScore", body.get("data", {}))


if __name__ == "__main__":
    unittest.main()
