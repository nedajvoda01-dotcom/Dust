"""test_net_stage24.py — Stage 24 ops-layer smoke tests.

Tests
-----
1. TestHealthOk
   — Spin up a NetworkServer, call /health, confirm status=ok and worldId.

2. TestCompactCreatesBaselineAndPrunes
   — Generate fake geo-event data.
   — Call compact().
   — Verify baseline folder exists with world.json and geo_events.snapshot.
   — Verify geo_events.delta_*.jsonl created, original geo_events.jsonl gone.

3. TestSoftResetNotifiesClients
   — Attach a simulated WebSocket client.
   — Trigger ops reset.
   — Confirm SERVER_WORLD_RESET message broadcast and new worldId differs.

4. TestLogRotation
   — Write a tiny rotate threshold so one log entry triggers rotation.
   — Write two entries; confirm a .gz file appears and count is pruned.

5. TestGuards
   — NaN/Inf guard returns False and logs ERROR.
   — Geo-event cap returns False when exceeded.
   — Patch-batch cap returns False when exceeded.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.net.NetworkServer import NetworkServer
from src.net.PlayerRegistry import PlayerRegistry
from src.net.WorldState import WorldState
from src.ops.OpsLayer import OpsLayer, CAT_OPS, CAT_ERROR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_world(tmp_dir: str) -> WorldState:
    ws = WorldState(tmp_dir)
    ws.load_or_create(default_seed=42)
    return ws


def _make_ops(ws: WorldState, state_dir: str, **kw) -> OpsLayer:
    return OpsLayer(world_state=ws, state_dir=state_dir, **kw)


def _make_server(state_dir: str) -> NetworkServer:
    ws = _make_world(state_dir)
    return NetworkServer(bootstrap=None, config=None, world_state=ws, state_dir=state_dir)


# ---------------------------------------------------------------------------
# 1. TestHealthOk
# ---------------------------------------------------------------------------

class TestHealthOk(unittest.IsolatedAsyncioTestCase):
    """GET /health returns ok status with valid worldId."""

    async def asyncSetUp(self) -> None:
        self._tmp      = tempfile.mkdtemp()
        self._state    = os.path.join(self._tmp, "world_state")
        self.server    = _make_server(self._state)

    async def test_health_ok_status(self) -> None:
        h = self.server._ops.health()
        self.assertIn(h["status"], ("ok", "degraded"),
                      "health status must be ok or degraded")

    async def test_health_has_world_id(self) -> None:
        h = self.server._ops.health()
        self.assertIsInstance(h["worldId"], str)
        self.assertGreater(len(h["worldId"]), 0, "worldId must not be empty")

    async def test_health_http_endpoint(self) -> None:
        """_handle_http('/health') returns 200 + valid JSON body."""
        result = await self.server._handle_http("/health", {})
        self.assertIsNotNone(result)
        status, headers, body = result
        self.assertEqual(status, 200)
        data = json.loads(body.decode("utf-8"))
        self.assertIn("status", data)
        self.assertIn("worldId", data)

    async def test_metrics_http_endpoint(self) -> None:
        """/metrics returns 200 and plain text."""
        result = await self.server._handle_http("/metrics", {})
        self.assertIsNotNone(result)
        status, _, body = result
        self.assertEqual(status, 200)
        text = body.decode("utf-8")
        self.assertIn("players_connected", text)
        self.assertIn("world_state_bytes_total", text)


# ---------------------------------------------------------------------------
# 2. TestCompactCreatesBaselineAndPrunes
# ---------------------------------------------------------------------------

class TestCompactCreatesBaselineAndPrunes(unittest.TestCase):
    """compact() creates baseline dir and rotates geo delta."""

    def setUp(self) -> None:
        self._tmp      = tempfile.mkdtemp()
        self._state    = os.path.join(self._tmp, "world_state")
        self.ws        = _make_world(self._state)
        # Inject some geo events
        for i in range(5):
            self.ws.append_geo_event({"eventId": i, "eventType": "QUAKE",
                                      "pos": [0.0, 1000.0, 0.0], "params": {}})
        self.ops = _make_ops(self.ws, self._state)

    def test_compact_returns_true(self) -> None:
        ok = self.ops.compact()
        self.assertTrue(ok, "compact() must return True on success")

    def test_baseline_dir_created(self) -> None:
        self.ops.compact()
        state = Path(self._state)
        baselines = list(state.glob("baseline_*"))
        self.assertGreater(len(baselines), 0, "No baseline_* directory created")

    def test_baseline_world_json_exists(self) -> None:
        self.ops.compact()
        state    = Path(self._state)
        baseline = sorted(state.glob("baseline_*"))[0]
        self.assertTrue((baseline / "world.json").exists(),
                        "baseline/world.json missing")

    def test_baseline_geo_events_snapshot_exists(self) -> None:
        self.ops.compact()
        state    = Path(self._state)
        baseline = sorted(state.glob("baseline_*"))[0]
        self.assertTrue((baseline / "geo_events.snapshot").exists(),
                        "baseline/geo_events.snapshot missing")

    def test_geo_events_rotated_to_delta(self) -> None:
        self.ops.compact()
        state  = Path(self._state)
        deltas = list(state.glob("geo_events.delta_*.jsonl"))
        self.assertGreater(len(deltas), 0,
                           "geo_events.delta_*.jsonl not created after compact")

    def test_original_geo_events_gone(self) -> None:
        self.ops.compact()
        self.assertFalse((Path(self._state) / "geo_events.jsonl").exists(),
                         "geo_events.jsonl should be rotated away after compact")

    def test_baseline_world_json_content(self) -> None:
        self.ops.compact()
        state    = Path(self._state)
        baseline = sorted(state.glob("baseline_*"))[0]
        data     = json.loads((baseline / "world.json").read_text())
        self.assertIn("worldId", data)
        self.assertIn("seed",    data)
        self.assertEqual(data["worldId"], self.ws.world_id)


# ---------------------------------------------------------------------------
# 3. TestSoftResetNotifiesClients
# ---------------------------------------------------------------------------

class TestSoftResetNotifiesClients(unittest.IsolatedAsyncioTestCase):
    """Soft reset broadcasts SERVER_WORLD_RESET and changes worldId."""

    async def asyncSetUp(self) -> None:
        self._tmp   = tempfile.mkdtemp()
        self._state = os.path.join(self._tmp, "world_state")
        self.ws     = _make_world(self._state)
        self.ops    = _make_ops(self.ws, self._state)

        self.received: list = []

        async def _on_reset(new_id, new_seed, new_sim_time):
            self.received.append({
                "newWorldId": new_id,
                "newSeed":    new_seed,
                "newSimTime": new_sim_time,
            })

        self.ops.set_reset_callback(_on_reset)
        self._old_world_id = self.ws.world_id

    async def test_reset_changes_world_id(self) -> None:
        self.ops.trigger_reset()
        await self.ops.maybe_reset()
        self.assertNotEqual(
            self.ws.world_id, self._old_world_id,
            "worldId must change after soft reset",
        )

    async def test_reset_callback_called(self) -> None:
        self.ops.trigger_reset()
        await self.ops.maybe_reset()
        self.assertEqual(len(self.received), 1,
                         "Reset callback must be called exactly once")

    async def test_reset_callback_new_world_id(self) -> None:
        self.ops.trigger_reset()
        await self.ops.maybe_reset()
        self.assertEqual(self.received[0]["newWorldId"], self.ws.world_id)

    async def test_reset_via_flag_file(self) -> None:
        flag = Path(self._state) / "RESET_NOW"
        flag.touch()
        self.ops._reset_flag_path = flag
        executed = await self.ops.maybe_reset()
        self.assertTrue(executed, "maybe_reset should return True when flag present")
        self.assertFalse(flag.exists(), "RESET_NOW flag must be removed after reset")

    async def test_server_world_reset_broadcast(self) -> None:
        """NetworkServer broadcasts SERVER_WORLD_RESET to connected clients."""
        messages: list = []

        class FakeWS:
            async def send(self, msg):
                messages.append(msg)

        server = _make_server(self._state)
        old_id = server._world_state.world_id
        # Simulate one connected client
        server._connections["p1"] = FakeWS()

        server._ops.trigger_reset()
        await server._ops.maybe_reset()

        reset_msgs = [
            json.loads(m) for m in messages
            if json.loads(m).get("type") == "SERVER_WORLD_RESET"
        ]
        self.assertGreater(len(reset_msgs), 0,
                           "SERVER_WORLD_RESET must be broadcast to clients")
        self.assertNotEqual(reset_msgs[0]["newWorldId"], old_id)


# ---------------------------------------------------------------------------
# 4. TestLogRotation
# ---------------------------------------------------------------------------

class TestLogRotation(unittest.TestCase):
    """Log files rotate when they exceed the size threshold."""

    def setUp(self) -> None:
        self._tmp   = tempfile.mkdtemp()
        self._state = os.path.join(self._tmp, "world_state")
        Path(self._state).mkdir(parents=True, exist_ok=True)
        self.ws  = _make_world(self._state)
        # Use tiny rotate threshold so a single entry triggers rotation
        self.ops = _make_ops(self.ws, self._state)
        self.ops._log_rotate_mb = 0.000_001   # ~1 byte — always rotate

    def test_rotation_creates_gz(self) -> None:
        self.ops.log("OPS", "entry one")
        self.ops.log("OPS", "entry two")
        gz_files = list(self.ops._log_dir.glob("server.log.*.jsonl.gz"))
        self.assertGreater(len(gz_files), 0,
                           "Rotation must create at least one .gz file")

    def test_pruning_removes_excess(self) -> None:
        self.ops._log_keep_files = 2
        # Write several entries to generate multiple rotated files
        for i in range(10):
            self.ops.log("OPS", f"entry {i}")
        gz_files = list(self.ops._log_dir.glob("server.log.*.jsonl.gz"))
        self.assertLessEqual(
            len(gz_files), self.ops._log_keep_files + 1,
            f"Should keep at most {self.ops._log_keep_files} rotated files "
            f"(got {len(gz_files)})",
        )


# ---------------------------------------------------------------------------
# 5. TestGuards
# ---------------------------------------------------------------------------

class TestGuards(unittest.TestCase):
    """Safety guard methods behave correctly."""

    def setUp(self) -> None:
        self._tmp   = tempfile.mkdtemp()
        self._state = os.path.join(self._tmp, "world_state")
        Path(self._state).mkdir(parents=True, exist_ok=True)
        self.ws  = _make_world(self._state)
        self.ops = _make_ops(self.ws, self._state)

    def test_nan_guard_finite_ok(self) -> None:
        self.assertTrue(self.ops.check_nan_in_value(1.0, "pos.x"))

    def test_nan_guard_nan_rejected(self) -> None:
        import math
        self.assertFalse(self.ops.check_nan_in_value(math.nan, "pos.x"))

    def test_nan_guard_inf_rejected(self) -> None:
        import math
        self.assertFalse(self.ops.check_nan_in_value(math.inf, "vel.z"))

    def test_geo_event_cap_allows_within_limit(self) -> None:
        self.ops._max_geo_per_hour = 100
        # Reset hour window
        self.ops._geo_hour_count = 0
        self.ops._geo_hour_start = time.monotonic()
        ok = self.ops.check_geo_event_cap()
        self.assertTrue(ok)

    def test_geo_event_cap_blocks_when_exceeded(self) -> None:
        self.ops._max_geo_per_hour = 1
        self.ops._geo_hour_count   = 2   # already at limit
        self.ops._geo_hour_start   = time.monotonic()
        ok = self.ops.check_geo_event_cap()
        self.assertFalse(ok)

    def test_patch_batch_cap_allows_within_limit(self) -> None:
        self.ops._max_patch_per_min = 100
        self.ops._patch_min_count   = 0
        self.ops._patch_min_start   = time.monotonic()
        ok = self.ops.check_patch_batch_cap()
        self.assertTrue(ok)

    def test_patch_batch_cap_blocks_when_exceeded(self) -> None:
        self.ops._max_patch_per_min = 1
        self.ops._patch_min_count   = 5
        self.ops._patch_min_start   = time.monotonic()
        ok = self.ops.check_patch_batch_cap()
        self.assertFalse(ok)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
