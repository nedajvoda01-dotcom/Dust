"""test_net_stage22.py — Stage 22 network-harness tests.

Tests
-----
1. TestInterestFiltering
   — Three players in different sectors.
   — Player A does NOT receive Player C's state when C is outside the
     interest radius.

2. TestRejoinCatchup
   — Server accumulates geo events while a "client" is disconnected.
   — REJOIN_RESYNC with the correct lastEventId receives only the delta.
   — REJOIN_RESYNC with a stale cursor triggers a full WORLD_SYNC.

3. TestRateLimiting
   — Rapid messages from one client are dropped beyond player_send_hz.
   — The rate window resets after 1 second.

4. TestPingPong
   — NetworkServer replies to a PING with a matching PONG.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.net.NetworkServer import NetworkServer
from src.net.PlayerRegistry import PlayerRegistry
from src.net.SpawnAnchor import SpawnAnchor
from src.net.WorldState import WorldState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_free_port() -> int:
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _make_server(state_dir: str) -> NetworkServer:
    ws_state = WorldState(state_dir)
    ws_state.load_or_create(default_seed=42)
    return NetworkServer(
        bootstrap   = None,
        config      = None,
        world_state = ws_state,
        state_dir   = state_dir,
    )


async def _recv_timeout(ws, timeout: float = 5.0):
    return await asyncio.wait_for(ws.recv(), timeout=timeout)


# ---------------------------------------------------------------------------
# 1. TestInterestFiltering
# ---------------------------------------------------------------------------

class TestInterestFiltering(unittest.TestCase):
    """Sector-based interest management excludes out-of-range players."""

    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp()
        self.reg  = PlayerRegistry()

    def _make_pos(self, lat_deg: float, lon_deg: float, r: float = 1001.8) -> list:
        """Convert lat/lon to a unit-sphere position scaled by *r*."""
        lat = math.radians(lat_deg)
        lon = math.radians(lon_deg)
        x   = r * math.cos(lat) * math.cos(lon)
        y   = r * math.sin(lat)
        z   = r * math.cos(lat) * math.sin(lon)
        return [x, y, z]

    def test_nearby_excludes_far_player(self) -> None:
        """Player at (0°,0°) must not see player at (80°,0°) within 5° sector."""
        pos_a = self._make_pos(0.0, 0.0)
        pos_b = self._make_pos(2.0, 2.0)   # within 5° of A
        pos_c = self._make_pos(80.0, 0.0)  # far away from A

        self.reg.add("a", pos_a)
        self.reg.add("b", pos_b)
        self.reg.add("c", pos_c)

        # sector_deg=5 × (sector_radius=2 + 0.5) = 12.5° view cone
        view_deg = 5.0 * (2 + 0.5)
        nearby = self.reg.get_nearby(pos_a, view_deg)
        ids = {r.player_id for r in nearby}

        self.assertIn("a", ids, "A should see itself")
        self.assertIn("b", ids, "B is close — A should see B")
        self.assertNotIn("c", ids, "C is far — A must NOT see C")

    def test_nearby_includes_all_when_pos_is_zero(self) -> None:
        """get_nearby on zero vector falls back to all players."""
        self.reg.add("x", [1.0, 0.0, 0.0])
        self.reg.add("y", [0.0, 1.0, 0.0])
        nearby = self.reg.get_nearby([0.0, 0.0, 0.0], 5.0)
        self.assertEqual(len(nearby), 2)

    def test_get_player_pos_returns_position(self) -> None:
        pos = [1.0, 1001.0, 2.0]
        self.reg.add("p1", pos)
        result = self.reg.get_player_pos("p1")
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result[0], 1.0, places=5)

    def test_get_player_pos_returns_none_for_unknown(self) -> None:
        result = self.reg.get_player_pos("ghost")
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# 2. TestRejoinCatchup
# ---------------------------------------------------------------------------

class TestRejoinCatchup(unittest.IsolatedAsyncioTestCase):
    """REJOIN_RESYNC delivers only the delta events the client missed."""

    async def asyncSetUp(self) -> None:
        self._tmp      = tempfile.mkdtemp()
        self._state_dir = os.path.join(self._tmp, "world_state")

        self.ws_state = WorldState(self._state_dir)
        self.ws_state.load_or_create(default_seed=42)

        # Pre-populate 5 geo events
        for i in range(1, 6):
            self.ws_state.append_geo_event({
                "eventId":   i,
                "eventType": "LANDSLIDE",
                "pos":       [0.0, 0.0, 0.0],
            })

        self.registry = PlayerRegistry()
        self.anchor   = SpawnAnchor()

        self.server = NetworkServer(
            bootstrap        = None,
            config           = None,
            world_state      = self.ws_state,
            player_registry  = self.registry,
            spawn_anchor     = self.anchor,
            state_dir        = self._state_dir,
        )

    async def test_delta_resync_sends_only_new_events(self) -> None:
        """lastEventId=2 should receive events 3, 4, 5 in catchupEvents."""
        messages: list = []

        class FakeWS:
            async def send(self, msg):
                messages.append(json.loads(msg))

        await self.server._send_rejoin(FakeWS(), "test_player", last_event_id=2)

        self.assertEqual(len(messages), 1)
        reply = messages[0]
        self.assertEqual(reply["type"], "REJOIN_RESYNC")
        ids = [e["eventId"] for e in reply["catchupEvents"]]
        self.assertEqual(sorted(ids), [3, 4, 5])

    async def test_delta_resync_empty_when_up_to_date(self) -> None:
        """lastEventId=5 should receive an empty catchupEvents list."""
        messages: list = []

        class FakeWS:
            async def send(self, msg):
                messages.append(json.loads(msg))

        await self.server._send_rejoin(FakeWS(), "test_player", last_event_id=5)

        self.assertEqual(len(messages), 1)
        reply = messages[0]
        self.assertEqual(reply["type"], "REJOIN_RESYNC")
        self.assertEqual(reply["catchupEvents"], [])

    async def test_stale_client_receives_world_sync(self) -> None:
        """When catchup would exceed the limit, a full WORLD_SYNC is sent."""
        # Add >100 events to trigger the full-sync path
        for i in range(6, 110):
            self.ws_state.append_geo_event({
                "eventId":   i,
                "eventType": "SCREE",
                "pos":       [0, 0, 0],
            })

        messages: list = []

        class FakeWS:
            async def send(self, msg):
                messages.append(json.loads(msg))

        self.registry.add("p_stale", [0.0, 1001.8, 0.0])
        await self.server._send_rejoin(FakeWS(), "p_stale", last_event_id=-1)

        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["type"], "WORLD_SYNC")


# ---------------------------------------------------------------------------
# 3. TestRateLimiting
# ---------------------------------------------------------------------------

class TestRateLimiting(unittest.IsolatedAsyncioTestCase):
    """Rate limiting allows up to player_send_hz messages per second."""

    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.mkdtemp()
        state_dir = os.path.join(self._tmp, "ws")

        ws_state = WorldState(state_dir)
        ws_state.load_or_create(default_seed=42)

        self.server = NetworkServer(
            bootstrap   = None,
            config      = None,
            world_state = ws_state,
            state_dir   = state_dir,
        )
        # Low limit to make the test fast
        self.server._player_send_hz = 5

    def test_within_limit(self) -> None:
        """First N messages within the window are accepted."""
        pid = "rate_test_player"
        accepted = sum(
            1 for _ in range(5)
            if self.server._check_rate_limit(pid)
        )
        self.assertEqual(accepted, 5)

    def test_beyond_limit_rejected(self) -> None:
        """Message N+1 in the same window is rejected."""
        pid = "rate_test_player2"
        for _ in range(5):
            self.server._check_rate_limit(pid)
        # 6th message should be rejected
        self.assertFalse(
            self.server._check_rate_limit(pid),
            "6th message in window should be rate-limited",
        )

    def test_window_resets_after_one_second(self) -> None:
        """After the window expires, messages are accepted again."""
        pid = "rate_test_player3"
        for _ in range(5):
            self.server._check_rate_limit(pid)

        # Manually back-date all timestamps by 2 seconds
        window = self.server._rate_windows[pid]
        for i in range(len(window)):
            window[i] -= 2.0

        # Now the window is cleared and a new message should be accepted
        self.assertTrue(
            self.server._check_rate_limit(pid),
            "After window reset, message should be accepted",
        )

    async def test_rate_limited_message_skipped(self) -> None:
        """PLAYER_STATE updates are ignored when rate-limited."""
        pid = "rl_update_test"
        self.server._registry.add(pid, [0.0, 1001.8, 0.0])
        self.server._player_send_hz = 1  # only 1 per second

        class _FakeWS:
            async def send(self, _): pass

        # First message accepted
        await self.server._on_client_message(
            pid, _FakeWS(),
            {"type": "PLAYER_STATE", "pos": [1.0, 1001.8, 0.0],
             "vel": [0.0, 0.0, 0.0], "flags": 0},
        )
        pos_after_first = list(self.server._registry._players[pid].pos)

        # Second message rate-limited — position must NOT change
        await self.server._on_client_message(
            pid, _FakeWS(),
            {"type": "PLAYER_STATE", "pos": [99.0, 1001.8, 0.0],
             "vel": [0.0, 0.0, 0.0], "flags": 0},
        )
        pos_after_second = list(self.server._registry._players[pid].pos)

        self.assertEqual(pos_after_first, pos_after_second,
                         "Rate-limited update must not change player position")

    async def test_ping_not_rate_limited(self) -> None:
        """PING messages are exempt from the rate limit."""
        pid = "rl_ping_test"
        self.server._player_send_hz = 1  # very low limit

        received: list = []

        class _FakeWS:
            async def send(self, msg):
                received.append(json.loads(msg))

        ws = _FakeWS()

        # Exhaust the rate limit with a PLAYER_STATE first
        self.server._registry.add(pid, [0.0, 1001.8, 0.0])
        await self.server._on_client_message(
            pid, ws,
            {"type": "PLAYER_STATE", "pos": [1.0, 1001.8, 0.0],
             "vel": [0.0, 0.0, 0.0], "flags": 0},
        )

        # Now a PING must still be answered despite rate limit
        await self.server._on_client_message(
            pid, ws,
            {"type": "PING", "t": 777.0},
        )

        pongs = [m for m in received if m.get("type") == "PONG"]
        self.assertEqual(len(pongs), 1, "PING must be answered even when rate-limited")


# ---------------------------------------------------------------------------
# 4. TestPingPong
# ---------------------------------------------------------------------------

class TestPingPong(unittest.IsolatedAsyncioTestCase):
    """Server echoes PING timestamp back in a PONG."""

    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.mkdtemp()
        state_dir = os.path.join(self._tmp, "ws")
        ws_state  = WorldState(state_dir)
        ws_state.load_or_create(default_seed=42)

        self.server = NetworkServer(
            bootstrap   = None,
            config      = None,
            world_state = ws_state,
            state_dir   = state_dir,
        )

    async def test_pong_echoes_timestamp(self) -> None:
        """PONG message must contain the same 't' sent in the PING."""
        received: list = []

        class FakeWS:
            async def send(self, msg):
                received.append(json.loads(msg))

        ping_t = 12345.678
        await self.server._on_client_message(
            "ping_player",
            FakeWS(),
            {"type": "PING", "t": ping_t},
        )

        self.assertEqual(len(received), 1)
        self.assertEqual(received[0]["type"], "PONG")
        self.assertAlmostEqual(received[0]["t"], ping_t, places=3)

    async def test_ping_without_timestamp(self) -> None:
        """PONG for a PING with no 't' must still be sent (t may be None)."""
        received: list = []

        class FakeWS:
            async def send(self, msg):
                received.append(json.loads(msg))

        await self.server._on_client_message(
            "ping_player2",
            FakeWS(),
            {"type": "PING"},
        )

        self.assertEqual(len(received), 1)
        self.assertEqual(received[0]["type"], "PONG")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
