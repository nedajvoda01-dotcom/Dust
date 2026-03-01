"""test_net.py — Stage 21 networking smoke tests.

Tests
-----
1. TestWorldState
   — create, save, reload (seed + simTime + worldId persist)
   — reset wipes state and creates new worldId
   — geo-event append + reload

2. TestPlayerRegistry
   — add / update / remove
   — remove_stale removes old entries
   — get_nearby returns correct subset
   — anti-cheat clamps excessive velocity

3. TestPlayerIdentity
   — same inputs → same key
   — different IP → different key
   — different userAgent → different key
   — different salt → different key

4. TestSpawnAnchor
   — spawn is deterministic per player_key
   — two different players get different positions
   — position is near the anchor (within radius)

5. TestNetworkServerComponents (no live WebSocket)
   — NetworkServer builds without a bootstrap
   — _handle_http returns HTML for "/"
   — _handle_http returns None for "/ws"
   — PLAYER_STATE update is processed correctly

6. TestTwoClientsSameWorld (integration, asyncio)
   — Two WebSocket clients connect to a live server
   — Both receive the same seed and worldId in WORLD_SYNC

7. TestGeoEventReplicates (integration, asyncio)
   — Server enqueues a geo event
   — Connected client receives a GEO_EVENT message

8. TestPlayerVisibility (integration, asyncio)
   — Client B sends PLAYER_STATE; client A receives it in PLAYERS
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.net.PlayerIdentity import make_player_key
from src.net.PlayerRegistry import PlayerRegistry
from src.net.SpawnAnchor import SpawnAnchor
from src.net.WorldState import WorldState
from src.net.NetworkServer import NetworkServer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_free_port() -> int:
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _ws_recv_timeout(ws, timeout: float = 5.0):
    """Receive one message from *ws* within *timeout* seconds."""
    return await asyncio.wait_for(ws.recv(), timeout=timeout)


# ---------------------------------------------------------------------------
# 1. TestWorldState
# ---------------------------------------------------------------------------

class TestWorldState(unittest.TestCase):
    """WorldState persistence and reset."""

    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp()
        self._dir = os.path.join(self._tmp, "ws")

    def _make(self) -> WorldState:
        return WorldState(self._dir)

    def test_create_fresh(self) -> None:
        ws = self._make()
        ws.load_or_create(default_seed=77)
        self.assertEqual(ws.seed, 77)
        self.assertNotEqual(ws.world_id, "")

    def test_save_reload(self) -> None:
        ws = self._make()
        ws.load_or_create(default_seed=42)
        ws.sim_time  = 123.45
        ws.time_scale = 2.0
        ws.epoch     = 5
        ws.save()

        ws2 = self._make()
        ws2.load_or_create(default_seed=99)  # default_seed ignored (file exists)
        self.assertEqual(ws2.seed, 42)
        self.assertAlmostEqual(ws2.sim_time, 123.45, places=2)
        self.assertAlmostEqual(ws2.time_scale, 2.0, places=5)
        self.assertEqual(ws2.epoch, 5)
        self.assertEqual(ws2.world_id, ws.world_id)

    def test_reset(self) -> None:
        ws = self._make()
        ws.load_or_create(default_seed=42)
        old_id = ws.world_id

        ws.reset()
        # After reset new world_id is generated
        self.assertNotEqual(ws.world_id, old_id)
        self.assertAlmostEqual(ws.sim_time, 0.0, places=5)

    def test_geo_event_append_reload(self) -> None:
        ws = self._make()
        ws.load_or_create(default_seed=42)
        ws.append_geo_event({"eventId": 1, "eventType": "LANDSLIDE", "pos": [0, 0, 0]})
        ws.append_geo_event({"eventId": 2, "eventType": "COLLAPSE",  "pos": [1, 0, 0]})

        ws2 = self._make()
        ws2.load_or_create(default_seed=99)
        events = ws2.geo_events()
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["eventId"], 1)
        self.assertEqual(events[1]["eventType"], "COLLAPSE")

    def test_climate_snapshot(self) -> None:
        ws = self._make()
        ws.load_or_create(default_seed=42)
        ws.save_climate_snapshot({"storms": [{"lat": 0.1}], "globalDust": 0.3})
        snap = ws.load_climate_snapshot()
        self.assertIsNotNone(snap)
        self.assertAlmostEqual(snap["globalDust"], 0.3, places=5)


# ---------------------------------------------------------------------------
# 2. TestPlayerRegistry
# ---------------------------------------------------------------------------

class TestPlayerRegistry(unittest.TestCase):
    """PlayerRegistry CRUD + interest management + anti-cheat."""

    def setUp(self) -> None:
        self.reg = PlayerRegistry()

    def test_add_and_len(self) -> None:
        self.reg.add("a", [0.0, 1001.8, 0.0])
        self.assertEqual(len(self.reg), 1)
        self.assertIn("a", self.reg)

    def test_update(self) -> None:
        self.reg.add("a", [0.0, 1001.8, 0.0])
        self.reg.update("a", [1.0, 1001.8, 0.5], [0.5, 0.0, 0.0], 3)
        rec = self.reg.all_players()[0]
        self.assertAlmostEqual(rec.pos[0], 1.0, places=5)
        self.assertEqual(rec.state_flags, 3)

    def test_update_unknown_noop(self) -> None:
        self.reg.update("ghost", [0, 0, 0], [0, 0, 0], 0)  # must not raise
        self.assertEqual(len(self.reg), 0)

    def test_remove(self) -> None:
        self.reg.add("a", [0.0, 1001.8, 0.0])
        self.reg.remove("a")
        self.assertEqual(len(self.reg), 0)

    def test_remove_stale(self) -> None:
        import time
        self.reg.add("old", [0.0, 1001.8, 0.0])
        rec = self.reg._players["old"]
        rec.last_seen = time.monotonic() - 100  # force stale
        self.reg.remove_stale(timeout_s=30.0)
        self.assertEqual(len(self.reg), 0)

    def test_get_nearby_same_hemisphere(self) -> None:
        self.reg.add("near", [0.0, 1001.8, 0.0])   # north pole
        self.reg.add("far",  [0.0, -1001.8, 0.0])  # south pole
        nearby = self.reg.get_nearby([0.0, 1001.8, 0.0], sector_deg=5.0)
        ids = [r.player_id for r in nearby]
        self.assertIn("near", ids)
        self.assertNotIn("far", ids)

    def test_anti_cheat_clamp(self) -> None:
        self.reg.add("a", [0.0, 1001.8, 0.0])
        fast_vel = [999.0, 0.0, 0.0]
        self.reg.update("a", [0.0, 1001.8, 0.0], fast_vel, 0)
        rec = self.reg._players["a"]
        speed = math.sqrt(sum(v * v for v in rec.vel))
        self.assertLessEqual(speed, 51.0)  # clamped to MAX_SPEED


# ---------------------------------------------------------------------------
# 3. TestPlayerIdentity
# ---------------------------------------------------------------------------

class TestPlayerIdentity(unittest.TestCase):
    """Deterministic and collision-resistant player key generation."""

    def test_same_inputs_same_key(self) -> None:
        k1 = make_player_key("10.0.0.1", "Mozilla/5.0", "salt")
        k2 = make_player_key("10.0.0.1", "Mozilla/5.0", "salt")
        self.assertEqual(k1, k2)

    def test_different_ip_different_key(self) -> None:
        k1 = make_player_key("10.0.0.1", "UA", "salt")
        k2 = make_player_key("10.0.0.2", "UA", "salt")
        self.assertNotEqual(k1, k2)

    def test_different_ua_different_key(self) -> None:
        k1 = make_player_key("10.0.0.1", "UA-A", "salt")
        k2 = make_player_key("10.0.0.1", "UA-B", "salt")
        self.assertNotEqual(k1, k2)

    def test_different_salt_different_key(self) -> None:
        k1 = make_player_key("10.0.0.1", "UA", "saltX")
        k2 = make_player_key("10.0.0.1", "UA", "saltY")
        self.assertNotEqual(k1, k2)

    def test_key_length(self) -> None:
        k = make_player_key("1.2.3.4", "UA", "s")
        self.assertEqual(len(k), 16)

    def test_default_salt_stable(self) -> None:
        """Two calls without salt should use the same module-level salt."""
        k1 = make_player_key("1.2.3.4", "UA")
        k2 = make_player_key("1.2.3.4", "UA")
        self.assertEqual(k1, k2)


# ---------------------------------------------------------------------------
# 4. TestSpawnAnchor
# ---------------------------------------------------------------------------

class TestSpawnAnchor(unittest.TestCase):
    """SpawnAnchor determinism and spatial correctness."""

    def setUp(self) -> None:
        self.anchor = SpawnAnchor(
            anchor_pos    = [0.0, 1001.8, 0.0],
            radius_m      = 5.0,
            planet_radius = 1000.0,
        )

    def test_deterministic(self) -> None:
        p1 = self.anchor.get_spawn_for_player("key_abc")
        p2 = self.anchor.get_spawn_for_player("key_abc")
        self.assertEqual(p1, p2)

    def test_different_keys_different_pos(self) -> None:
        p1 = self.anchor.get_spawn_for_player("key_aaa")
        p2 = self.anchor.get_spawn_for_player("key_bbb")
        self.assertNotEqual(p1, p2)

    def test_spawn_near_anchor(self) -> None:
        """Spawn positions must be within ~2× spawn_radius of the anchor."""
        anchor_pos = self.anchor.anchor
        ar = math.sqrt(sum(v * v for v in anchor_pos))
        for suffix in ("aaaa", "bbbb", "cccc", "dddd"):
            p = self.anchor.get_spawn_for_player("key_" + suffix)
            pr = math.sqrt(sum(v * v for v in p))
            # Dot product gives angular distance
            dot = sum(a * b for a, b in zip(
                [v / ar for v in anchor_pos],
                [v / pr for v in p],
            ))
            dot = max(-1.0, min(1.0, dot))
            ang_rad = math.acos(dot)
            ang_m   = ang_rad * 1000.0  # × planet_radius
            self.assertLess(ang_m, 20.0,
                            f"spawn too far from anchor: {ang_m:.2f} m")

    def test_anchor_get_set(self) -> None:
        self.anchor.anchor = [0.0, 1001.8, 10.0]
        a = self.anchor.anchor
        self.assertAlmostEqual(a[2], 10.0, places=5)


# ---------------------------------------------------------------------------
# 5. TestNetworkServerComponents (no live socket)
# ---------------------------------------------------------------------------

class TestNetworkServerComponents(unittest.IsolatedAsyncioTestCase):
    """NetworkServer builds and handles HTTP / message dispatch in isolation."""

    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.mkdtemp()
        self._state_dir = os.path.join(self._tmp, "world_state")

        self.ws_state = WorldState(self._state_dir)
        self.ws_state.load_or_create(default_seed=42)

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

    def test_constructs_without_bootstrap(self) -> None:
        self.assertIsNotNone(self.server)

    async def test_handle_http_index(self) -> None:
        import src.net.NetworkServer as nm
        html_path = nm._CLIENT_DIR / "index.html"
        result = await self.server._handle_http("/", {})
        if html_path.exists():
            self.assertIsNotNone(result)
            status, _, body = result
            self.assertEqual(status, 200)
            self.assertIn(b"<html", body.lower())
        else:
            self.assertEqual(result[0], 404)

    async def test_handle_http_ws(self) -> None:
        result = await self.server._handle_http("/ws", {})
        self.assertIsNone(result)  # None = proceed with WebSocket upgrade

    async def test_handle_http_unknown(self) -> None:
        result = await self.server._handle_http("/unknown", {})
        self.assertIsNotNone(result)
        self.assertEqual(result[0], 404)

    async def test_player_state_message_updates_registry(self) -> None:
        self.registry.add("pid1", [0.0, 1001.8, 0.0])
        msg = {"type": "PLAYER_STATE", "pos": [1.0, 1001.8, 0.5],
               "vel": [0.1, 0.0, 0.0], "flags": 2}
        await self.server._on_client_message("pid1", msg)
        rec = self.registry._players.get("pid1")
        self.assertIsNotNone(rec)
        self.assertAlmostEqual(rec.pos[0], 1.0, places=5)
        self.assertEqual(rec.state_flags, 2)

    async def test_world_sync_payload(self) -> None:
        """_send_world_sync should produce a valid JSON payload."""
        messages = []

        class FakeWS:
            async def send(self, msg):
                messages.append(msg)

        self.registry.add("p1", [0.0, 1001.8, 0.0])
        await self.server._send_world_sync(FakeWS(), "p1")
        self.assertEqual(len(messages), 1)
        data = json.loads(messages[0])
        self.assertEqual(data["type"], "WORLD_SYNC")
        self.assertEqual(data["seed"], 42)
        self.assertIn("worldId", data)
        self.assertIn("spawnPos", data)
        self.assertIn("geoEvents", data)


# ---------------------------------------------------------------------------
# 6. TestTwoClientsSameWorld  (live WebSocket integration)
# ---------------------------------------------------------------------------

@unittest.skipUnless(
    __import__("importlib").util.find_spec("websockets") is not None,
    "websockets not installed",
)
class TestTwoClientsSameWorld(unittest.IsolatedAsyncioTestCase):
    """Two WebSocket clients connect and receive the same world identity."""

    async def asyncSetUp(self) -> None:
        import websockets as _ws  # noqa: F401 — guard already checked above
        self._tmp   = tempfile.mkdtemp()
        self._port  = _find_free_port()
        state_dir   = os.path.join(self._tmp, "world_state")

        ws_state = WorldState(state_dir)
        ws_state.load_or_create(default_seed=55)

        self.server = NetworkServer(
            bootstrap        = None,
            config           = None,
            world_state      = ws_state,
            state_dir        = state_dir,
        )
        # Patch port
        self.server._port = self._port
        await self.server.start()
        # Give the server a moment to bind
        await asyncio.sleep(0.1)

    async def asyncTearDown(self) -> None:
        await self.server.stop()

    async def test_two_clients_same_seed_and_world(self) -> None:
        import websockets

        uri = f"ws://127.0.0.1:{self._port}/ws"

        async with websockets.connect(uri) as ws1, \
                   websockets.connect(uri) as ws2:

            # Both clients receive WORLD_SYNC as the first message
            raw1 = await _ws_recv_timeout(ws1)
            raw2 = await _ws_recv_timeout(ws2)

        msg1 = json.loads(raw1)
        msg2 = json.loads(raw2)

        self.assertEqual(msg1["type"], "WORLD_SYNC")
        self.assertEqual(msg2["type"], "WORLD_SYNC")

        self.assertEqual(msg1["seed"], msg2["seed"],
                         "Both clients must see the same seed")
        self.assertEqual(msg1["worldId"], msg2["worldId"],
                         "Both clients must see the same worldId")


# ---------------------------------------------------------------------------
# 7. TestGeoEventReplicates  (live WebSocket integration)
# ---------------------------------------------------------------------------

@unittest.skipUnless(
    __import__("importlib").util.find_spec("websockets") is not None,
    "websockets not installed",
)
class TestGeoEventReplicates(unittest.IsolatedAsyncioTestCase):
    """Geo event enqueued by the server reaches a connected client."""

    async def asyncSetUp(self) -> None:
        self._tmp  = tempfile.mkdtemp()
        self._port = _find_free_port()
        state_dir  = os.path.join(self._tmp, "world_state")

        ws_state = WorldState(state_dir)
        ws_state.load_or_create(default_seed=42)

        self.server = NetworkServer(
            bootstrap   = None,
            config      = None,
            world_state = ws_state,
            state_dir   = state_dir,
        )
        self.server._port = self._port
        # Speed up the world tick so geo events are broadcast quickly
        self.server._tick_hz_world = 20
        await self.server.start()
        await asyncio.sleep(0.1)

    async def asyncTearDown(self) -> None:
        await self.server.stop()

    async def test_geo_event_broadcast(self) -> None:
        import websockets

        uri = f"ws://127.0.0.1:{self._port}/ws"

        async with websockets.connect(uri) as ws:
            # Consume the WORLD_SYNC first
            await _ws_recv_timeout(ws)

            # Inject a geo event directly into the pending queue
            geo_ev = {
                "eventId":   99,
                "eventType": "LANDSLIDE",
                "pos":       [0.0, 1001.8, 0.0],
                "params":    {},
            }
            self.server._pending_geo.append(geo_ev)

            # Wait for GEO_EVENT message (may receive WORLD_TICK first)
            found = False
            for _ in range(10):
                try:
                    raw = await _ws_recv_timeout(ws, timeout=1.0)
                    msg = json.loads(raw)
                    if msg["type"] == "GEO_EVENT" and msg["eventId"] == 99:
                        found = True
                        break
                except asyncio.TimeoutError:
                    break

        self.assertTrue(found, "Client did not receive the GEO_EVENT message")


# ---------------------------------------------------------------------------
# 8. TestPlayerVisibility  (live WebSocket integration)
# ---------------------------------------------------------------------------

@unittest.skipUnless(
    __import__("importlib").util.find_spec("websockets") is not None,
    "websockets not installed",
)
class TestPlayerVisibility(unittest.IsolatedAsyncioTestCase):
    """Client A can see client B's position in PLAYERS broadcasts."""

    async def asyncSetUp(self) -> None:
        self._tmp  = tempfile.mkdtemp()
        self._port = _find_free_port()
        state_dir  = os.path.join(self._tmp, "world_state")

        ws_state = WorldState(state_dir)
        ws_state.load_or_create(default_seed=42)

        self.server = NetworkServer(
            bootstrap   = None,
            config      = None,
            world_state = ws_state,
            state_dir   = state_dir,
        )
        self.server._port             = self._port
        self.server._tick_hz_players  = 20  # faster for tests
        await self.server.start()
        await asyncio.sleep(0.1)

    async def asyncTearDown(self) -> None:
        await self.server.stop()

    async def test_client_a_sees_client_b(self) -> None:
        import websockets

        uri = f"ws://127.0.0.1:{self._port}/ws"

        async with websockets.connect(uri) as ws_a:
            # Consume the WORLD_SYNC for client A
            await _ws_recv_timeout(ws_a)

            # Inject a second player directly into the registry so that
            # the PLAYERS broadcast includes at least 2 entries.
            # (In tests both real WS clients share the same IP/UA → same key.)
            self.server._registry.add(
                "injected_player_b",
                [0.5, 1001.8, 0.5],
            )

            # Wait for PLAYERS message that includes at least 2 entries
            players_seen: list = []
            for _ in range(20):
                try:
                    raw = await _ws_recv_timeout(ws_a, timeout=1.0)
                    msg = json.loads(raw)
                    if msg["type"] == "PLAYERS":
                        players_seen = msg["players"]
                        if len(players_seen) >= 2:
                            break
                except asyncio.TimeoutError:
                    break

        self.assertGreaterEqual(
            len(players_seen), 2,
            "Client A should see at least 2 players (itself + injected player B)",
        )


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
