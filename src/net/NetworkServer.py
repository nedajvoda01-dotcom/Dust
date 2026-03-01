"""NetworkServer — Stage 22 authoritative WebSocket + HTTP server.

Serves the browser client (``client/index.html``) over HTTP and
maintains a WebSocket endpoint for real-time world + player
synchronisation.

Architecture
------------
* One asyncio event-loop drives everything.
* GameBootstrap (headless) ticks in a background asyncio task.
* New WebSocket clients receive a full ``WORLD_SYNC`` message.
* ``WORLD_TICK`` is broadcast at ``net.world_tick_hz`` Hz (default 5 Hz).
* ``PLAYERS`` is sent per-client at ``net.player_broadcast_hz`` Hz with
  interest-management filtering (only nearby players included).
* ``CLIMATE_SNAP`` is sent every ``net.snapshot_interval_sec`` seconds.
* ``GEO_EVENT`` is sent whenever the simulation fires a new event.
* ``PONG`` is returned to any ``PING`` from a client.
* ``REJOIN_RESYNC`` from a client triggers a delta or full re-sync.

Message types (JSON)
--------------------
Server → Client
  WORLD_SYNC   seed, worldId, simTime, timeScale, spawnPos, geoEvents, anchor
  WORLD_TICK   simTime, timeScale, epoch
  PLAYERS      players: [{id, pos, vel, flags}]
  GEO_EVENT    eventId, eventType, pos, params
  CLIMATE_SNAP storms: [{lat, lon, radius, intensity}], globalDust
  PONG         t (echoes client's PING timestamp)
  REJOIN_RESYNC simTime, epoch, catchupEvents (or full WORLD_SYNC if stale)

Client → Server
  JOIN         userAgent  (optional on reconnect)
  PLAYER_STATE pos, vel, flags
  PING         t
  REJOIN_RESYNC lastSeenTick, lastEventId, lastPatchId

Public API
----------
NetworkServer(bootstrap, config, ...)
  await .start()          — start WS server + background tasks
  await .stop()           — graceful shutdown
  await .serve_forever()  — start + block until KeyboardInterrupt
"""
from __future__ import annotations

import asyncio
import collections
import json
import math
import os
import secrets
import time
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

from src.core.Config import Config
from src.core.Logger import Logger
from src.net.PlayerIdentity import make_player_key
from src.net.PlayerRegistry import PlayerRegistry
from src.net.SpawnAnchor import SpawnAnchor
from src.net.WorldState import WorldState

_TAG        = "NetworkServer"
_CLIENT_DIR = Path(__file__).parent.parent.parent / "client"

# Config fallbacks
_DEFAULT_PORT             = 8765
_DEFAULT_TICK_HZ_WORLD    = 5
_DEFAULT_TICK_HZ_PLAYERS  = 20
_DEFAULT_SNAP_INTERVAL    = 30.0
_DEFAULT_SPAWN_RADIUS     = 5.0
_DEFAULT_SECTOR_DEG       = 5.0
_DEFAULT_SECTOR_RADIUS    = 2          # neighbouring sectors to include
_DEFAULT_PLAYER_SEND_HZ   = 20        # max client→server msgs/sec
_DEFAULT_INTERP_DELAY_MS  = 200
_DEFAULT_MAX_EXTRAP_MS    = 150
_DEFAULT_RESYNC_THRESHOLD = 5.0       # seconds of drift → hard resync
_PLAYER_TIMEOUT_S         = 30.0
_REJOIN_MAX_CATCHUP_EVENTS = 100      # if more, send full WORLD_SYNC


class NetworkServer:
    """Authoritative WebSocket server for a shared Dust world.

    Parameters
    ----------
    bootstrap:
        Headless :class:`GameBootstrap` instance.  May be *None* for unit
        tests (world state / player registry still work).
    config:
        Loaded game config.
    world_state:
        Pre-constructed :class:`WorldState`.  Created automatically when
        *None*.
    player_registry:
        Pre-constructed :class:`PlayerRegistry`.  Created automatically.
    spawn_anchor:
        Pre-constructed :class:`SpawnAnchor`.  Created automatically.
    state_dir:
        Directory for persistent world state (default ``"world_state"``).
    """

    def __init__(
        self,
        bootstrap                     = None,
        config:          Optional[Config]          = None,
        world_state:     Optional[WorldState]      = None,
        player_registry: Optional[PlayerRegistry]  = None,
        spawn_anchor:    Optional[SpawnAnchor]      = None,
        state_dir:       str = "world_state",
    ) -> None:
        self._bootstrap = bootstrap
        self._config    = config

        # Net parameters from config
        self._port              = _DEFAULT_PORT
        self._tick_hz_world     = _DEFAULT_TICK_HZ_WORLD
        self._tick_hz_players   = _DEFAULT_TICK_HZ_PLAYERS
        self._snap_interval     = _DEFAULT_SNAP_INTERVAL
        self._spawn_radius      = _DEFAULT_SPAWN_RADIUS
        self._sector_deg        = _DEFAULT_SECTOR_DEG
        self._sector_radius     = _DEFAULT_SECTOR_RADIUS
        self._player_send_hz    = _DEFAULT_PLAYER_SEND_HZ
        self._interp_delay_ms   = _DEFAULT_INTERP_DELAY_MS
        self._max_extrap_ms     = _DEFAULT_MAX_EXTRAP_MS
        self._resync_threshold  = _DEFAULT_RESYNC_THRESHOLD

        if config is not None:
            self._port            = int(config.get(
                "net", "ws_port",              default=_DEFAULT_PORT))
            self._tick_hz_world   = int(config.get(
                "net", "world_tick_hz",        default=_DEFAULT_TICK_HZ_WORLD))
            self._tick_hz_players = int(config.get(
                "net", "player_broadcast_hz",  default=_DEFAULT_TICK_HZ_PLAYERS))
            self._snap_interval   = float(config.get(
                "net", "snapshot_interval_sec", default=_DEFAULT_SNAP_INTERVAL))
            self._spawn_radius    = float(config.get(
                "net", "spawn_radius_m",        default=_DEFAULT_SPAWN_RADIUS))
            self._sector_deg      = float(config.get(
                "net", "sector_deg",            default=_DEFAULT_SECTOR_DEG))
            self._sector_radius   = int(config.get(
                "net", "sector_radius",         default=_DEFAULT_SECTOR_RADIUS))
            self._player_send_hz  = int(config.get(
                "net", "player_send_hz",        default=_DEFAULT_PLAYER_SEND_HZ))
            self._interp_delay_ms = float(config.get(
                "net", "interp_delay_ms",       default=_DEFAULT_INTERP_DELAY_MS))
            self._max_extrap_ms   = float(config.get(
                "net", "max_extrap_ms",         default=_DEFAULT_MAX_EXTRAP_MS))
            self._resync_threshold = float(config.get(
                "net", "resync_hard_threshold_sec", default=_DEFAULT_RESYNC_THRESHOLD))

        # Planet radius (for spawn anchor)
        planet_r = 1000.0
        if bootstrap is not None:
            planet_r = float(getattr(bootstrap, "planet_radius", 1000.0))

        # World state
        self._world_state = world_state or WorldState(state_dir)
        if not self._world_state.world_id:
            seed = int(getattr(bootstrap, "seed", 42)) if bootstrap else 42
            self._world_state.load_or_create(default_seed=seed)

        # Player registry
        self._registry: PlayerRegistry = (
            player_registry if player_registry is not None else PlayerRegistry()
        )

        # Spawn anchor
        if spawn_anchor is not None:
            self._anchor = spawn_anchor
        else:
            anchor_pos: Optional[List[float]] = None
            if bootstrap is not None:
                char = getattr(bootstrap, "character", None)
                if char is not None:
                    p = char.position
                    anchor_pos = [float(p.x), float(p.y), float(p.z)]
            self._anchor = SpawnAnchor(
                anchor_pos    = anchor_pos,
                radius_m      = self._spawn_radius,
                planet_radius = planet_r,
            )

        # Server salt (regenerated each run — no persistent identity)
        self._server_salt: str = secrets.token_hex(16)

        # WebSocket connection tables  (player_id → ws, ws → player_id)
        self._connections:   Dict[str, Any] = {}
        self._ws_to_player:  Dict[Any, str] = {}

        # Pending geo events to broadcast (filled by sim loop)
        self._pending_geo:   List[Dict[str, Any]] = []
        self._seen_event_count: int = 0

        # Rate limiting: player_id → deque of recent message timestamps
        # Uses collections.deque for O(1) popleft on window eviction.
        self._rate_windows:  Dict[str, Deque[float]] = {}

        # Background task handles
        self._ws_server = None
        self._tasks:    List[asyncio.Task] = []

        self._running    = False
        self._last_snap  = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the WebSocket server and all background tasks."""
        try:
            import websockets  # type: ignore[import]
        except ImportError:
            Logger.warn(_TAG, "websockets package not installed — server disabled")
            return

        self._running = True

        # Sync world state from bootstrap
        if self._bootstrap is not None:
            self._sync_from_bootstrap()

        Logger.info(_TAG, f"Starting on ws://0.0.0.0:{self._port}/ws")

        self._ws_server = await websockets.serve(
            self._handle_connection,
            "0.0.0.0",
            self._port,
            process_request=self._handle_http,
        )

        self._tasks = [
            asyncio.create_task(self._world_tick_loop(),   name="world_tick"),
            asyncio.create_task(self._player_bcast_loop(), name="player_bcast"),
            asyncio.create_task(self._sim_loop(),          name="sim"),
            asyncio.create_task(self._stale_loop(),        name="stale"),
        ]

        Logger.info(
            _TAG,
            f"Ready — open http://localhost:{self._port}/ in a browser",
        )

    async def stop(self) -> None:
        """Cancel background tasks and close the WebSocket server."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        if self._ws_server is not None:
            self._ws_server.close()
            await self._ws_server.wait_closed()

    async def serve_forever(self) -> None:
        """Start the server and block until a :exc:`KeyboardInterrupt`."""
        await self.start()
        try:
            await asyncio.Future()  # block indefinitely
        except (asyncio.CancelledError, KeyboardInterrupt):
            pass
        finally:
            await self.stop()

    # ------------------------------------------------------------------
    # HTTP handler — serve client/index.html
    # ------------------------------------------------------------------

    async def _handle_http(self, path: str, request_headers):  # noqa: ANN001
        """Serve the browser client for non-WebSocket requests.

        Returns *None* to let the WebSocket upgrade proceed for ``/ws``.
        """
        if path in ("/", "/index.html"):
            html_path = _CLIENT_DIR / "index.html"
            if html_path.exists():
                body = html_path.read_bytes()
                return (
                    200,
                    [
                        ("Content-Type",   "text/html; charset=utf-8"),
                        ("Content-Length", str(len(body))),
                    ],
                    body,
                )
            return (404, [], b"client/index.html not found")

        # /ws  → proceed with WebSocket upgrade (return None)
        if path.startswith("/ws"):
            return None

        return (404, [], b"Not found")

    # ------------------------------------------------------------------
    # WebSocket connection handler
    # ------------------------------------------------------------------

    async def _handle_connection(self, ws, path: str) -> None:  # noqa: ANN001
        """Lifecycle handler for one WebSocket client."""
        remote_ip = "unknown"
        try:
            addr = ws.remote_address
            if addr:
                remote_ip = addr[0]
        except Exception:
            pass

        user_agent = ""
        try:
            user_agent = ws.request_headers.get("User-Agent", "")
        except Exception:
            pass

        player_id  = make_player_key(remote_ip, user_agent, self._server_salt)
        Logger.info(_TAG, f"Connect: id={player_id} ip={remote_ip}")

        spawn_pos = self._anchor.get_spawn_for_player(player_id)
        self._registry.add(player_id, spawn_pos)
        self._connections[player_id] = ws
        self._ws_to_player[id(ws)]   = player_id

        try:
            await self._send_world_sync(ws, player_id)

            async for raw in ws:
                try:
                    msg = json.loads(raw)
                    await self._on_client_message(player_id, ws, msg)
                except Exception as exc:
                    # Log the exception type only — do not echo client-supplied
                    # data into the log to avoid leaking implementation details.
                    Logger.warn(_TAG, f"Bad message from {player_id}: {type(exc).__name__}")
        except Exception:
            pass
        finally:
            Logger.info(_TAG, f"Disconnect: id={player_id}")
            self._registry.remove(player_id)
            self._connections.pop(player_id, None)
            self._ws_to_player.pop(id(ws), None)
            self._rate_windows.pop(player_id, None)

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    def _check_rate_limit(self, player_id: str) -> bool:
        """Return *True* if this message is within the allowed rate.

        Implements a sliding-window counter over a 1-second window using a
        :class:`collections.deque` for O(1) popleft.
        """
        now = time.monotonic()
        window: Deque[float] = self._rate_windows.setdefault(
            player_id, collections.deque()
        )

        # Evict timestamps older than 1 second
        cutoff = now - 1.0
        while window and window[0] < cutoff:
            window.popleft()

        if len(window) >= self._player_send_hz:
            return False  # rate-limited

        window.append(now)
        return True

    # ------------------------------------------------------------------
    # Client message dispatch
    # ------------------------------------------------------------------

    async def _on_client_message(
        self,
        player_id: str,
        ws_or_msg,
        msg: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Dispatch a client message.

        Accepts two call signatures for backwards-compatibility with tests
        written before the ``ws`` parameter was added:

        * ``_on_client_message(player_id, ws, msg)``  — new
        * ``_on_client_message(player_id, msg)``       — legacy (ws=None)
        """
        # Resolve the two call signatures
        if msg is None:
            # Legacy 2-arg call: second positional is msg, ws is unknown
            msg = ws_or_msg  # type: ignore[assignment]
            ws  = self._connections.get(player_id)
        else:
            ws = ws_or_msg

        msg_type = msg.get("type", "")

        # PING and REJOIN_RESYNC are control messages — exempt from rate limiting
        # so that reconnects and RTT measurements are never silently dropped.
        if msg_type not in ("PING", "REJOIN_RESYNC"):
            if not self._check_rate_limit(player_id):
                return

        if msg_type == "PLAYER_STATE":
            pos   = msg.get("pos",   [0.0, 0.0, 0.0])
            vel   = msg.get("vel",   [0.0, 0.0, 0.0])
            flags = int(msg.get("flags", 0))
            if (
                isinstance(pos, list) and len(pos) == 3
                and isinstance(vel, list) and len(vel) == 3
            ):
                self._registry.update(player_id, pos, vel, flags)

        elif msg_type == "PING":
            # Reflect timestamp back so the client can measure RTT
            try:
                await ws.send(json.dumps({
                    "type": "PONG",
                    "t":    msg.get("t"),
                }))
            except Exception:
                pass

        elif msg_type == "REJOIN_RESYNC":
            last_event_id = int(msg.get("lastEventId", -1))
            await self._send_rejoin(ws, player_id, last_event_id)

        # JOIN: just a reconnect hint — no action needed

    # ------------------------------------------------------------------
    # World sync (initial message to new connections)
    # ------------------------------------------------------------------

    async def _send_world_sync(self, ws, player_id: str) -> None:
        spawn_pos = self._anchor.get_spawn_for_player(player_id)
        msg = json.dumps({
            "type":      "WORLD_SYNC",
            "seed":      self._world_state.seed,
            "worldId":   self._world_state.world_id,
            "simTime":   self._world_state.sim_time,
            "timeScale": self._world_state.time_scale,
            "spawnPos":  spawn_pos,
            "anchor":    self._anchor.anchor,
            "geoEvents": self._world_state.geo_events()[-100:],
        })
        await ws.send(msg)

    # ------------------------------------------------------------------
    # REJOIN_RESYNC — delta or full re-sync
    # ------------------------------------------------------------------

    async def _send_rejoin(self, ws, player_id: str, last_event_id: int) -> None:
        """Send catch-up events or a full WORLD_SYNC if the client is too stale."""
        all_events = self._world_state.geo_events()
        new_events = [
            e for e in all_events
            if int(e.get("eventId", -1)) > last_event_id
        ]

        if len(new_events) > _REJOIN_MAX_CATCHUP_EVENTS:
            # Too stale: send a full WORLD_SYNC instead
            await self._send_world_sync(ws, player_id)
        else:
            try:
                await ws.send(json.dumps({
                    "type":          "REJOIN_RESYNC",
                    "simTime":       self._world_state.sim_time,
                    "epoch":         self._world_state.epoch,
                    "catchupEvents": new_events,
                }))
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Broadcast helpers
    # ------------------------------------------------------------------

    async def _broadcast(self, msg: str) -> None:
        """Send *msg* to all connected clients, removing dead connections."""
        dead: List[str] = []
        for pid, ws in list(self._connections.items()):
            try:
                await ws.send(msg)
            except Exception:
                dead.append(pid)
        for pid in dead:
            self._connections.pop(pid, None)

    # ------------------------------------------------------------------
    # Background loops
    # ------------------------------------------------------------------

    async def _world_tick_loop(self) -> None:
        """Broadcast WORLD_TICK at *tick_hz_world* Hz."""
        interval = 1.0 / max(1, self._tick_hz_world)
        while self._running:
            await asyncio.sleep(interval)
            self._world_state.epoch += 1
            tick_msg = json.dumps({
                "type":      "WORLD_TICK",
                "simTime":   self._world_state.sim_time,
                "timeScale": self._world_state.time_scale,
                "epoch":     self._world_state.epoch,
            })
            await self._broadcast(tick_msg)

            # Broadcast pending geo events
            while self._pending_geo:
                ev = self._pending_geo.pop(0)
                await self._broadcast(json.dumps({"type": "GEO_EVENT", **ev}))

            # Periodic climate snapshot
            now = time.monotonic()
            if now - self._last_snap >= self._snap_interval:
                self._last_snap = now
                await self._broadcast_climate_snap()

    async def _player_bcast_loop(self) -> None:
        """Send PLAYERS to each client filtered by interest (sector proximity).

        Each client receives only the players that are within
        ``(sector_radius + 1) × sector_deg`` angular degrees of their own
        position, keeping per-client bandwidth proportional to local density.
        """
        interval = 1.0 / max(1, self._tick_hz_players)
        # Interest cone: sector_deg × (sector_radius + 0.5)
        view_deg = self._sector_deg * (self._sector_radius + 0.5)

        while self._running:
            await asyncio.sleep(interval)

            dead: List[str] = []
            for pid, ws in list(self._connections.items()):
                try:
                    # Find this client's current position for filtering
                    own_pos = self._registry.get_player_pos(pid)
                    if own_pos is not None:
                        nearby = self._registry.get_nearby(own_pos, view_deg)
                    else:
                        nearby = self._registry.all_players()

                    players = [
                        {
                            "id":    r.player_id,
                            "pos":   r.pos,
                            "vel":   r.vel,
                            "flags": r.state_flags,
                        }
                        for r in nearby
                    ]
                    await ws.send(
                        json.dumps({"type": "PLAYERS", "players": players})
                    )
                except Exception:
                    dead.append(pid)

            for pid in dead:
                self._connections.pop(pid, None)

    async def _sim_loop(self) -> None:
        """Advance the GameBootstrap simulation and sync world state."""
        if self._bootstrap is None:
            return
        last = time.monotonic()
        while self._running:
            await asyncio.sleep(0.016)  # ~60 Hz
            now = time.monotonic()
            dt  = min(now - last, 0.1)
            last = now
            self._bootstrap.tick(dt)
            self._sync_from_bootstrap()
            self._collect_geo_events()

    async def _stale_loop(self) -> None:
        """Remove stale players every 10 s."""
        while self._running:
            await asyncio.sleep(10.0)
            self._registry.remove_stale(timeout_s=_PLAYER_TIMEOUT_S)

    # ------------------------------------------------------------------
    # Climate snapshot
    # ------------------------------------------------------------------

    async def _broadcast_climate_snap(self) -> None:
        if self._bootstrap is None:
            return
        climate = getattr(self._bootstrap, "climate", None)
        if climate is None:
            return

        storms: List[Dict[str, Any]] = []
        storm_cells = getattr(climate, "_storm_cells", None)
        if storm_cells:
            for sc in storm_cells:
                storms.append({
                    "lat":       float(getattr(sc, "center_lat", 0.0)),
                    "lon":       float(getattr(sc, "center_lon", 0.0)),
                    "radius":    float(getattr(sc, "radius",     0.1)),
                    "intensity": float(getattr(sc, "intensity",  0.0)),
                })

        dust_field = getattr(climate, "_dust", None)
        global_dust = (
            sum(dust_field) / len(dust_field)
            if dust_field else 0.0
        )

        snap = {"storms": storms, "globalDust": float(global_dust)}
        self._world_state.save_climate_snapshot(snap)
        await self._broadcast(json.dumps({"type": "CLIMATE_SNAP", **snap}))

    # ------------------------------------------------------------------
    # Geo event collection
    # ------------------------------------------------------------------

    def _collect_geo_events(self) -> None:
        """Detect new geo events from bootstrap and queue for broadcast."""
        if self._bootstrap is None:
            return
        geo_sys = getattr(self._bootstrap, "geo_events", None)
        if geo_sys is None:
            return
        log = getattr(geo_sys, "event_log", None)
        if log is None:
            return

        all_records = log.records() if hasattr(log, "records") else []
        planet_r = float(getattr(self._bootstrap, "planet_radius", 1000.0))

        for record in all_records[self._seen_event_count:]:
            ev_type = (
                record.event_type.name
                if hasattr(record.event_type, "name")
                else str(record.event_type)
            )
            d = record.direction
            ev: Dict[str, Any] = {
                "eventId":   int(record.event_id),
                "eventType": ev_type,
                "pos":       [
                    float(d.x) * planet_r,
                    float(d.y) * planet_r,
                    float(d.z) * planet_r,
                ],
                "params":    dict(getattr(record, "params", {})),
            }
            self._pending_geo.append(ev)
            self._world_state.append_geo_event(ev)

        self._seen_event_count = len(all_records)

    # ------------------------------------------------------------------
    # Bootstrap sync
    # ------------------------------------------------------------------

    def _sync_from_bootstrap(self) -> None:
        """Copy live bootstrap state into the WorldState."""
        if self._bootstrap is None:
            return
        self._world_state.seed = int(getattr(self._bootstrap, "seed", 42))

        identity = getattr(self._bootstrap, "identity", None)
        if identity is not None:
            self._world_state.world_id = str(
                getattr(identity, "world_id", self._world_state.world_id)
            )

        clock = getattr(self._bootstrap, "clock", None)
        if clock is not None:
            self._world_state.sim_time   = float(getattr(clock, "sim_time",   0.0))
            self._world_state.time_scale = float(getattr(clock, "time_scale", 1.0))
