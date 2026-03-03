"""RegionShardServer — Stage 60 per-region authoritative simulation shard.

Manages the authoritative simulation for one region of the planet:

* Tracks which players are currently in this region.
* Exposes tick / update hooks (the actual physics integration is handled by
  the existing simulation stack; this class handles the shard bookkeeping).
* Detects when a player is approaching the handoff band and signals the
  HandoffManager.
* Applies hotspot degradation when player count exceeds the soft cap.

The class is intentionally *not* an asyncio server; it is a pure-Python
state machine so that it can be unit-tested synchronously.  The actual
network transport layer calls ``tick()`` on each simulation step.

Public API
----------
RegionShardServer(region_id, ga_service, indexing, config=None)
  .add_player(player_id, pos, vel)
  .remove_player(player_id)
  .update_player(player_id, pos, vel)
  .tick(dt_s) → list[HandoffRequest]
  .player_ids → list[str]
  .player_count → int
  .is_degraded → bool
  .accept_cross_region_events(events) → None
  .region_snapshot() → dict
  .load_region_snapshot(snap: dict) → None
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class _PlayerEntry:
    player_id: str
    pos: List[float]           # [x, y, z] or [lat, lon, alt]
    vel: List[float]
    last_seen: float = field(default_factory=time.monotonic)


@dataclass
class HandoffRequest:
    """Raised when a player approaches a region boundary."""
    player_id:       str
    source_region:   int
    target_region:   int
    player_state:    Dict[str, Any]


# ---------------------------------------------------------------------------
# RegionShardServer
# ---------------------------------------------------------------------------

class RegionShardServer:
    """Authoritative simulation shard for one planet region."""

    def __init__(
        self,
        region_id:  int,
        ga_service,                    # GlobalAuthorityService
        indexing,                      # RegionIndexing
        config=None,
    ) -> None:
        self._region_id   = region_id
        self._ga          = ga_service
        self._indexing    = indexing
        self._cfg         = config

        # Config values
        self._handoff_band_m:  float = self._cfg_float(
            ("scale", "handoff_band_m"), 2000.0
        )
        self._soft_cap:  int = self._cfg_int(
            ("scale", "max_players_per_region_soft"), 100
        )
        self._hard_cap:  int = self._cfg_int(
            ("scale", "max_players_per_region_hard"), 200
        )

        # State
        self._players:   Dict[str, _PlayerEntry] = {}
        self._is_degraded: bool = False
        self._pending_far_emitters: List[Dict[str, Any]] = []

        # Register with GA
        self._ga.register_region(region_id, f"rs:{region_id}")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def region_id(self) -> int:
        return self._region_id

    @property
    def player_ids(self) -> List[str]:
        return list(self._players.keys())

    @property
    def player_count(self) -> int:
        return len(self._players)

    @property
    def is_degraded(self) -> bool:
        return self._is_degraded

    # ------------------------------------------------------------------
    # Player management
    # ------------------------------------------------------------------

    def add_player(
        self,
        player_id: str,
        pos: List[float],
        vel: Optional[List[float]] = None,
    ) -> bool:
        """Add *player_id* to this shard.

        Returns ``False`` (and does not add) if the hard cap is reached.
        """
        if len(self._players) >= self._hard_cap:
            return False
        self._players[player_id] = _PlayerEntry(
            player_id=player_id,
            pos=list(pos),
            vel=list(vel) if vel else [0.0, 0.0, 0.0],
        )
        return True

    def remove_player(self, player_id: str) -> None:
        """Remove *player_id* from this shard (no-op if unknown)."""
        self._players.pop(player_id, None)

    def update_player(
        self,
        player_id: str,
        pos: List[float],
        vel: List[float],
    ) -> None:
        """Update position/velocity of *player_id*."""
        entry = self._players.get(player_id)
        if entry is not None:
            entry.pos = list(pos)
            entry.vel = list(vel)
            entry.last_seen = time.monotonic()

    # ------------------------------------------------------------------
    # Tick
    # ------------------------------------------------------------------

    def tick(self, dt_s: float) -> List[HandoffRequest]:
        """Advance shard by *dt_s* seconds.

        Returns a list of :class:`HandoffRequest` objects for players that
        have crossed into the handoff band and need migrating.
        """
        # Drain cross-region events from GA
        events = self._ga.drain_events_for_region(self._region_id)
        for ev in events:
            self._pending_far_emitters.append(ev)

        # Update degradation status
        self._is_degraded = len(self._players) > self._soft_cap

        # Heartbeat to GA
        self._ga.heartbeat(self._region_id)

        # Check for handoffs
        handoffs: List[HandoffRequest] = []
        bounds = self._indexing.region_bounds(self._region_id)
        # bounds = (lat_min, lon_min, lat_max, lon_max)
        lat_min, lon_min, lat_max, lon_max = bounds

        for pid, entry in list(self._players.items()):
            target = self._near_boundary_target(entry.pos, bounds)
            if target is not None:
                ps = self._serialise_player(entry)
                handoffs.append(HandoffRequest(
                    player_id=pid,
                    source_region=self._region_id,
                    target_region=target,
                    player_state=ps,
                ))

        return handoffs

    # ------------------------------------------------------------------
    # Cross-region events
    # ------------------------------------------------------------------

    def accept_cross_region_events(
        self, events: List[Dict[str, Any]]
    ) -> None:
        """Inject pre-routed cross-region events (bypassing GA queue)."""
        self._pending_far_emitters.extend(events)

    def drain_far_emitters(self) -> List[Dict[str, Any]]:
        """Consume pending far-emitter events."""
        out = list(self._pending_far_emitters)
        self._pending_far_emitters.clear()
        return out

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def region_snapshot(self) -> Dict[str, Any]:
        """Return a JSON-serialisable region state snapshot."""
        return {
            "region_id":  self._region_id,
            "players": {
                pid: {
                    "pos": list(e.pos),
                    "vel": list(e.vel),
                }
                for pid, e in self._players.items()
            },
            "is_degraded": self._is_degraded,
        }

    def load_region_snapshot(self, snap: Dict[str, Any]) -> None:
        """Restore region state from a snapshot dict."""
        self._players.clear()
        for pid, pdata in snap.get("players", {}).items():
            self._players[pid] = _PlayerEntry(
                player_id=pid,
                pos=list(pdata.get("pos", [0.0, 0.0, 0.0])),
                vel=list(pdata.get("vel", [0.0, 0.0, 0.0])),
            )
        self._is_degraded = bool(snap.get("is_degraded", False))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _near_boundary_target(
        self,
        pos: List[float],
        bounds: Tuple[float, float, float, float],
    ) -> Optional[int]:
        """Return target region ID if *pos* is within the handoff band."""
        if len(pos) < 2:
            return None

        lat, lon = float(pos[0]), float(pos[1])
        lat_min, lon_min, lat_max, lon_max = bounds

        # Convert handoff band from metres to approximate degrees
        band_deg = self._handoff_band_m / 111_320.0   # ≈ 1 deg lat = 111 320 m

        # Check each boundary direction
        if lat - lat_min < band_deg:
            # Near southern boundary
            target_lat = lat - band_deg * 2
            return self._indexing.region_id(target_lat, lon)
        if lat_max - lat < band_deg:
            # Near northern boundary
            target_lat = lat + band_deg * 2
            return self._indexing.region_id(target_lat, lon)
        if lon - lon_min < band_deg:
            # Near western boundary
            target_lon = lon - band_deg * 2
            return self._indexing.region_id(lat, target_lon)
        if lon_max - lon < band_deg:
            # Near eastern boundary
            target_lon = lon + band_deg * 2
            return self._indexing.region_id(lat, target_lon)

        return None

    @staticmethod
    def _serialise_player(entry: _PlayerEntry) -> Dict[str, Any]:
        return {
            "player_id": entry.player_id,
            "pos":       list(entry.pos),
            "vel":       list(entry.vel),
        }

    def _cfg_float(self, path: tuple, default: float) -> float:
        if self._cfg is None:
            return default
        return float(self._cfg.get(*path, default=default))

    def _cfg_int(self, path: tuple, default: int) -> int:
        if self._cfg is None:
            return default
        return int(self._cfg.get(*path, default=default))
