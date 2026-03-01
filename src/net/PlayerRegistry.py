"""PlayerRegistry — in-memory player state registry.

Tracks each connected player's world-space position, velocity, and
packed state flags.  Single-threaded for asyncio; no locking required.

Interest management
-------------------
:meth:`get_nearby` returns only players whose normalised surface position
is within *sector_deg* angular degrees of the query position, keeping
per-client bandwidth proportional to local density rather than total
player count.

Anti-cheat
----------
:meth:`update` silently clamps the velocity vector when its magnitude
exceeds ``MAX_SPEED_UNITS_PER_S`` so a rogue client cannot teleport.

Public API
----------
PlayerRegistry()
  .add(player_id, pos)
  .update(player_id, pos, vel, state_flags)
  .remove(player_id)
  .remove_stale(timeout_s)
  .all_players() → list[PlayerRecord]
  .get_nearby(pos, sector_deg) → list[PlayerRecord]
  len(registry) / player_id in registry
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Maximum plausible speed in simulation units per second.
# Derived from char.max_speed (6.0) with a generous anti-cheat headroom.
MAX_SPEED_UNITS_PER_S: float = 50.0


@dataclass
class PlayerRecord:
    """Live state for one connected player."""
    player_id:   str
    pos:         List[float]          # [x, y, z] world-space
    vel:         List[float]          # [vx, vy, vz]
    state_flags: int                  # packed bit-flags
    last_seen:   float = field(default_factory=time.monotonic)


class PlayerRegistry:
    """In-memory registry of all currently connected players."""

    def __init__(self) -> None:
        self._players: Dict[str, PlayerRecord] = {}

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add(self, player_id: str, pos: List[float]) -> None:
        """Register *player_id* at *pos* with zero velocity."""
        self._players[player_id] = PlayerRecord(
            player_id   = player_id,
            pos         = list(pos),
            vel         = [0.0, 0.0, 0.0],
            state_flags = 0,
        )

    def update(
        self,
        player_id:   str,
        pos:         List[float],
        vel:         List[float],
        state_flags: int,
    ) -> None:
        """Update state for *player_id* (clamped for anti-cheat).

        If *player_id* is not registered this call is a no-op.
        """
        rec = self._players.get(player_id)
        if rec is None:
            return

        # Anti-cheat: clamp velocity magnitude
        vx, vy, vz = float(vel[0]), float(vel[1]), float(vel[2])
        speed = math.sqrt(vx * vx + vy * vy + vz * vz)
        if speed > MAX_SPEED_UNITS_PER_S:
            scale = MAX_SPEED_UNITS_PER_S / speed
            vx, vy, vz = vx * scale, vy * scale, vz * scale

        rec.pos         = [float(pos[0]), float(pos[1]), float(pos[2])]
        rec.vel         = [vx, vy, vz]
        rec.state_flags = int(state_flags)
        rec.last_seen   = time.monotonic()

    def remove(self, player_id: str) -> None:
        """Unregister *player_id* (no-op if unknown)."""
        self._players.pop(player_id, None)

    def remove_stale(self, timeout_s: float = 30.0) -> None:
        """Remove players whose last update is older than *timeout_s*."""
        now   = time.monotonic()
        stale = [
            pid for pid, rec in self._players.items()
            if now - rec.last_seen > timeout_s
        ]
        for pid in stale:
            del self._players[pid]

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_player_pos(self, player_id: str) -> Optional[List[float]]:
        """Return position of *player_id*, or *None* if unknown."""
        rec = self._players.get(player_id)
        return list(rec.pos) if rec is not None else None

    def all_players(self) -> List[PlayerRecord]:
        """Snapshot of all registered player records."""
        return list(self._players.values())

    def get_nearby(
        self,
        pos: List[float],
        sector_deg: float = 5.0,
    ) -> List[PlayerRecord]:
        """Players within *sector_deg* angular degrees of *pos*.

        Falls back to all players when *pos* is the zero vector.
        """
        if len(pos) < 3:
            return self.all_players()

        qx, qy, qz = float(pos[0]), float(pos[1]), float(pos[2])
        ql = math.sqrt(qx * qx + qy * qy + qz * qz)
        if ql < 1e-9:
            return self.all_players()

        qx, qy, qz = qx / ql, qy / ql, qz / ql
        cos_thresh = math.cos(math.radians(sector_deg))
        result: List[PlayerRecord] = []

        for rec in self._players.values():
            px, py, pz = rec.pos[0], rec.pos[1], rec.pos[2]
            pl = math.sqrt(px * px + py * py + pz * pz)
            if pl < 1e-9:
                result.append(rec)
                continue
            dot = (px * qx + py * qy + pz * qz) / pl
            if dot >= cos_thresh:
                result.append(rec)

        return result

    # ------------------------------------------------------------------
    # Container protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._players)

    def __contains__(self, player_id: object) -> bool:
        return player_id in self._players
