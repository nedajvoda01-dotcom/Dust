"""StateBroadcaster — Stage 58 server-side authoritative state broadcaster.

Manages per-client send timers and emits ``AUTH_STATE`` (own player) and
``REMOTE_STATE`` (other players) messages at the LOD-appropriate rate via
the provided :class:`~src.net.LODStatePacker.LODStatePacker`.

Public API
----------
PlayerStateSnapshot
    Plain dataclass representing the authoritative state of one player.

StateBroadcaster(config)
    .update_player(player_id, snapshot)
        Update the authoritative state for a player.
    .tick(now_s) → dict[str, list[dict]]
        Advance time; return a mapping of ``recipient_player_id →
        [messages]`` containing only the messages that are due to be
        sent this tick.
    .remove(player_id)
        Discard a player's state on disconnect.
    .player_ids() → list[str]
        Currently registered players.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.net.LODStatePacker import LODStatePacker


# ---------------------------------------------------------------------------
# PlayerStateSnapshot
# ---------------------------------------------------------------------------

@dataclass
class PlayerStateSnapshot:
    """Authoritative server state for one player."""
    player_id:     str
    pos:           Tuple[float, float, float] = (0.0, 0.0, 0.0)
    vel:           Tuple[float, float, float] = (0.0, 0.0, 0.0)
    yaw:           float = 0.0
    contact_flags: int   = 0
    server_tick:   int   = 0
    last_input_seq: int  = -1
    pose_hash:     int   = 0
    timestamp_s:   float = 0.0


# ---------------------------------------------------------------------------
# StateBroadcaster
# ---------------------------------------------------------------------------

class StateBroadcaster:
    """Tick-driven server state broadcaster.

    Emits state updates at the correct LOD-aware rate for each
    (sender, recipient) pair.

    Parameters
    ----------
    config : dict
        Full game config dict; forwarded to :class:`LODStatePacker`.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        self._packer = LODStatePacker(config)
        self._snapshots: Dict[str, PlayerStateSnapshot] = {}
        # Last send time per (recipient, sender) pair
        self._last_sent: Dict[Tuple[str, str], float] = {}

    # ------------------------------------------------------------------

    def update_player(self, player_id: str, snapshot: PlayerStateSnapshot) -> None:
        """Register or update the authoritative state for *player_id*."""
        self._snapshots[player_id] = snapshot

    def remove(self, player_id: str) -> None:
        """Remove a player's state and all associated send timers."""
        self._snapshots.pop(player_id, None)
        to_drop = [k for k in self._last_sent if player_id in k]
        for k in to_drop:
            del self._last_sent[k]

    def player_ids(self) -> List[str]:
        """Currently registered player ids."""
        return list(self._snapshots.keys())

    # ------------------------------------------------------------------

    def tick(self, now_s: float) -> Dict[str, List[Dict[str, Any]]]:
        """Produce state messages due for delivery this tick.

        Returns
        -------
        dict
            ``{recipient_player_id: [msg, …]}`` — messages to send.
        """
        out: Dict[str, List[Dict[str, Any]]] = {}
        player_ids = list(self._snapshots.keys())

        for recipient_id in player_ids:
            msgs: List[Dict[str, Any]] = []

            for sender_id, snap in self._snapshots.items():
                key = (recipient_id, sender_id)
                dist = _distance(
                    self._snapshots[recipient_id].pos, snap.pos
                )

                if sender_id == recipient_id:
                    # Own player — AUTH_STATE at state_hz_self
                    hz = self._packer._hz_self
                else:
                    hz = self._packer.pack_state_hz(dist)

                interval = 1.0 / max(1.0, hz)
                last = self._last_sent.get(key, 0.0)
                if now_s - last < interval:
                    continue
                self._last_sent[key] = now_s

                if sender_id == recipient_id:
                    msg = self._packer.pack_own(
                        pos           = snap.pos,
                        vel           = snap.vel,
                        yaw           = snap.yaw,
                        contact_flags = snap.contact_flags,
                        server_tick   = snap.server_tick,
                        last_seq      = snap.last_input_seq,
                        pose_hash     = snap.pose_hash,
                    )
                else:
                    msg = self._packer.pack_remote(
                        player_id     = sender_id,
                        pos           = snap.pos,
                        vel           = snap.vel,
                        yaw           = snap.yaw,
                        contact_flags = snap.contact_flags,
                        server_tick   = snap.server_tick,
                        timestamp_s   = snap.timestamp_s,
                        distance_m    = dist,
                        pose_hash     = snap.pose_hash,
                    )

                msgs.append(msg)

            if msgs:
                out[recipient_id] = msgs

        return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _distance(
    a: Tuple[float, float, float],
    b: Tuple[float, float, float],
) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)
