"""LODStatePacker — Stage 58 server-side LOD-aware state packer.

Selects the appropriate level of detail for outgoing state messages based on
the distance between the server player and the receiving client, honouring
the ``net.lod_distance_thresholds`` config.

LOD levels
----------
NEAR (< thresholds[0])
    Full detail: pos, vel, yaw, contact_flags, per-finger pose hash.
MID  (< thresholds[1])
    Reduced: pos, vel, yaw, contact_flags (no fine pose).
FAR  (≥ thresholds[1])
    Minimal: pos, yaw only.

Public API
----------
LODLevel
    Enum-like constants: NEAR, MID, FAR.

LODStatePacker(config)
    .lod_for_distance(distance_m) → LODLevel
        Return the appropriate LOD level.
    .pack_own(pos, vel, yaw, contact_flags, server_tick, last_seq,
              pose_hash) → dict
        Build an ``AUTH_STATE`` message for the own player.
    .pack_remote(player_id, pos, vel, yaw, contact_flags, server_tick,
                 timestamp_s, distance_m, pose_hash) → dict
        Build a ``REMOTE_STATE`` message for a remote player, filtered by LOD.
    .pack_state_hz(distance_m) → float
        Return the appropriate send rate (Hz) for the given distance.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# LOD level constants
# ---------------------------------------------------------------------------

class LODLevel:
    NEAR = 0
    MID  = 1
    FAR  = 2


# ---------------------------------------------------------------------------
# LODStatePacker
# ---------------------------------------------------------------------------

class LODStatePacker:
    """Pack server-side state messages with LOD filtering.

    Parameters
    ----------
    config : dict
        Full game config dict; reads ``net.lod_distance_thresholds``,
        ``net.state_hz_remote_near``, and ``net.state_hz_remote_far``.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = (config or {}).get("net", {}) or {}
        thresholds = cfg.get("lod_distance_thresholds", [30.0, 100.0])
        self._near_dist: float = float(thresholds[0])
        self._far_dist:  float = float(thresholds[1])

        self._hz_self:   float = float(cfg.get("state_hz_self",         20))
        self._hz_near:   float = float(cfg.get("state_hz_remote_near",  20))
        self._hz_far:    float = float(cfg.get("state_hz_remote_far",   10))

    # ------------------------------------------------------------------

    def lod_for_distance(self, distance_m: float) -> int:
        """Return the appropriate :class:`LODLevel` for *distance_m*."""
        if distance_m < self._near_dist:
            return LODLevel.NEAR
        if distance_m < self._far_dist:
            return LODLevel.MID
        return LODLevel.FAR

    def pack_state_hz(self, distance_m: float) -> float:
        """Return the send rate (Hz) for the given recipient distance."""
        lod = self.lod_for_distance(distance_m)
        if lod == LODLevel.NEAR:
            return self._hz_near
        if lod == LODLevel.MID:
            return (self._hz_near + self._hz_far) * 0.5
        return self._hz_far

    # ------------------------------------------------------------------
    # Packers
    # ------------------------------------------------------------------

    def pack_own(
        self,
        pos:           Tuple[float, float, float],
        vel:           Tuple[float, float, float],
        yaw:           float,
        contact_flags: int,
        server_tick:   int,
        last_seq:      int,
        pose_hash:     int = 0,
    ) -> Dict[str, Any]:
        """Build an ``AUTH_STATE`` message for the own player (always full detail)."""
        return {
            "type":    "AUTH_STATE",
            "pos":     [pos[0], pos[1], pos[2]],
            "vel":     [vel[0], vel[1], vel[2]],
            "yaw":     yaw,
            "contact": contact_flags,
            "sTick":   server_tick,
            "lastSeq": last_seq,
            "poseHash": pose_hash,
        }

    def pack_remote(
        self,
        player_id:     str,
        pos:           Tuple[float, float, float],
        vel:           Tuple[float, float, float],
        yaw:           float,
        contact_flags: int,
        server_tick:   int,
        timestamp_s:   float,
        distance_m:    float,
        pose_hash:     int = 0,
    ) -> Dict[str, Any]:
        """Build a ``REMOTE_STATE`` message filtered to the appropriate LOD."""
        lod = self.lod_for_distance(distance_m)

        msg: Dict[str, Any] = {
            "type":     "REMOTE_STATE",
            "playerId": player_id,
            "ts":       timestamp_s,
            "pos":      [pos[0], pos[1], pos[2]],
            "yaw":      yaw,
            "sTick":    server_tick,
        }

        if lod <= LODLevel.MID:
            msg["vel"]     = [vel[0], vel[1], vel[2]]
            msg["contact"] = contact_flags

        if lod == LODLevel.NEAR:
            msg["poseHash"] = pose_hash

        return msg
