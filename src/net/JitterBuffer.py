"""JitterBuffer — Stage 58 adaptive jitter buffer for remote state frames.

Stores incoming state snapshots (indexed by their remote timestamp) and
exposes them for smooth interpolation at a configurable delay that adapts to
observed network jitter.

Public API
----------
StateFrame
    Dataclass: one server state snapshot for a remote entity.

JitterBuffer(config)
    .push(frame)
        Insert a new state frame.
    .interpolate(now_s) → StateFrame | None
        Return an interpolated frame at (now − delay) or *None* if not enough
        data.
    .update_delay(rtt_s, jitter_s)
        Update the adaptive interpolation delay based on measured RTT/jitter.
    .delay_s → float
        Current interpolation delay in seconds.
    .buffer_len() → int
        Number of frames currently in the buffer.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# StateFrame
# ---------------------------------------------------------------------------

@dataclass
class StateFrame:
    """One remote server state snapshot.

    Parameters
    ----------
    timestamp_s : float
        Server-side timestamp (seconds) when this state was captured.
    pos : tuple[float, float, float]
        World-space position.
    vel : tuple[float, float, float]
        Linear velocity.
    yaw : float
        Root yaw (radians).
    contact_flags : int
        Bitmask of foot-contact states.
    server_tick : int
        Server tick index.
    """
    timestamp_s:    float
    pos:            Tuple[float, float, float] = (0.0, 0.0, 0.0)
    vel:            Tuple[float, float, float] = (0.0, 0.0, 0.0)
    yaw:            float = 0.0
    contact_flags:  int   = 0
    server_tick:    int   = 0

    def to_dict(self) -> dict:
        return {
            "ts":      self.timestamp_s,
            "pos":     list(self.pos),
            "vel":     list(self.vel),
            "yaw":     self.yaw,
            "contact": self.contact_flags,
            "sTick":   self.server_tick,
        }

    @staticmethod
    def from_dict(d: dict) -> "StateFrame":
        p = d.get("pos", [0.0, 0.0, 0.0])
        v = d.get("vel", [0.0, 0.0, 0.0])
        return StateFrame(
            timestamp_s   = float(d["ts"]),
            pos           = (float(p[0]), float(p[1]), float(p[2])),
            vel           = (float(v[0]), float(v[1]), float(v[2])),
            yaw           = float(d.get("yaw", 0.0)),
            contact_flags = int(d.get("contact", 0)),
            server_tick   = int(d.get("sTick", 0)),
        )


# ---------------------------------------------------------------------------
# Interpolation helpers
# ---------------------------------------------------------------------------

def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _lerp_angle(a: float, b: float, t: float) -> float:
    """Lerp between two angles (radians) taking the shortest arc."""
    diff = (b - a + math.pi) % (2.0 * math.pi) - math.pi
    return a + diff * t


def _interpolate_frames(f0: StateFrame, f1: StateFrame, t: float) -> StateFrame:
    """Linearly interpolate between two state frames at blend factor *t* ∈ [0, 1]."""
    px = _lerp(f0.pos[0], f1.pos[0], t)
    py = _lerp(f0.pos[1], f1.pos[1], t)
    pz = _lerp(f0.pos[2], f1.pos[2], t)
    vx = _lerp(f0.vel[0], f1.vel[0], t)
    vy = _lerp(f0.vel[1], f1.vel[1], t)
    vz = _lerp(f0.vel[2], f1.vel[2], t)
    yaw = _lerp_angle(f0.yaw, f1.yaw, t)
    ts  = _lerp(f0.timestamp_s, f1.timestamp_s, t)
    contacts = f1.contact_flags if t >= 0.5 else f0.contact_flags
    return StateFrame(
        timestamp_s   = ts,
        pos           = (px, py, pz),
        vel           = (vx, vy, vz),
        yaw           = yaw,
        contact_flags = contacts,
        server_tick   = f1.server_tick,
    )


def _extrapolate_frame(f: StateFrame, dt: float) -> StateFrame:
    """Extrapolate *f* forward by *dt* seconds using constant velocity."""
    px = f.pos[0] + f.vel[0] * dt
    py = f.pos[1] + f.vel[1] * dt
    pz = f.pos[2] + f.vel[2] * dt
    return StateFrame(
        timestamp_s   = f.timestamp_s + dt,
        pos           = (px, py, pz),
        vel           = f.vel,
        yaw           = f.yaw,
        contact_flags = f.contact_flags,
        server_tick   = f.server_tick,
    )


# ---------------------------------------------------------------------------
# JitterBuffer
# ---------------------------------------------------------------------------

class JitterBuffer:
    """Adaptive jitter buffer for remote player state frames.

    Keeps an ordered list of state frames and interpolates at a render time
    of ``(now - delay)``.  The delay adapts up when jitter increases and
    trends down when the network is stable.

    Parameters
    ----------
    config : dict
        Full game config; reads ``net.interp_delay_base_ms`` and
        ``net.interp_delay_jitter_factor`` and ``net.extrapolation_max_ms``.
    """

    _MAX_BUFFER_SIZE = 64  # hard cap on stored frames per remote player

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = (config or {}).get("net", {}) or {}
        self._delay_base_s: float = float(cfg.get("interp_delay_base_ms", 100)) / 1000.0
        self._jitter_factor: float = float(cfg.get("interp_delay_jitter_factor", 1.5))
        self._extrap_max_s: float = float(cfg.get("extrapolation_max_ms", 200)) / 1000.0

        self._current_delay_s: float = self._delay_base_s
        self._frames: List[StateFrame] = []

    # ------------------------------------------------------------------
    # Push
    # ------------------------------------------------------------------

    def push(self, frame: StateFrame) -> None:
        """Insert a new state frame, maintaining ascending timestamp order."""
        # Insert in sorted order
        idx = len(self._frames)
        for i, f in enumerate(self._frames):
            if frame.timestamp_s < f.timestamp_s:
                idx = i
                break
        self._frames.insert(idx, frame)
        # Trim old frames
        if len(self._frames) > self._MAX_BUFFER_SIZE:
            self._frames = self._frames[-self._MAX_BUFFER_SIZE:]

    # ------------------------------------------------------------------
    # Interpolate / extrapolate
    # ------------------------------------------------------------------

    def interpolate(self, now_s: float) -> Optional[StateFrame]:
        """Return a state frame at ``now_s - delay`` via interpolation.

        Falls back to capped extrapolation if no future frame is available.
        Returns *None* if the buffer has fewer than 2 frames.

        Parameters
        ----------
        now_s : float
            Current monotonic clock time (seconds).
        """
        if not self._frames:
            return None

        target_ts = now_s - self._current_delay_s

        # Find bracketing frames
        before: Optional[StateFrame] = None
        after:  Optional[StateFrame] = None
        for f in self._frames:
            if f.timestamp_s <= target_ts:
                before = f
            elif after is None:
                after = f

        if before is not None and after is not None:
            span = after.timestamp_s - before.timestamp_s
            if span <= 0:
                return before
            t = (target_ts - before.timestamp_s) / span
            return _interpolate_frames(before, after, max(0.0, min(t, 1.0)))

        if before is not None:
            # No future frame: extrapolate from the last known state
            dt = target_ts - before.timestamp_s
            dt = min(dt, self._extrap_max_s)
            if dt <= 0:
                return before
            return _extrapolate_frame(before, dt)

        # All frames are in the future — too early, return earliest
        return self._frames[0]

    # ------------------------------------------------------------------
    # Adaptive delay
    # ------------------------------------------------------------------

    def update_delay(self, rtt_s: float, jitter_s: float) -> None:
        """Adjust interpolation delay based on measured RTT and jitter.

        The target delay is ``base + jitter * factor``.  The current delay
        is smoothed toward the target with a soft filter.

        Parameters
        ----------
        rtt_s : float
            Estimated round-trip time (seconds).
        jitter_s : float
            Estimated one-way jitter (seconds).
        """
        target = self._delay_base_s + jitter_s * self._jitter_factor
        # Asymmetric smoothing: jump up quickly, ease down slowly
        alpha = 0.3 if target > self._current_delay_s else 0.05
        self._current_delay_s += (target - self._current_delay_s) * alpha

    @property
    def delay_s(self) -> float:
        """Current interpolation delay (seconds)."""
        return self._current_delay_s

    def buffer_len(self) -> int:
        """Number of frames currently stored."""
        return len(self._frames)

    def clear(self) -> None:
        """Discard all buffered frames."""
        self._frames.clear()
