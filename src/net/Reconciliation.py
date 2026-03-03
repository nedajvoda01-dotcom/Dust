"""Reconciliation — Stage 58 client-side soft reconciliation for own player.

When the server sends an authoritative state that differs from the client's
predicted state, this module applies a smooth "error spring" correction
distributed over several ticks rather than teleporting.

If the error exceeds the hard-snap threshold the position is snapped
immediately, then smoothing resumes.

Public API
----------
ReconciliationState
    Mutable state object holding the current correction residuals.

Reconciliation(config)
    .receive_authoritative(server_pos, server_vel, server_yaw,
                           last_processed_seq, server_tick)
        Feed in an authoritative server state; computes error residuals.
    .apply(pos, vel, yaw) → (pos, vel, yaw)
        Apply the current spring correction to a predicted state and decay the
        residuals.  Call once per client tick.
    .has_active_correction() → bool
        True when a non-trivial correction is in progress.
    .last_correction_magnitude() → float
        Euclidean magnitude of the position error at the last server update.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

Vec3 = Tuple[float, float, float]


# ---------------------------------------------------------------------------
# ReconciliationState
# ---------------------------------------------------------------------------

@dataclass
class ReconciliationState:
    """Mutable correction residuals."""
    err_pos:  Vec3  = (0.0, 0.0, 0.0)
    err_vel:  Vec3  = (0.0, 0.0, 0.0)
    err_yaw:  float = 0.0
    snapped:  bool  = False
    last_correction_mag: float = 0.0


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------

class Reconciliation:
    """Soft error-spring reconciliation for the local player.

    Parameters
    ----------
    config : dict
        Full game config dict; reads ``net.reconcile.*``.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = ((config or {}).get("net", {}) or {}).get("reconcile", {}) or {}
        self._k_pos: float  = float(cfg.get("k_pos",  0.15))
        self._k_vel: float  = float(cfg.get("k_vel",  0.2))
        self._k_yaw: float  = float(cfg.get("k_yaw",  0.25))
        self._snap_thresh: float = float(cfg.get("hard_snap_threshold", 5.0))

        self._state = ReconciliationState()
        self._last_processed_seq: int = -1

    # ------------------------------------------------------------------
    # Receive server update
    # ------------------------------------------------------------------

    def receive_authoritative(
        self,
        server_pos:          Vec3,
        server_vel:          Vec3,
        server_yaw:          float,
        last_processed_seq:  int,
        server_tick:         int,
        predicted_pos:       Vec3,
        predicted_vel:       Vec3,
        predicted_yaw:       float,
    ) -> None:
        """Compute new error residuals from the latest server authoritative state.

        Parameters
        ----------
        server_pos, server_vel, server_yaw :
            Authoritative state from the server.
        last_processed_seq :
            The server's most recently processed client input sequence.
        server_tick :
            Server tick index for the snapshot.
        predicted_pos, predicted_vel, predicted_yaw :
            The client's current predicted state at the same simulation time.
        """
        self._last_processed_seq = last_processed_seq

        ep = (
            server_pos[0] - predicted_pos[0],
            server_pos[1] - predicted_pos[1],
            server_pos[2] - predicted_pos[2],
        )
        mag = math.sqrt(ep[0] ** 2 + ep[1] ** 2 + ep[2] ** 2)
        self._state.last_correction_mag = mag

        if mag >= self._snap_thresh:
            # Hard snap: immediately adopt server position, then reset errors
            self._state.err_pos  = ep
            self._state.err_vel  = (
                server_vel[0] - predicted_vel[0],
                server_vel[1] - predicted_vel[1],
                server_vel[2] - predicted_vel[2],
            )
            self._state.err_yaw  = _angle_diff(server_yaw, predicted_yaw)
            self._state.snapped  = True
            # Force full immediate application next apply() call
            self._state.err_pos  = (0.0, 0.0, 0.0)
            self._state.err_vel  = (0.0, 0.0, 0.0)
            self._state.err_yaw  = 0.0
            return

        self._state.snapped = False
        self._state.err_pos = (
            self._state.err_pos[0] + ep[0],
            self._state.err_pos[1] + ep[1],
            self._state.err_pos[2] + ep[2],
        )
        ev = (
            server_vel[0] - predicted_vel[0],
            server_vel[1] - predicted_vel[1],
            server_vel[2] - predicted_vel[2],
        )
        self._state.err_vel = (
            self._state.err_vel[0] + ev[0],
            self._state.err_vel[1] + ev[1],
            self._state.err_vel[2] + ev[2],
        )
        self._state.err_yaw += _angle_diff(server_yaw, predicted_yaw)

    # ------------------------------------------------------------------
    # Apply per tick
    # ------------------------------------------------------------------

    def apply(
        self,
        pos: Vec3,
        vel: Vec3,
        yaw: float,
    ) -> Tuple[Vec3, Vec3, float]:
        """Apply spring correction and decay residuals.

        Parameters
        ----------
        pos, vel, yaw :
            Current predicted state.

        Returns
        -------
        (corrected_pos, corrected_vel, corrected_yaw)
        """
        ep = self._state.err_pos
        ev = self._state.err_vel
        ey = self._state.err_yaw

        dp = (ep[0] * self._k_pos, ep[1] * self._k_pos, ep[2] * self._k_pos)
        dv = (ev[0] * self._k_vel, ev[1] * self._k_vel, ev[2] * self._k_vel)
        dy = ey * self._k_yaw

        new_pos = (pos[0] + dp[0], pos[1] + dp[1], pos[2] + dp[2])
        new_vel = (vel[0] + dv[0], vel[1] + dv[1], vel[2] + dv[2])
        new_yaw = yaw + dy

        # Decay residuals
        decay = 1.0 - self._k_pos  # same factor for simplicity
        self._state.err_pos = (ep[0] * decay, ep[1] * decay, ep[2] * decay)
        decay_v = 1.0 - self._k_vel
        self._state.err_vel = (ev[0] * decay_v, ev[1] * decay_v, ev[2] * decay_v)
        self._state.err_yaw = ey * (1.0 - self._k_yaw)

        return new_pos, new_vel, new_yaw

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    def has_active_correction(self) -> bool:
        """True when residuals are non-negligible (> 0.001 m)."""
        ep = self._state.err_pos
        return math.sqrt(ep[0] ** 2 + ep[1] ** 2 + ep[2] ** 2) > 1e-3

    def last_correction_magnitude(self) -> float:
        """Euclidean position-error magnitude at the last server update."""
        return self._state.last_correction_mag

    @property
    def last_processed_seq(self) -> int:
        """Most recent server-acknowledged input sequence."""
        return self._last_processed_seq


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _angle_diff(a: float, b: float) -> float:
    """Shortest signed difference between two angles (radians)."""
    diff = (a - b + math.pi) % (2.0 * math.pi) - math.pi
    return diff
