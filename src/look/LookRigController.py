"""LookRigController — Stage 43 head / torso / body look-at solver.

Distributes the player's ``FinalLookDir`` across three segments in order:

1. **Head** (visor) absorbs yaw up to ``head_yaw_max_deg`` and most of the
   pitch.
2. **Torso** takes the residual yaw up to ``torso_yaw_max_deg``.
3. **BodyYaw** (full-body rotation) handles any remaining yaw beyond the
   combined head+torso budget.

The solver runs once per tick and returns a :class:`LookRigResult` that
downstream systems (animation, camera, physics) read.

Standing vs. moving
-------------------
When the character is stationary the body can rotate quickly in place.
When moving, BodyYaw is realised through locomotion steering (an extra yaw
goal sent to the footstep planner).  When ``slip_risk`` is high the body
yaw rate is reduced to avoid dangerous direction changes.

Public API
----------
LookRigController(config=None)
  .update(dt, final_look_dir, body_forward, up,
          is_moving, slip_risk) → LookRigResult
  .debug_info() → dict
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from src.math.Vec3 import Vec3


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


def _normalize(v: Vec3) -> Vec3:
    l = v.length()
    return v * (1.0 / l) if l > 1e-9 else Vec3.zero()


def _signed_angle_around_axis(from_v: Vec3, to_v: Vec3, axis: Vec3) -> float:
    """Signed angle in radians from *from_v* to *to_v* rotating around *axis*."""
    cross = from_v.cross(to_v)
    sign  = 1.0 if cross.dot(axis) >= 0.0 else -1.0
    dot   = _clamp(from_v.dot(to_v), -1.0, 1.0)
    return sign * math.acos(dot)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class LookRigResult:
    """Per-tick output of the look-rig solver.

    All directions are world-space unit vectors.
    """
    head_dir:         Vec3  = field(default_factory=lambda: Vec3(0.0, 0.0, -1.0))
    torso_dir:        Vec3  = field(default_factory=lambda: Vec3(0.0, 0.0, -1.0))
    body_forward:     Vec3  = field(default_factory=lambda: Vec3(0.0, 0.0, -1.0))
    # How much residual yaw the footstep planner should steer toward [rad]
    body_yaw_goal_rad: float = 0.0
    # Fraction of head yaw used vs. limit (0 = eyes forward, 1 = max)
    head_yaw_fraction: float = 0.0
    torso_yaw_fraction: float = 0.0


# ---------------------------------------------------------------------------
# LookRigController
# ---------------------------------------------------------------------------

class LookRigController:
    """Distributes FinalLookDir across head, torso, and body.

    Parameters
    ----------
    config :
        Optional dict; reads ``look.*`` sub-keys.
    """

    _DEFAULT_HEAD_YAW_MAX_DEG    = 60.0
    _DEFAULT_TORSO_YAW_MAX_DEG   = 30.0
    _DEFAULT_BODY_YAW_RATE       = 90.0   # deg/s standing
    _DEFAULT_PITCH_MAX_DEG       = 80.0

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg  = config or {}
        lcfg = cfg.get("look", {}) or {}

        self._head_yaw_max  = math.radians(float(
            lcfg.get("head_yaw_max_deg",  self._DEFAULT_HEAD_YAW_MAX_DEG)))
        self._torso_yaw_max = math.radians(float(
            lcfg.get("torso_yaw_max_deg", self._DEFAULT_TORSO_YAW_MAX_DEG)))
        self._body_yaw_rate = math.radians(float(
            lcfg.get("body_yaw_rate_deg_per_sec", self._DEFAULT_BODY_YAW_RATE)))
        self._pitch_max     = math.radians(float(
            lcfg.get("pitch_max_deg", self._DEFAULT_PITCH_MAX_DEG)))

        # Maintained state
        self._body_yaw_offset: float = 0.0   # rad; current body yaw vs. look
        self._last_result = LookRigResult()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    @property
    def last_result(self) -> LookRigResult:
        return self._last_result

    def update(
        self,
        dt:              float,
        final_look_dir:  Vec3,
        body_forward:    Vec3,
        up:              Vec3,
        is_moving:       bool  = False,
        slip_risk:       float = 0.0,
    ) -> LookRigResult:
        """Solve the look rig for one tick.

        Parameters
        ----------
        dt :
            Elapsed time since last call [s].
        final_look_dir :
            World-space unit direction the player wants to look (from
            PlayerIntent.FinalLookDir).
        body_forward :
            Current body forward direction (world-space unit vector).
        up :
            Planet surface up at the character's position.
        is_moving :
            True if the character is actively locomoting.
        slip_risk :
            [0..1] from PerceptionState; reduces body-yaw rate when high.
        """
        # Project look direction onto horizontal plane for yaw calculation
        look_h  = _normalize(final_look_dir - up * final_look_dir.dot(up))
        body_h  = _normalize(body_forward   - up * body_forward.dot(up))

        if look_h.length() < 1e-9:
            look_h = body_h

        # Yaw angle from body to target look (horizontal)
        total_yaw = _signed_angle_around_axis(body_h, look_h, up)

        # --- Allocate yaw to head ---
        head_yaw  = _clamp(total_yaw, -self._head_yaw_max, self._head_yaw_max)
        residual  = total_yaw - head_yaw

        # --- Allocate remainder to torso ---
        torso_yaw = _clamp(residual, -self._torso_yaw_max, self._torso_yaw_max)
        body_yaw_needed = residual - torso_yaw

        # --- Body yaw: rate-limited ---
        slip_factor    = 1.0 - _clamp(slip_risk, 0.0, 0.8)   # reduce rate when slippery
        max_body_delta = self._body_yaw_rate * dt * slip_factor
        if is_moving:
            # Locomotion handles it more gently — half the rate limit
            max_body_delta *= 0.5

        body_yaw_delta = _clamp(body_yaw_needed, -max_body_delta, max_body_delta)
        self._body_yaw_offset += body_yaw_delta

        # --- Build output directions ---
        # Head direction: body_forward rotated by (head_yaw + torso_yaw + body_yaw_offset) in horizontal,
        # plus pitch from final_look_dir
        combined_yaw = head_yaw + torso_yaw + self._body_yaw_offset

        # Yaw rotation of body_h
        new_body_h = self._rotate_around_axis(body_h, up, self._body_yaw_offset)
        new_body_h = _normalize(new_body_h)

        torso_dir_h = self._rotate_around_axis(new_body_h, up, torso_yaw)
        torso_dir_h = _normalize(torso_dir_h)

        head_dir_h  = self._rotate_around_axis(torso_dir_h, up, head_yaw)
        head_dir_h  = _normalize(head_dir_h)

        # Add pitch to head dir only
        pitch = math.asin(_clamp(final_look_dir.dot(up), -1.0, 1.0))
        pitch = _clamp(pitch, -self._pitch_max, self._pitch_max)

        cos_p = math.cos(pitch)
        sin_p = math.sin(pitch)
        head_dir = _normalize(head_dir_h * cos_p + up * sin_p)

        # Fractions
        head_frac  = abs(head_yaw)  / max(self._head_yaw_max,  1e-6)
        torso_frac = abs(torso_yaw) / max(self._torso_yaw_max, 1e-6)

        result = LookRigResult(
            head_dir          = head_dir,
            torso_dir         = _normalize(torso_dir_h * cos_p + up * sin_p * 0.3),
            body_forward      = new_body_h,
            body_yaw_goal_rad = body_yaw_needed - body_yaw_delta,   # residual still needed
            head_yaw_fraction = _clamp(head_frac,  0.0, 1.0),
            torso_yaw_fraction= _clamp(torso_frac, 0.0, 1.0),
        )
        self._last_result = result
        return result

    def debug_info(self) -> dict:
        """Snapshot of look-rig state for gizmos / logging."""
        r = self._last_result
        return {
            "head_dir":          (r.head_dir.x,     r.head_dir.y,     r.head_dir.z),
            "torso_dir":         (r.torso_dir.x,    r.torso_dir.y,    r.torso_dir.z),
            "body_forward":      (r.body_forward.x, r.body_forward.y, r.body_forward.z),
            "body_yaw_goal_deg": math.degrees(r.body_yaw_goal_rad),
            "head_yaw_frac":     r.head_yaw_fraction,
            "torso_yaw_frac":    r.torso_yaw_fraction,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _rotate_around_axis(v: Vec3, axis: Vec3, angle_rad: float) -> Vec3:
        """Rodrigues' rotation formula."""
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        dot   = v.dot(axis)
        cross = axis.cross(v)
        return v * cos_a + cross * sin_a + axis * (dot * (1.0 - cos_a))
