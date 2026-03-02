"""InputSystem — Stage 43 WASD + mouse → PlayerIntent.

Converts raw per-frame axis deltas (WASD keys and mouse movement) into a
:class:`~src.input.PlayerIntent.PlayerIntent` that the rest of the motor
stack reads each tick.

Design
------
* WASD forms ``moveDir_local`` in **camera-relative** horizontal space (the
  ``input.move_frame = "camera"`` default from CONFIG_DEFAULTS).  W = forward
  along camera horizontal projection, S = back, A = strafe-left, D = right.
* Speed intent is always [0..1] walk/run; no sprint (design philosophy).
* Mouse delta accumulates into a yaw/pitch offset that is converted to a
  world-space unit direction via the caller-supplied ``up`` vector and a
  previous look direction.  Pitch is clamped to ±``pitch_max_deg``.
* Direction changes are smoothed with a simple exponential filter
  (``smooth_tau``) to avoid instant flips that would upset physics.

Public API
----------
InputSystem(config=None)
  .update(dt, wasd, mouse_dx, mouse_dy, camera_forward, camera_right, up)
      → PlayerIntent
  .current_intent  → PlayerIntent
  .debug_info()    → dict
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from src.math.Vec3 import Vec3
from src.input.PlayerIntent import (
    PlayerIntent, PrimaryMoveTarget, PrimaryLookTarget, ReflexOverlay,
)


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


def _normalize(v: Vec3) -> Vec3:
    l = v.length()
    return v * (1.0 / l) if l > 1e-9 else Vec3.zero()


# ---------------------------------------------------------------------------
# WASD key state helper
# ---------------------------------------------------------------------------

@dataclass
class WASDState:
    """Boolean key state for one tick.

    Positive axes: W = forward, D = right; A/S are negatives.
    """
    w: bool = False
    a: bool = False
    s: bool = False
    d: bool = False


# ---------------------------------------------------------------------------
# InputSystem
# ---------------------------------------------------------------------------

class InputSystem:
    """Converts raw WASD + mouse deltas into a :class:`PlayerIntent`.

    Parameters
    ----------
    config :
        Optional dict; reads ``input.*`` sub-keys.
    """

    _DEFAULT_PITCH_MAX_DEG  = 80.0
    _DEFAULT_MOUSE_SENS     = 0.15   # degrees per raw mouse unit
    _DEFAULT_SMOOTH_TAU     = 0.08   # seconds; move-dir smoothing
    _DEFAULT_MOVE_SPEED     = 1.0    # default speedIntent for any held key

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg  = config or {}
        icfg = cfg.get("input", {}) or {}

        self._pitch_max   = math.radians(float(icfg.get("pitch_max_deg",
                                                         self._DEFAULT_PITCH_MAX_DEG)))
        self._mouse_sens  = math.radians(float(icfg.get("mouse_sens_deg_per_unit",
                                                          self._DEFAULT_MOUSE_SENS)))
        self._smooth_tau  = float(icfg.get("smooth_tau_sec", self._DEFAULT_SMOOTH_TAU))

        # Look state maintained across ticks (yaw/pitch angles)
        self._yaw:   float = 0.0   # radians; accumulated
        self._pitch: float = 0.0   # radians; clamped

        # Smoothed move direction (local space)
        self._smooth_move = Vec3.zero()

        self._current_intent = PlayerIntent()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    @property
    def current_intent(self) -> PlayerIntent:
        """Most recently assembled PlayerIntent."""
        return self._current_intent

    def update(
        self,
        dt:             float,
        wasd:           WASDState,
        mouse_dx:       float,
        mouse_dy:       float,
        camera_forward: Vec3,
        camera_right:   Vec3,
        up:             Vec3,
    ) -> PlayerIntent:
        """Assemble and return a PlayerIntent for this tick.

        Parameters
        ----------
        dt :
            Elapsed time since last call [s].
        wasd :
            Current key hold state.
        mouse_dx, mouse_dy :
            Raw mouse delta this tick (arbitrary units; scaled by sens).
        camera_forward :
            World-space unit vector for camera horizontal forward.
        camera_right :
            World-space unit vector for camera right.
        up :
            Planet surface up at the character's position.
        """
        # --- 1. Build raw move direction from WASD ---
        fwd_w = 1.0 if wasd.w else 0.0
        bwd_w = 1.0 if wasd.s else 0.0
        rgt_w = 1.0 if wasd.d else 0.0
        lft_w = 1.0 if wasd.a else 0.0

        raw_fwd   = fwd_w - bwd_w
        raw_right = rgt_w - lft_w

        # Project camera axes onto horizontal plane
        cam_fwd_h   = _normalize(camera_forward - up * camera_forward.dot(up))
        cam_right_h = _normalize(camera_right   - up * camera_right.dot(up))

        raw_dir = cam_fwd_h * raw_fwd + cam_right_h * raw_right
        raw_len = raw_dir.length()

        if raw_len > 1e-9:
            raw_dir = raw_dir * (1.0 / raw_len)
            speed_intent = 1.0
        else:
            raw_dir      = Vec3.zero()
            speed_intent = 0.0

        # --- 2. Smooth move direction ---
        if dt > 1e-9 and self._smooth_tau > 1e-9:
            alpha = 1.0 - math.exp(-dt / self._smooth_tau)
        else:
            alpha = 1.0

        # Lerp the raw direction toward smooth_move
        sm = self._smooth_move * (1.0 - alpha) + raw_dir * alpha
        sm_len = sm.length()
        if sm_len > 1e-9:
            self._smooth_move = sm * (1.0 / sm_len)
        else:
            self._smooth_move = Vec3.zero()

        final_dir = self._smooth_move if speed_intent > 0.0 else Vec3.zero()

        # --- 3. Accumulate mouse look (yaw/pitch) ---
        self._yaw   += mouse_dx * self._mouse_sens
        self._pitch -= mouse_dy * self._mouse_sens   # invert Y (screen space)
        self._pitch  = _clamp(self._pitch, -self._pitch_max, self._pitch_max)

        # Convert yaw/pitch to world look direction.
        # Start from a canonical forward (0,0,-1), rotate by yaw around up,
        # then tilt by pitch away from horizontal.
        # Basis: forward is -Z, right is +X (right-hand). We build a local
        # horizontal forward from yaw, then add pitch component.
        look_dir = self._build_look_dir(up, camera_forward, camera_right)

        # --- 4. Assemble intent (reflex stays at defaults; filled by ReflexOverlaySystem) ---
        intent = PlayerIntent(
            move=PrimaryMoveTarget(
                moveDir_local=final_dir,
                speedIntent=speed_intent,
            ),
            look=PrimaryLookTarget(
                lookDir_world=look_dir,
            ),
            reflex=ReflexOverlay(),
        )
        self._current_intent = intent
        return intent

    def debug_info(self) -> dict:
        """Current input state snapshot for logging / gizmos."""
        i = self._current_intent
        return {
            "yaw_deg":        math.degrees(self._yaw),
            "pitch_deg":      math.degrees(self._pitch),
            "moveDir_local":  (i.move.moveDir_local.x,
                               i.move.moveDir_local.y,
                               i.move.moveDir_local.z),
            "speedIntent":    i.move.speedIntent,
            "lookDir_world":  (i.look.lookDir_world.x,
                               i.look.lookDir_world.y,
                               i.look.lookDir_world.z),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_look_dir(
        self,
        up:             Vec3,
        camera_forward: Vec3,
        camera_right:   Vec3,
    ) -> Vec3:
        """Convert current yaw + pitch angles into a world-space look dir.

        Yaw rotates around ``up``; pitch tilts up/down from the horizontal
        plane.  The pitch component is added along the *body up* axis so
        looking up tilts toward the sky naturally.
        """
        # Horizontal component: yaw around up
        # We use the camera_forward as the zero-yaw reference each frame
        # (absolute yaw is accumulated in self._yaw).
        cos_y = math.cos(self._yaw)
        sin_y = math.sin(self._yaw)

        # Horizontal forward (perpendicular to up)
        fwd_h   = _normalize(camera_forward - up * camera_forward.dot(up))
        right_h = _normalize(camera_right   - up * camera_right.dot(up))

        # Rotate fwd_h by yaw in horizontal plane
        horiz = fwd_h * cos_y + right_h * sin_y
        horiz = _normalize(horiz)

        # Tilt by pitch: blend horiz with up
        cos_p = math.cos(self._pitch)
        sin_p = math.sin(self._pitch)
        look  = horiz * cos_p + up * sin_p

        return _normalize(look)
