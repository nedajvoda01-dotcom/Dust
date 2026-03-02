"""StabilityCameraController — Stage 41 Stability-Driven Cinematic Camera.

Architecture (section 3)::

    MotorState + PerceptionState + SocialState
                ↓
          CameraIntent           (via CinematicBias)
                ↓
          CameraController       (this module)
                ↓
          CameraRig              (CameraFrame output)

Design principles
-----------------
* Behaviour is a pure function of inputs — deterministic, no ``random()``.
* Critically damped spring smoothing on position, FOV, and roll (section 13).
* Predictive offset accounts for character velocity (section 13).
* Cinematic constraints enforced: tilt ≤ 5°, roll ≤ 5°, FOV 60–75° (section 11).
* No scripted cut-scenes, no UI markers (section 21).
* Camera state is local only — not network-synchronised (section 16).

Public API
----------
CameraFrame (dataclass)
StabilityCameraController(config=None, player_id=0)
  .update(dt, stability_input) → CameraFrame
  .debug_info → dict
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from src.math.Vec3 import Vec3
from src.math.Quat import Quat
from src.math.PlanetMath import PlanetMath
from src.systems.CharacterPhysicalController import CharacterState
from src.camera.CameraConfig import CameraConfig
from src.camera.CameraIntent import CameraIntent
from src.camera.CinematicBias import CinematicBias, StabilityInput
from src.camera.ShakeModel import ShakeModel


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


_MAX_FRAME_DT: float = 0.1  # guard against spiral-of-death in spring integrators


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class CameraFrame:
    """Per-tick output of StabilityCameraController."""

    cam_pos: Vec3
    cam_rot: Quat
    fov_deg: float
    shake_intensity: float = 0.0
    roll_deg: float = 0.0    # applied roll (for debug / tests)
    tilt_deg: float = 0.0    # applied tilt (for debug / tests)


# ---------------------------------------------------------------------------
# Spring integrators (critically damped)
# ---------------------------------------------------------------------------

class _SpringVec3:
    """Critically-damped spring on Vec3."""

    def __init__(self, value: Vec3) -> None:
        self.value = value
        self.velocity = Vec3.zero()

    def update(self, target: Vec3, dt: float, freq: float, damp: float) -> Vec3:
        omega = 2.0 * math.pi * freq
        df = 1.0 + 2.0 * dt * damp * omega
        osq = omega * omega
        hosq = dt * osq
        det_inv = 1.0 / (df + dt * hosq)
        new_val = (self.value * df + self.velocity * dt + target * (hosq * dt)) * det_inv
        self.velocity = (self.velocity + (target - self.value) * hosq) * det_inv
        self.value = new_val
        return self.value


class _SpringFloat:
    """Critically-damped spring on a scalar float."""

    def __init__(self, value: float) -> None:
        self.value = value
        self.velocity = 0.0

    def update(self, target: float, dt: float, freq: float, damp: float) -> float:
        omega = 2.0 * math.pi * freq
        df = 1.0 + 2.0 * dt * damp * omega
        osq = omega * omega
        hosq = dt * osq
        det_inv = 1.0 / (df + dt * hosq)
        new_val = (self.value * df + self.velocity * dt + target * hosq * dt) * det_inv
        self.velocity = (self.velocity + (target - self.value) * hosq) * det_inv
        self.value = new_val
        return self.value


# ---------------------------------------------------------------------------
# StabilityCameraController
# ---------------------------------------------------------------------------

class StabilityCameraController:
    """Stability-driven cinematic third-person camera (Stage 41).

    Reacts to balance risk, slip state, wind, grasp, social context, and
    macro events.  Motion is always smooth and deterministic.

    Parameters
    ----------
    config :
        :class:`CameraConfig` instance.  Uses defaults if not provided.
    player_id :
        Stable integer player identifier used to seed the shake model.
    """

    def __init__(
        self,
        config: Optional[CameraConfig] = None,
        player_id: int = 0,
    ) -> None:
        self._cfg = config or CameraConfig()
        self._bias = CinematicBias()
        self._shake = ShakeModel(player_id=player_id)

        cfg = self._cfg

        # Position spring — initialised at a reasonable default above surface
        _init_pos = Vec3(0.0, 1000.0 + cfg.base_height, 0.0)
        self._spring_pos = _SpringVec3(_init_pos)

        # Scalar springs
        self._spring_fov = _SpringFloat(cfg.base_fov)
        self._spring_roll = _SpringFloat(0.0)
        self._spring_dist = _SpringFloat(cfg.base_distance)
        self._spring_height = _SpringFloat(cfg.base_height)

        # Last known tangent forward (used when character is stationary)
        self._last_forward: Vec3 = Vec3(0.0, 0.0, -1.0)

        self._time: float = 0.0
        self._debug: dict = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, dt: float, inp: StabilityInput) -> CameraFrame:
        """Advance camera by *dt* seconds and return the new frame.

        Parameters
        ----------
        dt :
            Elapsed game-seconds (clamped internally to 0.1 s maximum).
        inp :
            Full :class:`StabilityInput` snapshot for this tick.
        """
        dt = _clamp(dt, 1e-6, _MAX_FRAME_DT)
        self._time += dt
        self._shake.update(dt)

        cfg = self._cfg

        # 1. Compute camera intent from all bias sources
        intent = self._bias.compute(inp, cfg)

        # 2. Planet up at character position
        up = PlanetMath.up_at_position(inp.position)
        if up.is_near_zero():
            up = inp.up if not inp.up.is_near_zero() else Vec3(0.0, 1.0, 0.0)

        # 3. Character forward tangent
        fwd = self._resolve_forward(inp, up)

        # 4. Target distance and height (apply intent, with safety clamps)
        target_dist = _clamp(
            cfg.base_distance * intent.distance_scale,
            cfg.base_distance * 0.5,
            cfg.base_distance * cfg.macro_pullback_k,
        )
        target_height = cfg.base_height + intent.height_bias

        # 5. Spring-smooth distance and height
        smooth_dist = self._spring_dist.update(
            target_dist, dt, cfg.spring_freq, cfg.spring_damp
        )
        smooth_height = self._spring_height.update(
            target_height, dt, cfg.spring_freq, cfg.spring_damp
        )

        # 6. Wind lateral sway (section 7 — low-frequency, smooth)
        side = up.cross(fwd)
        if side.is_near_zero():
            side = Vec3(1.0, 0.0, 0.0)
        side = side.normalized()
        sway_offset = side * intent.lateral_sway

        # 7. Look-at point (head level, possibly shifted by focus intent)
        look_at = inp.position + up * cfg.head_height
        if not intent.focus_dir.is_near_zero():
            # Subtle focus shift (not a hard look-at redirect)
            look_at = look_at + intent.focus_dir * 0.5

        # 8. Predictive offset: lean ahead of velocity (section 13)
        vel_tangent = PlanetMath.tangent_forward(up, inp.velocity)
        vel_speed = inp.velocity.length()
        if vel_speed > 0.1 and not vel_tangent.is_near_zero():
            predictive = vel_tangent * min(
                vel_speed * cfg.predictive_velocity_scale, cfg.predictive_offset_max
            )
        else:
            predictive = Vec3.zero()

        # 9. Desired camera position (boom)
        desired_cam = (
            inp.position
            - fwd * smooth_dist
            + up * smooth_height
            + side * cfg.shoulder_offset
            + sway_offset
            + predictive
        )

        # 10. Collision avoidance: push back above surface
        corrected_cam = self._collide(look_at, desired_cam, inp.position)

        # 11. Spring-smooth camera position
        smooth_pos = self._spring_pos.update(
            corrected_cam, dt, cfg.spring_freq, cfg.spring_damp
        )

        # 12. Attention look-aside (section 10, max attention_max_deg)
        if intent.attention_offset_deg > 0.01 and not inp.attention_dir.is_near_zero():
            attn_angle = math.radians(
                _clamp(intent.attention_offset_deg, 0.0, cfg.attention_max_deg)
            )
            attn_tan = PlanetMath.tangent_forward(up, inp.attention_dir)
            if not attn_tan.is_near_zero():
                q_attn = Quat.from_axis_angle(up, attn_angle * cfg.attention_rotation_weight)
                look_at = look_at + q_attn.rotate_vec(attn_tan) * cfg.attention_position_weight

        # 13. Shake (deterministic, section 14)
        shake_offset, shake_intensity = self._shake.compute(
            vibration_level=inp.vibration_level,
            constraint_force=0.4 if inp.grasp_active else 0.0,
            landing_impulse=inp.landing_impulse,
            up=up,
            forward=fwd,
        )
        cam_pos = smooth_pos + shake_offset

        # Ensure camera stays above planet surface
        cam_pos = self._bias_above_surface(cam_pos, inp.position)

        # 14. Base rotation: look-at with planet-up constraint
        cam_rot_base = self._compute_rotation(cam_pos, look_at, up)

        # 15. Roll bias (section 6.2, constrained to ±roll_max_deg)
        target_roll_rad = math.radians(
            _clamp(intent.roll_bias, -cfg.roll_max_deg, cfg.roll_max_deg)
        )
        smooth_roll_rad = self._spring_roll.update(
            target_roll_rad, dt, cfg.roll_freq, cfg.spring_damp
        )
        applied_roll_deg = math.degrees(smooth_roll_rad)

        if abs(smooth_roll_rad) > 1e-4:
            cam_fwd = cam_rot_base.rotate_vec(Vec3(0.0, 0.0, -1.0))
            q_roll = Quat.from_axis_angle(cam_fwd, smooth_roll_rad)
            cam_rot = q_roll * cam_rot_base
        else:
            cam_rot = cam_rot_base

        # 16. FOV spring (section 11: FOV clamped to [min_fov, max_fov])
        target_fov = _clamp(cfg.base_fov + intent.fov_bias, cfg.min_fov, cfg.max_fov)
        fov = self._spring_fov.update(target_fov, dt, cfg.fov_freq, cfg.spring_damp)
        fov = _clamp(fov, cfg.min_fov, cfg.max_fov)

        # 17. Store debug info (no UI)
        self._debug = {
            "intent_distance_scale": round(intent.distance_scale, 3),
            "intent_height_bias":    round(intent.height_bias, 3),
            "intent_fov_bias":       round(intent.fov_bias, 2),
            "intent_roll_bias":      round(intent.roll_bias, 2),
            "intent_shake_level":    round(intent.shake_level, 3),
            "smooth_dist":           round(smooth_dist, 3),
            "smooth_height":         round(smooth_height, 3),
            "fov":                   round(fov, 2),
            "roll_deg":              round(applied_roll_deg, 3),
            "shake_intensity":       round(shake_intensity, 4),
            "cam_pos":               cam_pos,
            "look_at":               look_at,
        }

        return CameraFrame(
            cam_pos=cam_pos,
            cam_rot=cam_rot,
            fov_deg=fov,
            shake_intensity=shake_intensity,
            roll_deg=applied_roll_deg,
            tilt_deg=intent.tilt_bias,
        )

    @property
    def debug_info(self) -> dict:
        """Current camera debug state (no UI, section 19)."""
        return dict(self._debug)

    # ------------------------------------------------------------------
    # Forward tangent resolution
    # ------------------------------------------------------------------

    def _resolve_forward(self, inp: StabilityInput, up: Vec3) -> Vec3:
        vel_len = inp.velocity.length()
        if vel_len > 0.1:
            fwd = PlanetMath.tangent_forward(up, inp.velocity)
            if not fwd.is_near_zero():
                self._last_forward = fwd
                return fwd
        fwd = PlanetMath.tangent_forward(up, self._last_forward)
        if fwd.is_near_zero():
            ref = (
                Vec3(1.0, 0.0, 0.0)
                if abs(up.dot(Vec3(0.0, 1.0, 0.0))) > 0.9
                else Vec3(0.0, 1.0, 0.0)
            )
            fwd = up.cross(ref).normalized()
        return fwd

    # ------------------------------------------------------------------
    # Rotation: look-at with planet-up constraint
    # ------------------------------------------------------------------

    def _compute_rotation(self, cam_pos: Vec3, look_at: Vec3, up: Vec3) -> Quat:
        fwd = look_at - cam_pos
        if fwd.is_near_zero():
            return Quat.identity()
        fwd = fwd.normalized()
        right = fwd.cross(up)
        if right.is_near_zero():
            right = fwd.cross(Vec3(1.0, 0.0, 0.0))
        right = right.normalized()
        cam_up = right.cross(fwd).normalized()
        return _mat3_to_quat(right, cam_up, -fwd)

    # ------------------------------------------------------------------
    # Terrain collision avoidance
    # ------------------------------------------------------------------

    def _collide(self, origin: Vec3, desired: Vec3, char_pos: Vec3) -> Vec3:
        """Pull camera back if it would pass below the planet surface."""
        planet_r = char_pos.length()
        ray = desired - origin
        ray_len = ray.length()
        if ray_len < 1e-6:
            return desired
        ray_dir = ray / ray_len
        t_close = _clamp(-origin.dot(ray_dir), 0.0, ray_len)
        closest = origin + ray_dir * t_close
        min_r = planet_r + self._cfg.collision_radius
        if closest.length() >= min_r:
            return desired
        safe_t = 0.0
        lo, hi = 0.0, ray_len
        for _ in range(8):
            mid = (lo + hi) * 0.5
            p = origin + ray_dir * mid
            if p.length() > min_r:
                lo = mid
                safe_t = mid
            else:
                hi = mid
        return origin + ray_dir * safe_t

    def _bias_above_surface(self, cam_pos: Vec3, char_pos: Vec3) -> Vec3:
        planet_r = char_pos.length()
        min_r = planet_r + self._cfg.collision_radius
        if cam_pos.length() < min_r:
            cam_pos = cam_pos.normalized() * min_r
        return cam_pos


# ---------------------------------------------------------------------------
# Helper: 3×3 column-vector matrix → unit Quat
# ---------------------------------------------------------------------------

def _mat3_to_quat(right: Vec3, up: Vec3, back: Vec3) -> Quat:
    """Convert orthonormal basis (right, up, back) to unit quaternion."""
    m00, m01, m02 = right.x, up.x, back.x
    m10, m11, m12 = right.y, up.y, back.y
    m20, m21, m22 = right.z, up.z, back.z
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m21 - m12) * s
        y = (m02 - m20) * s
        z = (m10 - m01) * s
    elif m00 > m11 and m00 > m22:
        s = 2.0 * math.sqrt(1.0 + m00 - m11 - m22)
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * math.sqrt(1.0 + m11 - m00 - m22)
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = 2.0 * math.sqrt(1.0 + m22 - m00 - m11)
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    return Quat(w, x, y, z).normalized()
