"""CinematicCameraSystem — Stage 18 autonomous cinematic third-person camera.

Implements a fully automatic camera operator that:
* follows the character in 3rd-person without HUD
* selects one of three modes (Exploration / Struggle / Awe) deterministically
* uses spring-damper physics for smooth, heavy-feeling motion
* respects planetary Up (horizon always correct)
* reacts to geo-events (shake, pan)
* applies wind micro-shake
* avoids terrain penetration via sphere-cast collision

Public API
----------
update(dt, char_input, env, astro, geo_signal=None) → CameraFrame
    Advance camera by *dt* game-seconds and return the new frame.

debug_info → dict
    Current mode, target parameters, shake, and spring state (no UI).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from src.core.Config import Config
from src.math.Vec3 import Vec3
from src.math.Quat import Quat
from src.math.PlanetMath import PlanetMath
from src.systems.CharacterPhysicalController import CharacterState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _lerp_vec(a: Vec3, b: Vec3, t: float) -> Vec3:
    return a.lerp(b, t)


def _sign(x: float) -> float:
    return 1.0 if x >= 0.0 else -1.0


def _curl_noise_2d(x: float, y: float, seed: int = 0) -> float:
    """Cheap deterministic pseudo-noise in [-1, 1] using sine products."""
    s = float(seed) * 1.618033
    return math.sin(x * 7.3 + s) * math.cos(y * 5.1 + s * 0.7)


# ---------------------------------------------------------------------------
# Input / output dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CharacterInput:
    """Snapshot of character state for camera consumption."""
    position: Vec3
    velocity: Vec3
    up: Vec3                           # planet up at character position
    state: CharacterState = CharacterState.GROUNDED
    head_height: float = 1.6           # metres above position for look-at


@dataclass
class EnvContext:
    """Environmental context from ClimateSystem."""
    wind_speed: float = 0.0            # m/s
    dust: float = 0.0                  # 0–1
    visibility: float = 1.0            # 0–1  (1 = clear)
    storm_intensity: float = 0.0       # 0–1


@dataclass
class AstroContext:
    """Astronomical context from AstroSystem."""
    sun1_dir: Vec3 = field(default_factory=lambda: Vec3(0.7, 0.7, 0.0).normalized())
    sun2_dir: Vec3 = field(default_factory=lambda: Vec3(-0.5, 0.5, 0.5).normalized())
    eclipse_factor: float = 0.0        # 0=no eclipse, 1=max overlap
    ring_shadow: float = 0.0           # 0=clear, 1=full shadow
    both_suns_visible: bool = False    # True when both suns above horizon


@dataclass
class GeoSignal:
    """Camera impulse from a geo-event."""
    intensity: float = 0.0            # 0–1  (>0 = active event)
    position: Vec3 = field(default_factory=Vec3.zero)
    radius: float = 50.0              # event influence radius
    is_impact: bool = False           # True = sharp kick; False = pre-event rumble


@dataclass
class CameraFrame:
    """Output produced by CinematicCameraSystem each tick."""
    cam_pos: Vec3
    cam_rot: Quat
    fov_deg: float
    shake_intensity: float = 0.0      # for future post-fx (0–1)
    exposure_target: float = 1.0      # sRGB linear exposure scale (smoothed)


# ---------------------------------------------------------------------------
# CameraMode
# ---------------------------------------------------------------------------

class CameraMode(Enum):
    EXPLORATION = auto()
    STRUGGLE    = auto()
    AWE         = auto()


# ---------------------------------------------------------------------------
# _SpringVec3 / _SpringFloat — critically-damped spring integrators
# ---------------------------------------------------------------------------

class _SpringVec3:
    """Critically-damped spring on a Vec3 value.

    Integrates: x'' + 2ζω x' + ω² x = ω² target
    """

    def __init__(self, value: Vec3) -> None:
        self.value: Vec3 = value
        self.velocity: Vec3 = Vec3.zero()

    def update(self, target: Vec3, dt: float, freq: float, damp: float) -> Vec3:
        omega = 2.0 * math.pi * freq
        damping_factor = 1.0 + 2.0 * dt * damp * omega
        omega_sq = omega * omega
        h_omega_sq = dt * omega_sq
        det_inv = 1.0 / (damping_factor + dt * h_omega_sq)
        new_val = (self.value * damping_factor + self.velocity * dt + target * h_omega_sq * dt) * det_inv
        self.velocity = (self.velocity + (target - self.value) * h_omega_sq) * det_inv
        self.value = new_val
        return self.value


class _SpringFloat:
    """Critically-damped spring on a scalar float."""

    def __init__(self, value: float) -> None:
        self.value: float = value
        self.velocity: float = 0.0

    def update(self, target: float, dt: float, freq: float, damp: float) -> float:
        omega = 2.0 * math.pi * freq
        damping_factor = 1.0 + 2.0 * dt * damp * omega
        omega_sq = omega * omega
        h_omega_sq = dt * omega_sq
        det_inv = 1.0 / (damping_factor + dt * h_omega_sq)
        new_val = (self.value * damping_factor + self.velocity * dt + target * h_omega_sq * dt) * det_inv
        self.velocity = (self.velocity + (target - self.value) * h_omega_sq) * det_inv
        self.value = new_val
        return self.value


# ---------------------------------------------------------------------------
# _ModeParams — target parameters for a given mode
# ---------------------------------------------------------------------------

@dataclass
class _ModeParams:
    dist: float
    height: float
    fov: float


# ---------------------------------------------------------------------------
# CinematicCameraSystem
# ---------------------------------------------------------------------------

class CinematicCameraSystem:
    """Autonomous cinematic camera for Stage 18.

    Usage::

        cam = CinematicCameraSystem(config)
        frame = cam.update(dt, char_input, env, astro, geo_signal)

    All spring state is maintained internally; the caller only supplies
    per-frame input data and receives a CameraFrame.
    """

    def __init__(self, config: Optional[Config] = None) -> None:
        if config is None:
            config = Config()
        self._cfg = config
        self._load_config()

        # Current mode and timers
        self._mode: CameraMode = CameraMode.EXPLORATION
        self._mode_timer: float = 0.0       # seconds spent in current mode
        self._awe_cooldown: float = 0.0     # seconds remaining on awe cooldown

        # Spring state — position and FOV
        # Initial position: sensible default 1m above a unit-sphere surface point
        _init_pos = Vec3(0.0, 1001.8 + self._p_explore.height, 0.0)
        self._spring_pos = _SpringVec3(_init_pos)
        self._spring_fov = _SpringFloat(self._p_explore.fov)

        # Smoothed exposure
        self._exposure_spring = _SpringFloat(1.0)

        # Last known character forward (for low-speed framing)
        self._last_forward: Vec3 = Vec3(0.0, 0.0, -1.0)

        # Smoothed mode param blending
        self._cur_dist:   float = self._p_explore.dist
        self._cur_height: float = self._p_explore.height

        # Geo-event accumulated shake state
        self._shake_vel: float = 0.0
        self._shake_pos: float = 0.0   # oscillating scalar (drives offsets)
        self._pan_angle: float = 0.0   # current pan toward event (radians)
        self._pan_spring = _SpringFloat(0.0)

        # Accumulated time (for noise seeding)
        self._time: float = 0.0

        # Debug info storage
        self._debug: dict = {}

    # ------------------------------------------------------------------
    # Config loading
    # ------------------------------------------------------------------

    def _load_config(self) -> None:
        c = self._cfg

        # Mode parameters
        self._p_explore = _ModeParams(
            dist=c.get("cam", "explore", "dist", default=4.5),
            height=c.get("cam", "explore", "height", default=1.8),
            fov=c.get("cam", "explore", "fov", default=62.0),
        )
        self._p_struggle = _ModeParams(
            dist=c.get("cam", "struggle", "dist", default=2.8),
            height=c.get("cam", "struggle", "height", default=1.2),
            fov=c.get("cam", "struggle", "fov", default=58.0),
        )
        self._p_awe = _ModeParams(
            dist=c.get("cam", "awe", "dist", default=7.0),
            height=c.get("cam", "awe", "height", default=3.2),
            fov=c.get("cam", "awe", "fov", default=68.0),
        )

        # Spring parameters
        self._freq_pos: float = c.get("cam", "spring", "freq_pos", default=3.0)
        self._damp_pos: float = c.get("cam", "spring", "damp_pos", default=0.9)
        self._freq_rot: float = c.get("cam", "spring", "freq_rot", default=4.0)
        self._damp_rot: float = c.get("cam", "spring", "damp_rot", default=0.85)

        # Collision
        self._collision_radius: float = c.get("cam", "collision_radius", default=0.25)

        # Shake / pan
        self._wind_shake_k: float = c.get("cam", "wind_shake_k", default=0.004)
        self._geo_shake_pre_k: float = c.get("cam", "geo_shake_k_pre", default=0.015)
        self._geo_shake_impact_k: float = c.get("cam", "geo_shake_k_impact", default=0.06)
        self._pan_max_rad: float = math.radians(
            c.get("cam", "pan_to_event_max_deg", default=15.0)
        )

        # Shoulder offset
        self._shoulder: float = c.get("cam", "shoulder_offset", default=0.35)

        # Mode switch cooldown
        self._mode_cooldown: float = c.get("cam", "mode_cooldown_sec", default=5.0)
        self._awe_total_cooldown: float = c.get("cam", "awe", "cooldown_sec", default=90.0)
        self._awe_min_hold: float = c.get("cam", "awe", "min_hold_sec", default=8.0)

        # Hysteresis thresholds
        self._struggle_enter_vis: float = c.get("cam", "struggle_enter_visibility", default=0.35)
        self._struggle_exit_vis: float = c.get("cam", "struggle_exit_visibility", default=0.45)
        self._struggle_enter_storm: float = c.get("cam", "struggle_enter_storm", default=0.6)
        self._struggle_exit_storm: float = c.get("cam", "struggle_exit_storm", default=0.45)
        self._awe_enter_eclipse: float = c.get("cam", "awe_enter_eclipse", default=0.7)
        self._awe_enter_vis: float = c.get("cam", "awe_enter_visibility", default=0.5)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        dt: float,
        char: CharacterInput,
        env: EnvContext,
        astro: AstroContext,
        geo: Optional[GeoSignal] = None,
    ) -> CameraFrame:
        """Advance camera by *dt* seconds and return the new CameraFrame."""
        dt = _clamp(dt, 1e-6, 0.1)   # guard against spiral-of-death
        self._time += dt

        # 1. Update mode
        self._update_mode(dt, char, env, astro)

        # 2. Resolve target params for this mode (smoothly blended)
        target_params = self._mode_params()
        blend_speed = 3.0 * dt
        self._cur_dist   = _lerp(self._cur_dist,   target_params.dist,   blend_speed)
        self._cur_height = _lerp(self._cur_height, target_params.height, blend_speed)

        # 3. Planet Up at character
        up = PlanetMath.up_at_position(char.position)
        if up.is_near_zero():
            up = char.up if not char.up.is_near_zero() else Vec3(0.0, 1.0, 0.0)

        # 4. Forward tangent (direction character is moving / last intent)
        fwd = self._resolve_forward(char, up)

        # 5. Desired camera position (boom)
        look_at = char.position + up * char.head_height
        # In Struggle, look at legs/ground more (look_at lower)
        if self._mode == CameraMode.STRUGGLE:
            look_at = char.position + up * (char.head_height * 0.5)
        # In Awe, keep look_at at head
        side = up.cross(fwd).normalized()

        desired_cam = (
            char.position
            - fwd * self._cur_dist
            + up * self._cur_height
            + side * self._shoulder
        )

        # 6. Collision avoidance: sphere-cast from look_at to desired_cam
        corrected_cam = self._collide(look_at, desired_cam, up)

        # 7. Spring toward corrected position
        smooth_pos = self._spring_pos.update(
            corrected_cam, dt, self._freq_pos, self._damp_pos
        )

        # 8. Compute shake offset
        shake_offset, shake_intensity = self._compute_shake(dt, env, geo, up)

        # 9. Final camera position (shake added in world-tangent space)
        cam_pos = smooth_pos + shake_offset

        # Ensure camera stays above planet surface
        cam_pos = self._bias_above_surface(cam_pos, up, char.position)

        # 10. Camera rotation: look at (smoothed look_at via pan)
        pan_fwd = self._compute_pan(dt, char, geo, fwd, up)
        pan_look_at = look_at + pan_fwd * 0.5

        cam_rot = self._compute_rotation(cam_pos, pan_look_at, up)

        # 11. FOV spring
        fov = self._spring_fov.update(
            target_params.fov, dt, self._freq_rot * 0.5, self._damp_rot
        )

        # 12. Exposure target
        exposure = self._compute_exposure(astro, env)
        smooth_exp = self._exposure_spring.update(exposure, dt, 0.5, 1.0)

        # 13. Debug
        self._debug = {
            "mode": self._mode.name,
            "mode_timer": round(self._mode_timer, 2),
            "awe_cooldown": round(self._awe_cooldown, 2),
            "dist": round(self._cur_dist, 3),
            "height": round(self._cur_height, 3),
            "fov": round(fov, 2),
            "shake_intensity": round(shake_intensity, 4),
            "desired_cam_pos": desired_cam,
            "corrected_cam_pos": corrected_cam,
            "look_at": look_at,
            "up": up,
        }

        return CameraFrame(
            cam_pos=cam_pos,
            cam_rot=cam_rot,
            fov_deg=fov,
            shake_intensity=shake_intensity,
            exposure_target=smooth_exp,
        )

    @property
    def debug_info(self) -> dict:
        """Current camera debug state (no UI)."""
        return dict(self._debug)

    @property
    def mode(self) -> CameraMode:
        return self._mode

    # ------------------------------------------------------------------
    # Mode selection with hysteresis
    # ------------------------------------------------------------------

    def _update_mode(
        self,
        dt: float,
        char: CharacterInput,
        env: EnvContext,
        astro: AstroContext,
    ) -> None:
        self._mode_timer += dt
        if self._awe_cooldown > 0.0:
            self._awe_cooldown -= dt

        struggling = (
            char.state in (CharacterState.SLIDING, CharacterState.STUMBLING,
                           CharacterState.FALLING_CONTROLLED)
            or env.visibility < self._struggle_enter_vis
            or env.storm_intensity > self._struggle_enter_storm
        )
        awe_trigger = (
            astro.eclipse_factor > self._awe_enter_eclipse
            and env.visibility > self._awe_enter_vis
            and self._awe_cooldown <= 0.0
        )

        current = self._mode
        min_hold = self._mode_cooldown

        if current == CameraMode.EXPLORATION:
            if awe_trigger:
                self._switch_mode(CameraMode.AWE)
            elif struggling:
                self._switch_mode(CameraMode.STRUGGLE)

        elif current == CameraMode.STRUGGLE:
            if self._mode_timer >= min_hold:
                # Check exit conditions (hysteresis: higher thresholds to exit)
                can_exit = (
                    env.visibility > self._struggle_exit_vis
                    and env.storm_intensity < self._struggle_exit_storm
                    and char.state not in (CharacterState.SLIDING, CharacterState.STUMBLING,
                                           CharacterState.FALLING_CONTROLLED)
                )
                if can_exit:
                    if awe_trigger:
                        self._switch_mode(CameraMode.AWE)
                    else:
                        self._switch_mode(CameraMode.EXPLORATION)

        elif current == CameraMode.AWE:
            if self._mode_timer >= self._awe_min_hold:
                if struggling:
                    self._switch_mode(CameraMode.STRUGGLE)
                elif astro.eclipse_factor <= self._awe_enter_eclipse * 0.7:
                    # Eclipse fading — return to exploration
                    self._switch_mode(CameraMode.EXPLORATION)

    def _switch_mode(self, new_mode: CameraMode) -> None:
        if new_mode == self._mode:
            return
        if new_mode == CameraMode.EXPLORATION:
            pass   # no cooldown
        elif new_mode == CameraMode.AWE:
            pass   # cooldown applied on leaving AWE
        # Start cooldown when LEAVING awe
        if self._mode == CameraMode.AWE:
            self._awe_cooldown = self._awe_total_cooldown
        self._mode = new_mode
        self._mode_timer = 0.0

    def _mode_params(self) -> _ModeParams:
        if self._mode == CameraMode.EXPLORATION:
            return self._p_explore
        elif self._mode == CameraMode.STRUGGLE:
            return self._p_struggle
        else:
            return self._p_awe

    # ------------------------------------------------------------------
    # Forward tangent resolution
    # ------------------------------------------------------------------

    def _resolve_forward(self, char: CharacterInput, up: Vec3) -> Vec3:
        """Return the camera's forward tangent direction (character's facing)."""
        vel_len = char.velocity.length()
        if vel_len > 0.1:
            fwd = PlanetMath.tangent_forward(up, char.velocity)
            if not fwd.is_near_zero():
                self._last_forward = fwd
                return fwd
        # Fall back to last known forward
        fwd = PlanetMath.tangent_forward(up, self._last_forward)
        if fwd.is_near_zero():
            # Construct a default tangent
            ref = Vec3(1.0, 0.0, 0.0) if abs(up.dot(Vec3(0.0, 1.0, 0.0))) > 0.9 else Vec3(0.0, 1.0, 0.0)
            fwd = up.cross(ref).normalized()
        return fwd

    # ------------------------------------------------------------------
    # Rotation: look at target, constrain up
    # ------------------------------------------------------------------

    def _compute_rotation(self, cam_pos: Vec3, look_at: Vec3, up: Vec3) -> Quat:
        """Build a quaternion that looks from cam_pos toward look_at with planet up."""
        fwd = (look_at - cam_pos)
        if fwd.is_near_zero():
            return Quat.identity()
        fwd = fwd.normalized()

        # Re-orthogonalise: up ← up - fwd*(fwd·up)
        right = fwd.cross(up)
        if right.is_near_zero():
            right = fwd.cross(Vec3(1.0, 0.0, 0.0))
        right = right.normalized()
        cam_up = right.cross(fwd).normalized()

        # Build rotation matrix → quaternion
        # Column vectors: right, cam_up, -fwd  (OpenGL convention)
        return _mat3_to_quat(right, cam_up, -fwd)

    # ------------------------------------------------------------------
    # Shake
    # ------------------------------------------------------------------

    def _compute_shake(
        self,
        dt: float,
        env: EnvContext,
        geo: Optional[GeoSignal],
        up: Vec3,
    ) -> tuple[Vec3, float]:
        """Return (world-space shake offset, scalar intensity 0-1)."""
        total = 0.0

        # Wind micro-shake
        wind_amp = env.wind_speed * self._wind_shake_k
        total += wind_amp

        # Geo pre/impact shake
        if geo is not None and geo.intensity > 0.0:
            if geo.is_impact:
                k = self._geo_shake_impact_k
                # Damped kick: decays quickly
                self._shake_vel += geo.intensity * k * 60.0
            else:
                k = self._geo_shake_pre_k
                total += geo.intensity * k

        # Integrate shake oscillator (spring-damper with low damping)
        omega_shake = 2.0 * math.pi * 12.0   # ~12 Hz shake
        self._shake_vel -= (omega_shake * omega_shake * self._shake_pos + 2.0 * 0.25 * omega_shake * self._shake_vel) * dt
        self._shake_pos += self._shake_vel * dt

        # Total amplitude
        intensity = _clamp(abs(self._shake_pos) + total, 0.0, 1.0)

        # Direction: curl noise-based tangent perturbation
        right = up.cross(self._last_forward).normalized()
        n1 = _curl_noise_2d(self._time * 7.3, 0.0)
        n2 = _curl_noise_2d(0.0, self._time * 5.1)
        offset = (right * n1 + self._last_forward * n2) * intensity * 0.5
        return offset, intensity

    # ------------------------------------------------------------------
    # Pan toward geo-event
    # ------------------------------------------------------------------

    def _compute_pan(
        self,
        dt: float,
        char: CharacterInput,
        geo: Optional[GeoSignal],
        fwd: Vec3,
        up: Vec3,
    ) -> Vec3:
        """Return a (potentially panned) look-at forward direction."""
        if geo is None or geo.intensity < 0.05:
            # Decay pan back to zero
            self._pan_spring.update(0.0, dt, 2.0, 0.8)
            target_angle = 0.0
        else:
            # Direction to geo-event from character
            to_event = geo.position - char.position
            to_event_tan = PlanetMath.tangent_forward(up, to_event)
            if to_event_tan.is_near_zero():
                self._pan_spring.update(0.0, dt, 2.0, 0.8)
                target_angle = 0.0
            else:
                # Signed angle between fwd and to_event_tan around up axis
                dot = _clamp(fwd.dot(to_event_tan), -1.0, 1.0)
                cross_y = fwd.cross(to_event_tan).dot(up)
                raw_angle = math.atan2(cross_y, dot)
                target_angle = _clamp(raw_angle, -self._pan_max_rad, self._pan_max_rad)
                self._pan_spring.update(target_angle, dt, 2.0, 0.8)

        # Apply smoothed pan angle as rotation of fwd around up
        angle = self._pan_spring.value
        if abs(angle) < 1e-6:
            return fwd
        q = Quat.from_axis_angle(up, angle)
        return q.rotate_vec(fwd)

    # ------------------------------------------------------------------
    # Collision avoidance
    # ------------------------------------------------------------------

    def _collide(self, origin: Vec3, desired: Vec3, up: Vec3) -> Vec3:
        """Move camera toward origin if terrain blocks the line to desired.

        Simple sphere-cast approximation: check if desired is too close to
        the planet surface; if so, pull back along the origin→desired ray.
        """
        planet_r = origin.length()   # approximate: use character's orbital radius
        # Find the closest point on ray (origin→desired) to planet center
        ray = desired - origin
        ray_len = ray.length()
        if ray_len < 1e-6:
            return desired

        ray_dir = ray / ray_len
        # t at closest approach to origin (planet center = 0)
        t_close = _clamp(-origin.dot(ray_dir), 0.0, ray_len)
        closest = origin + ray_dir * t_close
        dist_to_center = closest.length()
        min_r = planet_r + self._collision_radius

        if dist_to_center >= min_r:
            return desired   # no collision

        # Pull safe: binary-search along ray for a safe t
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

    # ------------------------------------------------------------------
    # Bias above surface
    # ------------------------------------------------------------------

    def _bias_above_surface(self, cam_pos: Vec3, up: Vec3, char_pos: Vec3) -> Vec3:
        """Ensure the camera stays at least collision_radius above the planet surface."""
        planet_r = char_pos.length()
        cam_r = cam_pos.length()
        min_r = planet_r + self._collision_radius
        if cam_r < min_r:
            cam_pos = cam_pos.normalized() * min_r
        return cam_pos

    # ------------------------------------------------------------------
    # Exposure target
    # ------------------------------------------------------------------

    def _compute_exposure(self, astro: AstroContext, env: EnvContext) -> float:
        """Compute target linear exposure scale (not applied to frame; for post-fx)."""
        # Base: bright day
        base = 1.0
        # Eclipse dims the scene
        base *= 1.0 - astro.eclipse_factor * 0.5
        # Ring shadow dims slightly
        base *= 1.0 - astro.ring_shadow * 0.15
        # Storm/dust = milky-white lift (higher exposure for diffuse haze)
        if env.visibility < 0.4:
            whiteout = (0.4 - env.visibility) / 0.4
            base = _lerp(base, 1.3, whiteout * 0.6)
        return _clamp(base, 0.2, 2.0)


# ---------------------------------------------------------------------------
# Helper: 3×3 orthonormal matrix (column vectors) → Quat
# ---------------------------------------------------------------------------

def _mat3_to_quat(right: Vec3, up: Vec3, back: Vec3) -> Quat:
    """Convert a right-handed orthonormal basis (column vectors) to a unit Quat.

    right = +X axis, up = +Y axis, back = +Z axis in camera space.
    """
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
