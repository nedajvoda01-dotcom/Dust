"""CharacterPhysicalController — Stage 13 physical character controller.

Represents a capsule-shaped character body that:
  * stays on the surface of a spherical planet (gravity toward center)
  * detects ground contact and corrects orientation ("Up" = away from center)
  * accepts movement *intent* (desired direction + speed) in the tangent plane
  * responds to environmental forces: wind drag, dust viscosity, ice slip
  * transitions between states: Grounded, Airborne, Sliding, Stumbling
  * reacts to GeoEventSignal IMPACT with a stumble impulse

No animation, no IK, no ragdoll, no jump/action buttons.

Public API
----------
update(dt, desired_dir, desired_speed)   — advance physics one tick
position  → Vec3                         — current world position
velocity  → Vec3                         — current velocity
state     → CharacterState               — current physics state
orientation → Quat                       — current orientation (Up-aligned)
debug_info → dict                        — gizmo / log data (no UI)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from src.math.PlanetMath import PlanetMath
from src.math.Quat import Quat
from src.math.Vec3 import Vec3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


# ---------------------------------------------------------------------------
# CharacterState
# ---------------------------------------------------------------------------

class CharacterState(Enum):
    GROUNDED           = auto()   # on surface, can move normally
    AIRBORNE           = auto()   # not touching ground
    SLIDING            = auto()   # on too-steep slope or too-slippery surface
    STUMBLING          = auto()   # destabilised by geo-event or gust
    BRACED             = auto()   # leaning on a surface for support
    CROUCHED           = auto()   # crouching (e.g. against gust)
    HANGING            = auto()   # hanging from a ledge after grab
    FALLING_CONTROLLED = auto()   # controlled fall / unrecoverable slide


# ---------------------------------------------------------------------------
# GroundHit — result of a ground query
# ---------------------------------------------------------------------------

@dataclass
class GroundHit:
    """Result from the ground sampler."""
    hit:      bool         # True if ground found within probe distance
    normal:   Vec3         # surface normal (world space), valid when hit=True
    point:    Vec3         # ground contact point (world space)
    distance: float        # signed distance: positive = above surface


# ---------------------------------------------------------------------------
# IGroundSampler — abstract interface for ground detection
# ---------------------------------------------------------------------------

class IGroundSampler:
    """Abstract ground detection interface.

    Subclass and override ``query_ground`` to plug in SDF / mesh queries.
    The default implementation uses a perfect sphere of *planet_radius*.
    """

    def __init__(self, planet_radius: float = 1000.0) -> None:
        self.planet_radius = planet_radius

    def query_ground(self, world_pos: Vec3, probe_dist: float) -> GroundHit:
        """Return ground information at *world_pos*.

        Default: perfect-sphere ground.
        """
        up    = world_pos.normalized()
        r_pos = world_pos.length()
        dist  = r_pos - self.planet_radius
        point = up * self.planet_radius
        return GroundHit(
            hit      = dist <= probe_dist,
            normal   = up,
            point    = point,
            distance = dist,
        )


# ---------------------------------------------------------------------------
# EnvironmentSampler — adapter over ClimateSystem + GeoEventSystem
# ---------------------------------------------------------------------------

class EnvironmentSampler:
    """Wraps ClimateSystem and GeoEventSystem to provide per-position samples.

    Parameters
    ----------
    climate:
        Any object with ``sample_wind(Vec3)->Vec3``, ``sample_dust(Vec3)->float``,
        and optionally ``get_wetness(Vec3)->float`` and
        ``sample_temperature(Vec3)->float``.  Pass *None* for calm conditions.
    geo_events:
        Any object with ``query_signals_near(Vec3, float)->list``.
        Pass *None* for no geo signals.
    freeze_threshold:
        Temperature (K) below which surface is considered icy.
    """

    def __init__(
        self,
        climate=None,
        geo_events=None,
        freeze_threshold: float = 270.0,
    ) -> None:
        self._climate          = climate
        self._geo              = geo_events
        self._freeze_threshold = freeze_threshold

    # --- Climate proxies ---

    def sample_wind(self, world_pos: Vec3) -> Vec3:
        """Wind velocity vector (m/s) at *world_pos*."""
        if self._climate is None:
            return Vec3.zero()
        return self._climate.sample_wind(world_pos)

    def sample_dust(self, world_pos: Vec3) -> float:
        """Dust suspension [0, 1] at *world_pos*."""
        if self._climate is None:
            return 0.0
        return _clamp(self._climate.sample_dust(world_pos), 0.0, 1.0)

    def sample_ice_wetness(self, world_pos: Vec3) -> float:
        """Ice/wetness factor [0, 1] — higher → slippery surface."""
        if self._climate is None:
            return 0.0
        # Use get_wetness if available, otherwise fall back to temperature proxy
        if hasattr(self._climate, "get_wetness"):
            wetness = _clamp(self._climate.get_wetness(world_pos), 0.0, 1.0)
        else:
            wetness = 0.0
        if hasattr(self._climate, "sample_temperature"):
            t   = self._climate.sample_temperature(world_pos)
            ice = _clamp((self._freeze_threshold - t) / 30.0, 0.0, 1.0)
        else:
            ice = 0.0
        return _clamp(wetness * 0.4 + ice * 0.8, 0.0, 1.0)

    def sample_ground_material(self, world_pos: Vec3) -> dict:
        """Return a dict with proxy friction and viscosity values.

        Keys:
          'friction'   — base friction coefficient [0, 1]
          'viscosity'  — movement resistance [0, 1] (deep dust / snow)
        """
        dust      = self.sample_dust(world_pos)
        ice_slip  = self.sample_ice_wetness(world_pos)
        # Full ice (ice_slip=1) → near-zero friction; full dust → moderate slip
        friction  = _clamp(1.0 - 0.95 * ice_slip - 0.15 * dust, 0.01, 1.0)
        viscosity = _clamp(dust * 0.6, 0.0, 1.0)
        return {"friction": friction, "viscosity": viscosity}

    # --- Geo event proxies ---

    def query_geo_signals(self, world_pos: Vec3, radius: float = 500.0):
        """Return active GeoEventSignal list near *world_pos*."""
        if self._geo is None:
            return []
        return self._geo.query_signals_near(world_pos, radius)


# ---------------------------------------------------------------------------
# CharacterPhysicalController
# ---------------------------------------------------------------------------

class CharacterPhysicalController:
    """Physical body controller for a capsule-shaped character on a sphere.

    Parameters
    ----------
    position:
        Initial world-space position (must be outside planet radius).
    planet_radius:
        Radius of the planet (simulation units).
    config:
        Optional Config object.  When *None*, hard-coded defaults are used.
    ground_sampler:
        IGroundSampler implementation.  Defaults to perfect-sphere sampler.
    env_sampler:
        EnvironmentSampler instance.  Defaults to no-climate sampler.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        position:       Vec3,
        planet_radius:  float = 1000.0,
        config=None,
        ground_sampler: Optional[IGroundSampler] = None,
        env_sampler:    Optional[EnvironmentSampler] = None,
        reflex_system=None,
    ) -> None:
        self._planet_r = planet_radius

        # --- Physical state ---
        self.position:    Vec3  = Vec3(position.x, position.y, position.z)
        self.velocity:    Vec3  = Vec3.zero()
        self.orientation: Quat  = Quat.identity()
        self.state:       CharacterState = CharacterState.AIRBORNE

        # --- Capsule geometry ---
        self.capsule_radius: float = _cfg(config, "char", "capsule_radius", 0.4)
        self.capsule_height: float = _cfg(config, "char", "capsule_height", 1.8)

        # --- Physics parameters ---
        self.mass:              float = _cfg(config, "char", "mass",              75.0)
        self.gravity:           float = _cfg(config, "char",  "gravity",           9.2)
        self.max_speed:         float = _cfg(config, "char",  "max_speed",         6.0)
        self.accel:             float = _cfg(config, "char",  "accel",             8.0)
        self.ground_friction:   float = _cfg(config, "char",  "base_friction",     0.8)
        self.air_drag:          float = _cfg(config, "char",  "air_drag",          0.1)
        self.wind_drag:         float = _cfg(config, "char",  "wind_drag",         0.4)
        self.max_slope_angle:   float = math.radians(
            _cfg(config, "char", "max_slope_deg", 40.0))
        self.slide_friction:    float = _cfg(config, "char",  "slide_friction",    0.15)
        self.stumble_threshold: float = _cfg(config, "char",  "stumble_thresholds_shock", 0.3)
        self.stumble_min:       float = _cfg(config, "char",  "stumble_duration_min",     0.2)
        self.stumble_max:       float = _cfg(config, "char",  "stumble_duration_max",     0.8)
        self.gust_strength:     float = _cfg(config, "char",  "gust_strength",     2.5)
        self.ice_friction_mul:  float = _cfg(config, "char",  "ice_friction_multiplier",  0.15)
        self.dust_slip_mul:     float = _cfg(config, "char",  "dust_slip_multiplier",     0.25)

        # Ground probe distance (small gap so we detect ground before embedding)
        self._probe_dist: float = self.capsule_radius + 0.3

        # --- Environment modifier scales (set by CharacterEnvironmentIntegration) ---
        self._speed_scale:              float = 1.0
        self._accel_scale:              float = 1.0
        self._turn_responsiveness:      float = 1.0
        self._effective_friction_scale: float = 1.0
        self._wind_drag_scale:          float = 1.0

        # --- Sub-systems ---
        self._ground    = ground_sampler or IGroundSampler(planet_radius)
        self._env       = env_sampler    or EnvironmentSampler()
        self._reflex    = reflex_system  # Optional[ReflexSystem]

        # --- Stumble timer ---
        self._stumble_remaining: float = 0.0

        # --- Debug info (filled each tick) ---
        self._debug: dict = {}

        # --- Last ground hit ---
        self._last_ground: GroundHit = GroundHit(False, Vec3.zero(), Vec3.zero(), 999.0)

        # Align orientation immediately
        self._align_orientation(1.0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        dt:            float,
        desired_dir:   Vec3  = Vec3(0.0, 0.0, 0.0),
        desired_speed: float = 0.0,
    ) -> None:
        """Advance the character physics by *dt* seconds.

        Parameters
        ----------
        dt:
            Elapsed game-time in seconds.
        desired_dir:
            World-space direction the player *wants* to move.
            Need not be normalised; zero vector means no movement intent.
        desired_speed:
            Desired speed magnitude [0, max_speed].
        """
        if dt <= 0.0:
            return

        # 1. Compute local Up/Down
        up   = PlanetMath.up_at_position(self.position)
        down = -up

        # 2. Ground query
        gnd  = self._ground.query_ground(self.position, self._probe_dist)
        self._last_ground = gnd

        # 3. Determine state (transitions)
        self._update_state(gnd, up, dt)

        # 4. Apply gravity
        self._apply_gravity(gnd, up, dt)

        # 5. Sample environment
        wind     = self._env.sample_wind(self.position)
        material = self._env.sample_ground_material(self.position)
        dust     = self._env.sample_dust(self.position)
        mu       = material["friction"]
        viscosity = material["viscosity"]

        # 6. Apply wind force
        self._apply_wind(wind, dust, dt)

        # 7. Handle geo-event signals (stumble / impulse)
        self._handle_geo_signals(dt)

        # 8. Movement intent
        self._apply_movement(desired_dir, desired_speed, up, mu, viscosity, dt)

        # 9. Stumble drift
        if self.state == CharacterState.STUMBLING:
            self._apply_stumble_drift(up, dt)

        # 9b. ReflexSystem (between intent and final velocity integration)
        if self._reflex is not None:
            self._reflex.update(self, dt, wind, mu, gnd)

        # 10. Speed cap
        self._cap_speed(up)

        # 11. Integrate position
        self.position = self.position + self.velocity * dt

        # 12. Ground correction (prevent sinking)
        self._correct_position(gnd, up)

        # 13. Align orientation
        self._align_orientation(dt)

        # 14. Fill debug info
        self._debug = {
            "up":           up,
            "ground_normal": gnd.normal if gnd.hit else up,
            "wind":         wind,
            "state":        self.state.name,
            "mu":           mu,
            "slope_angle":  self._slope_angle(gnd, up),
            "speed":        self.velocity.length(),
            "balance":      self._reflex.balance_model.balance if self._reflex else 1.0,
        }

    @property
    def debug_info(self) -> dict:
        """Gizmo / log data filled after each update()."""
        return dict(self._debug)

    def set_ground_sampler(self, sampler: IGroundSampler) -> None:
        """Replace the ground sampler (e.g., to switch terrain mid-simulation)."""
        self._ground = sampler

    # Environment modifier setters (called by CharacterEnvironmentIntegration)

    def set_speed_scale(self, x: float) -> None:
        """Set speed scale [0..1] from environment (wind, dust, slope, etc.)."""
        self._speed_scale = _clamp(x, 0.0, 1.0)

    def set_accel_scale(self, x: float) -> None:
        """Set acceleration scale [0..1] from environment."""
        self._accel_scale = _clamp(x, 0.0, 1.0)

    def set_effective_friction_scale(self, x: float) -> None:
        """Set friction scale [0..2] from environment (ice lowers, rough raises)."""
        self._effective_friction_scale = _clamp(x, 0.0, 2.0)

    def set_wind_drag_scale(self, x: float) -> None:
        """Set wind drag scale [0..2] — storm/dust boosts drag."""
        self._wind_drag_scale = _clamp(x, 0.0, 2.0)

    def set_turn_responsiveness(self, x: float) -> None:
        """Set turn responsiveness [0..1] — whiteout reduces agility."""
        self._turn_responsiveness = _clamp(x, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Private: state transitions
    # ------------------------------------------------------------------

    def _update_state(self, gnd: GroundHit, up: Vec3, dt: float) -> None:
        """Evaluate and update CharacterState."""

        # Stumble timer
        if self.state == CharacterState.STUMBLING:
            self._stumble_remaining -= dt
            if self._stumble_remaining <= 0.0:
                self._stumble_remaining = 0.0
                # Fall back to grounded or airborne after stumble
                self.state = CharacterState.GROUNDED if gnd.hit else CharacterState.AIRBORNE
            return   # don't override stumble until timer expires

        if not gnd.hit:
            self.state = CharacterState.AIRBORNE
            return

        slope = self._slope_angle(gnd, up)
        mat   = self._env.sample_ground_material(self.position)
        mu    = mat["friction"]

        # Slide condition: too steep OR too slippery
        if slope > self.max_slope_angle or mu < 0.12:
            self.state = CharacterState.SLIDING
        else:
            self.state = CharacterState.GROUNDED

    def _slope_angle(self, gnd: GroundHit, up: Vec3) -> float:
        """Angle (radians) between ground normal and local Up."""
        if not gnd.hit:
            return 0.0
        cos_a = _clamp(gnd.normal.dot(up), -1.0, 1.0)
        return math.acos(cos_a)

    # ------------------------------------------------------------------
    # Private: gravity
    # ------------------------------------------------------------------

    def _apply_gravity(self, gnd: GroundHit, up: Vec3, dt: float) -> None:
        """Apply gravitational acceleration toward planet center."""
        down = -up
        grav_accel = down * self.gravity

        if gnd.hit and self.state != CharacterState.AIRBORNE:
            # On ground: remove velocity component into the ground normal
            normal = gnd.normal
            v_into = min(0.0, self.velocity.dot(normal))
            self.velocity = self.velocity - normal * v_into
        else:
            # Airborne: apply full gravity
            self.velocity = self.velocity + grav_accel * dt

    # ------------------------------------------------------------------
    # Private: wind
    # ------------------------------------------------------------------

    def _apply_wind(self, wind: Vec3, dust: float, dt: float) -> None:
        """Apply wind force scaled by dust density (heavier dust storms)."""
        dust_factor = 1.0 + dust * 1.5
        force       = wind * (self.wind_drag * dust_factor * self._wind_drag_scale)
        self.velocity = self.velocity + (force / self.mass) * dt

    # ------------------------------------------------------------------
    # Private: geo-event signals
    # ------------------------------------------------------------------

    def _handle_geo_signals(self, dt: float) -> None:
        """Process nearby geo-event signals: vibration in PRE, impulse in IMPACT."""
        # Import here to avoid circular dependency; these are lightweight enums
        try:
            from src.systems.GeoEventSystem import GeoEventPhase
        except ImportError:
            return

        signals = self._env.query_geo_signals(self.position, 500.0)
        for sig in signals:
            if sig.phase == GeoEventPhase.PRE:
                # Micro-vibration proportional to growing intensity
                vib_amp = 0.05 * sig.intensity
                # Deterministic based on position hash
                h = (hash((round(self.position.x, 1),
                           round(self.position.z, 1),
                           round(dt, 3))) & 0xFF) / 255.0
                self.velocity = self.velocity + Vec3(
                    (h * 2.0 - 1.0) * vib_amp,
                    0.0,
                    ((h * 137.0 % 1.0) * 2.0 - 1.0) * vib_amp,
                )

            elif sig.phase == GeoEventPhase.IMPACT:
                # Direction away from event (shock wave)
                shock_dir = (self.position - sig.position)
                if not shock_dir.is_near_zero():
                    shock_dir = shock_dir.normalized()
                else:
                    shock_dir = Vec3(0.0, 1.0, 0.0)
                impulse_mag = sig.intensity * 4.0
                self.velocity = self.velocity + shock_dir * impulse_mag

                # Trigger stumble if impulse is strong enough
                if sig.intensity >= self.stumble_threshold:
                    duration = self.stumble_min + (
                        self.stumble_max - self.stumble_min
                    ) * _clamp(sig.intensity, 0.0, 1.0)
                    if self._stumble_remaining < duration:
                        self._stumble_remaining = duration
                        self.state = CharacterState.STUMBLING

    # ------------------------------------------------------------------
    # Private: movement intent
    # ------------------------------------------------------------------

    def _apply_movement(
        self,
        desired_dir:   Vec3,
        desired_speed: float,
        up:            Vec3,
        mu:            float,
        viscosity:     float,
        dt:            float,
    ) -> None:
        """Apply player intent (tangent-plane constrained) and friction."""

        if self.state == CharacterState.AIRBORNE:
            # Air drag only — no direct control
            drag_k = self.air_drag
            self.velocity = self.velocity * max(0.0, 1.0 - drag_k * dt)
            return

        if self.state == CharacterState.SLIDING:
            self._apply_slide(up, mu, dt)
            # Limited lateral steering while sliding
            self._apply_limited_steering(desired_dir, desired_speed, up, mu, dt, factor=0.25)
            return

        if self.state == CharacterState.STUMBLING:
            # Very limited control while stumbling
            self._apply_limited_steering(desired_dir, desired_speed, up, mu, dt, factor=0.1)
            self._apply_tangent_friction(up, mu, viscosity, dt)
            return

        # --- Grounded normal movement ---
        # Project desired direction onto tangent plane
        if not desired_dir.is_near_zero() and desired_speed > 0.0:
            tang_dir = _project_tangent(desired_dir, up)
            if not tang_dir.is_near_zero():
                eff_speed = min(desired_speed, self.max_speed) * self._speed_scale
                v_desired = tang_dir * eff_speed
                # Tangent component of current velocity
                v_tang = _project_tangent(self.velocity, up)
                # Acceleration toward desired
                dv = v_desired - v_tang
                dv_mag = dv.length()
                if dv_mag > 1e-6:
                    accel_step = min(self.accel * self._accel_scale * dt, dv_mag)
                    self.velocity = self.velocity + dv.normalized() * accel_step

        # Friction
        self._apply_tangent_friction(up, mu, viscosity, dt)

    def _apply_slide(self, up: Vec3, mu: float, dt: float) -> None:
        """Accelerate down-slope during slide state."""
        gnd    = self._last_ground
        normal = gnd.normal if gnd.hit else up

        # Down-slope direction: project Down onto the surface plane
        down = -up
        down_slope = _project_tangent(down, normal)
        if down_slope.is_near_zero():
            return
        down_slope = down_slope.normalized()

        # Slide acceleration from gravity component along slope
        slope_angle   = self._slope_angle(gnd, up)
        slide_accel   = self.gravity * math.sin(slope_angle)
        self.velocity = self.velocity + down_slope * slide_accel * dt

        # Slide friction
        eff_slide_friction = self.slide_friction * mu
        v_tang = _project_tangent(self.velocity, normal)
        v_tang_mag = v_tang.length()
        if v_tang_mag > 1e-6:
            friction_decel = eff_slide_friction * self.gravity * dt
            reduction      = min(friction_decel, v_tang_mag)
            self.velocity  = self.velocity - v_tang.normalized() * reduction

    def _apply_limited_steering(
        self,
        desired_dir:   Vec3,
        desired_speed: float,
        up:            Vec3,
        mu:            float,
        dt:            float,
        factor:        float = 0.25,
    ) -> None:
        """Apply a reduced-authority version of normal movement intent."""
        if desired_dir.is_near_zero() or desired_speed <= 0.0:
            return
        tang_dir = _project_tangent(desired_dir, up)
        if tang_dir.is_near_zero():
            return
        eff_speed = min(desired_speed, self.max_speed) * self._speed_scale
        v_desired = tang_dir * eff_speed
        v_tang    = _project_tangent(self.velocity, up)
        dv        = v_desired - v_tang
        dv_mag    = dv.length()
        if dv_mag > 1e-6:
            accel_step = min(self.accel * self._accel_scale * factor
                             * self._turn_responsiveness * dt, dv_mag)
            self.velocity = self.velocity + dv.normalized() * accel_step

    def _apply_tangent_friction(
        self,
        up:        Vec3,
        mu:        float,
        viscosity: float,
        dt:        float,
    ) -> None:
        """Reduce tangential velocity by friction and viscosity."""
        # Effective friction = base * material_mu * env_scale * (1 - viscosity_drag)
        friction_k = self.ground_friction * mu * self._effective_friction_scale
        viscosity_k = viscosity * 2.0   # viscosity adds extra drag
        total_drag  = _clamp((friction_k + viscosity_k) * dt, 0.0, 1.0)
        v_tang = _project_tangent(self.velocity, up)
        v_tang_reduced = v_tang * (1.0 - total_drag)
        # Keep the normal component; replace tangential
        v_normal = up * self.velocity.dot(up)
        self.velocity = v_normal + v_tang_reduced

    def _apply_stumble_drift(self, up: Vec3, dt: float) -> None:
        """Add small lateral drift during stumble to simulate loss of balance."""
        # Deterministic pseudo-random drift perpendicular to velocity
        v_tang = _project_tangent(self.velocity, up)
        if v_tang.is_near_zero():
            return
        perp = up.cross(v_tang).normalized()
        drift_amp  = 1.5
        t_frac     = self._stumble_remaining / max(self.stumble_max, 1e-6)
        self.velocity = self.velocity + perp * (
            drift_amp * t_frac * dt
        )

    # ------------------------------------------------------------------
    # Private: speed cap
    # ------------------------------------------------------------------

    def _cap_speed(self, up: Vec3) -> None:
        """Clamp tangential speed to max_speed * speed_scale; leave vertical component."""
        v_tang   = _project_tangent(self.velocity, up)
        v_normal = up * self.velocity.dot(up)
        tang_mag = v_tang.length()
        cap = self.max_speed * self._speed_scale
        if tang_mag > cap:
            v_tang = v_tang * (cap / tang_mag)
        self.velocity = v_normal + v_tang

    # ------------------------------------------------------------------
    # Private: position correction
    # ------------------------------------------------------------------

    def _correct_position(self, gnd: GroundHit, up: Vec3) -> None:
        """Push character up so capsule bottom does not intersect ground."""
        if not gnd.hit:
            return
        dist = gnd.distance
        # Amount the capsule bottom penetrates the surface
        penetration = self.capsule_radius - dist
        if penetration > 0.0:
            # Push up to resolve overlap
            self.position = self.position + up * penetration
            # Kill downward velocity component
            v_down = min(0.0, self.velocity.dot(up))
            self.velocity = self.velocity - up * v_down
        elif dist < self._probe_dist and self.state != CharacterState.AIRBORNE:
            # Ground-snapping: small gap → snap toward surface
            snap = min(0.05 * (self._probe_dist - dist), 0.1)
            self.position = self.position - up * snap

    # ------------------------------------------------------------------
    # Private: orientation
    # ------------------------------------------------------------------

    def _align_orientation(self, dt: float) -> None:
        """Smoothly rotate orientation so character 'up' aligns with planet up."""
        target_up = PlanetMath.up_at_position(self.position)
        self.orientation = PlanetMath.align_up(
            self.orientation, target_up, dt, stiffness=5.0
        )


# ---------------------------------------------------------------------------
# Module-level tangent projection helper
# ---------------------------------------------------------------------------

def _project_tangent(vec: Vec3, normal: Vec3) -> Vec3:
    """Project *vec* onto the tangent plane defined by *normal*.

    Returns the projection (not normalised).
    """
    return vec - normal * vec.dot(normal)


# ---------------------------------------------------------------------------
# Config helper
# ---------------------------------------------------------------------------

def _cfg(config, section: str, key: str, default: float) -> float:
    """Safe config read with fallback."""
    if config is None:
        return default
    return config.get(section, key, default=default)
