"""CharacterEnvironmentIntegration — Stage 15 environment-to-character layer.

Samples climate, geology, and surface state each tick and produces:
  * EnvContext          — unified per-character environment snapshot
  * LocomotionModifiers — physics/reflex parameter set derived from EnvContext
  * AnimParamFrame      — continuous animation parameters for future AnimController

Applies LocomotionModifiers to CharacterPhysicalController and optionally
updates ReflexSystem thresholds for environment-aware reflexes.

Public API
----------
CharacterEnvironmentIntegration(config, global_seed, character_id, ...)
  .update(ctrl, reflex, dt, game_time) — one tick
  .env_context     → EnvContext
  .loco_modifiers  → LocomotionModifiers
  .anim_frame      → AnimParamFrame
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from src.math.Vec3 import Vec3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * _clamp(t, 0.0, 1.0)


# ---------------------------------------------------------------------------
# EnvContext — unified per-tick environment snapshot around the character
# ---------------------------------------------------------------------------

@dataclass
class EnvContext:
    """Sampled environment conditions at the character's current position."""
    wind_vec_world:         Vec3  = field(default_factory=Vec3.zero)
    wind_speed:             float = 0.0
    dust:                   float = 0.0
    visibility:             float = 1.0
    temp:                   float = 290.0
    ice:                    float = 0.0
    wetness:                float = 0.0
    slope_angle:            float = 0.0        # radians
    ground_normal:          Vec3  = field(default_factory=lambda: Vec3(0.0, 1.0, 0.0))
    ground_roughness_proxy: float = 0.5
    friction_mu:            float = 1.0
    ground_stability:       float = 1.0        # 1 = fully stable
    fracture:               float = 0.0
    hardness:               float = 1.0
    storm_intensity:        float = 0.0
    geo_signal_pre:         float = 0.0        # max PRE signal intensity nearby
    geo_signal_impact:      float = 0.0        # max IMPACT signal intensity nearby


# ---------------------------------------------------------------------------
# LocomotionModifiers — derived physics/reflex parameter adjustments
# ---------------------------------------------------------------------------

@dataclass
class LocomotionModifiers:
    """All parameter modifiers that affect the physical controller and reflexes."""
    speed_scale:              float = 1.0
    accel_scale:              float = 1.0
    turn_responsiveness:      float = 1.0
    effective_friction_scale: float = 1.0
    wind_drag_scale:          float = 1.0
    stance:                   str   = "normal"   # "normal" | "braced" | "crouched"
    balance_loss_scale:       float = 1.0
    balance_recovery_scale:   float = 1.0
    stumble_rate:             float = 0.0
    brace_rate:               float = 0.0
    slip_rate:                float = 0.0
    grab_rate:                float = 0.0
    step_height_scale:        float = 1.0
    upper_body_lean:          float = 0.0        # scalar lean magnitude
    arm_support_bias:         float = 0.0        # 0..1


# ---------------------------------------------------------------------------
# AnimParamFrame — continuous animation parameters for future AnimController
# ---------------------------------------------------------------------------

@dataclass
class AnimParamFrame:
    """Per-tick animation parameters derived from environment."""
    stride_length: float = 1.0
    cadence:       float = 1.0
    arm_swing_amp: float = 1.0
    torso_twist:   float = 0.0
    head_bob:      float = 1.0
    micro_jitter:  float = 0.0
    effort:        float = 0.0


# ---------------------------------------------------------------------------
# CharacterRngStream — deterministic seeded RNG per character × position × time
# ---------------------------------------------------------------------------

class CharacterRngStream:
    """Provides deterministic micro-variation values that are stable within
    each *variation_window_sec* bucket and change smoothly across buckets.

    Parameters
    ----------
    global_seed:
        Master world seed.
    character_id:
        Unique integer identifier for this character.
    variation_window_sec:
        Duration (game-seconds) of each RNG bucket.  Within a window the
        values remain constant; they are re-drawn at each window boundary.
    """

    def __init__(
        self,
        global_seed:          int   = 42,
        character_id:         int   = 0,
        variation_window_sec: float = 2.0,
    ) -> None:
        self._global_seed    = global_seed
        self._character_id   = character_id
        self._window_sec     = max(0.01, variation_window_sec)
        self._current_bucket: int   = -1
        self._values:         list  = []

    # ------------------------------------------------------------------

    def update(self, game_time: float, lat_cell: int, lon_cell: int) -> None:
        """Recompute RNG draw if the time bucket has advanced.

        Parameters
        ----------
        game_time:
            Current simulation time (seconds).
        lat_cell, lon_cell:
            Coarse integer lat/lon cell of the character's position.
        """
        bucket = int(game_time / self._window_sec)
        if bucket == self._current_bucket:
            return
        self._current_bucket = bucket
        # Derive seed deterministically
        seed = (
            self._global_seed
            ^ (self._character_id * 1664525 + 1013904223)
            ^ (lat_cell * 2971 + lon_cell * 1731)
            ^ (bucket * 131071)
        ) & 0xFFFFFFFF
        import random
        rng = random.Random(seed)
        # Draw a small pool of values in [0, 1)
        self._values = [rng.random() for _ in range(16)]

    def value(self, index: int) -> float:
        """Return a stable value [0, 1) for the given index."""
        if not self._values:
            return 0.5
        return self._values[index % len(self._values)]

    def signed(self, index: int) -> float:
        """Return a stable value [-1, 1) for the given index."""
        return self.value(index) * 2.0 - 1.0


# ---------------------------------------------------------------------------
# CharacterEnvironmentIntegration — main facade
# ---------------------------------------------------------------------------

class CharacterEnvironmentIntegration:
    """Bridges climate + geology + surface with the character controller.

    Parameters
    ----------
    config:
        Config object (reads ``env.*`` keys).  Pass *None* for defaults.
    global_seed:
        Master world seed for deterministic micro-variation.
    character_id:
        Unique integer for this character (used in RNG seed).
    climate:
        Optional climate system with ``sample_wind``, ``sample_dust``,
        ``sample_temperature``, ``get_wetness``, ``get_visibility``.
    geo_field_sampler:
        Optional ``GeoFieldSampler`` for stress/fracture/stability/hardness.
    geo_event_system:
        Optional geo event system with ``query_signals_near``.
    planet_radius:
        Planet radius (simulation units).
    dev_log_enabled:
        When True, logs EnvContext and LocomotionModifiers periodically.
    """

    def __init__(
        self,
        config=None,
        global_seed:      int   = 42,
        character_id:     int   = 0,
        climate=None,
        geo_field_sampler=None,
        geo_event_system=None,
        planet_radius:    float = 1000.0,
        dev_log_enabled:  bool  = False,
    ) -> None:
        def _c(key: str, default: float) -> float:
            if config is None:
                return default
            return config.get("env", key, default=default)

        self._wind_lean_max        = _c("wind_lean_max",        20.0)
        self._headwind_speed_k     = _c("headwind_speed_k",     0.04)
        self._whiteout_turn_k      = _c("whiteout_turn_k",      0.6)
        self._whiteout_stride_scale= _c("whiteout_stride_scale",0.5)
        self._ice_friction_min     = _c("ice_friction_scale_min",0.15)
        self._dust_viscosity_k     = _c("dust_viscosity_k",     0.5)
        self._slope_speed_k        = _c("slope_speed_k",        0.6)
        self._stability_balance_k  = _c("stability_balance_k",  0.5)
        self._variation_window_sec = _c("variation_window_sec", 2.0)
        self._variation_strength   = _c("variation_strength",   0.08)

        self._climate              = climate
        self._geo_field            = geo_field_sampler
        self._geo_events           = geo_event_system
        self._planet_radius        = planet_radius
        self._dev_log              = dev_log_enabled
        self._dev_log_timer:float  = 0.0

        # Ice derived from temperature; freeze threshold matches climate default
        freeze_k = (
            config.get("climate", "freeze_threshold", default=270.0)
            if config is not None
            else 270.0
        )
        self._freeze_threshold = freeze_k

        self._rng = CharacterRngStream(
            global_seed          = global_seed,
            character_id         = character_id,
            variation_window_sec = self._variation_window_sec,
        )

        self._env_ctx:   EnvContext          = EnvContext()
        self._loco_mods: LocomotionModifiers = LocomotionModifiers()
        self._anim_frame:AnimParamFrame      = AnimParamFrame()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def env_context(self) -> EnvContext:
        return self._env_ctx

    @property
    def loco_modifiers(self) -> LocomotionModifiers:
        return self._loco_mods

    @property
    def anim_frame(self) -> AnimParamFrame:
        return self._anim_frame

    # ------------------------------------------------------------------
    # Main per-tick update
    # ------------------------------------------------------------------

    def update(
        self,
        ctrl,               # CharacterPhysicalController
        reflex=None,        # Optional[ReflexSystem]
        dt:        float = 0.016,
        game_time: float = 0.0,
    ) -> None:
        """Advance one tick.

        Samples the environment, computes modifiers, applies them to
        *ctrl* (and optionally *reflex*).

        Parameters
        ----------
        ctrl:
            CharacterPhysicalController to read position / ground from.
        reflex:
            Optional ReflexSystem whose thresholds can be updated.
        dt:
            Elapsed game time (seconds).
        game_time:
            Absolute game time (seconds), used for RNG buckets.
        """
        pos  = ctrl.position
        gnd  = ctrl._last_ground
        up   = pos.normalized()

        # 1. Sample EnvContext
        self._sample_env(pos, up, gnd, game_time)

        # 2. Compute LocomotionModifiers from EnvContext
        self._compute_modifiers()

        # 3. Apply modifiers to controller
        self._apply_to_controller(ctrl)

        # 4. Update reflex thresholds if available
        if reflex is not None:
            self._apply_to_reflex(reflex)

        # 5. Compute AnimParamFrame
        self._compute_anim_frame()

        # 6. Dev logging (every 2 seconds)
        if self._dev_log:
            self._dev_log_timer -= dt
            if self._dev_log_timer <= 0.0:
                self._dev_log_timer = 2.0
                print(
                    f"[EnvInteg] t={game_time:.1f} "
                    f"wind={self._env_ctx.wind_speed:.1f} "
                    f"dust={self._env_ctx.dust:.2f} "
                    f"ice={self._env_ctx.ice:.2f} "
                    f"stability={self._env_ctx.ground_stability:.2f} "
                    f"speedScale={self._loco_mods.speed_scale:.2f} "
                    f"stance={self._loco_mods.stance} "
                    f"lean={self._loco_mods.upper_body_lean:.2f}"
                )

    # ------------------------------------------------------------------
    # Private: sample environment
    # ------------------------------------------------------------------

    def _sample_env(self, pos: Vec3, up: Vec3, gnd, game_time: float) -> None:
        """Fill self._env_ctx from all available systems."""
        ctx = self._env_ctx

        # --- Climate ---
        if self._climate is not None:
            wind_vec = self._climate.sample_wind(pos)
            ctx.wind_vec_world = wind_vec
            ctx.wind_speed     = wind_vec.length()
            ctx.dust           = _clamp(self._climate.sample_dust(pos), 0.0, 1.0)
            ctx.visibility     = (
                self._climate.get_visibility(pos)
                if hasattr(self._climate, "get_visibility")
                else math.exp(-ctx.dust * 5.0)
            )
            ctx.temp           = (
                self._climate.sample_temperature(pos)
                if hasattr(self._climate, "sample_temperature")
                else 290.0
            )
            ctx.wetness        = (
                _clamp(self._climate.get_wetness(pos), 0.0, 1.0)
                if hasattr(self._climate, "get_wetness")
                else 0.0
            )
            # Ice derived from temperature or explicit wetness proxy
            freeze_k = self._freeze_threshold
            ctx.ice  = _clamp((freeze_k - ctx.temp) / 30.0, 0.0, 1.0)
            # Storm intensity from active storms (ClimateSystem attribute)
            ctx.storm_intensity = self._storm_intensity(pos)
        else:
            ctx.wind_vec_world = Vec3.zero()
            ctx.wind_speed     = 0.0
            ctx.dust           = 0.0
            ctx.visibility     = 1.0
            ctx.temp           = 290.0
            ctx.wetness        = 0.0
            ctx.ice            = 0.0
            ctx.storm_intensity= 0.0

        # --- Ground ---
        if gnd.hit:
            ctx.ground_normal  = gnd.normal
            cos_a              = _clamp(gnd.normal.dot(up), -1.0, 1.0)
            ctx.slope_angle    = math.acos(cos_a)
        else:
            ctx.ground_normal  = up
            ctx.slope_angle    = 0.0

        # Friction proxy from dust and ice
        ice_slip = ctx.ice * 0.8 + ctx.wetness * 0.2
        ctx.friction_mu = _clamp(1.0 - 0.85 * ice_slip - 0.10 * ctx.dust, 0.05, 1.0)

        # Ground roughness: inversely proportional to dust smoothness
        ctx.ground_roughness_proxy = _clamp(1.0 - ctx.dust * 0.5, 0.3, 1.0)

        # --- Geology ---
        if self._geo_field is not None:
            try:
                sample             = self._geo_field.sample(pos.normalized())
                ctx.ground_stability = sample.stability
                ctx.fracture         = sample.fracture
                ctx.hardness         = sample.hardness
            except Exception:
                ctx.ground_stability = 1.0
                ctx.fracture         = 0.0
                ctx.hardness         = 1.0
        else:
            ctx.ground_stability = 1.0
            ctx.fracture         = 0.0
            ctx.hardness         = 1.0

        # --- Geo event signals ---
        ctx.geo_signal_pre    = 0.0
        ctx.geo_signal_impact = 0.0
        if self._geo_events is not None:
            try:
                from src.systems.GeoEventSystem import GeoEventPhase
                signals = self._geo_events.query_signals_near(pos, 500.0)
                for sig in signals:
                    if sig.phase == GeoEventPhase.PRE:
                        ctx.geo_signal_pre = max(ctx.geo_signal_pre, sig.intensity)
                    elif sig.phase == GeoEventPhase.IMPACT:
                        ctx.geo_signal_impact = max(ctx.geo_signal_impact, sig.intensity)
            except Exception:
                pass

        # --- Update RNG bucket ---
        lat_cell = int(math.asin(_clamp(pos.y / max(pos.length(), 1e-9), -1.0, 1.0)) * 10)
        lon_cell = int(math.atan2(pos.x, pos.z) * 10)
        self._rng.update(game_time, lat_cell, lon_cell)

    def _storm_intensity(self, pos: Vec3) -> float:
        """Query storm intensity at *pos* from ClimateSystem if available."""
        if self._climate is None:
            return 0.0
        if not hasattr(self._climate, "storms"):
            return 0.0
        import math as _math
        from src.math.PlanetMath import PlanetMath
        try:
            ll = PlanetMath.from_direction(pos)
        except Exception:
            return 0.0
        best = 0.0
        for storm in self._climate.storms:
            dlat = storm.center_lat - ll.lat_rad
            dlon = storm.center_lon - ll.lon_rad
            a = (_math.sin(dlat * 0.5) ** 2
                 + _math.cos(ll.lat_rad) * _math.cos(storm.center_lat)
                 * _math.sin(dlon * 0.5) ** 2)
            dist = 2.0 * _math.asin(_math.sqrt(_clamp(a, 0.0, 1.0)))
            if dist < storm.radius:
                frac = 1.0 - dist / storm.radius
                best = max(best, storm.intensity * frac)
        return _clamp(best, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Private: compute LocomotionModifiers from EnvContext
    # ------------------------------------------------------------------

    def _compute_modifiers(self) -> None:
        """Apply all environment rules to produce LocomotionModifiers."""
        ctx  = self._env_ctx
        mods = LocomotionModifiers()

        # --- 6.1 Wind ---
        # Upper body lean (into headwind)
        lean = _clamp(ctx.wind_speed / max(self._wind_lean_max, 1e-3), 0.0, 1.0)
        mods.upper_body_lean = lean

        # Headwind speed penalty
        head_factor = self._headwind_factor(ctx)
        mods.speed_scale *= max(0.0, 1.0 - self._headwind_speed_k
                                * ctx.wind_speed * head_factor)

        # Wind drag rises with both storm intensity and wind speed
        wind_scale_factor = _clamp(ctx.wind_speed / 20.0, 0.0, 1.0)
        mods.wind_drag_scale = 1.0 + ctx.storm_intensity * 0.8 + wind_scale_factor * 0.5

        # Stance: crouched in gust/storm
        gust_threshold = 12.0
        if ctx.wind_speed > gust_threshold or ctx.storm_intensity > 0.5:
            mods.stance = "crouched"

        # --- 6.2 Dust / whiteout ---
        whiteout_factor = 1.0 - ctx.visibility          # 0 = clear, 1 = whiteout
        mods.turn_responsiveness = _lerp(1.0, 1.0 - self._whiteout_turn_k, whiteout_factor)
        stride_scale = _lerp(1.0, self._whiteout_stride_scale, whiteout_factor)
        mods.speed_scale *= stride_scale
        mods.balance_loss_scale  += whiteout_factor * 0.3
        mods.brace_rate          += whiteout_factor * 0.4

        # --- 6.3 Ice / wetness ---
        ice_factor = ctx.ice
        mods.effective_friction_scale = _lerp(1.0, self._ice_friction_min, ice_factor)
        mods.slip_rate    += ice_factor * 0.6
        mods.stumble_rate += ice_factor * 0.3
        if ice_factor > 0.3 and mods.stance == "normal":
            mods.stance = "braced"

        # --- 6.4 Dust viscosity ---
        # Treat high dust + low slope as "deep dust" proxy
        deep_dust = ctx.dust * (1.0 - _clamp(ctx.slope_angle / 0.5, 0.0, 1.0))
        mods.speed_scale       *= max(0.3, 1.0 - self._dust_viscosity_k * deep_dust)
        mods.step_height_scale  = 1.0 + deep_dust * 0.6   # "lift feet higher"

        # --- 6.5 Slope ---
        slope_factor = _clamp(ctx.slope_angle / (math.pi * 0.5), 0.0, 1.0)
        mods.speed_scale      *= max(0.3, 1.0 - self._slope_speed_k * slope_factor)
        mods.arm_support_bias  = _clamp(slope_factor * 1.2, 0.0, 1.0)
        mods.brace_rate       += slope_factor * 0.5
        mods.grab_rate        += slope_factor * 0.3

        # --- 6.6 Geological instability ---
        instability = 1.0 - ctx.ground_stability
        fracture_factor = ctx.fracture
        mods.balance_loss_scale += instability * self._stability_balance_k
        mods.stumble_rate       += instability * 0.4
        mods.stumble_rate       += fracture_factor * 0.3

        # --- 6.7 Geo PRE / IMPACT signals ---
        pre    = ctx.geo_signal_pre
        impact = ctx.geo_signal_impact

        if pre > 0.1:
            # Pre-brace: anticipatory crouch
            if mods.stance == "normal":
                mods.stance = "braced"
            mods.speed_scale  *= max(0.5, 1.0 - pre * 0.5)
            mods.brace_rate   += pre * 0.8

        if impact > 0.1:
            # Forced stumble/brace on impact
            mods.stumble_rate += impact * 0.9
            mods.brace_rate   += impact * 0.5
            mods.grab_rate    += impact * 0.4

        # --- Final clamp ---
        mods.speed_scale             = _clamp(mods.speed_scale, 0.1, 1.0)
        mods.accel_scale             = _clamp(mods.accel_scale, 0.1, 1.0)
        mods.turn_responsiveness     = _clamp(mods.turn_responsiveness, 0.1, 1.0)
        mods.effective_friction_scale= _clamp(mods.effective_friction_scale, 0.05, 2.0)
        mods.wind_drag_scale         = _clamp(mods.wind_drag_scale, 0.0, 3.0)
        mods.balance_loss_scale      = _clamp(mods.balance_loss_scale, 1.0, 4.0)
        mods.stumble_rate            = _clamp(mods.stumble_rate, 0.0, 1.0)
        mods.brace_rate              = _clamp(mods.brace_rate, 0.0, 1.0)
        mods.slip_rate               = _clamp(mods.slip_rate, 0.0, 1.0)
        mods.grab_rate               = _clamp(mods.grab_rate, 0.0, 1.0)
        mods.arm_support_bias        = _clamp(mods.arm_support_bias, 0.0, 1.0)

        self._loco_mods = mods

    def _headwind_factor(self, ctx: EnvContext) -> float:
        """Return fraction of wind that is a headwind [0..1] (always 0..1 scalar)."""
        # Without knowledge of character forward direction we use a constant
        # scalar that represents "some fraction of wind opposes motion".
        # Full integration with character forward vector happens in apply.
        return 0.5   # conservative neutral assumption

    # ------------------------------------------------------------------
    # Private: apply modifiers to controller
    # ------------------------------------------------------------------

    def _apply_to_controller(self, ctrl) -> None:
        mods = self._loco_mods
        ctrl.set_speed_scale(mods.speed_scale)
        ctrl.set_accel_scale(mods.accel_scale)
        ctrl.set_effective_friction_scale(mods.effective_friction_scale)
        ctrl.set_wind_drag_scale(mods.wind_drag_scale)
        ctrl.set_turn_responsiveness(mods.turn_responsiveness)

    # ------------------------------------------------------------------
    # Private: update reflex thresholds
    # ------------------------------------------------------------------

    def _apply_to_reflex(self, reflex) -> None:
        """Adjust ReflexSystem thresholds based on current environment."""
        ctx  = self._env_ctx
        mods = self._loco_mods

        # Lower gust threshold in storm (character braces more readily)
        gust_thresh = reflex._planner.gust_threshold
        storm_k     = ctx.storm_intensity * 0.5
        reflex._planner.gust_threshold = max(3.0, gust_thresh * (1.0 - storm_k))

        # Whiteout: reduce grab_max_speed (hard to see ledges)
        whiteout = 1.0 - ctx.visibility
        base_grab = reflex._planner.grab_max_speed
        reflex._planner.grab_max_speed = max(1.0, base_grab * (1.0 - whiteout * 0.4))

        # Slope: increase brace reach
        slope_factor = _clamp(ctx.slope_angle / (math.pi * 0.5), 0.0, 1.0)
        reflex._prober.brace_reach = 1.2 + slope_factor * 0.8

        # Geo PRE: lower brace threshold (brace sooner)
        if ctx.geo_signal_pre > 0.2:
            reflex._planner.brace_slope_threshold = max(
                0.2,
                reflex._planner.brace_slope_threshold - ctx.geo_signal_pre * 0.15,
            )

        # Scale BalanceModel loss factor
        reflex.balance_model._wind_k = 0.04 * mods.balance_loss_scale

    # ------------------------------------------------------------------
    # Private: compute AnimParamFrame
    # ------------------------------------------------------------------

    def _compute_anim_frame(self) -> None:
        """Derive AnimParamFrame from EnvContext + RNG micro-variation."""
        ctx   = self._env_ctx
        mods  = self._loco_mods
        rng   = self._rng
        var   = self._variation_strength

        # Base values driven by modifiers
        base_stride  = mods.speed_scale
        base_cadence = _lerp(1.0, 0.6, 1.0 - ctx.visibility)
        base_effort  = _clamp(
            (1.0 - mods.speed_scale) * 0.5
            + mods.arm_support_bias * 0.3
            + ctx.dust * 0.2,
            0.0, 1.0,
        )

        # Seeded micro-variation
        stride_var  = rng.signed(0) * var
        cadence_var = rng.signed(1) * var * 0.5
        jitter_k    = ctx.storm_intensity * 0.6 + (1.0 - ctx.ground_stability) * 0.3

        frame = AnimParamFrame(
            stride_length = _clamp(base_stride + stride_var, 0.1, 1.5),
            cadence       = _clamp(base_cadence + cadence_var, 0.3, 1.5),
            arm_swing_amp = _clamp(mods.arm_support_bias + 0.3 + rng.value(2) * var, 0.1, 1.5),
            torso_twist   = mods.upper_body_lean * 0.5 + rng.signed(3) * var * 0.3,
            head_bob      = _clamp(base_cadence * 0.8 + rng.signed(4) * var * 0.2, 0.1, 1.2),
            micro_jitter  = _clamp(jitter_k + rng.value(5) * var, 0.0, 1.0),
            effort        = base_effort,
        )
        self._anim_frame = frame
