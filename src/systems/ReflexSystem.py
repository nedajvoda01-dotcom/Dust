"""ReflexSystem — Stage 14 reflex and balance-recovery system.

Sits between player intent and the final velocity integration in
CharacterPhysicalController.  Provides "character aliveness" without any
animation or IK:

  * BalanceModel       — tracks stability [0..1] and stance
  * EnvironmentProber  — raycasts for hand-contact points and ledge edges
  * ReflexPlanner      — selects which reflex to trigger each tick
  * ReflexActuator     — applies reflex effects to the controller
  * AnimationEventStream — event bus for future AnimationController

Public API
----------
ReflexSystem(config=None, ground_sampler=None)
  .update(ctrl, dt, wind, mu, gnd)   — advance one tick
  .balance_model  → BalanceModel
  .event_stream   → AnimationEventStream
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Deque, List, Optional

from src.math.Vec3 import Vec3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


# ---------------------------------------------------------------------------
# AnimationEventStream
# ---------------------------------------------------------------------------

class AnimEventType(Enum):
    ON_BRACE        = auto()
    ON_STUMBLE_STEP = auto()
    ON_HAND_CONTACT = auto()
    ON_SLIP_RECOVER = auto()
    ON_GRAB_LEDGE   = auto()
    ON_KNEEL_IN_GUST= auto()
    ON_FALL         = auto()


@dataclass
class AnimEvent:
    """A single animation event emitted by ReflexSystem."""
    type:          AnimEventType
    time:          float
    intensity:     float           = 1.0
    dir:           Optional[Vec3]  = None
    contact_point: Optional[Vec3]  = None


class AnimationEventStream:
    """Ring-buffer event bus: Push / ConsumeAll / dev logging."""

    _MAX_DEV_LOG: int = 32

    def __init__(self) -> None:
        self._pending:  List[AnimEvent]       = []
        self._dev_log:  Deque[AnimEvent]      = deque(maxlen=self._MAX_DEV_LOG)

    def push(self, event: AnimEvent) -> None:
        """Append an event to the pending queue."""
        self._pending.append(event)
        self._dev_log.append(event)

    def consume_all(self) -> List[AnimEvent]:
        """Return and clear the pending event list."""
        events          = list(self._pending)
        self._pending   = []
        return events

    @property
    def dev_log(self) -> List[AnimEvent]:
        """Last N events (for debug)."""
        return list(self._dev_log)

    def __len__(self) -> int:
        return len(self._pending)


# ---------------------------------------------------------------------------
# Stance enum
# ---------------------------------------------------------------------------

class Stance(Enum):
    NORMAL  = auto()
    BRACED  = auto()
    CROUCHED= auto()


# ---------------------------------------------------------------------------
# BalanceModel
# ---------------------------------------------------------------------------

class BalanceModel:
    """Tracks character stability [0..1].

    Parameters (all optional; keyword-only after *config* positional):
      balance_recovery_rate  — recovery speed per second while still & braced
      balance_loss_wind_k    — multiplier for wind-induced balance loss
      balance_loss_shock_k   — multiplier for shock-induced balance loss
      balance_loss_slope_k   — multiplier for slope-induced balance loss
      fall_threshold         — balance level at which ControlledFall triggers
    """

    def __init__(
        self,
        balance_recovery_rate: float = 0.3,
        balance_loss_wind_k:   float = 0.04,
        balance_loss_shock_k:  float = 0.6,
        balance_loss_slope_k:  float = 0.08,
        fall_threshold:        float = 0.15,
    ) -> None:
        self.balance:   float  = 1.0
        self.fatigue:   float  = 0.0
        self.stance:    Stance = Stance.NORMAL

        self._recovery_rate  = balance_recovery_rate
        self._wind_k         = balance_loss_wind_k
        self._shock_k        = balance_loss_shock_k
        self._slope_k        = balance_loss_slope_k
        self.fall_threshold  = fall_threshold

    # ------------------------------------------------------------------

    def update(
        self,
        dt:              float,
        wind_speed:      float,
        shock_intensity: float,
        slip_risk:       float,
        slope_angle_rad: float,
        is_grounded:     bool,
        speed:           float,
    ) -> None:
        """Advance balance by one tick."""
        # Loss factors
        loss = (
            self._wind_k  * wind_speed
            + self._shock_k * shock_intensity
            + 0.15         * slip_risk
            + self._slope_k * max(0.0, slope_angle_rad)
        ) * dt

        self.balance = _clamp(self.balance - loss, 0.0, 1.0)

        # Recovery: grounded, slow, braced/crouched
        recovering = (
            is_grounded
            and speed < 2.0
            and self.stance in (Stance.BRACED, Stance.CROUCHED, Stance.NORMAL)
        )
        if recovering and loss < self._recovery_rate * dt:
            # Only recover if environment pressure is low
            self.balance = _clamp(
                self.balance + self._recovery_rate * dt, 0.0, 1.0
            )

    def apply_reflex_bonus(self, bonus: float) -> None:
        """Boost balance after a successful reflex."""
        self.balance = _clamp(self.balance + bonus, 0.0, 1.0)

    @property
    def is_falling(self) -> bool:
        return self.balance <= self.fall_threshold


# ---------------------------------------------------------------------------
# ContactResult — returned by EnvironmentProber
# ---------------------------------------------------------------------------

@dataclass
class ContactResult:
    hit:            bool
    point:          Vec3
    normal:         Vec3
    distance:       float
    strength:       float   = 0.0   # 0..1, how reliable the contact is


@dataclass
class LedgeResult:
    found:          bool
    ledge_point:    Vec3
    ledge_normal:   Vec3


# ---------------------------------------------------------------------------
# EnvironmentProber
# ---------------------------------------------------------------------------

class EnvironmentProber:
    """Raycasts/sphere queries to find hand-contact points and ledge edges.

    Uses the same IGroundSampler interface as CharacterPhysicalController so
    it works with flat/sphere ground as well as custom SDF samplers.
    """

    def __init__(
        self,
        ground_sampler,
        brace_reach:    float = 1.2,
        ledge_step_len: float = 1.5,
        ledge_threshold:float = 0.8,
    ) -> None:
        self._gs             = ground_sampler
        self.brace_reach     = brace_reach
        self.ledge_step_len  = ledge_step_len
        self.ledge_threshold = ledge_threshold

    @property
    def has_ground_sampler(self) -> bool:
        """Return True when a ground sampler is available for probing."""
        return self._gs is not None

    def set_ground_sampler(self, gs) -> None:
        """Replace the ground sampler (e.g., at runtime during testing)."""
        self._gs = gs

    # ------------------------------------------------------------------
    # Brace contact
    # ------------------------------------------------------------------

    def probe_brace_contact(
        self,
        world_pos: Vec3,
        up:        Vec3,
        forward:   Vec3,
    ) -> ContactResult:
        """Check forward/diagonal directions for a surface to brace against.

        Probes radially outward from *world_pos* so that the sphere ground
        geometry (where the surface is at planet_radius in every direction)
        is reliably detected even on steep slopes.
        """
        right = up.cross(forward).normalized()
        # Probe directions: forward, diagonals, forward-down
        candidates = [
            forward,
            (forward + right * 0.5).normalized(),
            (forward - right * 0.5).normalized(),
            (forward - up * 0.5).normalized(),
        ]

        best: Optional[ContactResult] = None
        for d in candidates:
            probe_end = world_pos + d * self.brace_reach
            # Use brace_reach as probe_dist so any surface within reach is found
            hit = self._gs.query_ground(probe_end, probe_dist=self.brace_reach)
            if hit.hit:
                # Strength based on how close the probe endpoint is to the surface
                surface_dist = max(0.0, hit.distance)
                strength = _clamp(1.0 - surface_dist / self.brace_reach, 0.0, 1.0)
                result = ContactResult(
                    hit      = True,
                    point    = hit.point,
                    normal   = hit.normal,
                    distance = surface_dist,
                    strength = strength,
                )
                if best is None or strength > best.strength:
                    best = result

        if best is not None:
            return best
        return ContactResult(hit=False, point=Vec3.zero(),
                             normal=up, distance=self.brace_reach)

    # ------------------------------------------------------------------
    # Ledge detection
    # ------------------------------------------------------------------

    def probe_ledge(
        self,
        world_pos: Vec3,
        up:        Vec3,
        forward:   Vec3,
    ) -> LedgeResult:
        """Detect an edge/ledge ahead by checking for a ground drop-off.

        Algorithm:
          1. Cast a probe step_len ahead.
          2. If ground is absent or > ledge_threshold below current height →
             cast straight down to find the ledge surface.
          3. If a surface is found within 2 * step_len, it's a ledge.
        """
        step_ahead = world_pos + forward * self.ledge_step_len
        ahead_hit  = self._gs.query_ground(step_ahead, probe_dist=0.6)

        current_r = world_pos.length()

        if ahead_hit.hit:
            # Check height drop relative to current position
            drop = current_r - ahead_hit.point.length()
            if drop < self.ledge_threshold:
                return LedgeResult(found=False,
                                   ledge_point=Vec3.zero(),
                                   ledge_normal=up)
            # Significant drop → treat the forward-surface as a ledge top
            return LedgeResult(
                found        = True,
                ledge_point  = step_ahead,
                ledge_normal = ahead_hit.normal,
            )

        # No ground ahead — look down from the step point for a wall / ledge
        down_probe = step_ahead - up * (self.ledge_step_len * 2.0)
        down_hit   = self._gs.query_ground(down_probe, probe_dist=self.ledge_step_len)
        if down_hit.hit:
            drop = current_r - down_hit.point.length()
            if drop > self.ledge_threshold:
                return LedgeResult(
                    found        = True,
                    ledge_point  = step_ahead,
                    ledge_normal = down_hit.normal,
                )

        return LedgeResult(found=False, ledge_point=Vec3.zero(), ledge_normal=up)


# ---------------------------------------------------------------------------
# ReflexPlanner — decides which reflex fires
# ---------------------------------------------------------------------------

class ReflexPlanner:
    """Evaluates conditions and returns a list of reflex actions to execute."""

    def __init__(
        self,
        gust_threshold:        float = 8.0,
        brace_slope_threshold: float = 0.45,   # ~26 deg
        slip_recover_slope_max:float = 0.65,   # ~37 deg — too steep = unrecoverable
        grab_max_speed:        float = 4.0,
    ) -> None:
        self.gust_threshold         = gust_threshold
        self.brace_slope_threshold  = brace_slope_threshold
        self.slip_recover_slope_max = slip_recover_slope_max
        self.grab_max_speed         = grab_max_speed

    # ------------------------------------------------------------------

    def plan(
        self,
        balance:       float,
        fall_threshold:float,
        is_grounded:   bool,
        is_sliding:    bool,
        is_airborne:   bool,
        wind_speed:    float,
        mu:            float,
        slope_angle:   float,
        speed:         float,
        contact:       ContactResult,
        ledge:         LedgeResult,
    ) -> List[str]:
        """Return ordered list of reflex identifiers to execute."""
        actions: List[str] = []

        # ControlledFall — highest priority, overrides everything
        if balance <= fall_threshold:
            actions.append("controlled_fall")
            return actions

        # GrabLedge — when airborne and ledge in reach and speed low enough
        if is_airborne and ledge.found and speed <= self.grab_max_speed:
            actions.append("grab_ledge")
            return actions

        # CrouchInGust
        if wind_speed > self.gust_threshold:
            actions.append("crouch_in_gust")

        # Brace — steep slope or very slippery, and contact available
        if (slope_angle > self.brace_slope_threshold or mu < 0.25) and contact.hit:
            actions.append("brace")

        # SlipRecover — sliding but not too steep
        if is_sliding and slope_angle < self.slip_recover_slope_max:
            actions.append("slip_recover")

        # StumbleStep — balance dropping but still above fall threshold
        if is_grounded and balance < 0.55 and not is_sliding:
            actions.append("stumble_step")

        return actions


# ---------------------------------------------------------------------------
# ReflexActuator — applies reflex effects to CharacterPhysicalController
# ---------------------------------------------------------------------------

class ReflexActuator:
    """Modifies controller state / velocity to implement reflexes."""

    def __init__(
        self,
        brace_duration:       float = 0.8,
        grab_hold_time:       float = 2.0,
        stumble_step_boost:   float = 1.2,
        slip_recover_strength:float = 3.0,
    ) -> None:
        self.brace_duration        = brace_duration
        self.grab_hold_time        = grab_hold_time
        self.stumble_step_boost    = stumble_step_boost
        self.slip_recover_strength = slip_recover_strength

        # Internal timers
        self._brace_timer:        float           = 0.0
        self._grab_timer:         float           = 0.0
        self._grab_anchor:        Optional[Vec3]  = None
        self._crouch_event_timer: float           = 0.0   # rate-limit crouch events
        self._stumble_event_timer:float           = 0.0   # rate-limit stumble events

    # ------------------------------------------------------------------
    # Per-tick timer advancement (called once per frame from ReflexSystem)
    # ------------------------------------------------------------------

    def tick_timers(
        self,
        ctrl,
        dt:      float,
        balance: BalanceModel,
    ) -> None:
        """Advance all internal timers; must be called exactly once per tick."""
        from src.systems.CharacterPhysicalController import CharacterState

        # Grab hold timer
        if self._grab_timer > 0.0:
            self._grab_timer -= dt
            if self._grab_timer <= 0.0:
                self._grab_timer  = 0.0
                self._grab_anchor = None
                if ctrl.state == CharacterState.HANGING:
                    ctrl.state = CharacterState.AIRBORNE

        # Brace timer
        if self._brace_timer > 0.0:
            self._brace_timer -= dt
            if self._brace_timer <= 0.0:
                self._brace_timer = 0.0
                if balance.stance == Stance.BRACED:
                    balance.stance = Stance.NORMAL

        # Rate-limit event timers
        if self._crouch_event_timer > 0.0:
            self._crouch_event_timer -= dt
        if self._stumble_event_timer > 0.0:
            self._stumble_event_timer -= dt

    # ------------------------------------------------------------------
    # Action dispatcher
    # ------------------------------------------------------------------

    def execute(
        self,
        action:   str,
        ctrl,                # CharacterPhysicalController
        balance:  BalanceModel,
        contact:  ContactResult,
        ledge:    LedgeResult,
        wind_dir: Vec3,
        up:       Vec3,
        dt:       float,
        events:   AnimationEventStream,
        game_time:float,
    ) -> None:
        """Dispatch a single reflex action."""
        if action == "brace":
            self._do_brace(ctrl, balance, contact, up, dt, events, game_time)

        elif action == "crouch_in_gust":
            self._do_crouch_in_gust(ctrl, balance, wind_dir, up, dt, events, game_time)

        elif action == "slip_recover":
            self._do_slip_recover(ctrl, balance, up, dt, events, game_time)

        elif action == "stumble_step":
            self._do_stumble_step(ctrl, balance, wind_dir, up, dt, events, game_time)

        elif action == "grab_ledge":
            self._do_grab_ledge(ctrl, balance, ledge, up, dt, events, game_time)

        elif action == "controlled_fall":
            self._do_controlled_fall(ctrl, balance, events, game_time)

    # ------------------------------------------------------------------
    # Individual reflex implementations
    # ------------------------------------------------------------------

    def _do_brace(self, ctrl, balance: BalanceModel, contact: ContactResult,
                  up: Vec3, dt: float, events: AnimationEventStream,
                  game_time: float) -> None:
        from src.systems.CharacterPhysicalController import CharacterState
        # Slow down tangentially
        v_tang = ctrl.velocity - up * ctrl.velocity.dot(up)
        ctrl.velocity = ctrl.velocity - v_tang * _clamp(2.5 * dt, 0.0, 1.0)
        balance.stance = Stance.BRACED
        ctrl.state     = CharacterState.BRACED
        # Only emit event and grant bonus once per brace_duration
        if self._brace_timer <= 0.0:
            self._brace_timer = self.brace_duration
            events.push(AnimEvent(
                type          = AnimEventType.ON_BRACE,
                time          = game_time,
                intensity     = contact.strength,
                contact_point = contact.point,
            ))
            events.push(AnimEvent(
                type          = AnimEventType.ON_HAND_CONTACT,
                time          = game_time,
                intensity     = contact.strength,
                contact_point = contact.point,
            ))
            balance.apply_reflex_bonus(0.05)

    def _do_crouch_in_gust(self, ctrl, balance: BalanceModel, wind_dir: Vec3,
                            up: Vec3, dt: float, events: AnimationEventStream,
                            game_time: float) -> None:
        from src.systems.CharacterPhysicalController import CharacterState
        # Reduce tangential speed to simulate crouching
        v_tang = ctrl.velocity - up * ctrl.velocity.dot(up)
        if v_tang.length() > 1.5:
            ctrl.velocity = ctrl.velocity - v_tang * _clamp(1.5 * dt, 0.0, 1.0)
        balance.stance = Stance.CROUCHED
        ctrl.state     = CharacterState.CROUCHED
        # Rate-limited event (not every tick)
        if self._crouch_event_timer <= 0.0:
            self._crouch_event_timer = 0.5   # at most 2 events per second
            events.push(AnimEvent(
                type      = AnimEventType.ON_KNEEL_IN_GUST,
                time      = game_time,
                intensity = _clamp(ctrl.velocity.length() / 6.0, 0.0, 1.0),
                dir       = wind_dir,
            ))

    def _do_slip_recover(self, ctrl, balance: BalanceModel, up: Vec3,
                          dt: float, events: AnimationEventStream,
                          game_time: float) -> None:
        # Reduce down-slope velocity component
        v_tang = ctrl.velocity - up * ctrl.velocity.dot(up)
        ctrl.velocity = ctrl.velocity - v_tang * _clamp(
            self.slip_recover_strength * dt, 0.0, 0.6
        )
        events.push(AnimEvent(
            type      = AnimEventType.ON_SLIP_RECOVER,
            time      = game_time,
            intensity = _clamp(v_tang.length() / 6.0, 0.0, 1.0),
        ))

    def _do_stumble_step(self, ctrl, balance: BalanceModel, wind_dir: Vec3,
                          up: Vec3, dt: float, events: AnimationEventStream,
                          game_time: float) -> None:
        # Short stabilising push against wind direction
        if not wind_dir.is_near_zero():
            stabilise_dir = (-wind_dir).normalized()
            stabilise_dir = stabilise_dir - up * stabilise_dir.dot(up)
            if not stabilise_dir.is_near_zero():
                ctrl.velocity = ctrl.velocity + stabilise_dir.normalized() * (
                    self.stumble_step_boost * dt
                )
        # Rate-limited event emission
        if self._stumble_event_timer <= 0.0:
            self._stumble_event_timer = 0.3   # at most ~3 events per second
            events.push(AnimEvent(
                type      = AnimEventType.ON_STUMBLE_STEP,
                time      = game_time,
                intensity = 1.0 - balance.balance,
                dir       = wind_dir,
            ))

    def _do_grab_ledge(self, ctrl, balance: BalanceModel, ledge: LedgeResult,
                        up: Vec3, dt: float, events: AnimationEventStream,
                        game_time: float) -> None:
        from src.systems.CharacterPhysicalController import CharacterState
        # Maintain existing grab
        if self._grab_anchor is not None and self._grab_timer > 0.0:
            ctrl.position = Vec3(self._grab_anchor.x,
                                 self._grab_anchor.y,
                                 self._grab_anchor.z)
            ctrl.velocity = Vec3.zero()
            ctrl.state    = CharacterState.HANGING
            return

        # Begin new grab
        self._grab_anchor = Vec3(ctrl.position.x, ctrl.position.y, ctrl.position.z)
        self._grab_timer  = self.grab_hold_time
        ctrl.velocity     = Vec3.zero()
        ctrl.state        = CharacterState.HANGING
        events.push(AnimEvent(
            type          = AnimEventType.ON_GRAB_LEDGE,
            time          = game_time,
            intensity     = 1.0,
            contact_point = ledge.ledge_point,
        ))
        balance.apply_reflex_bonus(0.2)

    def _do_controlled_fall(self, ctrl, balance: BalanceModel,
                             events: AnimationEventStream,
                             game_time: float) -> None:
        from src.systems.CharacterPhysicalController import CharacterState
        ctrl.state = CharacterState.FALLING_CONTROLLED
        events.push(AnimEvent(
            type      = AnimEventType.ON_FALL,
            time      = game_time,
            intensity = 1.0 - balance.balance,
        ))


# ---------------------------------------------------------------------------
# ReflexSystem — main facade
# ---------------------------------------------------------------------------

class ReflexSystem:
    """Coordinates BalanceModel, EnvironmentProber, ReflexPlanner, ReflexActuator.

    Call ``update(ctrl, dt, wind, mu, gnd)`` once per physics tick, *after*
    movement intent is applied but *before* position integration.

    Parameters
    ----------
    config:
        Optional mapping; falls back to hard-coded defaults.
    ground_sampler:
        The same IGroundSampler used by CharacterPhysicalController.
    game_time_ref:
        Optional callable ``() -> float`` for event timestamps.
        When *None*, cumulative dt is used.
    dev_log_enabled:
        When True, reflex state is printed each tick (no UI).
    """

    def __init__(
        self,
        config=None,
        ground_sampler=None,
        game_time_ref=None,
        dev_log_enabled: bool = False,
    ) -> None:
        def _c(key: str, default: float) -> float:
            if config is None:
                return default
            return config.get("reflex", key, default=default)

        self.balance_model = BalanceModel(
            balance_recovery_rate = _c("balance_recovery_rate", 0.3),
            balance_loss_wind_k   = _c("balance_loss_wind_k",   0.04),
            balance_loss_shock_k  = _c("balance_loss_shock_k",  0.6),
            balance_loss_slope_k  = _c("balance_loss_slope_k",  0.08),
            fall_threshold        = _c("fall_threshold",        0.15),
        )

        self._prober = EnvironmentProber(
            ground_sampler = ground_sampler,
            brace_reach    = _c("brace_reach",    1.2),
            ledge_step_len = _c("ledge_step_len", 1.5),
            ledge_threshold= _c("ledge_threshold",0.8),
        )

        self._planner = ReflexPlanner(
            gust_threshold         = _c("gust_threshold",          8.0),
            brace_slope_threshold  = _c("brace_slope_threshold",   0.45),
            slip_recover_slope_max = _c("slip_recover_slope_max",  0.65),
            grab_max_speed         = _c("grab_max_speed",          4.0),
        )

        self._actuator = ReflexActuator(
            brace_duration        = _c("brace_duration",        0.8),
            grab_hold_time        = _c("grab_hold_time",        2.0),
            stumble_step_boost    = _c("stumble_step_boost",    1.2),
            slip_recover_strength = _c("slip_recover_strength", 3.0),
        )

        self.event_stream    = AnimationEventStream()
        self._game_time_ref  = game_time_ref
        self._elapsed:float  = 0.0
        self._dev_log        = dev_log_enabled

    # ------------------------------------------------------------------

    def update(
        self,
        ctrl,               # CharacterPhysicalController
        dt:        float,
        wind:      Vec3,
        mu:        float,
        gnd,                # GroundHit
    ) -> None:
        """One reflex tick.

        Parameters
        ----------
        ctrl:
            The CharacterPhysicalController to read/modify.
        dt:
            Elapsed time in seconds.
        wind:
            Current wind vector (world space).
        mu:
            Effective friction coefficient.
        gnd:
            Most recent GroundHit from the ground sampler.
        """
        from src.systems.CharacterPhysicalController import CharacterState

        self._elapsed += dt
        now = self._game_time_ref() if self._game_time_ref else self._elapsed

        up    = ctrl.position.normalized()
        speed = ctrl.velocity.length()

        # ---- Classify state ----
        state        = ctrl.state
        is_grounded  = state in (
            CharacterState.GROUNDED,
            CharacterState.BRACED,
            CharacterState.CROUCHED,
            CharacterState.STUMBLING,   # stumbling is still on-ground
        )
        is_sliding   = state == CharacterState.SLIDING
        is_airborne  = state in (CharacterState.AIRBORNE, CharacterState.HANGING)

        # ---- Slope angle ----
        if gnd.hit:
            cos_a       = _clamp(gnd.normal.dot(up), -1.0, 1.0)
            slope_angle = math.acos(cos_a)
        else:
            slope_angle = 0.0

        # ---- Wind speed (tangential component) ----
        wind_tang  = wind - up * wind.dot(up)
        wind_speed = wind_tang.length()
        wind_dir   = wind_tang.normalized() if wind_speed > 1e-6 else Vec3.zero()

        # ---- Shock intensity from any geo IMPACT signals ----
        shock_intensity = self._read_shock(ctrl)

        # ---- Slip risk ----
        slip_risk = _clamp((1.0 - mu) + (speed / 8.0) * (1.0 - mu), 0.0, 1.0)

        # ---- Update BalanceModel ----
        self.balance_model.update(
            dt              = dt,
            wind_speed      = wind_speed,
            shock_intensity = shock_intensity,
            slip_risk       = slip_risk,
            slope_angle_rad = slope_angle,
            is_grounded     = is_grounded,
            speed           = speed,
        )

        # ---- Probe environment (only when grounded or near-grounded) ----
        if self._prober.has_ground_sampler:
            # Determine "forward" from velocity or default to a tangent
            v_tang = ctrl.velocity - up * ctrl.velocity.dot(up)
            forward = (v_tang.normalized()
                       if not v_tang.is_near_zero()
                       else _perpendicular(up))

            contact = self._prober.probe_brace_contact(ctrl.position, up, forward)
            ledge   = self._prober.probe_ledge(ctrl.position, up, forward)
        else:
            contact = ContactResult(hit=False, point=Vec3.zero(),
                                    normal=up, distance=1.0)
            ledge   = LedgeResult(found=False, ledge_point=Vec3.zero(),
                                  ledge_normal=up)

        # ---- Plan reflexes ----
        actions = self._planner.plan(
            balance        = self.balance_model.balance,
            fall_threshold = self.balance_model.fall_threshold,
            is_grounded    = is_grounded,
            is_sliding     = is_sliding,
            is_airborne    = is_airborne,
            wind_speed     = wind_speed,
            mu             = mu,
            slope_angle    = slope_angle,
            speed          = speed,
            contact        = contact,
            ledge          = ledge,
        )

        # ---- Execute reflexes ----
        for action in actions:
            self._actuator.execute(
                action    = action,
                ctrl      = ctrl,
                balance   = self.balance_model,
                contact   = contact,
                ledge     = ledge,
                wind_dir  = wind_dir,
                up        = up,
                dt        = dt,
                events    = self.event_stream,
                game_time = now,
            )

        # ---- Advance actuator timers (once per tick) ----
        self._actuator.tick_timers(ctrl, dt, self.balance_model)

        # ---- Dev logging ----
        if self._dev_log:
            print(
                f"[Reflex] t={now:.2f} balance={self.balance_model.balance:.2f} "
                f"stance={self.balance_model.stance.name} state={ctrl.state.name} "
                f"actions={actions}"
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _read_shock(self, ctrl) -> float:
        """Extract max shock intensity from any active IMPACT geo signals."""
        try:
            from src.systems.GeoEventSystem import GeoEventPhase
        except ImportError:
            return 0.0

        signals = ctrl._env.query_geo_signals(ctrl.position, 500.0)
        shock   = 0.0
        for sig in signals:
            if sig.phase == GeoEventPhase.IMPACT:
                shock = max(shock, sig.intensity)
        return shock


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _perpendicular(v: Vec3) -> Vec3:
    """Return an arbitrary vector perpendicular to *v* (normalised)."""
    if abs(v.x) < 0.9:
        return v.cross(Vec3(1.0, 0.0, 0.0)).normalized()
    return v.cross(Vec3(0.0, 1.0, 0.0)).normalized()
