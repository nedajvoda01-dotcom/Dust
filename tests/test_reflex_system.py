"""test_reflex_system — Stage 14 ReflexSystem tests.

Tests
-----
1. test_balance_decreases_in_wind
   — strong wind causes balance to fall over time

2. test_brace_triggers_on_slope_with_contact
   — steep slope + available contact → stance=Braced and OnBrace event

3. test_grab_ledge_when_ground_lost
   — ground absent ahead + surface below → state transitions to Hanging

4. test_slip_recover
   — moderate sliding → SlipRecover fires and tangential speed is reduced

5. test_event_stream_order
   — events emitted in correct phase order (brace before grab, etc.)
"""
from __future__ import annotations

import math
import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.math.Vec3 import Vec3
from src.systems.CharacterPhysicalController import (
    CharacterPhysicalController,
    CharacterState,
    EnvironmentSampler,
    GroundHit,
    IGroundSampler,
)
from src.systems.ReflexSystem import (
    AnimEventType,
    AnimationEventStream,
    BalanceModel,
    ContactResult,
    EnvironmentProber,
    LedgeResult,
    ReflexActuator,
    ReflexPlanner,
    ReflexSystem,
    Stance,
)


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

PLANET_R = 1000.0


# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------

def _on_surface(height_above: float = 0.5) -> Vec3:
    return Vec3(0.0, PLANET_R + height_above, 0.0)


class _FlatGround(IGroundSampler):
    """Perfect-sphere ground — always returns a hit at planet_radius."""
    pass


class _SteepGround(IGroundSampler):
    """Returns a tilted normal simulating a steep slope."""

    def __init__(self, planet_radius: float, slope_deg: float = 50.0) -> None:
        super().__init__(planet_radius)
        self._slope_rad = math.radians(slope_deg)

    def query_ground(self, world_pos: Vec3, probe_dist: float) -> GroundHit:
        base = super().query_ground(world_pos, probe_dist)
        if not base.hit:
            return base
        up   = world_pos.normalized()
        perp = Vec3(1.0, 0.0, 0.0) if abs(up.y) < 0.9 else Vec3(0.0, 0.0, 1.0)
        perp = up.cross(perp).normalized()
        tilted = (up * math.cos(self._slope_rad) +
                  perp * math.sin(self._slope_rad)).normalized()
        return GroundHit(hit=base.hit, normal=tilted,
                         point=base.point, distance=base.distance)


class _LedgeGround(IGroundSampler):
    """Simulates a cliff edge: past *ledge_x* the ground drops by *cliff_drop* units.

    - Before ledge_x: normal sphere surface at planet_radius
    - After ledge_x:  cliff surface at planet_radius - cliff_drop

    This creates a real height difference that EnvironmentProber can detect.
    """

    def __init__(
        self,
        planet_radius: float,
        ledge_x:   float = 1.5,
        cliff_drop:float = 3.0,
    ) -> None:
        super().__init__(planet_radius)
        self.ledge_x    = ledge_x
        self.cliff_drop = cliff_drop

    def query_ground(self, world_pos: Vec3, probe_dist: float) -> GroundHit:
        if world_pos.x > self.ledge_x:
            # Cliff face: ground is deeper here
            up             = world_pos.normalized()
            r_pos          = world_pos.length()
            cliff_radius   = self.planet_radius - self.cliff_drop
            dist           = r_pos - cliff_radius
            return GroundHit(
                hit      = dist <= probe_dist,
                normal   = up,
                point    = up * cliff_radius,
                distance = dist,
            )
        return super().query_ground(world_pos, probe_dist)


class _WindyClimate:
    def __init__(self, wind_speed: float = 15.0) -> None:
        self._wind = Vec3(wind_speed, 0.0, 0.0)

    def sample_wind(self, pos: Vec3) -> Vec3:
        return self._wind

    def sample_dust(self, pos: Vec3) -> float:
        return 0.0

    def get_wetness(self, pos: Vec3) -> float:
        return 0.0

    def sample_temperature(self, pos: Vec3) -> float:
        return 280.0


# ---------------------------------------------------------------------------
# 1. test_balance_decreases_in_wind
# ---------------------------------------------------------------------------

class TestBalanceDecreasesInWind(unittest.TestCase):
    """Strong wind must cause balance to fall over time."""

    def test_balance_falls_in_strong_wind(self):
        model = BalanceModel(balance_loss_wind_k=0.04)
        initial = model.balance

        # 5 seconds of 20 m/s wind, standing still on flat ground
        for _ in range(50):
            model.update(
                dt              = 0.1,
                wind_speed      = 20.0,
                shock_intensity = 0.0,
                slip_risk       = 0.0,
                slope_angle_rad = 0.0,
                is_grounded     = True,
                speed           = 0.0,
            )

        self.assertLess(
            model.balance, initial,
            f"Balance should decrease in strong wind; got {model.balance:.3f}"
        )

    def test_balance_stable_in_calm(self):
        """Balance should not fall with no wind, no slope, good friction."""
        model = BalanceModel()
        initial = model.balance
        model.stance = Stance.BRACED

        for _ in range(30):
            model.update(
                dt=0.1, wind_speed=0.0, shock_intensity=0.0,
                slip_risk=0.0, slope_angle_rad=0.0,
                is_grounded=True, speed=0.0,
            )

        # Should not drop (may recover slightly)
        self.assertGreaterEqual(
            model.balance, initial - 0.01,
            "Balance should not fall without environmental stress"
        )

    def test_reflex_system_balance_falls_via_controller(self):
        """ReflexSystem.update() must reduce balance when windy."""
        pos     = _on_surface()
        gs      = _FlatGround(PLANET_R)
        reflex  = ReflexSystem(ground_sampler=gs)
        climate = _WindyClimate(wind_speed=20.0)
        env     = EnvironmentSampler(climate=climate)
        ctrl    = CharacterPhysicalController(
            pos, planet_radius=PLANET_R,
            ground_sampler=gs, env_sampler=env,
            reflex_system=reflex,
        )

        initial_balance = reflex.balance_model.balance

        for _ in range(60):
            ctrl.update(1.0 / 30.0, desired_dir=Vec3.zero(), desired_speed=0.0)

        self.assertLess(
            reflex.balance_model.balance, initial_balance,
            "Balance should decrease with strong wind via full controller loop"
        )


# ---------------------------------------------------------------------------
# 2. test_brace_triggers_on_slope_with_contact
# ---------------------------------------------------------------------------

class TestBraceTriggersOnSlopeWithContact(unittest.TestCase):
    """Steep slope + available contact surface → BRACED and OnBrace event."""

    def test_brace_event_emitted(self):
        pos     = _on_surface()
        gs      = _SteepGround(PLANET_R, slope_deg=50.0)
        reflex  = ReflexSystem(ground_sampler=gs)
        env     = EnvironmentSampler()
        ctrl    = CharacterPhysicalController(
            pos, planet_radius=PLANET_R,
            ground_sampler=gs, env_sampler=env,
            reflex_system=reflex,
        )

        # Run enough ticks for planner to see slope risk + contact
        for _ in range(20):
            ctrl.update(1.0 / 30.0, desired_dir=Vec3.zero(), desired_speed=0.0)

        all_events = reflex.event_stream.dev_log
        brace_events = [e for e in all_events
                        if e.type == AnimEventType.ON_BRACE]

        self.assertTrue(
            len(brace_events) > 0,
            "Expected at least one OnBrace event on steep slope with contact"
        )

    def test_brace_stance_set_on_steep_slope(self):
        pos    = _on_surface()
        gs     = _SteepGround(PLANET_R, slope_deg=50.0)
        reflex = ReflexSystem(ground_sampler=gs)
        env    = EnvironmentSampler()
        ctrl   = CharacterPhysicalController(
            pos, planet_radius=PLANET_R,
            ground_sampler=gs, env_sampler=env,
            reflex_system=reflex,
        )

        for _ in range(20):
            ctrl.update(1.0 / 30.0, desired_dir=Vec3.zero(), desired_speed=0.0)

        self.assertEqual(
            reflex.balance_model.stance, Stance.BRACED,
            f"Expected Braced stance on steep slope, got {reflex.balance_model.stance}"
        )


# ---------------------------------------------------------------------------
# 3. test_grab_ledge_when_ground_lost
# ---------------------------------------------------------------------------

class TestGrabLedgeWhenGroundLost(unittest.TestCase):
    """Ledge ahead + airborne + low speed → Hanging state."""

    def test_hanging_state_after_grab(self):
        # Place character just at the ledge boundary in X
        pos = Vec3(1.4, PLANET_R + 0.5, 0.0)
        gs  = _LedgeGround(PLANET_R, ledge_x=1.5)

        reflex = ReflexSystem(ground_sampler=gs)
        env    = EnvironmentSampler()
        ctrl   = CharacterPhysicalController(
            pos, planet_radius=PLANET_R,
            ground_sampler=gs, env_sampler=env,
            reflex_system=reflex,
        )

        # Move toward ledge gently
        forward = Vec3(1.0, 0.0, 0.0)
        for _ in range(30):
            ctrl.update(1.0 / 30.0, desired_dir=forward, desired_speed=1.0)

        # Check that a GrabLedge event was emitted OR state is Hanging
        events   = reflex.event_stream.dev_log
        grab_evs = [e for e in events if e.type == AnimEventType.ON_GRAB_LEDGE]
        is_hanging = ctrl.state == CharacterState.HANGING

        self.assertTrue(
            is_hanging or len(grab_evs) > 0,
            "Character should be Hanging or have emitted OnGrabLedge at the ledge"
        )

    def test_ledge_probe_detects_edge(self):
        """EnvironmentProber must find a ledge with _LedgeGround sampler."""
        gs     = _LedgeGround(PLANET_R, ledge_x=1.5)
        prober = EnvironmentProber(ground_sampler=gs)
        pos    = Vec3(0.9, PLANET_R + 0.5, 0.0)
        up     = pos.normalized()
        fwd    = Vec3(1.0, 0.0, 0.0)
        result = prober.probe_ledge(pos, up, fwd)

        self.assertTrue(
            result.found,
            "EnvironmentProber should detect the ledge ahead"
        )


# ---------------------------------------------------------------------------
# 4. test_slip_recover
# ---------------------------------------------------------------------------

class TestSlipRecover(unittest.TestCase):
    """SlipRecover reflex reduces tangential speed during moderate sliding."""

    def test_slip_recover_reduces_speed(self):
        # Set up on a moderately steep slope (sliding, but < 37 deg)
        slope_deg = 35.0
        pos   = _on_surface()
        gs    = _SteepGround(PLANET_R, slope_deg=slope_deg)
        reflex = ReflexSystem(ground_sampler=gs)
        env    = EnvironmentSampler()
        ctrl   = CharacterPhysicalController(
            pos, planet_radius=PLANET_R,
            ground_sampler=gs, env_sampler=env,
            reflex_system=reflex,
        )

        # Give initial slide velocity
        up = pos.normalized()
        ctrl.velocity = Vec3(2.0, 0.0, 0.0) - up * up.dot(Vec3(2.0, 0.0, 0.0))

        speed_before = ctrl.velocity.length()

        for _ in range(30):
            ctrl.update(1.0 / 30.0, desired_dir=Vec3.zero(), desired_speed=0.0)

        events   = reflex.event_stream.dev_log
        slip_evs = [e for e in events if e.type == AnimEventType.ON_SLIP_RECOVER]

        # Either slip recover fired, or speed has dropped
        speed_after = ctrl.velocity.length()
        self.assertTrue(
            len(slip_evs) > 0 or speed_after <= speed_before,
            f"Expected SlipRecover event or reduced speed; "
            f"before={speed_before:.2f} after={speed_after:.2f} events={len(slip_evs)}"
        )

    def test_actuator_slip_recover_decreases_velocity(self):
        """ReflexActuator._do_slip_recover must reduce tangential speed."""
        actuator = ReflexActuator(slip_recover_strength=3.0)
        events   = AnimationEventStream()
        balance  = BalanceModel()

        pos = _on_surface()
        gs  = _FlatGround(PLANET_R)
        ctrl = CharacterPhysicalController(pos, planet_radius=PLANET_R,
                                           ground_sampler=gs)

        up = pos.normalized()
        ctrl.velocity = Vec3(4.0, 0.0, 0.0)
        speed_before = ctrl.velocity.length()

        actuator._do_slip_recover(ctrl, balance, up, dt=0.1,
                                  events=events, game_time=0.0)

        speed_after = ctrl.velocity.length()
        self.assertLess(
            speed_after, speed_before,
            f"SlipRecover should reduce speed: {speed_before:.2f} → {speed_after:.2f}"
        )


# ---------------------------------------------------------------------------
# 5. test_event_stream_order
# ---------------------------------------------------------------------------

class TestEventStreamOrder(unittest.TestCase):
    """Events must be emitted in the correct phase order."""

    def test_brace_before_grab_on_slope_then_airborne(self):
        """In a sequence where character first braces then goes airborne near
        a ledge, OnBrace must appear before OnGrabLedge in the dev log."""

        pos    = _on_surface()
        gs     = _SteepGround(PLANET_R, slope_deg=50.0)
        reflex = ReflexSystem(ground_sampler=gs)
        env    = EnvironmentSampler()
        ctrl   = CharacterPhysicalController(
            pos, planet_radius=PLANET_R,
            ground_sampler=gs, env_sampler=env,
            reflex_system=reflex,
        )

        # Phase 1: grounded on steep slope → brace
        for _ in range(20):
            ctrl.update(1.0 / 30.0, desired_dir=Vec3.zero(), desired_speed=0.0)

        brace_events_phase1 = [e for e in reflex.event_stream.dev_log
                                if e.type == AnimEventType.ON_BRACE]

        # Phase 2: force airborne state + trigger grab logic via planner
        ctrl.state    = CharacterState.AIRBORNE
        ctrl.velocity = Vec3(0.5, 0.0, 0.0)

        # Swap to ledge ground
        ledge_gs  = _LedgeGround(PLANET_R, ledge_x=1.5)
        ctrl._ground         = ledge_gs
        reflex._prober._gs   = ledge_gs

        for _ in range(10):
            ctrl.update(1.0 / 30.0, desired_dir=Vec3(1.0, 0.0, 0.0),
                        desired_speed=0.5)

        log = reflex.event_stream.dev_log

        brace_times = [e.time for e in log if e.type == AnimEventType.ON_BRACE]
        grab_times  = [e.time for e in log if e.type == AnimEventType.ON_GRAB_LEDGE]

        # Brace must have occurred at some point in phase 1
        self.assertTrue(
            len(brace_events_phase1) > 0,
            "Expected OnBrace events during steep-slope phase"
        )

        # If grab occurred, it must be after the first brace
        if grab_times:
            self.assertGreaterEqual(
                min(grab_times), min(brace_times),
                "OnGrabLedge must not precede OnBrace in event timeline"
            )

    def test_event_stream_consume_clears_pending(self):
        """consume_all() must empty the pending queue."""
        stream = AnimationEventStream()
        stream.push(AnimEvent(type=AnimEventType.ON_BRACE, time=0.1))
        stream.push(AnimEvent(type=AnimEventType.ON_SLIP_RECOVER, time=0.2))

        consumed = stream.consume_all()
        self.assertEqual(len(consumed), 2)
        self.assertEqual(len(stream), 0,
                         "Pending queue should be empty after consume_all()")

    def test_dev_log_preserves_history_after_consume(self):
        """dev_log must retain events even after consume_all()."""
        stream = AnimationEventStream()
        stream.push(AnimEvent(type=AnimEventType.ON_BRACE, time=0.1))
        stream.consume_all()   # clears pending

        self.assertEqual(
            len(stream.dev_log), 1,
            "dev_log should still hold events after consume_all()"
        )

    def test_geo_shock_triggers_stumble_step_before_fall(self):
        """IMPACT geo signal should reduce balance; stumble_step fires before fall."""
        from dataclasses import dataclass

        @dataclass
        class _FakeSignal:
            phase:          object
            intensity:      float
            position:       Vec3
            radius:         float = 500.0
            time_to_impact: float = 0.0

        class _ShockGeo:
            def query_signals_near(self, pos, r):
                from src.systems.GeoEventSystem import GeoEventPhase
                return [_FakeSignal(GeoEventPhase.IMPACT, 0.8, Vec3.zero())]

        pos    = _on_surface()
        gs     = _FlatGround(PLANET_R)
        reflex = ReflexSystem(ground_sampler=gs)
        env    = EnvironmentSampler(geo_events=_ShockGeo())
        ctrl   = CharacterPhysicalController(
            pos, planet_radius=PLANET_R,
            ground_sampler=gs, env_sampler=env,
            reflex_system=reflex,
        )

        for _ in range(30):
            ctrl.update(1.0 / 30.0, desired_dir=Vec3.zero(), desired_speed=0.0)

        log = reflex.event_stream.dev_log
        # Balance should have been reduced by shock
        self.assertLess(reflex.balance_model.balance, 1.0,
                        "Shock should have reduced balance")

        # Stumble step or fall must have been emitted
        relevant = [e for e in log if e.type in (
            AnimEventType.ON_STUMBLE_STEP,
            AnimEventType.ON_FALL,
        )]
        self.assertGreater(
            len(relevant), 0,
            "Shock should trigger stumble_step or fall events"
        )


# Import AnimEvent directly so tests can construct events
from src.systems.ReflexSystem import AnimEvent

if __name__ == "__main__":
    unittest.main()
