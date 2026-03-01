"""test_character_physics — Stage 13 CharacterPhysicalController tests.

Tests
-----
1. test_tangent_projection       — projected desired direction is ⊥ Up (dot ≈ 0)
2. test_ground_stability         — |pos| − R stays within tolerance while walking
3. test_sliding_trigger          — slope > maxSlopeAngle → Sliding state
4. test_wind_influence           — nonzero wind shifts average velocity into wind dir
5. test_stumble_on_shock         — GeoEventSignal IMPACT at sufficient intensity → Stumbling
"""
from __future__ import annotations

import math
import sys
import os
import unittest
from dataclasses import dataclass
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.math.Vec3 import Vec3
from src.systems.CharacterPhysicalController import (
    CharacterPhysicalController,
    CharacterState,
    EnvironmentSampler,
    GroundHit,
    IGroundSampler,
    _project_tangent,
)


# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------

PLANET_R = 1000.0


def _spawn_on_surface(lat_deg: float = 0.0, extra_height: float = 1.0) -> Vec3:
    """Return a world position on (or just above) the planet surface."""
    lat = math.radians(lat_deg)
    r   = PLANET_R + extra_height
    return Vec3(0.0, r * math.sin(lat), r * math.cos(lat))


class _FlatGround(IGroundSampler):
    """Perfect-sphere ground sampler (same as default)."""
    pass


class _SteepGround(IGroundSampler):
    """Returns a tilted normal to simulate a steep slope."""

    def __init__(self, planet_radius: float, slope_deg: float = 50.0) -> None:
        super().__init__(planet_radius)
        self._slope_rad = math.radians(slope_deg)

    def query_ground(self, world_pos: Vec3, probe_dist: float) -> GroundHit:
        """Override normal with a tilted one."""
        base = super().query_ground(world_pos, probe_dist)
        if not base.hit:
            return base
        # Tilt the normal away from Up by slope_rad
        up     = world_pos.normalized()
        perp   = Vec3(1.0, 0.0, 0.0) if abs(up.y) < 0.9 else Vec3(0.0, 0.0, 1.0)
        perp   = up.cross(perp).normalized()  # tangent vector
        tilt   = math.cos(self._slope_rad)
        tilted = (up * tilt + perp * math.sin(self._slope_rad)).normalized()
        return GroundHit(
            hit      = base.hit,
            normal   = tilted,
            point    = base.point,
            distance = base.distance,
        )


class _WindyClimate:
    """Stub climate that returns a constant wind in +X direction."""

    def __init__(self, wind_speed: float = 12.0) -> None:
        self._wind = Vec3(wind_speed, 0.0, 0.0)

    def sample_wind(self, pos: Vec3) -> Vec3:
        return self._wind

    def sample_dust(self, pos: Vec3) -> float:
        return 0.0

    def get_wetness(self, pos: Vec3) -> float:
        return 0.0

    def sample_temperature(self, pos: Vec3) -> float:
        return 280.0


class _IcyClimate:
    """Stub climate: cold and icy (low friction)."""

    def sample_wind(self, pos: Vec3) -> Vec3:
        return Vec3.zero()

    def sample_dust(self, pos: Vec3) -> float:
        return 0.0

    def get_wetness(self, pos: Vec3) -> float:
        return 0.9

    def sample_temperature(self, pos: Vec3) -> float:
        return 200.0   # well below freeze threshold → ice factor = 1.0


class _FakeGeoSignal:
    """Minimal stand-in for GeoEventSignal."""
    def __init__(self, phase_impact: bool, intensity: float, pos: Vec3) -> None:
        from src.systems.GeoEventSystem import GeoEventPhase, GeoEventType
        self.phase     = GeoEventPhase.IMPACT if phase_impact else GeoEventPhase.PRE
        self.intensity = intensity
        self.position  = pos
        self.radius    = 500.0
        self.type      = GeoEventType.COLLAPSE
        self.time_to_impact = 0.0


class _ShockGeoEvents:
    """Stub geo system that always returns a strong IMPACT signal."""

    def __init__(self, intensity: float = 0.9) -> None:
        self._intensity = intensity

    def query_signals_near(self, world_pos: Vec3, radius: float) -> list:
        return [_FakeGeoSignal(True, self._intensity, Vec3.zero())]


# ---------------------------------------------------------------------------
# 1. test_tangent_projection
# ---------------------------------------------------------------------------

class TestTangentProjection(unittest.TestCase):
    """Desired direction projected onto tangent plane must be ⊥ Up (dot ≈ 0)."""

    def _check(self, pos: Vec3, desired: Vec3) -> None:
        up   = pos.normalized()
        tang = _project_tangent(desired, up)
        # tang may be zero if desired is exactly parallel to up
        if tang.is_near_zero():
            return
        self.assertAlmostEqual(
            tang.dot(up), 0.0, places=10,
            msg=f"Projection not tangent: dot={tang.dot(up):.2e} at pos={pos}"
        )

    def test_equator_north(self):
        pos     = Vec3(0.0, 0.0, PLANET_R)
        desired = Vec3(0.0, 0.0, 1.0)    # along Up direction at equator
        self._check(pos, desired)

    def test_equator_east(self):
        pos     = Vec3(0.0, 0.0, PLANET_R)
        desired = Vec3(1.0, 0.0, 0.0)
        self._check(pos, desired)

    def test_pole(self):
        pos     = Vec3(0.0, PLANET_R, 0.0)
        desired = Vec3(1.0, 0.0, 0.0)
        self._check(pos, desired)

    def test_diagonal_desired(self):
        pos     = Vec3(PLANET_R * 0.6, PLANET_R * 0.8, 0.0)
        desired = Vec3(1.0, 1.0, 1.0)
        self._check(pos, desired)

    def test_many_latitudes(self):
        for lat_deg in range(-80, 90, 20):
            lat = math.radians(lat_deg)
            pos = Vec3(PLANET_R * math.cos(lat),
                       PLANET_R * math.sin(lat), 0.0)
            desired = Vec3(0.0, 0.0, 1.0)
            self._check(pos, desired)


# ---------------------------------------------------------------------------
# 2. test_ground_stability
# ---------------------------------------------------------------------------

class TestGroundStability(unittest.TestCase):
    """Walking character should stay on the surface (|pos| − R within tolerance)."""

    _TOLERANCE = 1.5   # allow up to 1.5 units deviation (capsule radius margin)

    def _run_steps(
        self,
        pos: Vec3,
        desired: Vec3,
        steps: int = 60,
        dt: float = 1.0 / 30.0,
    ) -> List[float]:
        ctrl = CharacterPhysicalController(pos, planet_radius=PLANET_R)
        radii = []
        for _ in range(steps):
            ctrl.update(dt, desired_dir=desired, desired_speed=3.0)
            r = ctrl.position.length()
            radii.append(r - PLANET_R)
        return radii

    def test_equator_walk(self):
        pos     = Vec3(0.0, 0.0, PLANET_R + 0.4)
        desired = Vec3(1.0, 0.0, 0.0)
        deviations = self._run_steps(pos, desired)
        for d in deviations:
            self.assertLess(abs(d), self._TOLERANCE,
                            f"Height deviation {d:.3f} exceeds tolerance")

    def test_north_pole_walk(self):
        pos     = Vec3(0.0, PLANET_R + 0.4, 0.0)
        desired = Vec3(1.0, 0.0, 0.0)
        deviations = self._run_steps(pos, desired)
        for d in deviations:
            self.assertLess(abs(d), self._TOLERANCE,
                            f"Height deviation {d:.3f} exceeds tolerance")

    def test_mid_latitude_walk(self):
        lat = math.radians(45.0)
        r   = PLANET_R + 0.4
        pos = Vec3(r * math.cos(lat), r * math.sin(lat), 0.0)
        desired = Vec3(0.0, 0.0, 1.0)
        deviations = self._run_steps(pos, desired)
        for d in deviations:
            self.assertLess(abs(d), self._TOLERANCE,
                            f"Height deviation {d:.3f} exceeds tolerance")

    def test_no_input_stability(self):
        """Standing still should not drift off the surface."""
        pos = Vec3(0.0, PLANET_R + 0.4, 0.0)
        deviations = self._run_steps(pos, Vec3.zero(), steps=120)
        for d in deviations:
            self.assertLess(abs(d), self._TOLERANCE)


# ---------------------------------------------------------------------------
# 3. test_sliding_trigger
# ---------------------------------------------------------------------------

class TestSlidingTrigger(unittest.TestCase):
    """slopeAngle > maxSlopeAngle must trigger Sliding state."""

    def test_steep_slope_triggers_sliding(self):
        pos      = Vec3(0.0, PLANET_R + 0.5, 0.0)
        steep_gs = _SteepGround(PLANET_R, slope_deg=55.0)   # 55° > default 40°
        ctrl     = CharacterPhysicalController(
            pos,
            planet_radius=PLANET_R,
            ground_sampler=steep_gs,
        )
        # Run a few frames to allow state to settle
        for _ in range(5):
            ctrl.update(1.0 / 30.0, desired_dir=Vec3.zero(), desired_speed=0.0)

        self.assertEqual(ctrl.state, CharacterState.SLIDING,
                         f"Expected SLIDING but got {ctrl.state.name}")

    def test_gentle_slope_stays_grounded(self):
        pos     = Vec3(0.0, PLANET_R + 0.5, 0.0)
        flat_gs = _FlatGround(PLANET_R)   # perfectly flat → 0° slope
        ctrl    = CharacterPhysicalController(
            pos,
            planet_radius=PLANET_R,
            ground_sampler=flat_gs,
        )
        for _ in range(5):
            ctrl.update(1.0 / 30.0, desired_dir=Vec3.zero(), desired_speed=0.0)

        self.assertEqual(ctrl.state, CharacterState.GROUNDED,
                         f"Expected GROUNDED but got {ctrl.state.name}")

    def test_icy_surface_triggers_sliding(self):
        """Ice friction < 0.12 should also trigger Sliding even on flat ground."""
        pos    = Vec3(0.0, PLANET_R + 0.5, 0.0)
        env    = EnvironmentSampler(climate=_IcyClimate(), freeze_threshold=270.0)
        ctrl   = CharacterPhysicalController(
            pos,
            planet_radius=PLANET_R,
            env_sampler=env,
        )
        for _ in range(5):
            ctrl.update(1.0 / 30.0, desired_dir=Vec3.zero(), desired_speed=0.0)

        self.assertEqual(ctrl.state, CharacterState.SLIDING,
                         f"Icy surface should trigger SLIDING, got {ctrl.state.name}")


# ---------------------------------------------------------------------------
# 4. test_wind_influence
# ---------------------------------------------------------------------------

class TestWindInfluence(unittest.TestCase):
    """Non-zero wind should shift average velocity into the wind direction."""

    _WIND_SPEED = 18.0   # m/s — strong enough to overcome inertia quickly

    def test_wind_shifts_velocity(self):
        pos     = Vec3(0.0, PLANET_R + 0.5, 0.0)
        climate = _WindyClimate(self._WIND_SPEED)
        env     = EnvironmentSampler(climate=climate)
        ctrl    = CharacterPhysicalController(
            pos,
            planet_radius=PLANET_R,
            env_sampler=env,
        )
        # Stand still, let wind push the character
        for _ in range(60):
            ctrl.update(1.0 / 30.0, desired_dir=Vec3.zero(), desired_speed=0.0)

        # Velocity should have a component in the +X direction (wind direction)
        vx = ctrl.velocity.x
        self.assertGreater(vx, 0.0,
                           f"Wind should push in +X but vx={vx:.4f}")

    def test_no_wind_no_drift(self):
        """Without wind, a standing character should not gain horizontal velocity."""
        pos  = Vec3(0.0, PLANET_R + 0.5, 0.0)
        env  = EnvironmentSampler()   # no climate → no wind
        ctrl = CharacterPhysicalController(
            pos,
            planet_radius=PLANET_R,
            env_sampler=env,
        )
        for _ in range(60):
            ctrl.update(1.0 / 30.0, desired_dir=Vec3.zero(), desired_speed=0.0)

        speed = ctrl.velocity.length()
        self.assertLess(speed, 0.5,
                        f"No wind should not cause significant drift; speed={speed:.4f}")


# ---------------------------------------------------------------------------
# 5. test_stumble_on_shock
# ---------------------------------------------------------------------------

class TestStumbleOnShock(unittest.TestCase):
    """GeoEventSignal IMPACT at sufficient intensity must trigger Stumbling."""

    def test_strong_impact_triggers_stumble(self):
        pos     = Vec3(0.0, PLANET_R + 0.5, 0.0)
        geo     = _ShockGeoEvents(intensity=0.9)   # intensity > stumble_threshold (0.3)
        env     = EnvironmentSampler(geo_events=geo)
        ctrl    = CharacterPhysicalController(
            pos,
            planet_radius=PLANET_R,
            env_sampler=env,
        )
        ctrl.update(1.0 / 30.0, desired_dir=Vec3.zero(), desired_speed=0.0)

        self.assertEqual(ctrl.state, CharacterState.STUMBLING,
                         f"Strong IMPACT should trigger STUMBLING, got {ctrl.state.name}")

    def test_weak_impact_no_stumble(self):
        """Below-threshold intensity should NOT trigger stumble."""
        pos = Vec3(0.0, PLANET_R + 0.5, 0.0)
        geo = _ShockGeoEvents(intensity=0.05)   # well below 0.3
        env = EnvironmentSampler(geo_events=geo)
        ctrl = CharacterPhysicalController(
            pos,
            planet_radius=PLANET_R,
            env_sampler=env,
        )
        ctrl.update(1.0 / 30.0, desired_dir=Vec3.zero(), desired_speed=0.0)

        self.assertNotEqual(ctrl.state, CharacterState.STUMBLING,
                            "Weak IMPACT should not trigger STUMBLING")

    def test_stumble_expires(self):
        """Stumble state should expire and return to Grounded after its duration."""
        pos  = Vec3(0.0, PLANET_R + 0.5, 0.0)
        geo  = _ShockGeoEvents(intensity=0.9)
        env  = EnvironmentSampler(geo_events=geo)
        ctrl = CharacterPhysicalController(
            pos,
            planet_radius=PLANET_R,
            env_sampler=env,
        )
        # Trigger stumble
        ctrl.update(1.0 / 30.0)

        self.assertEqual(ctrl.state, CharacterState.STUMBLING)

        # Remove geo signals
        ctrl._env = EnvironmentSampler()

        # Advance past maximum stumble duration (0.8 s default)
        for _ in range(50):   # 50 * 1/30 ≈ 1.67 s
            ctrl.update(1.0 / 30.0)

        self.assertNotEqual(ctrl.state, CharacterState.STUMBLING,
                            "Stumble should have expired by now")


if __name__ == "__main__":
    unittest.main()
