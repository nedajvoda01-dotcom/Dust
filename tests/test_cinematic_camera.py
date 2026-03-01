"""test_cinematic_camera — Stage 18 CinematicCameraSystem tests.

Tests
-----
1. test_mode_hysteresis
   — Camera mode does not oscillate when env metrics hover around thresholds;
     cooldown prevents rapid switching.

2. test_collision_prevents_penetration
   — When the desired camera position is below the planet surface the
     corrected position is kept above (or at) the surface.

3. test_awe_trigger
   — When eclipseFactor is high and visibility is adequate, the camera
     eventually transitions to Awe mode.

4. test_up_alignment
   — The camera 'up' axis (reconstructed from the rotation) stays close to
     the planet Up vector except during extreme shake.
"""
from __future__ import annotations

import math
import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.math.Vec3 import Vec3
from src.math.PlanetMath import PlanetMath
from src.systems.CharacterPhysicalController import CharacterState
from src.systems.CinematicCameraSystem import (
    CinematicCameraSystem,
    CameraMode,
    CharacterInput,
    EnvContext,
    AstroContext,
    GeoSignal,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PLANET_R = 1000.0


def _char_on_surface(pos_dir: Vec3 = None) -> CharacterInput:
    """Return a CharacterInput sitting still on the planet surface."""
    if pos_dir is None:
        pos_dir = Vec3(0.0, 1.0, 0.0)
    pos_dir = pos_dir.normalized()
    pos = pos_dir * _PLANET_R
    up = PlanetMath.up_at_position(pos)
    return CharacterInput(
        position=pos,
        velocity=Vec3.zero(),
        up=up,
        state=CharacterState.GROUNDED,
    )


def _env_clear() -> EnvContext:
    return EnvContext(wind_speed=0.0, dust=0.0, visibility=1.0, storm_intensity=0.0)


def _astro_clear() -> AstroContext:
    return AstroContext(eclipse_factor=0.0, ring_shadow=0.0)


def _astro_eclipse() -> AstroContext:
    return AstroContext(eclipse_factor=0.85, ring_shadow=0.0)


def _tick(cam: CinematicCameraSystem, n: int, dt: float,
          char: CharacterInput, env: EnvContext, astro: AstroContext,
          geo=None):
    frame = None
    for _ in range(n):
        frame = cam.update(dt, char, env, astro, geo)
    return frame


# ---------------------------------------------------------------------------
# 1. Hysteresis: mode does not bounce when metrics oscillate near threshold
# ---------------------------------------------------------------------------

class TestModeHysteresis(unittest.TestCase):

    def test_mode_does_not_oscillate_at_threshold(self):
        """Visibility alternating just around the struggle threshold must not
        flip the camera mode on every frame — cooldown prevents it."""
        cam = CinematicCameraSystem()
        char = _char_on_surface()
        astro = _astro_clear()
        dt = 0.05  # 20 fps

        # Drive into Struggle with clearly low visibility
        env_in = EnvContext(wind_speed=5.0, dust=0.8, visibility=0.2, storm_intensity=0.7)
        _tick(cam, 40, dt, char, env_in, astro)
        self.assertEqual(cam.mode, CameraMode.STRUGGLE)

        # Now oscillate right at the *exit* threshold — should NOT immediately
        # switch back to Exploration because the cooldown has not elapsed.
        env_border = EnvContext(wind_speed=2.0, dust=0.3, visibility=0.5, storm_intensity=0.3)
        mode_changes = 0
        prev = cam.mode
        # Run for only a brief period (shorter than cooldown of 5 s)
        for _ in range(int(0.5 / dt)):  # 0.5 seconds  (< 5 s cooldown)
            cam.update(dt, char, env_border, astro)
            if cam.mode != prev:
                mode_changes += 1
                prev = cam.mode

        self.assertEqual(mode_changes, 0,
                         f"Mode changed {mode_changes} time(s) within cooldown window")

    def test_mode_can_switch_after_cooldown(self):
        """Mode CAN switch after the cooldown has elapsed."""
        cam = CinematicCameraSystem()
        char = _char_on_surface()
        astro = _astro_clear()
        dt = 0.1

        # Enter Struggle
        env_struggle = EnvContext(wind_speed=5.0, dust=0.9, visibility=0.1, storm_intensity=0.8)
        _tick(cam, 50, dt, char, env_struggle, astro)  # 5 s
        self.assertEqual(cam.mode, CameraMode.STRUGGLE)

        # Wait long enough for cooldown (> 5 s), then provide clear env
        env_clear = _env_clear()
        _tick(cam, 80, dt, char, env_clear, astro)  # 8 s
        self.assertEqual(cam.mode, CameraMode.EXPLORATION)


# ---------------------------------------------------------------------------
# 2. Collision: camera must not go below the surface
# ---------------------------------------------------------------------------

class TestCollisionPreventsPenetration(unittest.TestCase):

    def test_camera_stays_above_surface(self):
        """After any number of updates the camera position must be at or
        above the planet surface (length >= planet_radius)."""
        cam = CinematicCameraSystem()
        # Character slightly above surface
        char = _char_on_surface()
        env = _env_clear()
        astro = _astro_clear()

        for _ in range(60):
            frame = cam.update(0.016, char, env, astro)
            cam_r = frame.cam_pos.length()
            self.assertGreaterEqual(
                cam_r,
                _PLANET_R - 0.5,   # allow tiny floating-point tolerance
                f"Camera penetrated surface: cam_r={cam_r:.4f} < planet_r={_PLANET_R}",
            )

    def test_collision_corrects_underground_desired(self):
        """The _collide method (internal) must move the camera back toward
        origin when the desired position would be underground."""
        # Direct unit test of internal collision logic
        cam = CinematicCameraSystem()
        up = Vec3(0.0, 1.0, 0.0)
        char_pos = Vec3(0.0, _PLANET_R, 0.0)

        # Look-at is on the surface; desired is *below* the surface
        origin = char_pos + up * 1.6
        desired_underground = Vec3(0.0, _PLANET_R - 5.0, 0.0)

        corrected = cam._collide(origin, desired_underground, up)
        # Corrected should be above the raw desired position
        self.assertGreater(
            corrected.length(),
            desired_underground.length(),
            "Corrected position should be higher than underground desired position",
        )


# ---------------------------------------------------------------------------
# 3. Awe trigger
# ---------------------------------------------------------------------------

class TestAweTrigger(unittest.TestCase):

    def test_awe_triggers_on_eclipse(self):
        """When eclipseFactor is high (> 0.7) and visibility is good the
        camera must transition to Awe mode within a reasonable time."""
        cam = CinematicCameraSystem()
        char = _char_on_surface()
        env = EnvContext(wind_speed=0.5, dust=0.05, visibility=0.9, storm_intensity=0.0)
        astro = _astro_eclipse()   # eclipse_factor = 0.85

        # Run for up to 10 seconds; Awe should kick in
        for _ in range(200):   # 200 * 0.05 = 10 s
            cam.update(0.05, char, env, astro)
            if cam.mode == CameraMode.AWE:
                break

        self.assertEqual(cam.mode, CameraMode.AWE,
                         "Camera should enter Awe mode during strong eclipse")

    def test_awe_does_not_trigger_during_storm(self):
        """Awe mode must NOT trigger when visibility is low (storm)."""
        cam = CinematicCameraSystem()
        char = _char_on_surface()
        # Eclipse present but storm obscures sky
        env = EnvContext(wind_speed=10.0, dust=0.9, visibility=0.2, storm_intensity=0.8)
        astro = _astro_eclipse()

        for _ in range(200):
            cam.update(0.05, char, env, astro)

        self.assertNotEqual(cam.mode, CameraMode.AWE,
                            "Awe must not trigger when visibility is too low")

    def test_awe_respects_cooldown(self):
        """After leaving Awe mode the cooldown must prevent immediate re-entry."""
        cam = CinematicCameraSystem()
        char = _char_on_surface()
        env_clear = EnvContext(wind_speed=0.0, dust=0.0, visibility=1.0, storm_intensity=0.0)
        astro_eclipse = _astro_eclipse()
        astro_clear = _astro_clear()

        # Trigger Awe
        for _ in range(200):
            cam.update(0.05, char, env_clear, astro_eclipse)
            if cam.mode == CameraMode.AWE:
                break
        self.assertEqual(cam.mode, CameraMode.AWE)

        # Hold Awe for min_hold_sec (8 s default)
        for _ in range(int(8.0 / 0.05)):
            cam.update(0.05, char, env_clear, astro_eclipse)

        # Eclipse fades
        for _ in range(int(5.0 / 0.05)):
            cam.update(0.05, char, env_clear, astro_clear)
        self.assertNotEqual(cam.mode, CameraMode.AWE)

        # Immediately try to re-trigger — cooldown should block
        for _ in range(int(2.0 / 0.05)):  # only 2 s, cooldown is 90 s
            cam.update(0.05, char, env_clear, astro_eclipse)

        self.assertNotEqual(cam.mode, CameraMode.AWE,
                            "Awe should be on cooldown after just leaving it")


# ---------------------------------------------------------------------------
# 4. Up alignment
# ---------------------------------------------------------------------------

class TestUpAlignment(unittest.TestCase):

    def test_camera_up_close_to_planet_up(self):
        """The camera's 'up' axis (Y column of rotation) should stay close to
        the planet Up vector.  We allow a generous threshold because of
        look-at geometry, but it should not be wildly off."""
        cam = CinematicCameraSystem()
        char = _char_on_surface()
        env = _env_clear()
        astro = _astro_clear()

        for _ in range(120):
            frame = cam.update(0.016, char, env, astro)

        # Extract camera 'up' from the rotation quaternion (local Y axis)
        cam_up_world = frame.cam_rot.rotate_vec(Vec3(0.0, 1.0, 0.0))
        planet_up = PlanetMath.up_at_position(char.position)

        dot = cam_up_world.dot(planet_up)
        self.assertGreater(
            dot,
            0.5,   # within 60 degrees
            f"Camera up {cam_up_world} diverged from planet up {planet_up} (dot={dot:.3f})",
        )

    def test_camera_up_on_different_positions(self):
        """Camera up alignment holds regardless of character position on
        the sphere."""
        cam = CinematicCameraSystem()
        env = _env_clear()
        astro = _astro_clear()

        positions = [
            Vec3(1.0, 0.0, 0.0),
            Vec3(0.0, 0.0, 1.0),
            Vec3(0.577, 0.577, 0.577),
            Vec3(-1.0, 0.5, 0.3),
        ]
        for pos_dir in positions:
            char = _char_on_surface(pos_dir)
            cam2 = CinematicCameraSystem()
            for _ in range(120):
                frame = cam2.update(0.016, char, env, astro)

            cam_up = frame.cam_rot.rotate_vec(Vec3(0.0, 1.0, 0.0))
            planet_up = PlanetMath.up_at_position(char.position)
            dot = cam_up.dot(planet_up)
            self.assertGreater(
                dot, 0.5,
                f"Camera up misaligned at pos {pos_dir}: dot={dot:.3f}",
            )


if __name__ == "__main__":
    unittest.main()
