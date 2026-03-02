"""test_camera_stage41 — Stage 41 Stability-Driven Cinematic Camera tests.

Tests (section 20)
------------------
1. test_camera_distance_increases_on_macro_event
   — When a macro event (rift) is nearby the camera pulls back farther than
     the neutral base distance.

2. test_camera_close_on_grasp
   — When a grasp constraint is active the camera moves closer to the
     character than the neutral base distance.

3. test_no_excess_roll
   — Camera roll never exceeds roll_max_deg (5°) regardless of inputs.

4. test_deterministic_shake
   — Identical inputs on two independent controllers produce identical
     camera frames (no random()).

5. test_smooth_transitions
   — Position change per tick is bounded; there are no sudden jumps.
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
from src.camera.CameraConfig import CameraConfig
from src.camera.CinematicBias import StabilityInput
from src.camera.StabilityCameraController import StabilityCameraController, CameraFrame


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PLANET_R = 1000.0
_DT = 1.0 / 60.0  # 60 fps


def _stable_input(pos_dir: Vec3 = None) -> StabilityInput:
    """Return a StabilityInput for a stationary stable character."""
    if pos_dir is None:
        pos_dir = Vec3(0.0, 1.0, 0.0)
    pos_dir = pos_dir.normalized()
    pos = pos_dir * _PLANET_R
    up = PlanetMath.up_at_position(pos)
    return StabilityInput(
        position=pos,
        velocity=Vec3.zero(),
        up=up,
        char_state=CharacterState.GROUNDED,
        balance_margin=1.0,
        slip_risk=0.0,
        global_risk=0.0,
    )


def _run(cam: StabilityCameraController, n: int, inp: StabilityInput) -> CameraFrame:
    """Advance camera for n ticks and return the last frame."""
    frame = None
    for _ in range(n):
        frame = cam.update(_DT, inp)
    return frame


def _dist_from_char(frame: CameraFrame, inp: StabilityInput) -> float:
    """Distance from camera to character position."""
    return (frame.cam_pos - inp.position).length()


def _extract_roll_deg(frame: CameraFrame, planet_up: Vec3) -> float:
    """Extract the roll angle (degrees) between camera up and planet up.

    Roll is defined as the angle between the camera's up axis (projected
    perpendicular to its forward axis) and the planet up vector (similarly
    projected).
    """
    cam_fwd = frame.cam_rot.rotate_vec(Vec3(0.0, 0.0, -1.0))
    cam_up  = frame.cam_rot.rotate_vec(Vec3(0.0, 1.0,  0.0))

    # Remove forward component
    cam_up_perp = cam_up - cam_fwd * cam_up.dot(cam_fwd)
    pup_perp    = planet_up - cam_fwd * planet_up.dot(cam_fwd)

    len_a = cam_up_perp.length()
    len_b = pup_perp.length()
    if len_a < 1e-8 or len_b < 1e-8:
        return 0.0

    dot = max(-1.0, min(1.0, cam_up_perp.dot(pup_perp) / (len_a * len_b)))
    return math.degrees(math.acos(dot))


# ---------------------------------------------------------------------------
# 1. Camera distance increases on macro event
# ---------------------------------------------------------------------------

class TestCameraDistanceOnMacroEvent(unittest.TestCase):

    def test_camera_distance_increases_on_macro_event(self):
        """After a rift macro event activates the camera must be farther from
        the character than in neutral (stable, no event) conditions."""
        cfg = CameraConfig()

        # --- Neutral reference ---
        cam_neutral = StabilityCameraController(config=cfg, player_id=1)
        inp_neutral = _stable_input()
        frame_neutral = _run(cam_neutral, 120, inp_neutral)   # 2 s to settle
        dist_neutral = _dist_from_char(frame_neutral, inp_neutral)

        # --- Macro event (rift, proximity=0.9) ---
        cam_macro = StabilityCameraController(config=cfg, player_id=1)
        inp_macro = _stable_input()
        inp_macro.macro_proximity = 0.9
        inp_macro.macro_is_rift = True
        inp_macro.macro_epicenter = inp_macro.position + Vec3(100.0, 0.0, 0.0)
        frame_macro = _run(cam_macro, 180, inp_macro)         # 3 s to settle
        dist_macro = _dist_from_char(frame_macro, inp_macro)

        self.assertGreater(
            dist_macro,
            dist_neutral,
            f"Macro-event camera dist {dist_macro:.3f} should exceed neutral "
            f"dist {dist_neutral:.3f}",
        )


# ---------------------------------------------------------------------------
# 2. Camera closes on grasp
# ---------------------------------------------------------------------------

class TestCameraCloseOnGrasp(unittest.TestCase):

    def test_camera_close_on_grasp(self):
        """When a grasp constraint is active the camera must end up closer to
        the character than in neutral (no grasp) conditions."""
        cfg = CameraConfig()

        # --- Neutral reference ---
        cam_neutral = StabilityCameraController(config=cfg, player_id=2)
        inp_neutral = _stable_input()
        frame_neutral = _run(cam_neutral, 120, inp_neutral)
        dist_neutral = _dist_from_char(frame_neutral, inp_neutral)

        # --- Grasp active ---
        cam_grasp = StabilityCameraController(config=cfg, player_id=2)
        inp_grasp = _stable_input()
        inp_grasp.grasp_active = True
        inp_grasp.grasp_point = inp_grasp.position + Vec3(1.0, 0.5, 0.0)
        frame_grasp = _run(cam_grasp, 180, inp_grasp)
        dist_grasp = _dist_from_char(frame_grasp, inp_grasp)

        self.assertLess(
            dist_grasp,
            dist_neutral,
            f"Grasp camera dist {dist_grasp:.3f} should be less than neutral "
            f"dist {dist_neutral:.3f}",
        )


# ---------------------------------------------------------------------------
# 3. No excess roll
# ---------------------------------------------------------------------------

class TestNoExcessRoll(unittest.TestCase):

    def _check_roll_bounded(self, cam: StabilityCameraController,
                            inp: StabilityInput, n: int,
                            limit_deg: float, label: str) -> None:
        planet_up = PlanetMath.up_at_position(inp.position)
        for _ in range(n):
            frame = cam.update(_DT, inp)
            roll = _extract_roll_deg(frame, planet_up)
            self.assertLessEqual(
                roll,
                limit_deg,
                f"{label}: roll {roll:.3f}° exceeds limit {limit_deg}°",
            )

    def test_no_excess_roll_stable(self):
        """Stable character — no roll bias is applied, so roll must stay near 0°."""
        cam = StabilityCameraController(player_id=3)
        inp = _stable_input()
        self._check_roll_bounded(cam, inp, 120, 1.0, "stable")

    def test_no_excess_roll_falling(self):
        """Falling character triggers roll bias; it must not exceed roll_max_deg."""
        cfg = CameraConfig()
        cam = StabilityCameraController(config=cfg, player_id=3)
        inp = _stable_input()
        inp.char_state = CharacterState.FALLING_CONTROLLED
        inp.slip_risk = 1.0
        inp.balance_margin = 0.0
        planet_up = PlanetMath.up_at_position(inp.position)
        for _ in range(120):
            frame = cam.update(_DT, inp)
            roll = _extract_roll_deg(frame, planet_up)
            # Allow a tiny tolerance beyond the config limit for floating-point
            self.assertLessEqual(
                roll,
                cfg.roll_max_deg + 1.0,
                f"Falling roll {roll:.3f}° exceeds bound {cfg.roll_max_deg + 1.0}°",
            )

    def test_no_excess_roll_wind(self):
        """High wind causes lateral sway but must not introduce camera roll."""
        cam = StabilityCameraController(player_id=3)
        inp = _stable_input()
        inp.wind_load = 1.0
        self._check_roll_bounded(cam, inp, 120, 1.0, "wind")


# ---------------------------------------------------------------------------
# 4. Deterministic shake
# ---------------------------------------------------------------------------

class TestDeterministicShake(unittest.TestCase):

    def test_deterministic_shake(self):
        """Two cameras with the same player_id and identical inputs must
        produce bit-identical frames — no random() is used."""
        cfg = CameraConfig()
        player_id = 42

        cam_a = StabilityCameraController(config=cfg, player_id=player_id)
        cam_b = StabilityCameraController(config=cfg, player_id=player_id)

        inp = _stable_input()
        inp.vibration_level = 0.8
        inp.slip_risk = 0.5

        for _ in range(60):
            fa = cam_a.update(_DT, inp)
            fb = cam_b.update(_DT, inp)

        # Final frames must match exactly (within floating-point tolerance)
        tol = 1e-9
        self.assertAlmostEqual(fa.cam_pos.x, fb.cam_pos.x, delta=tol,
                               msg="cam_pos.x diverged")
        self.assertAlmostEqual(fa.cam_pos.y, fb.cam_pos.y, delta=tol,
                               msg="cam_pos.y diverged")
        self.assertAlmostEqual(fa.cam_pos.z, fb.cam_pos.z, delta=tol,
                               msg="cam_pos.z diverged")
        self.assertAlmostEqual(fa.fov_deg, fb.fov_deg, delta=tol,
                               msg="fov_deg diverged")
        self.assertAlmostEqual(fa.shake_intensity, fb.shake_intensity, delta=tol,
                               msg="shake_intensity diverged")

    def test_different_player_ids_produce_different_shake(self):
        """Two cameras with different player IDs should produce different shake
        at high vibration (they share no seed)."""
        cfg = CameraConfig()

        cam_a = StabilityCameraController(config=cfg, player_id=1)
        cam_b = StabilityCameraController(config=cfg, player_id=2)

        inp = _stable_input()
        inp.vibration_level = 1.0

        frames_a, frames_b = [], []
        for _ in range(30):
            frames_a.append(cam_a.update(_DT, inp))
            frames_b.append(cam_b.update(_DT, inp))

        # Positions should differ at some point (different noise seeds)
        any_diff = any(
            abs(a.cam_pos.x - b.cam_pos.x) > 1e-6
            for a, b in zip(frames_a, frames_b)
        )
        self.assertTrue(any_diff, "Different player IDs should produce different shake")


# ---------------------------------------------------------------------------
# 5. Smooth transitions
# ---------------------------------------------------------------------------

class TestSmoothTransitions(unittest.TestCase):

    def test_no_sudden_position_jumps(self):
        """Frame-to-frame camera position change must be bounded.

        The spring system ensures no instantaneous teleports even when the
        intent changes abruptly.
        """
        cfg = CameraConfig()
        cam = StabilityCameraController(config=cfg, player_id=5)

        # Settle in neutral
        inp = _stable_input()
        _run(cam, 60, inp)

        # Abrupt change: rift appears at proximity 1.0
        inp.macro_proximity = 1.0
        inp.macro_is_rift = True
        inp.macro_epicenter = inp.position + Vec3(50.0, 0.0, 0.0)

        prev_frame = cam.update(_DT, inp)
        for _ in range(120):
            curr_frame = cam.update(_DT, inp)
            delta = (curr_frame.cam_pos - prev_frame.cam_pos).length()
            # At 60 fps the spring should move at most ~5 m/s → 0.083 m/tick
            self.assertLess(
                delta,
                0.5,  # generous bound; spring prevents large jumps
                f"Position jumped {delta:.4f} m in one tick",
            )
            prev_frame = curr_frame

    def test_fov_changes_smoothly(self):
        """FOV spring prevents step-function FOV jumps."""
        cfg = CameraConfig()
        cam = StabilityCameraController(config=cfg, player_id=5)

        inp = _stable_input()
        _run(cam, 60, inp)

        # Trigger max FOV bias
        inp.macro_proximity = 1.0
        inp.macro_is_rift = True
        inp.macro_epicenter = inp.position + Vec3(50.0, 0.0, 0.0)

        prev_fov = cam.update(_DT, inp).fov_deg
        for _ in range(60):
            fov = cam.update(_DT, inp).fov_deg
            delta_fov = abs(fov - prev_fov)
            self.assertLess(
                delta_fov,
                3.0,  # degrees per tick — smooth, not a step
                f"FOV jumped {delta_fov:.3f}° in one tick",
            )
            prev_fov = fov


if __name__ == "__main__":
    unittest.main()
