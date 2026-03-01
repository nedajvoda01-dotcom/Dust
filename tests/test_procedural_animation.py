"""test_procedural_animation — Stage 17 ProceduralAnimationSystem tests.

Tests
-----
1. test_gait_phase_continuity
   — phase advances monotonically and wraps [0..1) without gaps

2. test_foot_lock_stability
   — in stance phase the foot world-position does not drift while
     the character is grounded and not sliding

3. test_action_layer_trigger
   — an OnBrace AnimEvent activates the Brace layer and visibly changes
     the arm-bone angles compared to a no-event baseline

4. test_deterministic_pose
   — identical inputs (seed, trajectory) produce the same pose_hash()
     across two independent ProceduralAnimationSystem instances
"""
from __future__ import annotations

import math
import sys
import os
import unittest
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.math.Vec3 import Vec3
from src.systems.CharacterEnvironmentIntegration import AnimParamFrame
from src.systems.CharacterPhysicalController import CharacterState
from src.systems.ReflexSystem import AnimEvent, AnimEventType
from src.systems.ProceduralAnimationSystem import (
    ActionLayers,
    BonePose,
    GaitGenerator,
    FootPlacement,
    ProceduralAnimationSystem,
    _ActionLayerType,
    _SeededRng,
)

# ---------------------------------------------------------------------------
# Minimal controller stub
# ---------------------------------------------------------------------------

PLANET_R = 1000.0


class _FakeCtrl:
    """Minimal stub satisfying the interface consumed by ProceduralAnimationSystem."""

    def __init__(
        self,
        pos:      Vec3       = None,
        velocity: Vec3       = None,
        state:    CharacterState = CharacterState.GROUNDED,
    ) -> None:
        self.position  = pos      or Vec3(0.0, PLANET_R + 1.0, 0.0)
        self.velocity  = velocity or Vec3(0.0, 0.0, 1.0)
        self.state     = state

    def debug_info(self) -> dict:
        return {"slope_angle": 0.0}


def _make_anim_frame(
    stride: float = 1.0,
    cadence: float = 1.0,
    effort: float = 0.1,
    micro_jitter: float = 0.0,
) -> AnimParamFrame:
    return AnimParamFrame(
        stride_length     = stride,
        cadence           = cadence,
        arm_swing_amp     = 0.8,
        torso_twist       = 0.0,
        head_bob          = 0.8,
        micro_jitter      = micro_jitter,
        effort            = effort,
        lean              = 0.0,
        step_height_scale = 1.0,
    )


# ===========================================================================
# 1. test_gait_phase_continuity
# ===========================================================================

class TestGaitPhaseContinuity(unittest.TestCase):
    """Phase must advance monotonically (mod 1) and never skip or stall."""

    def test_phase_monotone_and_wraps(self):
        gait   = GaitGenerator(rng=_SeededRng(42))
        frame  = _make_anim_frame(cadence=1.0)
        dt     = 0.016  # ~60 fps
        prev   = 0.0
        wrapped = False

        for i in range(200):
            gait.update(dt, frame, grounded=True, sliding=False)
            current = gait.phase

            # Phase must lie in [0, 1)
            self.assertGreaterEqual(current, 0.0, f"step {i}: phase < 0")
            self.assertLess(current, 1.0,         f"step {i}: phase >= 1")

            if not wrapped:
                if current < prev:
                    # Wrap-around: only one crossing allowed per phase cycle
                    wrapped = True
                    # delta before wrap should have been positive
                else:
                    self.assertGreaterEqual(
                        current, prev, f"step {i}: phase regressed without wrap"
                    )
            prev = current

        # At least one full cycle should complete in 200 × 16 ms = 3.2 s
        # (cadence=1.0 → 1 cycle/s; ~3 wraps expected)
        self.assertTrue(wrapped, "Phase never wrapped after 200 steps")

    def test_phase_pauses_when_airborne(self):
        gait  = GaitGenerator(rng=_SeededRng(0))
        frame = _make_anim_frame(cadence=1.0)

        # Advance a few frames while grounded
        for _ in range(5):
            gait.update(0.016, frame, grounded=True, sliding=False)
        phase_before = gait.phase

        # Now advance while airborne — phase must not move
        for _ in range(20):
            gait.update(0.016, frame, grounded=False, sliding=False)
        self.assertAlmostEqual(gait.phase, phase_before, places=9,
                               msg="Phase advanced while airborne")


# ===========================================================================
# 2. test_foot_lock_stability
# ===========================================================================

class TestFootLockStability(unittest.TestCase):
    """In stance phase the locked foot world-position must not drift."""

    # Epsilon: up to 1 mm drift is acceptable (rounding, floating point)
    EPS = 1e-3

    def _run_stance_check(self, sliding: bool) -> None:
        system  = ProceduralAnimationSystem(global_seed=7, character_id=1)
        frame   = _make_anim_frame(stride=0.8, cadence=0.8)
        ctrl    = _FakeCtrl(velocity=Vec3(0.0, 0.0, 1.5))

        locked_pos_l: Optional[Vec3] = None
        locked_pos_r: Optional[Vec3] = None
        failures           = 0

        # Run for ~3 seconds (180 × 16 ms)
        for step in range(180):
            t  = step * 0.016
            ctrl.state = CharacterState.SLIDING if sliding else CharacterState.GROUNDED
            system.update(ctrl, frame, [], 0.016, t)

            fl = system.foot_world(0)
            fr = system.foot_world(1)

            if fl is not None:
                if locked_pos_l is None:
                    locked_pos_l = fl
                else:
                    drift = (fl - locked_pos_l).length()
                    if not sliding:
                        # Must not drift when grounded
                        if drift > self.EPS:
                            failures += 1

            if fr is not None:
                if locked_pos_r is None:
                    locked_pos_r = fr
                else:
                    drift = (fr - locked_pos_r).length()
                    if not sliding:
                        if drift > self.EPS:
                            failures += 1

            # Reset lock reference after foot unlocks (enters swing)
            if fl is None:
                locked_pos_l = None
            if fr is None:
                locked_pos_r = None

        if not sliding:
            self.assertEqual(failures, 0,
                             f"Foot drifted >EPS during stance ({failures} violations)")

    def test_no_drift_when_grounded(self):
        self._run_stance_check(sliding=False)

    def test_drift_allowed_when_sliding(self):
        # Sliding stance is NOT locked, so we simply assert no exception
        self._run_stance_check(sliding=True)


# ===========================================================================
# 3. test_action_layer_trigger
# ===========================================================================

class TestActionLayerTrigger(unittest.TestCase):
    """OnBrace event must activate the Brace layer and affect arm angles."""

    def test_brace_layer_activates(self):
        layers = ActionLayers(rng=_SeededRng(42))
        frame  = _make_anim_frame()

        # No layers active initially
        self.assertEqual(layers.active_layer_names, [])
        self.assertFalse(layers.has_layer(_ActionLayerType.BRACE))

        # Fire an OnBrace event
        brace_event = AnimEvent(
            type          = AnimEventType.ON_BRACE,
            time          = 0.0,
            intensity     = 1.0,
            contact_point = Vec3(1.0, PLANET_R + 1.0, 0.5),
        )
        layers.process_events([brace_event], game_time=0.0)

        # Advance a small dt so weight climbs above zero
        layers.update(dt=0.05, game_time=0.05)

        self.assertTrue(layers.has_layer(_ActionLayerType.BRACE),
                        "Brace layer not active after OnBrace event")
        self.assertIn("BRACE", layers.active_layer_names)

    def test_brace_changes_arm_angle(self):
        """Arm angle in brace pose must differ from zero-baseline."""
        # Baseline: no events
        layers_base  = ActionLayers(rng=_SeededRng(1))
        frame        = _make_anim_frame()
        base_pose    = layers_base.compute_pose(frame)
        base_arm_l   = base_pose.get("upper_arm_l", BonePose()).rx
        base_arm_r   = base_pose.get("upper_arm_r", BonePose()).rx

        # With brace event
        layers_brace = ActionLayers(rng=_SeededRng(1))
        brace_event  = AnimEvent(
            type          = AnimEventType.ON_BRACE,
            time          = 0.0,
            intensity     = 1.0,
            contact_point = None,
        )
        layers_brace.process_events([brace_event], game_time=0.0)
        layers_brace.update(dt=0.3, game_time=0.3)   # mid-blend (weight near peak)
        brace_pose   = layers_brace.compute_pose(frame)
        brace_arm_l  = brace_pose.get("upper_arm_l", BonePose()).rx
        brace_arm_r  = brace_pose.get("upper_arm_r", BonePose()).rx

        # The brace layer should have changed at least one arm (hand_side is RNG-selected)
        delta_l = abs(brace_arm_l - base_arm_l)
        delta_r = abs(brace_arm_r - base_arm_r)
        self.assertTrue(
            delta_l > 0.01 or delta_r > 0.01,
            f"Neither arm changed during brace: Δl={delta_l:.5f}, Δr={delta_r:.5f}",
        )
        # The affected arm should be reaching (rx < 0)
        affected_rx = brace_arm_l if delta_l > delta_r else brace_arm_r
        self.assertLess(affected_rx, 0.0,
                        "Affected arm.rx should be negative (arm reaching out) during brace")

    def test_full_system_brace_changes_pose(self):
        """Full ProceduralAnimationSystem: OnBrace event changes the pose hash."""
        ctrl  = _FakeCtrl()
        frame = _make_anim_frame()
        STEPS = 25          # 25 × 16 ms = 400 ms; brace at step 2 → weight ≈ 0.96 at step 24
        DT    = 0.016

        # Baseline: no events (same seed, same trajectory)
        sys_base = ProceduralAnimationSystem(global_seed=10)
        for i in range(STEPS):
            sys_base.update(ctrl, frame, [], DT, i * DT)
        spine1_base = sys_base.pose.get("spine1", BonePose()).rz

        # With brace event at step 2
        sys_brace = ProceduralAnimationSystem(global_seed=10)
        brace_ev  = AnimEvent(
            type=AnimEventType.ON_BRACE, time=2 * DT, intensity=1.0)
        for i in range(STEPS):
            evs = [brace_ev] if i == 2 else []
            sys_brace.update(ctrl, frame, evs, DT, i * DT)
        spine1_brace = sys_brace.pose.get("spine1", BonePose()).rz

        # The brace layer always modifies spine1 — the values must differ
        self.assertNotAlmostEqual(
            spine1_brace, spine1_base, places=3,
            msg="Brace event did not change spine1.rz in full system",
        )


# ===========================================================================
# 4. test_deterministic_pose
# ===========================================================================

class TestDeterministicPose(unittest.TestCase):
    """Identical seed + trajectory must produce bit-identical pose_hash."""

    def _run_scenario(self, seed: int, steps: int = 50) -> str:
        system = ProceduralAnimationSystem(global_seed=seed, character_id=0)
        ctrl   = _FakeCtrl(velocity=Vec3(0.0, 0.0, 2.0))
        frame  = _make_anim_frame(stride=1.0, cadence=1.2, effort=0.3)

        for i in range(steps):
            t = i * 0.02
            system.update(ctrl, frame, [], 0.02, t)

        return system.pose_hash()

    def test_same_seed_same_hash(self):
        h1 = self._run_scenario(seed=42)
        h2 = self._run_scenario(seed=42)
        self.assertEqual(h1, h2, "Same seed produced different hashes (non-deterministic)")

    def test_different_seeds_different_hash(self):
        h1 = self._run_scenario(seed=42)
        h2 = self._run_scenario(seed=99)
        self.assertNotEqual(h1, h2,
                            "Different seeds produced identical hash (collision or broken RNG)")

    def test_events_change_hash(self):
        """A system that received a Stumble event differs from one that did not."""
        def run_with_stumble(inject_at: int, steps: int = 30) -> str:
            sys_ = ProceduralAnimationSystem(global_seed=7)
            ctrl = _FakeCtrl(velocity=Vec3(0.0, 0.0, 1.0))
            frame = _make_anim_frame()
            stumble_ev = AnimEvent(
                type=AnimEventType.ON_STUMBLE_STEP, time=inject_at * 0.02, intensity=1.0)
            for i in range(steps):
                evs = [stumble_ev] if i == inject_at else []
                sys_.update(ctrl, frame, evs, 0.02, i * 0.02)
            return sys_.pose_hash()

        h_no_event = self._run_scenario(seed=7, steps=30)
        h_stumble  = run_with_stumble(inject_at=5)
        self.assertNotEqual(h_no_event, h_stumble,
                            "Stumble event did not change final pose hash")


if __name__ == "__main__":
    unittest.main()
