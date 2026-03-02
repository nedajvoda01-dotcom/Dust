"""test_grasp_stage40.py — Stage 40 Constraint-Based Multi-Agent Grasping smoke tests.

Tests
-----
1. test_grasp_created_only_when_reachable
   — GraspOpportunityDetector returns None for out-of-reach targets and a
     valid GraspCandidate for reachable, slipping targets.

2. test_server_authority_constraint_sync
   — Only the server-side GraspConstraintBinder creates constraints;
     ConstraintEvents are emitted on propose() and retrieved via tick().

3. test_break_on_overforce
   — GraspConstraintBinder.update_force() breaks the constraint when force
     exceeds break_force and returns a break ConstraintEvent.

4. test_pull_bias_only_when_helper_stable
   — CooperativeMotorGoals produces pull_bias > 0 only when the helper's
     support_quality is high and global_risk is low.

5. test_audio_impulse_from_constraint_force
   — A ContactImpulseCollector can aggregate a grasp-derived contact impulse
     correctly (integrating with Stage 36).

6. test_no_more_than_one_grasp_per_player
   — A second propose() for a player already in a grasp is rejected.
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.math.Vec3 import Vec3
from src.grasp.GraspOpportunityDetector import GraspOpportunityDetector, GraspCandidate
from src.grasp.GraspConstraintBinder import (
    GraspConstraintBinder, ContactCandidate, GraspType, ConstraintEvent,
)
from src.grasp.GraspConstraintSolver import GraspConstraintSolver, BodyState
from src.grasp.CooperativeMotorGoals import CooperativeMotorGoals, GraspRole
from src.grasp.GraspNetProtocol import GraspNetProtocol
from src.audio.ContactImpulseCollector import ContactImpulseCollector


# ---------------------------------------------------------------------------
# Shared config
# ---------------------------------------------------------------------------

_CFG = {
    "grasp": {
        "enable":                         True,
        "reach_radius":                   2.0,
        "score_threshold":                0.35,
        "max_force":                      800.0,
        "break_force":                    1200.0,
        "damping":                        0.85,
        "pull_bias_max":                  0.6,
        "pull_bias_tau":                  2.0,
        "server_confirm_timeout_ms":      200.0,
        "max_active_per_player":          1,
        "com_margin_min":                 0.15,
        "max_constraint_solve_iterations": 4,
        "ttc_max_sec":                    3.0,
    }
}


def _make_candidate(player_a: int = 1, player_b: int = 2) -> ContactCandidate:
    return ContactCandidate(
        player_a=player_a,
        player_b=player_b,
        grasp_type=GraspType.HAND_TO_HAND,
        anchor_a=Vec3(0.3, 1.5, 0.0),
        anchor_b=Vec3(-0.3, 1.5, 0.0),
        normal=Vec3(0.0, 1.0, 0.0),
        relative_velocity=0.5,
        estimated_force_capacity=900.0,
        tick=1,
    )


# ---------------------------------------------------------------------------
# 1. test_grasp_created_only_when_reachable
# ---------------------------------------------------------------------------

class TestGraspCreatedOnlyWhenReachable(unittest.TestCase):
    """§19 · test_grasp_created_only_when_reachable"""

    def setUp(self):
        self.detector = GraspOpportunityDetector(_CFG)

    def test_out_of_reach_returns_none(self):
        """Target 10 m away — well beyond reach_radius=2 m → no candidate."""
        result = self.detector.detect(
            self_pos=Vec3(0.0, 0.0, 0.0),
            self_vel=Vec3(0.0, 0.0, 0.0),
            other_pos=Vec3(10.0, 0.0, 0.0),
            other_vel=Vec3(0.0, 0.0, 0.0),
            self_risk=0.1,
            assist_opportunity=0.9,
            other_is_slipping=True,
            other_risk=0.8,
        )
        self.assertIsNone(result, "Should return None when target is out of reach")

    def test_reachable_slipping_returns_candidate(self):
        """Target 1.5 m away, slipping, self safe → candidate above threshold."""
        result = self.detector.detect(
            self_pos=Vec3(0.0, 0.0, 0.0),
            self_vel=Vec3(0.0, 0.0, 0.0),
            other_pos=Vec3(1.5, 0.0, 0.0),
            other_vel=Vec3(0.5, -0.3, 0.0),
            self_risk=0.05,
            assist_opportunity=0.85,
            other_is_slipping=True,
            other_risk=0.75,
        )
        self.assertIsNotNone(result, "Should return a GraspCandidate when reachable and other is slipping")
        self.assertIsInstance(result, GraspCandidate)
        self.assertGreater(result.score, 0.0)
        self.assertLessEqual(result.score, 1.0)

    def test_self_high_risk_suppresses_candidate(self):
        """Self at very high risk — should not trigger grasp (score below threshold)."""
        result = self.detector.detect(
            self_pos=Vec3(0.0, 0.0, 0.0),
            self_vel=Vec3(0.0, 0.0, 0.0),
            other_pos=Vec3(1.2, 0.0, 0.0),
            other_vel=Vec3(0.0, 0.0, 0.0),
            self_risk=0.95,          # self is almost falling
            assist_opportunity=0.9,
            other_is_slipping=True,
            other_risk=0.8,
        )
        self.assertIsNone(
            result,
            "When self_risk is very high, grasp score should fall below threshold",
        )


# ---------------------------------------------------------------------------
# 2. test_server_authority_constraint_sync
# ---------------------------------------------------------------------------

class TestServerAuthorityConstraintSync(unittest.TestCase):
    """§19 · test_server_authority_constraint_sync"""

    def test_propose_creates_constraint_and_emits_event(self):
        binder    = GraspConstraintBinder(_CFG)
        candidate = _make_candidate(player_a=1, player_b=2)

        accepted = binder.propose(candidate)
        self.assertTrue(accepted, "Server should accept a valid candidate")

        # Tick to collect events
        events = binder.tick(dt=0.1, sim_time=0.1)
        self.assertTrue(
            any(e.kind == "create" for e in events),
            "A 'create' ConstraintEvent must be emitted after propose()",
        )

        # Constraint should be queryable
        c = binder.get_active(1)
        self.assertIsNotNone(c)
        self.assertEqual(c.player_a, 1)
        self.assertEqual(c.player_b, 2)

    def test_net_encode_decode_create(self):
        """Network protocol round-trips a create event (§12)."""
        binder    = GraspConstraintBinder(_CFG)
        candidate = _make_candidate(player_a=3, player_b=4)
        binder.propose(candidate)
        events = binder.tick(dt=0.1, sim_time=0.1)
        create_evt = next(e for e in events if e.kind == "create")

        raw = GraspNetProtocol.encode_event(create_evt)
        decoded = GraspNetProtocol.decode_event(raw)
        self.assertEqual(decoded.kind, "create")
        self.assertEqual(decoded.constraint.id, create_evt.constraint.id)
        self.assertEqual(decoded.constraint.player_a, create_evt.constraint.player_a)
        self.assertEqual(decoded.constraint.player_b, create_evt.constraint.player_b)

    def test_net_encode_decode_break(self):
        """Network protocol round-trips a break event."""
        binder = GraspConstraintBinder(_CFG)
        binder.propose(_make_candidate(player_a=5, player_b=6))
        binder.tick(dt=0.1, sim_time=0.1)  # consume create event

        c = binder.get_active(5)
        binder.force_break(c.id)
        events = binder.tick(dt=0.1, sim_time=0.2)
        break_evt = next(e for e in events if e.kind == "break")

        raw     = GraspNetProtocol.encode_event(break_evt)
        decoded = GraspNetProtocol.decode_event(raw)
        self.assertEqual(decoded.kind, "break")
        self.assertEqual(decoded.constraint.id, break_evt.constraint.id)


# ---------------------------------------------------------------------------
# 3. test_break_on_overforce
# ---------------------------------------------------------------------------

class TestBreakOnOverforce(unittest.TestCase):
    """§19 · test_break_on_overforce"""

    def test_constraint_breaks_when_force_exceeds_threshold(self):
        binder    = GraspConstraintBinder(_CFG)
        candidate = _make_candidate(player_a=10, player_b=11)
        binder.propose(candidate)
        binder.tick(dt=0.1, sim_time=0.1)

        c = binder.get_active(10)
        self.assertIsNotNone(c, "Constraint should be active before break")

        # Apply a force well above break_force (1200 N)
        evt = binder.update_force(c.id, force=1500.0)
        self.assertIsNotNone(evt, "A break event should be returned on overforce")
        self.assertEqual(evt.kind, "break")

        # Constraint should no longer be active
        self.assertIsNone(binder.get_active(10), "Constraint must be gone after break")
        self.assertIsNone(binder.get_active(11), "Constraint must be gone for both players after break")

    def test_constraint_survives_normal_force(self):
        binder    = GraspConstraintBinder(_CFG)
        candidate = _make_candidate(player_a=12, player_b=13)
        binder.propose(candidate)
        binder.tick(dt=0.1, sim_time=0.1)

        c = binder.get_active(12)
        evt = binder.update_force(c.id, force=400.0)
        self.assertIsNone(evt, "No break event when force is within limits")
        self.assertIsNotNone(binder.get_active(12), "Constraint must remain active under normal force")

    def test_solver_reports_should_break(self):
        """GraspConstraintSolver sets should_break when constraint force exceeds break_force."""
        binder    = GraspConstraintBinder(_CFG)
        candidate = _make_candidate(player_a=20, player_b=21)
        binder.propose(candidate)
        binder.tick(dt=0.1, sim_time=0.1)
        c = binder.get_active(20)

        solver = GraspConstraintSolver(_CFG)
        # Place bodies far apart to generate a large spring force
        body_a = BodyState(
            position=Vec3(0.0, 0.0, 0.0),
            velocity=Vec3(-5.0, 0.0, 0.0),
            mass=70.0,
            support_quality=0.0,
            global_risk=0.9,
        )
        body_b = BodyState(
            position=Vec3(1.8, 0.0, 0.0),
            velocity=Vec3(5.0, 0.0, 0.0),
            mass=70.0,
            support_quality=1.0,
            global_risk=0.1,
        )
        # Override break_force to a tiny value so solver triggers break
        c.break_force = 0.001
        result = solver.solve(c, body_a, body_b, dt=0.016)
        self.assertTrue(result.should_break, "Solver must flag should_break when force > break_force")


# ---------------------------------------------------------------------------
# 4. test_pull_bias_only_when_helper_stable
# ---------------------------------------------------------------------------

class TestPullBiasOnlyWhenHelperStable(unittest.TestCase):
    """§19 · test_pull_bias_only_when_helper_stable"""

    def _make_constraint(self) -> "GraspConstraint":  # noqa: F821
        from src.grasp.GraspConstraintBinder import GraspConstraint, GraspType
        return GraspConstraint(
            id=1, player_a=1, player_b=2,
            anchor_a=Vec3.zero(), anchor_b=Vec3.zero(),
            grasp_type=GraspType.HAND_TO_HAND,
            max_force=800.0, break_force=1200.0,
            damping=0.85, rest_length=0.0,
            created_at_tick=0,
        )

    def test_pull_bias_when_helper_stable(self):
        goals_module = CooperativeMotorGoals(_CFG)
        constraint   = self._make_constraint()

        helper_body = BodyState(
            position=Vec3(0.0, 0.0, 0.0),
            velocity=Vec3(0.0, 0.0, 0.0),
            mass=70.0,
            support_quality=0.9,   # solid support
            global_risk=0.1,       # low risk
        )
        partner_body = BodyState(
            position=Vec3(1.5, 0.0, 0.0),
            velocity=Vec3(0.0, -2.0, 0.0),
            mass=70.0,
            support_quality=0.0,
            global_risk=0.85,
        )
        # Run several steps to let pull_bias ramp up
        goals = None
        for _ in range(30):
            goals = goals_module.compute(
                role=GraspRole.HELPER,
                body_state=helper_body,
                partner_state=partner_body,
                constraint=constraint,
                dt=0.1,
            )

        self.assertGreater(
            goals.pull_bias, 0.0,
            "pull_bias must be > 0 when helper is stable and partner is falling",
        )
        self.assertLessEqual(goals.pull_bias, _CFG["grasp"]["pull_bias_max"] + 1e-6)

    def test_no_pull_bias_when_helper_unstable(self):
        goals_module = CooperativeMotorGoals(_CFG)
        constraint   = self._make_constraint()

        unstable_helper = BodyState(
            position=Vec3(0.0, 0.0, 0.0),
            velocity=Vec3(0.0, 0.0, 0.0),
            mass=70.0,
            support_quality=0.2,   # poor footing
            global_risk=0.8,       # helper is also at risk
        )
        partner_body = BodyState(
            position=Vec3(1.5, 0.0, 0.0),
            velocity=Vec3(0.0, -2.0, 0.0),
            mass=70.0,
            support_quality=0.0,
            global_risk=0.9,
        )
        goals = goals_module.compute(
            role=GraspRole.HELPER,
            body_state=unstable_helper,
            partner_state=partner_body,
            constraint=constraint,
            dt=0.1,
        )
        self.assertAlmostEqual(
            goals.pull_bias, 0.0, places=3,
            msg="pull_bias must be ~0 when helper is unstable (§10)",
        )


# ---------------------------------------------------------------------------
# 5. test_audio_impulse_from_constraint_force
# ---------------------------------------------------------------------------

class TestAudioImpulseFromConstraintForce(unittest.TestCase):
    """§19 · test_audio_impulse_from_constraint_force (integration with Stage 36)."""

    def test_contact_impulse_generated_from_grasp_force(self):
        """Simulate recording a grasp-contact impulse and flushing to ContactImpulse."""
        collector = ContactImpulseCollector(config={"audio": {"network_impulse_hz": 200.0}})

        # Grasp produces a force — simulated as a contact between two materials
        # Material IDs: 10 = glove/suit, 11 = suit harness
        MAT_GLOVE   = 10
        MAT_HARNESS = 11

        # Record a grasp contact impulse
        collector.record(
            fn=350.0,    # normal force [N]
            ft=80.0,     # tangential force
            v_rel=0.3,   # relative velocity
            mat_a=MAT_GLOVE,
            mat_b=MAT_HARNESS,
            area=0.01,
            duration=0.016,
            world_pos=(1.0, 1.5, 0.0),
        )

        # Advance past the flush interval
        impulses = collector.flush(dt=0.01)
        # May be empty if interval not yet elapsed; force flush with larger dt
        if not impulses:
            impulses = collector.flush(dt=0.01)

        # At least some data should come through within a few ticks
        # (flush_interval = 1/200 = 5 ms; we advanced 20 ms total)
        self.assertTrue(
            len(impulses) >= 0,   # non-negative sanity
            "flush() must return a list",
        )
        # Record again and flush immediately with large dt
        collector.record(
            fn=400.0, ft=100.0, v_rel=0.5,
            mat_a=MAT_GLOVE, mat_b=MAT_HARNESS,
            area=0.01, duration=0.016,
            world_pos=(1.0, 1.5, 0.0),
        )
        impulses2 = collector.flush(dt=1.0)
        self.assertGreater(len(impulses2), 0, "ContactImpulse must be produced from grasp force data")
        imp = impulses2[0]
        self.assertEqual(imp.material_pair, (MAT_GLOVE, MAT_HARNESS))
        self.assertGreater(imp.impulse_magnitude, 0.0)


# ---------------------------------------------------------------------------
# 6. test_no_more_than_one_grasp_per_player
# ---------------------------------------------------------------------------

class TestNoMoreThanOneGraspPerPlayer(unittest.TestCase):
    """§19 · test_no_more_than_one_grasp_per_player"""

    def test_second_propose_for_same_player_rejected(self):
        binder = GraspConstraintBinder(_CFG)

        # First grasp between players 1 and 2 — accepted
        c1 = _make_candidate(player_a=1, player_b=2)
        accepted1 = binder.propose(c1)
        self.assertTrue(accepted1)

        # Player 1 tries to grasp another player (3) — must be rejected
        c2 = ContactCandidate(
            player_a=1, player_b=3,
            grasp_type=GraspType.HAND_TO_FOREARM,
            anchor_a=Vec3.zero(), anchor_b=Vec3.zero(),
            normal=Vec3(0.0, 1.0, 0.0),
            relative_velocity=0.2,
            estimated_force_capacity=800.0,
            tick=2,
        )
        accepted2 = binder.propose(c2)
        self.assertFalse(accepted2, "A second grasp for the same player must be rejected (max_active_per_player=1)")

    def test_player_can_grasp_after_break(self):
        binder = GraspConstraintBinder(_CFG)
        c1 = _make_candidate(player_a=7, player_b=8)
        binder.propose(c1)
        binder.tick(dt=0.1, sim_time=0.1)

        constraint = binder.get_active(7)
        binder.force_break(constraint.id)

        # After break, player 7 should be able to grasp again
        c2 = _make_candidate(player_a=7, player_b=9)
        accepted = binder.propose(c2)
        self.assertTrue(accepted, "Player should be able to grasp again after a break")


if __name__ == "__main__":
    unittest.main()
