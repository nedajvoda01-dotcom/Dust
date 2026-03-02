"""test_social_stage39.py — Stage 39 Emergent Social Coupling smoke tests.

Tests
-----
1. test_first_contact_acknowledge
   — A player appearing within presence radius triggers AcknowledgePresence
     or a higher-priority social mode (not Ignore).

2. test_parallel_walk_speed_alignment
   — Two players moving in the same direction produce positive
     alignmentStrength in SocialBiases.

3. test_collision_avoidance_yield
   — Two agents on a collision course produce a non-zero yieldBias and a
     non-zero yieldDir from CollisionYield.

4. test_assist_prep_when_other_slips
   — An adjacent slipping player (with self at low risk) causes SocialCoupler
     to enter AssistPrep mode and raise assistBias.

5. test_high_risk_disables_social_modes
   — When own globalRisk is very high, SocialCoupler falls back to Ignore
     (or at most AcknowledgePresence) and alignmentStrength ≈ 0.

6. test_deterministic_social_mode
   — Two identical SocialCoupler runs with the same player_id and inputs
     produce bit-identical SocialBiases values.
"""
from __future__ import annotations

import hashlib
import os
import struct
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.math.Vec3 import Vec3
from src.perception.PerceptionSystem import PerceptionState
from src.social.SocialCoupler import SocialCoupler, SocialBiases
from src.social.SocialNetInputs import SocialAgentState, SocialNetInputs
from src.social.PersonalitySeed import PersonalitySeed
from src.social.ModeSelector import SocialMode
from src.social.CollisionYield import CollisionYield
from src.social.GroupCohesion import GroupCohesion


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_BASE_CONFIG = {
    "social": {
        "tick_hz":                          10,
        "presence_radius":                  20.0,
        "personal_space_min":               1.2,
        "personal_space_max":               3.5,
        "alignment_strength_max":           0.8,
        "follow_in_low_visibility_enable":  True,
        "assist_prep_threshold":            0.4,
        "yield_prediction_horizon_sec":     1.5,
        "collision_min_dist_m":             1.0,
        "mode_hysteresis_margin":           0.0,   # no margin → easier to enter modes
    }
}


def _make_perception(**kwargs) -> PerceptionState:
    defaults = dict(
        globalRisk         = 0.0,
        attentionDir       = Vec3(0, 0, 0),
        movementConfidence = 1.0,
        braceBias          = 0.0,
        slipRisk           = 0.0,
        windLoad           = 0.0,
        visibility         = 1.0,
        presenceNear       = 0.0,
        assistOpportunity  = 0.0,
        vibrationLevel     = 0.0,
        audioUrgency       = 0.0,
    )
    defaults.update(kwargs)
    return PerceptionState(**defaults)


def _make_coupler(player_id: int = 1, config: dict | None = None) -> SocialCoupler:
    return SocialCoupler(config or _BASE_CONFIG, sim_time=0.0, player_id=player_id)


def _run_coupler(
    coupler:    SocialCoupler,
    perception: PerceptionState,
    others:     list,
    self_pos:   Vec3,
    self_vel:   Vec3,
    steps:      int = 30,
) -> SocialBiases:
    t = 0.0
    biases = None
    for _ in range(steps):
        t += 0.1
        biases = coupler.update(
            dt=0.1,
            sim_time=t,
            perception=perception,
            others=others,
            self_pos=self_pos,
            self_vel=self_vel,
        )
    return biases


def _hash_biases(b: SocialBiases) -> str:
    buf = bytearray()
    for val in [
        b.preferredDistance, b.maxApproachSpeedScale, b.alignmentStrength,
        b.attentionToOther, b.assistBias, b.yieldBias,
        b.yieldDir.x, b.yieldDir.y, b.yieldDir.z,
        b.jointHazardBias,
    ]:
        buf += struct.pack("d", float(val))
    return hashlib.sha256(buf).hexdigest()


# ---------------------------------------------------------------------------
# 1. test_first_contact_acknowledge
# ---------------------------------------------------------------------------

class TestFirstContactAcknowledge(unittest.TestCase):
    """Spec §17 · test_first_contact_acknowledge"""

    def test_first_contact_acknowledge(self):
        coupler    = _make_coupler(player_id=1)
        # Ensure high sociability so AcknowledgePresence score clears the threshold
        coupler._personality.sociability = 0.8
        self_pos   = Vec3(0.0, 0.0, 0.0)
        self_vel   = Vec3(0.0, 0.0, 0.0)
        perception = _make_perception(presenceNear=0.6)

        nearby = [SocialAgentState(position=Vec3(5.0, 0.0, 0.0))]
        biases = _run_coupler(coupler, perception, nearby, self_pos, self_vel)

        mode = coupler.current_mode
        self.assertNotEqual(
            mode, SocialMode.Ignore,
            f"Expected mode != Ignore on first contact, got {mode.name}",
        )
        self.assertGreater(
            biases.attentionToOther, 0.0,
            "attentionToOther must be > 0 when another player is nearby",
        )


# ---------------------------------------------------------------------------
# 2. test_parallel_walk_speed_alignment
# ---------------------------------------------------------------------------

class TestParallelWalkAlignment(unittest.TestCase):
    """Spec §17 · test_parallel_walk_speed_alignment"""

    def test_parallel_walk_speed_alignment(self):
        coupler  = _make_coupler(player_id=2)
        # Ensure high sociability so alignment scores and mode selection fire reliably
        coupler._personality.sociability = 0.85
        self_pos = Vec3(0.0, 0.0, 0.0)
        self_vel = Vec3(1.5, 0.0, 0.0)   # moving in +X

        # Other player also moving in +X at similar speed, close by
        other = SocialAgentState(
            position=Vec3(3.0, 0.0, 0.0),
            velocity=Vec3(1.4, 0.0, 0.0),
        )
        perception = _make_perception(presenceNear=0.7, movementConfidence=0.9)
        biases = _run_coupler(coupler, perception, [other], self_pos, self_vel)

        self.assertGreater(
            biases.alignmentStrength, 0.0,
            "alignmentStrength must be > 0 when moving parallel to another player",
        )
        self.assertLessEqual(
            biases.preferredDistance, 4.5,
            "preferredDistance should not be excessively large during parallel walk",
        )


# ---------------------------------------------------------------------------
# 3. test_collision_avoidance_yield
# ---------------------------------------------------------------------------

class TestCollisionAvoidanceYield(unittest.TestCase):
    """Spec §17 · test_collision_avoidance_yield"""

    def test_collision_avoidance_yield(self):
        cy = CollisionYield(_BASE_CONFIG)

        # Head-on collision in 0.5 s
        self_pos   = Vec3(0.0, 0.0, 0.0)
        self_vel   = Vec3(2.0, 0.0, 0.0)    # moving toward +X
        other_pos  = Vec3(1.0, 0.0, 0.0)    # just 1 m away
        other_vel  = Vec3(-2.0, 0.0, 0.0)   # moving toward -X

        yield_bias, yield_dir = cy.compute(
            self_pos=self_pos,
            self_vel=self_vel,
            other_pos=other_pos,
            other_vel=other_vel,
            self_risk=0.05,
            other_risk=0.05,
            caution=0.7,
        )

        self.assertGreater(
            yield_bias, 0.0,
            "yield_bias must be > 0 on a head-on collision course",
        )
        # Avoidance direction should be non-zero and lateral (non-zero XZ)
        self.assertGreater(
            yield_dir.length(), 0.0,
            "yield_dir must be non-zero on a collision course",
        )

    def test_no_yield_when_paths_diverge(self):
        cy = CollisionYield(_BASE_CONFIG)
        self_pos  = Vec3(0.0, 0.0, 0.0)
        self_vel  = Vec3(0.0, 0.0, 1.0)    # moving in +Z
        other_pos = Vec3(5.0, 0.0, 0.0)    # 5 m away in +X
        other_vel = Vec3(1.0, 0.0, 0.0)    # moving in +X (diverging)

        yield_bias, _ = cy.compute(
            self_pos=self_pos, self_vel=self_vel,
            other_pos=other_pos, other_vel=other_vel,
        )
        self.assertAlmostEqual(
            yield_bias, 0.0, places=3,
            msg="yield_bias should be 0 when paths do not intersect",
        )


# ---------------------------------------------------------------------------
# 4. test_assist_prep_when_other_slips
# ---------------------------------------------------------------------------

class TestAssistPrepWhenOtherSlips(unittest.TestCase):
    """Spec §17 · test_assist_prep_when_other_slips"""

    def test_assist_prep_when_other_slips(self):
        coupler  = _make_coupler(player_id=3)
        self_pos = Vec3(0.0, 0.0, 0.0)
        self_vel = Vec3(0.3, 0.0, 0.0)

        # Override personality with high helpfulness so AssistPrep fires reliably
        coupler._personality.helpfulness = 0.95
        coupler._personality.sociability = 0.8

        slipping = SocialAgentState(
            position=Vec3(4.0, 0.0, 0.0),
            velocity=Vec3(0.0, 0.0, 0.0),
            is_slipping=True,
            global_risk=0.8,
        )
        perception = _make_perception(
            globalRisk=0.05,          # self is safe
            presenceNear=0.75,
            assistOpportunity=0.85,   # Stage 37 already signalled assist opportunity
            movementConfidence=0.9,
        )
        biases = _run_coupler(coupler, perception, [slipping], self_pos, self_vel)

        self.assertEqual(
            coupler.current_mode, SocialMode.AssistPrep,
            f"Expected AssistPrep, got {coupler.current_mode.name}",
        )
        self.assertGreater(
            biases.assistBias, 0.0,
            "assistBias must be > 0 in AssistPrep mode",
        )
        self.assertLess(
            biases.maxApproachSpeedScale, 1.0,
            "maxApproachSpeedScale should be reduced in AssistPrep mode",
        )


# ---------------------------------------------------------------------------
# 5. test_high_risk_disables_social_modes
# ---------------------------------------------------------------------------

class TestHighRiskDisablesSocialModes(unittest.TestCase):
    """Spec §17 · test_high_risk_disables_social_modes"""

    def test_high_risk_disables_social_modes(self):
        coupler  = _make_coupler(player_id=4)
        self_pos = Vec3(0.0, 0.0, 0.0)
        self_vel = Vec3(0.5, 0.0, 0.0)

        nearby = [
            SocialAgentState(
                position=Vec3(3.0, 0.0, 0.0),
                velocity=Vec3(0.5, 0.0, 0.0),
            )
        ]
        perception = _make_perception(
            globalRisk=0.95,    # very high personal risk
            windLoad=0.9,
            slipRisk=0.85,
            presenceNear=0.7,
            assistOpportunity=0.5,
        )
        biases = _run_coupler(coupler, perception, nearby, self_pos, self_vel)

        mode = coupler.current_mode
        self.assertIn(
            mode, (SocialMode.Ignore, SocialMode.AcknowledgePresence),
            f"Under high risk, mode should be Ignore or AcknowledgePresence, got {mode.name}",
        )
        self.assertAlmostEqual(
            biases.alignmentStrength, 0.0, places=3,
            msg="alignmentStrength must be ~0 under very high personal risk",
        )
        self.assertAlmostEqual(
            biases.assistBias, 0.0, places=3,
            msg="assistBias must be ~0 when own risk is critical",
        )


# ---------------------------------------------------------------------------
# 6. test_deterministic_social_mode
# ---------------------------------------------------------------------------

class TestDeterministicSocialMode(unittest.TestCase):
    """Spec §17 · test_deterministic_social_mode"""

    def test_deterministic_social_mode(self):
        perception = _make_perception(
            globalRisk=0.1,
            presenceNear=0.6,
            assistOpportunity=0.0,
            visibility=0.8,
        )
        others = [
            SocialAgentState(
                position=Vec3(4.0, 0.0, 0.0),
                velocity=Vec3(1.0, 0.0, 0.0),
                global_risk=0.1,
            )
        ]
        self_pos = Vec3(0.0, 0.0, 0.0)
        self_vel = Vec3(1.0, 0.0, 0.0)

        def _run() -> str:
            coupler = _make_coupler(player_id=99)
            biases = _run_coupler(coupler, perception, others, self_pos, self_vel, steps=20)
            return _hash_biases(biases)

        h1 = _run()
        h2 = _run()
        self.assertEqual(h1, h2, "Identical runs must produce identical SocialBiases")


# ---------------------------------------------------------------------------
# Bonus: SocialNetInputs encode/decode round-trip
# ---------------------------------------------------------------------------

class TestSocialNetInputsRoundTrip(unittest.TestCase):
    """Encode and decode social flags over the network codec."""

    def test_encode_decode_roundtrip(self):
        data = SocialNetInputs.encode_flags(
            is_slipping=True, is_stumbling=False, global_risk=0.6
        )
        self.assertEqual(len(data), 2)
        is_slip, is_stum, risk = SocialNetInputs.decode_flags(data)
        self.assertTrue(is_slip)
        self.assertFalse(is_stum)
        self.assertAlmostEqual(risk, 0.6, delta=0.005)

    def test_all_flags(self):
        data = SocialNetInputs.encode_flags(
            is_slipping=True, is_stumbling=True, global_risk=1.0
        )
        is_slip, is_stum, risk = SocialNetInputs.decode_flags(data)
        self.assertTrue(is_slip)
        self.assertTrue(is_stum)
        self.assertAlmostEqual(risk, 1.0, delta=0.005)

    def test_invalid_length_raises(self):
        with self.assertRaises(ValueError):
            SocialNetInputs.decode_flags(b"\x00")


# ---------------------------------------------------------------------------
# Bonus: PersonalitySeed determinism
# ---------------------------------------------------------------------------

class TestPersonalitySeedDeterminism(unittest.TestCase):
    """PersonalitySeed must always yield identical params for the same player_id."""

    def test_deterministic_personality(self):
        seed = PersonalitySeed(_BASE_CONFIG)
        p1 = seed.generate(42)
        p2 = seed.generate(42)
        self.assertEqual(p1.sociability,   p2.sociability)
        self.assertEqual(p1.caution,       p2.caution)
        self.assertEqual(p1.helpfulness,   p2.helpfulness)
        self.assertEqual(p1.personalSpace, p2.personalSpace)

    def test_different_players_differ(self):
        seed = PersonalitySeed(_BASE_CONFIG)
        p1 = seed.generate(1)
        p2 = seed.generate(2)
        # Very unlikely to be identical for different IDs
        self.assertFalse(
            p1.sociability == p2.sociability
            and p1.caution == p2.caution
            and p1.helpfulness == p2.helpfulness,
            "Different player_ids should produce different personalities",
        )


if __name__ == "__main__":
    unittest.main()
