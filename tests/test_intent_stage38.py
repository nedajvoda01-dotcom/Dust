"""test_intent_stage38.py — Stage 38 Autonomous Micro-Intent smoke tests.

Tests
-----
1. test_high_wind_selects_wide_stance
   — Strong wind + high slipRisk causes WideStanceStabilize to be selected
     and stanceWidthBias to increase significantly.

2. test_low_visibility_reduces_speed
   — Low visibility (high dust) causes desiredVelocity magnitude to drop
     compared to a clear-sky scenario.

3. test_audio_event_triggers_attention_shift
   — A strong audio event sets attentionTargetDir toward the source.

4. test_assist_preparation_when_other_slips
   — Another player slipping nearby (and self at low risk) triggers
     AssistPreparation mode with assistWillingness > 0.

5. test_hysteresis_prevents_mode_oscillation
   — ModeHysteresis does not toggle on score values inside the hysteresis
     band (enter=0.6, exit=0.4).

6. test_deterministic_intent_hash
   — Two identical IntentArbitrator runs (same player_id, same inputs)
     produce identical MotorIntent values — no random() used.
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
from src.intent.ModeHysteresis   import ModeHysteresis
from src.intent.CostEvaluator    import CostEvaluator
from src.intent.StrategySelector import StrategySelector, MotorMode, MotorIntent
from src.intent.IntentArbitrator import IntentArbitrator
from src.intent.IntentNetSync    import IntentNetSync


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_BASE_CONFIG = {
    "intent": {
        "tick_hz":                   10,
        "blend_tau_sec":             0.01,  # near-instant blend for tests
        "w_balance":                 1.0,
        "w_slip":                    1.0,
        "w_visibility":              1.0,
        "w_wind":                    1.0,
        "w_vibration":               1.0,
        "w_proximity":               0.5,
        "w_uncertainty":             0.8,
        "mode_hysteresis_margin":    0.0,
        "assist_threshold":          0.4,
        "max_speed_scale_under_risk": 0.2,
        "max_stance_bias":           0.9,
    }
}


def _make_state(**kwargs) -> PerceptionState:
    """Build a PerceptionState with neutral defaults, overriding supplied keys."""
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


def _run_arbitrator(state: PerceptionState, steps: int = 30,
                    input_vel: Vec3 | None = None) -> IntentArbitrator:
    arb = IntentArbitrator(_BASE_CONFIG, sim_time=0.0, player_id=1)
    t = 0.0
    for _ in range(steps):
        t += 0.1
        arb.update(0.1, t, state, input_velocity=input_vel)
    return arb


def _hash_intent(intent: MotorIntent) -> str:
    buf = bytearray()
    for val in [
        intent.desiredVelocity.x, intent.desiredVelocity.y, intent.desiredVelocity.z,
        intent.stanceWidthBias, intent.stepLengthBias, intent.bracePreference,
        intent.attentionTargetDir.x, intent.attentionTargetDir.y, intent.attentionTargetDir.z,
        intent.proximityPreference, intent.assistWillingness,
    ]:
        buf += struct.pack("d", float(val))
    return hashlib.sha256(buf).hexdigest()


# ---------------------------------------------------------------------------
# 1. test_high_wind_selects_wide_stance
# ---------------------------------------------------------------------------

class TestHighWindWideStance(unittest.TestCase):
    """Spec §15 · test_high_wind_selects_wide_stance"""

    def test_high_wind_selects_wide_stance(self):
        state = _make_state(
            windLoad    = 0.85,
            slipRisk    = 0.70,
            globalRisk  = 0.75,
            braceBias   = 0.60,
            movementConfidence = 0.5,
        )
        arb = _run_arbitrator(state, input_vel=Vec3(1, 0, 0))

        self.assertEqual(
            arb.current_mode, MotorMode.WideStanceStabilize,
            f"Expected WideStanceStabilize, got {arb.current_mode.name}",
        )
        intent = arb.current_intent
        self.assertGreater(
            intent.stanceWidthBias, 0.3,
            "stanceWidthBias should increase significantly under high wind",
        )
        self.assertLess(
            intent.stepLengthBias, 0.8,
            "stepLengthBias should decrease under high wind",
        )
        self.assertGreater(
            intent.bracePreference, 0.3,
            "bracePreference should be elevated under high wind",
        )


# ---------------------------------------------------------------------------
# 2. test_low_visibility_reduces_speed
# ---------------------------------------------------------------------------

class TestLowVisibilityReducesSpeed(unittest.TestCase):
    """Spec §15 · test_low_visibility_reduces_speed"""

    def test_low_visibility_reduces_speed(self):
        input_vel = Vec3(2.0, 0.0, 0.0)

        state_clear = _make_state(visibility=1.0, movementConfidence=1.0)
        state_dark  = _make_state(visibility=0.05, movementConfidence=0.1)

        arb_clear = _run_arbitrator(state_clear, input_vel=input_vel)
        arb_dark  = _run_arbitrator(state_dark,  input_vel=input_vel)

        speed_clear = arb_clear.current_intent.desiredVelocity.length()
        speed_dark  = arb_dark.current_intent.desiredVelocity.length()

        self.assertGreater(
            speed_clear, speed_dark,
            f"Speed in clear ({speed_clear:.3f}) should exceed speed in dark ({speed_dark:.3f})",
        )
        self.assertGreater(
            speed_dark, 0.0,
            "desiredVelocity should not be exactly zero even in low visibility",
        )


# ---------------------------------------------------------------------------
# 3. test_audio_event_triggers_attention_shift
# ---------------------------------------------------------------------------

class TestAudioAttentionShift(unittest.TestCase):
    """Spec §15 · test_audio_event_triggers_attention_shift"""

    def test_audio_event_triggers_attention_shift(self):
        audio_dir = Vec3(0.0, 0.0, 1.0)   # sound from +Z

        state = _make_state(
            audioUrgency  = 0.9,
            attentionDir  = audio_dir,
            globalRisk    = 0.3,
        )
        arb = _run_arbitrator(state, input_vel=Vec3(1, 0, 0))

        attn = arb.current_intent.attentionTargetDir
        # Dot product with audio_dir should be strongly positive
        dot = attn.dot(audio_dir)
        self.assertGreater(
            dot, 0.5,
            f"attentionTargetDir should point toward audio source, dot={dot:.3f}",
        )


# ---------------------------------------------------------------------------
# 4. test_assist_preparation_when_other_slips
# ---------------------------------------------------------------------------

class TestAssistPreparation(unittest.TestCase):
    """Spec §15 · test_assist_preparation_when_other_slips"""

    def test_assist_preparation_when_other_slips(self):
        state = _make_state(
            assistOpportunity = 0.9,
            presenceNear      = 0.7,
            globalRisk        = 0.1,   # self is safe
            movementConfidence = 0.9,
        )
        arb = _run_arbitrator(state, input_vel=Vec3(0.5, 0, 0))

        self.assertEqual(
            arb.current_mode, MotorMode.AssistPreparation,
            f"Expected AssistPreparation, got {arb.current_mode.name}",
        )
        intent = arb.current_intent
        self.assertGreater(
            intent.assistWillingness, 0.0,
            "assistWillingness must be > 0 in AssistPreparation mode",
        )
        self.assertGreater(
            intent.bracePreference, 0.3,
            "bracePreference should be elevated in AssistPreparation mode",
        )


# ---------------------------------------------------------------------------
# 5. test_hysteresis_prevents_mode_oscillation
# ---------------------------------------------------------------------------

class TestHysteresisPreventsOscillation(unittest.TestCase):
    """Spec §15 · test_hysteresis_prevents_mode_oscillation"""

    def test_hysteresis_prevents_mode_oscillation(self):
        h = ModeHysteresis(enter_threshold=0.6, exit_threshold=0.4)

        # Below enter: stays inactive
        self.assertFalse(h.update(0.3))
        self.assertFalse(h.update(0.55))

        # Cross enter: activates
        self.assertTrue(h.update(0.65))

        # Inside band: stays active
        self.assertTrue(h.update(0.5))    # 0.4 <= 0.5 <= 0.6
        self.assertTrue(h.update(0.42))

        # Below exit: deactivates
        self.assertFalse(h.update(0.35))

        # Re-enters only when crossing enter again
        self.assertFalse(h.update(0.5))   # in band but was deactivated → need ≥ enter
        self.assertTrue(h.update(0.62))

    def test_arbitrator_does_not_oscillate(self):
        """Mode should not flip each tick under a borderline risk value."""
        # Create a PerceptionState that is right on the edge of WideStanceStabilize
        state = _make_state(windLoad=0.45, slipRisk=0.45, globalRisk=0.45)
        arb = IntentArbitrator(_BASE_CONFIG, sim_time=0.0)
        modes = []
        t = 0.0
        for _ in range(50):
            t += 0.1
            arb.update(0.1, t, state, input_velocity=Vec3(1, 0, 0))
            modes.append(arb.current_mode)

        # Count mode transitions — should be at most 2 (enter once, stay)
        transitions = sum(1 for a, b in zip(modes, modes[1:]) if a != b)
        self.assertLessEqual(
            transitions, 2,
            f"Mode oscillated {transitions} times; hysteresis should prevent this",
        )


# ---------------------------------------------------------------------------
# 6. test_deterministic_intent_hash
# ---------------------------------------------------------------------------

class TestDeterministicIntentHash(unittest.TestCase):
    """Spec §15 · test_deterministic_intent_hash"""

    def test_deterministic_intent_hash(self):
        state = _make_state(
            windLoad           = 0.6,
            slipRisk           = 0.5,
            globalRisk         = 0.55,
            braceBias          = 0.4,
            movementConfidence = 0.6,
            visibility         = 0.7,
            audioUrgency       = 0.3,
            attentionDir       = Vec3(0.5, 0.0, 0.5).normalized(),
        )

        def _run() -> str:
            arb = IntentArbitrator(_BASE_CONFIG, sim_time=0.0, player_id=42)
            t = 0.0
            for _ in range(20):
                t += 0.1
                arb.update(0.1, t, state, input_velocity=Vec3(1, 0, 0))
            return _hash_intent(arb.current_intent)

        h1 = _run()
        h2 = _run()
        self.assertEqual(h1, h2, "Identical runs must produce identical MotorIntent")


# ---------------------------------------------------------------------------
# Bonus: IntentNetSync encode/decode round-trip
# ---------------------------------------------------------------------------

class TestIntentNetSync(unittest.TestCase):
    """Encode and decode a MotorIntent over the network codec."""

    def test_encode_decode_roundtrip(self):
        mode   = MotorMode.WideStanceStabilize
        intent = MotorIntent(
            desiredVelocity     = Vec3(1.5, 0.0, -0.5),
            stanceWidthBias     = 0.72,
            stepLengthBias      = 0.40,
            bracePreference     = 0.65,
            attentionTargetDir  = Vec3(0.0, 0.0, 1.0),
            proximityPreference = 1.2,
            assistWillingness   = 0.0,
        )
        data = IntentNetSync.encode(mode, intent)
        self.assertEqual(len(data), 38)

        decoded_mode, decoded_intent = IntentNetSync.decode(data)
        self.assertEqual(decoded_mode, mode)
        self.assertAlmostEqual(decoded_intent.stanceWidthBias,  intent.stanceWidthBias,  delta=0.005)
        self.assertAlmostEqual(decoded_intent.stepLengthBias,   intent.stepLengthBias,   delta=0.005)
        self.assertAlmostEqual(decoded_intent.bracePreference,  intent.bracePreference,  delta=0.005)
        self.assertAlmostEqual(decoded_intent.assistWillingness, intent.assistWillingness, delta=0.005)

    def test_invalid_length_raises(self):
        with self.assertRaises(ValueError):
            IntentNetSync.decode(b"\x00" * 10)


if __name__ == "__main__":
    unittest.main()
