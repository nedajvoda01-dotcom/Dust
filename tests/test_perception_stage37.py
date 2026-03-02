"""test_perception_stage37.py — Stage 37 Perception Field smoke tests.

Tests
-----
1. test_risk_increases_in_low_friction
   — GroundStabilityField raises slip_risk when friction is near zero.

2. test_attention_points_to_loud_impulse
   — AudioSalienceField.audio_dir points toward the dominant audio source.

3. test_visibility_affects_confidence
   — PerceptionSystem.movementConfidence drops with high dust_density.

4. test_presence_assist_opportunity_when_other_slips
   — PresenceField.assist_opportunity > 0 when a nearby player is slipping.

5. test_deterministic_perception_hash
   — Two identical PerceptionSystem instances with the same inputs produce
     identical PerceptionState values (determinism requirement §8).
"""
from __future__ import annotations

import hashlib
import math
import os
import struct
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.math.Vec3 import Vec3
from src.perception.AudioSalience       import AudioSalienceField, AudioSource
from src.perception.GroundStability     import GroundStabilityField
from src.perception.PresenceField       import PresenceField, OtherPlayerState
from src.perception.PerceptionSystem    import PerceptionSystem, PerceptionEnv
from src.perception.ThreatAggregator    import ThreatAggregator
from src.perception.PerceptionNetInputs import PerceptionNetInputs
from src.perception.VibrationField      import VibrationField, GeoVibrationSignal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_CONFIG = {
    "perception": {
        "tick_hz":            20,
        "audio_hz":           20,
        "ground_hz":          10,
        "wind_hz":            10,
        "vis_hz":              5,
        "vibration_hz":       10,
        "presence_hz":        20,
        "smoothing_tau_sec":  0.05,   # fast convergence for tests
        "audio": {
            "salience_radius":   100.0,
            "lowfreq_urgency_k": 2.0,
        },
        "ground": {
            "slip_mu_weight": 1.5,
            "sink_weight":    1.2,
        },
        "wind":       {"load_k": 1.0},
        "visibility": {"weight": 1.0},
        "vibration":  {"weight": 1.0},
        "presence":   {"radius": 20.0},
        "assist":     {"opportunity_radius": 8.0},
    }
}


def _run_ground(friction: float, steps: int = 30) -> GroundStabilityField:
    field = GroundStabilityField(_BASE_CONFIG)
    for _ in range(steps):
        field.update(friction=friction, softness=0.0, slope_deg=0.0, roughness=0.5, dt=0.1)
    return field


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGroundStability(unittest.TestCase):
    """test_risk_increases_in_low_friction"""

    def test_risk_increases_in_low_friction(self):
        high_friction = _run_ground(friction=0.9)
        low_friction  = _run_ground(friction=0.05)

        self.assertLess(
            high_friction.slip_risk,
            low_friction.slip_risk,
            "slip_risk must be higher when friction is low",
        )
        self.assertGreater(
            low_friction.slip_risk, 0.3,
            "slip_risk should be substantial at near-zero friction",
        )
        self.assertGreater(
            high_friction.support_quality,
            low_friction.support_quality,
            "support_quality must be higher when friction is good",
        )


class TestAudioSalience(unittest.TestCase):
    """test_attention_points_to_loud_impulse"""

    def test_attention_points_to_loud_impulse(self):
        field = AudioSalienceField(_BASE_CONFIG)
        listener = Vec3(0.0, 0.0, 0.0)

        # One loud source to the North (+X)
        sources = [
            AudioSource(position=Vec3(50.0, 0.0, 0.0), energy=0.9, low_freq_ratio=0.5),
        ]

        for _ in range(20):
            field.update(
                listener_pos=listener,
                sources=sources,
                dust_density=0.0,
                cave_factor=0.0,
                dt=0.05,
            )

        self.assertGreater(field.audio_salience, 0.0, "salience should be > 0 near a source")

        # Direction should be roughly toward +X
        d = field.audio_dir
        self.assertGreater(d.x, 0.5, f"audio_dir.x should be strongly positive, got {d.x:.3f}")

    def test_dust_reduces_salience(self):
        field_clear = AudioSalienceField(_BASE_CONFIG)
        field_dusty = AudioSalienceField(_BASE_CONFIG)
        listener = Vec3(0.0, 0.0, 0.0)
        sources = [
            AudioSource(position=Vec3(30.0, 0.0, 0.0), energy=1.0, low_freq_ratio=0.0),
        ]
        for _ in range(20):
            field_clear.update(listener, sources, dust_density=0.0, dt=0.05)
            field_dusty.update(listener, sources, dust_density=0.9, dt=0.05)

        self.assertGreater(
            field_clear.audio_salience,
            field_dusty.audio_salience,
            "high dust should cut high-frequency salience",
        )


class TestVisibilityConfidence(unittest.TestCase):
    """test_visibility_affects_confidence"""

    def test_visibility_affects_confidence(self):
        sys_clear = PerceptionSystem(_BASE_CONFIG, sim_time=0.0)
        sys_dusty = PerceptionSystem(_BASE_CONFIG, sim_time=0.0)

        env_clear = PerceptionEnv(dust_density=0.0, position=Vec3(0, 0, 0))
        env_dusty = PerceptionEnv(dust_density=0.95, position=Vec3(0, 0, 0))

        t = 0.0
        for _ in range(30):
            t += 0.1
            sys_clear.update(0.1, t, env_clear)
            sys_dusty.update(0.1, t, env_dusty)

        state_clear = sys_clear.debug_info()
        state_dusty = sys_dusty.debug_info()

        self.assertGreater(
            state_clear["visibility"],
            state_dusty["visibility"],
            "visibility must be lower in heavy dust",
        )
        self.assertGreater(
            state_clear["movementConfidence"],
            state_dusty["movementConfidence"],
            "movementConfidence must be lower when visibility is poor",
        )


class TestPresenceAssist(unittest.TestCase):
    """test_presence_assist_opportunity_when_other_slips"""

    def test_presence_assist_opportunity_when_other_slips(self):
        field = PresenceField(_BASE_CONFIG)
        listener = Vec3(0.0, 0.0, 0.0)

        # Nearby player is slipping
        slipping_player = OtherPlayerState(
            position=Vec3(5.0, 0.0, 0.0),
            velocity=Vec3(1.0, 0.0, 0.0),
            is_slipping=True,
        )

        for _ in range(20):
            field.update(
                listener_pos=listener,
                others=[slipping_player],
                self_global_risk=0.1,  # low self-risk → can assist
                dt=0.05,
            )

        self.assertGreater(
            field.assist_opportunity, 0.0,
            "assist_opportunity must be > 0 when a nearby player is slipping",
        )
        self.assertGreater(
            field.presence_near, 0.0,
            "presence_near must be > 0 when another player is close",
        )

    def test_no_assist_when_self_at_risk(self):
        field = PresenceField(_BASE_CONFIG)
        listener = Vec3(0.0, 0.0, 0.0)
        slipping_player = OtherPlayerState(
            position=Vec3(3.0, 0.0, 0.0),
            is_slipping=True,
        )
        for _ in range(20):
            field.update(
                listener_pos=listener,
                others=[slipping_player],
                self_global_risk=1.0,  # self is fully at risk
                dt=0.05,
            )

        self.assertAlmostEqual(
            field.assist_opportunity, 0.0, places=3,
            msg="assist_opportunity must be ~0 when self_global_risk is 1.0",
        )


class TestDeterminism(unittest.TestCase):
    """test_deterministic_perception_hash"""

    @staticmethod
    def _hash_state(state_dict: dict) -> str:
        buf = bytearray()
        for key in sorted(state_dict.keys()):
            v = state_dict[key]
            if isinstance(v, (int, float)):
                buf += struct.pack("d", float(v))
            elif isinstance(v, tuple):
                for x in v:
                    buf += struct.pack("d", float(x))
        return hashlib.sha256(buf).hexdigest()

    def test_deterministic_perception_hash(self):
        def _run() -> str:
            sys = PerceptionSystem(_BASE_CONFIG, sim_time=0.0)
            env = PerceptionEnv(
                dust_density=0.3,
                friction=0.5,
                wind_vec=Vec3(5.0, 0.0, 2.0),
                position=Vec3(0.0, 0.0, 0.0),
                audio_sources=[
                    AudioSource(Vec3(20.0, 0.0, 0.0), energy=0.7, low_freq_ratio=0.4)
                ],
            )
            t = 0.0
            for _ in range(20):
                t += 0.05
                sys.update(0.05, t, env)
            return self._hash_state(sys.debug_info())

        h1 = _run()
        h2 = _run()
        self.assertEqual(h1, h2, "Identical runs must produce identical PerceptionState hashes")


class TestNetInputs(unittest.TestCase):
    """Encode/decode motor flags round-trip."""

    def test_encode_decode_roundtrip(self):
        flags = PerceptionNetInputs.encode_flags(
            is_slipping=True, is_stumbling=False, support_ok=True
        )
        decoded = PerceptionNetInputs.decode_flags(flags)
        self.assertTrue(decoded.is_slipping)
        self.assertFalse(decoded.is_stumbling)
        self.assertTrue(decoded.support_ok)

    def test_all_flags_false(self):
        flags = PerceptionNetInputs.encode_flags(
            is_slipping=False, is_stumbling=False, support_ok=False
        )
        decoded = PerceptionNetInputs.decode_flags(flags)
        self.assertFalse(decoded.is_slipping)
        self.assertFalse(decoded.is_stumbling)
        self.assertFalse(decoded.support_ok)


if __name__ == "__main__":
    unittest.main()
