"""test_injury_stage48.py — Stage 48 Injury & Recovery Micro-Physics smoke tests.

Tests
-----
1. test_strain_accumulates_under_high_tau
   — Sustained high joint torque causes strain to increase over time.

2. test_acute_spike_on_impact_impulse
   — An impact impulse above the threshold produces an acute spike.

3. test_injury_reduces_tau_limits
   — InjuryToQPAdapter reduces per-joint tau_scale when strain is present.

4. test_behavior_shifts_weight_away_from_injured_leg
   — InjuryToFootstepBias shifts weight preference away from the injured side.

5. test_recovery_reduces_strain_over_time
   — Strain and acute decrease during low-load rest conditions.

6. test_network_authoritative_injury_state
   — InjuryReplicator encode/decode roundtrip is lossless within quantisation,
     and a corrupted checksum returns a neutral snapshot.

7. test_determinism_replay_injury_hash
   — Two identical InjurySystem tick sequences produce identical state.
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.injury.InjurySystem import (
    InjurySystem, InjuryState, JointInjury,
    LoadInput, JointLoad, InjuryEnvInput, JOINT_NAMES,
)
from src.injury.JointLoadEstimator import JointLoadEstimator, PhysicsFrame
from src.injury.InjuryToQPAdapter import InjuryToQPAdapter
from src.injury.InjuryToFootstepBias import InjuryToFootstepBias
from src.injury.RecoveryModel import RecoveryModel
from src.net.InjuryReplicator import InjuryReplicator, InjurySnapshot


# ---------------------------------------------------------------------------
# Shared config
# ---------------------------------------------------------------------------

_CFG = {
    "injury": {
        "enable":               True,
        "tick_hz":              5,
        "k_strain":             0.04,
        "k_acute":              0.6,
        "impact_threshold":     0.25,
        "tau_floor":            0.75,
        "stiffness_min":        0.70,
        "recover_k":            0.015,
        "acute_decay_k":        0.05,
        "pain_avoidance_k":     0.08,
        "max_total_influence":  0.60,
        "strain_power":         2.0,
        "acute_to_strain":      0.15,
        "repl_hz":              2,
        "load_window_sec":      1.0,
        "max_weight_shift":     0.20,
        "max_step_caution":     0.80,
        "lean_threshold":       0.30,
        "rest_condition_thresholds": {
            "speed":   0.5,
            "wind":    0.25,
            "support": 0.65,
        },
    }
}

_LOAD_ENV  = InjuryEnvInput(speed=2.0, windLoad=0.5, supportQuality=0.8, isGrasping=False)
_REST_ENV  = InjuryEnvInput(speed=0.0, windLoad=0.0, supportQuality=1.0, isGrasping=False)
_DT        = 0.2   # 5 Hz tick


def _make_high_tau_load(joint: str = "ankle_l", tau: float = 0.9) -> LoadInput:
    """Build a LoadInput with high torque on one joint."""
    return LoadInput(joints={joint: JointLoad(tau=tau, tau_max=1.0)})


def _make_impact_load(joint: str = "knee_l", impulse: float = 0.8) -> LoadInput:
    """Build a LoadInput with an impact impulse on one joint."""
    return LoadInput(joints={joint: JointLoad(impactImpulse=impulse)})


def _make_rest_load() -> LoadInput:
    """Build a zero-load LoadInput (rest)."""
    return LoadInput(joints={})


# ---------------------------------------------------------------------------
# 1. test_strain_accumulates_under_high_tau
# ---------------------------------------------------------------------------

class TestStrainAccumulatesUnderHighTau(unittest.TestCase):
    """Sustained high torque must increase strain on the loaded joint."""

    def test_strain_increases_with_high_tau(self):
        """60 s of high-torque load must raise ankle strain above zero."""
        sys_ = InjurySystem(_CFG)
        load = _make_high_tau_load("ankle_l", tau=0.95)

        for _ in range(300):  # 60 s at 5 Hz
            sys_.tick(_DT, load, _LOAD_ENV)

        strain = sys_.state.joints["ankle_l"].strain
        self.assertGreater(strain, 0.0, "Strain must accumulate under high torque")

    def test_higher_tau_causes_more_strain(self):
        """Quadratic exponent: tau=0.9 must cause more strain than tau=0.5."""
        sys_high = InjurySystem(_CFG)
        sys_low  = InjurySystem(_CFG)

        for _ in range(100):  # 20 s
            sys_high.tick(_DT, _make_high_tau_load("knee_r", tau=0.9), _LOAD_ENV)
            sys_low.tick( _DT, _make_high_tau_load("knee_r", tau=0.5), _LOAD_ENV)

        self.assertGreater(
            sys_high.state.joints["knee_r"].strain,
            sys_low.state.joints["knee_r"].strain,
            "Higher torque must cause more strain (quadratic exponent)",
        )

    def test_zero_load_no_strain(self):
        """Zero torque must not accumulate any strain."""
        sys_ = InjurySystem(_CFG)
        for _ in range(50):
            sys_.tick(_DT, _make_rest_load(), _LOAD_ENV)

        for name in JOINT_NAMES:
            self.assertAlmostEqual(
                sys_.state.joints[name].strain, 0.0, places=5,
                msg=f"Zero load must not accumulate strain on {name}",
            )


# ---------------------------------------------------------------------------
# 2. test_acute_spike_on_impact_impulse
# ---------------------------------------------------------------------------

class TestAcuteSpikeOnImpact(unittest.TestCase):
    """An impact impulse above threshold must produce an acute spike."""

    def test_above_threshold_creates_acute(self):
        """Impulse of 0.8 (threshold=0.25) must immediately produce acute > 0."""
        sys_ = InjurySystem(_CFG)
        load = _make_impact_load("knee_l", impulse=0.8)
        sys_.tick(_DT, load, _LOAD_ENV)

        acute = sys_.state.joints["knee_l"].acute
        self.assertGreater(acute, 0.0, "Impact above threshold must produce acute > 0")

    def test_below_threshold_no_acute(self):
        """Impulse of 0.1 (below threshold=0.25) must not produce acute."""
        sys_ = InjurySystem(_CFG)
        load = _make_impact_load("knee_l", impulse=0.1)
        sys_.tick(_DT, load, _LOAD_ENV)

        acute = sys_.state.joints["knee_l"].acute
        self.assertAlmostEqual(acute, 0.0, places=3,
                               msg="Impulse below threshold must not create acute spike")

    def test_larger_impulse_causes_larger_acute(self):
        """Larger impact must produce proportionally larger acute."""
        sys_lo = InjurySystem(_CFG)
        sys_hi = InjurySystem(_CFG)

        sys_lo.tick(_DT, _make_impact_load("hip_l", impulse=0.4), _LOAD_ENV)
        sys_hi.tick(_DT, _make_impact_load("hip_l", impulse=0.9), _LOAD_ENV)

        self.assertGreater(
            sys_hi.state.joints["hip_l"].acute,
            sys_lo.state.joints["hip_l"].acute,
            "Larger impulse must produce larger acute spike",
        )

    def test_acute_decays_over_time(self):
        """Acute value must decay when no further impact is applied."""
        sys_ = InjurySystem(_CFG)
        sys_.tick(_DT, _make_impact_load("ankle_r", impulse=0.9), _LOAD_ENV)
        acute_after_impact = sys_.state.joints["ankle_r"].acute
        self.assertGreater(acute_after_impact, 0.0)

        # Rest for 30 s — acute should decrease
        for _ in range(150):
            sys_.tick(_DT, _make_rest_load(), _REST_ENV)

        self.assertLess(
            sys_.state.joints["ankle_r"].acute, acute_after_impact,
            "Acute must decay over time without further impact",
        )


# ---------------------------------------------------------------------------
# 3. test_injury_reduces_tau_limits
# ---------------------------------------------------------------------------

class TestInjuryReducesTauLimits(unittest.TestCase):
    """InjuryToQPAdapter must reduce tau_scale for injured joints."""

    def test_strained_joint_has_lower_tau_scale(self):
        """A joint with high strain must have tau_scale below 1.0."""
        sys_ = InjurySystem(_CFG)
        sys_.force_strain("ankle_l", 0.8)

        adapter = InjuryToQPAdapter(_CFG)
        adj     = adapter.adapt(sys_.state)

        self.assertLess(
            adj.tau_scale["ankle_l"], 1.0,
            "Strained joint must have reduced tau_scale",
        )
        self.assertGreaterEqual(
            adj.tau_scale["ankle_l"],
            _CFG["injury"]["tau_floor"] - 0.01,
            "tau_scale must not fall below configured tau_floor",
        )

    def test_healthy_joint_full_tau_scale(self):
        """An uninjured joint must have tau_scale = 1.0."""
        sys_ = InjurySystem(_CFG)
        # Only injure left ankle; right ankle should be unaffected
        sys_.force_strain("ankle_l", 0.8)

        adapter = InjuryToQPAdapter(_CFG)
        adj     = adapter.adapt(sys_.state)

        self.assertAlmostEqual(
            adj.tau_scale["ankle_r"], 1.0, places=3,
            msg="Healthy joint must retain full tau_scale",
        )

    def test_stiffness_scale_also_reduced(self):
        """Stiffness scale must also decrease for an injured joint."""
        sys_ = InjurySystem(_CFG)
        sys_.force_strain("knee_l", 0.7)

        adapter = InjuryToQPAdapter(_CFG)
        adj     = adapter.adapt(sys_.state)

        self.assertLess(
            adj.stiffness_scale["knee_l"], 1.0,
            "Stiffness must decrease for an injured joint",
        )

    def test_step_length_scale_decreases_with_global_injury(self):
        """Step length scale must be reduced when globalInjuryIndex is high."""
        sys_healthy  = InjurySystem(_CFG)
        sys_injured  = InjurySystem(_CFG)
        sys_injured.force_strain("ankle_l", 0.9)
        sys_injured.force_strain("knee_l",  0.7)

        adapter = InjuryToQPAdapter(_CFG)
        adj_h   = adapter.adapt(sys_healthy.state)
        adj_i   = adapter.adapt(sys_injured.state)

        self.assertLess(
            adj_i.stepLengthScale, adj_h.stepLengthScale,
            "Step length must be shorter when globally injured",
        )


# ---------------------------------------------------------------------------
# 4. test_behavior_shifts_weight_away_from_injured_leg
# ---------------------------------------------------------------------------

class TestBehaviorShiftsWeight(unittest.TestCase):
    """InjuryToFootstepBias must shift weight away from the injured leg."""

    def test_left_ankle_injury_shifts_weight_right(self):
        """Left ankle strain must increase right-leg weight preference."""
        sys_ = InjurySystem(_CFG)
        sys_.force_strain("ankle_l", 0.8)

        bias_module = InjuryToFootstepBias(_CFG)
        bias        = bias_module.bias(sys_.state)

        self.assertGreater(
            bias.weightBias_r, bias.weightBias_l,
            "Left ankle injury must shift weight preference to the right leg",
        )
        self.assertGreater(
            bias.weightBias_r, 0.5,
            "Right-leg weight bias must exceed 0.5 with left ankle injury",
        )

    def test_right_hip_injury_shifts_weight_left(self):
        """Right hip strain must increase left-leg weight preference."""
        sys_ = InjurySystem(_CFG)
        sys_.force_strain("hip_r", 0.7)

        bias_module = InjuryToFootstepBias(_CFG)
        bias        = bias_module.bias(sys_.state)

        self.assertGreater(
            bias.weightBias_l, bias.weightBias_r,
            "Right hip injury must shift weight preference to the left leg",
        )

    def test_step_caution_increases_with_injury(self):
        """Step caution must be higher on the injured side."""
        sys_healthy = InjurySystem(_CFG)
        sys_injured = InjurySystem(_CFG)
        sys_injured.force_strain("ankle_l", 0.7)

        bias_module = InjuryToFootstepBias(_CFG)
        b_h = bias_module.bias(sys_healthy.state)
        b_i = bias_module.bias(sys_injured.state)

        self.assertGreater(
            b_i.stepCaution_l, b_h.stepCaution_l,
            "Step caution must increase on injured side",
        )

    def test_avoid_lean_hint_set_with_high_pain(self):
        """avoidLeanLeft must be True when left-side pain avoidance is high."""
        sys_ = InjurySystem(_CFG)
        sys_.force_strain("knee_l", 0.9)  # sets painAvoidance ~0.72

        # Advance tick once to let painAvoidance update
        sys_.tick(_DT, _make_rest_load(), _LOAD_ENV)

        bias_module = InjuryToFootstepBias(_CFG)
        bias        = bias_module.bias(sys_.state)

        self.assertTrue(
            bias.avoidLeanLeft,
            "avoidLeanLeft should be True when left knee pain avoidance exceeds threshold",
        )


# ---------------------------------------------------------------------------
# 5. test_recovery_reduces_strain_over_time
# ---------------------------------------------------------------------------

class TestRecoveryReducesStrain(unittest.TestCase):
    """Strain and acute must decrease under rest conditions."""

    def test_strain_decreases_at_rest(self):
        """After loading, resting must reduce strain on the injured joint."""
        sys_ = InjurySystem(_CFG)
        # Apply load for 20 s
        for _ in range(100):
            sys_.tick(_DT, _make_high_tau_load("ankle_l", tau=0.95), _LOAD_ENV)
        peak_strain = sys_.state.joints["ankle_l"].strain
        self.assertGreater(peak_strain, 0.0)

        # Rest for 60 s
        for _ in range(300):
            sys_.tick(_DT, _make_rest_load(), _REST_ENV)

        self.assertLess(
            sys_.state.joints["ankle_l"].strain, peak_strain,
            "Strain must decrease during rest",
        )

    def test_recovery_model_returns_resting_true(self):
        """RecoveryModel must signal is_resting=True in calm conditions."""
        rm  = RecoveryModel(_CFG)
        env = _REST_ENV
        rc  = rm.evaluate(env)
        self.assertTrue(rc.is_resting,    "Must be resting in calm conditions")
        self.assertGreater(rc.recovery_multiplier, 0.0)

    def test_recovery_model_not_resting_in_high_load(self):
        """RecoveryModel must signal is_resting=False during hard activity."""
        rm  = RecoveryModel(_CFG)
        env = InjuryEnvInput(speed=3.0, windLoad=0.8, supportQuality=0.3)
        rc  = rm.evaluate(env)
        self.assertFalse(rc.is_resting, "Must not be resting during high load")
        self.assertEqual(rc.recovery_multiplier, 0.0)

    def test_continued_load_slows_recovery(self):
        """Resting with high load must recover slower than full rest."""
        sys_full  = InjurySystem(_CFG)
        sys_load  = InjurySystem(_CFG)

        # Give both the same starting strain
        for _ in range(50):
            sys_full.tick(_DT, _make_high_tau_load("ankle_l", 0.9), _LOAD_ENV)
            sys_load.tick(_DT, _make_high_tau_load("ankle_l", 0.9), _LOAD_ENV)

        strain_before = sys_full.state.joints["ankle_l"].strain

        # sys_full recovers at full rest; sys_load continues under load
        for _ in range(150):
            sys_full.tick(_DT, _make_rest_load(), _REST_ENV)
            sys_load.tick(_DT, _make_high_tau_load("ankle_l", 0.9), _LOAD_ENV)

        self.assertLess(
            sys_full.state.joints["ankle_l"].strain,
            strain_before,
            "Full rest must reduce strain",
        )
        self.assertGreater(
            sys_load.state.joints["ankle_l"].strain,
            sys_full.state.joints["ankle_l"].strain,
            "Continued load must prevent recovery",
        )


# ---------------------------------------------------------------------------
# 6. test_network_authoritative_injury_state
# ---------------------------------------------------------------------------

class TestNetworkAuthoritativeInjury(unittest.TestCase):
    """InjuryReplicator encode/decode must be lossless within quantisation."""

    def _make_state(self) -> InjuryState:
        state = InjuryState()
        state.joints["ankle_l"] = JointInjury(strain=0.6, acute=0.3, painAvoidance=0.5)
        state.joints["knee_r"]  = JointInjury(strain=0.2, acute=0.0, painAvoidance=0.1)
        state.joints["shoulder_l"] = JointInjury(strain=0.1, acute=0.4, painAvoidance=0.2)
        state.globalInjuryIndex = 0.6
        return state

    def test_encode_decode_roundtrip(self):
        """Encode then decode must yield values within 1/255 of originals."""
        state    = self._make_state()
        raw      = InjuryReplicator.encode(state)
        snapshot = InjuryReplicator.decode(raw)

        tolerance = 1.0 / 255.0 + 1e-6
        for name in JOINT_NAMES:
            self.assertAlmostEqual(
                snapshot.joint_strain[name],
                state.joints[name].strain,
                delta=tolerance,
                msg=f"Strain mismatch for {name}",
            )
            self.assertAlmostEqual(
                snapshot.joint_acute[name],
                state.joints[name].acute,
                delta=tolerance,
                msg=f"Acute mismatch for {name}",
            )
        self.assertAlmostEqual(
            snapshot.globalInjuryIndex, state.globalInjuryIndex, delta=tolerance
        )

    def test_corrupted_packet_returns_neutral(self):
        """A packet with a bad checksum must return a neutral snapshot."""
        state = self._make_state()
        raw   = bytearray(InjuryReplicator.encode(state))
        raw[-1] ^= 0xFF   # corrupt checksum
        snap  = InjuryReplicator.decode(bytes(raw))
        # Neutral: all strain and acute should be 0
        for name in JOINT_NAMES:
            self.assertAlmostEqual(snap.joint_strain[name], 0.0, places=3)
            self.assertAlmostEqual(snap.joint_acute[name],  0.0, places=3)

    def test_short_packet_returns_neutral(self):
        """A short packet must return a neutral snapshot without error."""
        snap = InjuryReplicator.decode(b"\x00\x01")
        self.assertAlmostEqual(snap.globalInjuryIndex, 0.0, places=3)

    def test_should_send_respects_rate(self):
        """should_send must respect the configured replication interval."""
        repl  = InjuryReplicator(_CFG)
        sends = sum(1 for t in range(100) if repl.should_send(t * 0.1))
        # At 2 Hz over 10 s we expect ~20 sends
        self.assertGreaterEqual(sends, 18)
        self.assertLessEqual(sends, 22)

    def test_apply_snapshot_reconstructs_state(self):
        """apply_server_snapshot must reproduce the original strain values."""
        state = self._make_state()
        raw   = InjuryReplicator.encode(state)
        snap  = InjuryReplicator.decode(raw)
        reconstructed = InjuryReplicator.apply_server_snapshot(snap)

        tolerance = 1.0 / 255.0 + 1e-6
        for name in JOINT_NAMES:
            self.assertAlmostEqual(
                reconstructed.joints[name].strain,
                state.joints[name].strain,
                delta=tolerance,
            )


# ---------------------------------------------------------------------------
# 7. test_determinism_replay_injury_hash
# ---------------------------------------------------------------------------

class TestDeterminismReplayInjury(unittest.TestCase):
    """Two identical InjurySystem tick sequences must produce identical state."""

    @staticmethod
    def _run_sequence(ticks: int = 200) -> InjuryState:
        sys_ = InjurySystem(_CFG)
        dt   = 0.2
        for i in range(ticks):
            if i % 50 == 49:
                # Rest tick
                sys_.tick(dt, _make_rest_load(), _REST_ENV)
            elif i % 10 == 5:
                # Impact tick
                sys_.tick(dt, _make_impact_load("knee_l", 0.7), _LOAD_ENV)
            else:
                sys_.tick(dt, _make_high_tau_load("ankle_l", 0.85), _LOAD_ENV)
        return sys_.state

    def test_same_inputs_same_state(self):
        """Two runs with identical inputs must yield identical InjuryState."""
        s1 = self._run_sequence()
        s2 = self._run_sequence()

        self.assertAlmostEqual(s1.globalInjuryIndex, s2.globalInjuryIndex, places=10)
        for name in JOINT_NAMES:
            self.assertAlmostEqual(
                s1.joints[name].strain, s2.joints[name].strain, places=10,
                msg=f"strain mismatch for {name}",
            )
            self.assertAlmostEqual(
                s1.joints[name].acute, s2.joints[name].acute, places=10,
                msg=f"acute mismatch for {name}",
            )

    def test_joint_load_estimator_deterministic(self):
        """JointLoadEstimator.flush must be deterministic for same inputs."""
        cfg = _CFG
        est1 = JointLoadEstimator(cfg)
        est2 = JointLoadEstimator(cfg)

        frame = PhysicsFrame(
            joint_tau={"ankle_l": 0.8, "knee_r": 0.5},
            impact_impulse={"knee_r": 0.6},
        )
        for _ in range(5):
            est1.update(0.02, frame)
            est2.update(0.02, frame)

        li1 = est1.flush()
        li2 = est2.flush()

        self.assertAlmostEqual(
            li1.joints["ankle_l"].tau, li2.joints["ankle_l"].tau, places=10,
            msg="JointLoadEstimator must be deterministic",
        )


if __name__ == "__main__":
    unittest.main()
