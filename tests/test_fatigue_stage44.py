"""test_fatigue_stage44.py — Stage 44 Embodied Fatigue & Recovery smoke tests.

Tests
-----
1. test_energy_decreases_with_work
   — Energy drains when mechanical work and wind work are applied.

2. test_recovery_in_safe_conditions
   — Energy recovers when all rest-condition thresholds are satisfied.

3. test_motor_params_degrade_with_fatigue
   — FatigueToMotorAdapter scales torque, stiffness, and step length down,
     and brace bias and step width up, as fatigue increases.

4. test_grasp_strength_scales_with_gripReserve
   — GripForceScale from the adapter decreases as gripReserve is depleted.

5. test_network_authoritative_fatigue
   — FatigueReplicator encodes/decodes the authoritative state without
     error, and the checksum detects corruption.

6. test_determinism_replay_fatigue_hash
   — Two identical FatigueSystem tick sequences produce identical state.
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.fatigue.FatigueSystem import FatigueSystem, FatigueState, WorkInput, EnvInput
from src.fatigue.WorkEstimator import WorkEstimator
from src.fatigue.RecoveryDetector import RecoveryDetector
from src.fatigue.FatigueToMotorAdapter import FatigueToMotorAdapter, MotorParams
from src.net.FatigueReplicator import FatigueReplicator, FatigueSnapshot


# ---------------------------------------------------------------------------
# Shared config
# ---------------------------------------------------------------------------

_CFG = {
    "fatigue": {
        "enable":              True,
        "tick_hz":             5,
        "k_work":              0.08,
        "k_wind":              0.06,
        "k_recovery":          0.12,
        "k_recover":           0.04,
        "torque_floor":        0.65,
        "stiffness_scale_min": 0.60,
        "noise_max":           0.60,
        "grip_scale_min":      0.50,
        "work_window_sec":     3.0,
        "single_recovery_cost": 0.25,
        "grasp_drain_rate":    0.05,
        "grip_recover_rate":   0.08,
        "max_reaction_delay":  0.08,
        "repl_hz":             2,
        "repl_smooth_tau":     0.5,
        "rest_condition_thresholds": {
            "wind":    0.2,
            "slope":   5.0,
            "support": 0.7,
            "speed":   0.3,
        },
    }
}


# ---------------------------------------------------------------------------
# 1. test_energy_decreases_with_work
# ---------------------------------------------------------------------------

class TestEnergyDecreasesWithWork(unittest.TestCase):
    """Energy must drain when mechanical and wind work are applied."""

    def test_energy_drains_under_load(self):
        """Simulate 60 s of hard walking in wind; energy must fall."""
        fs  = FatigueSystem(_CFG)
        env = EnvInput(windLoad=0.5, slopeDeg=10.0, supportQuality=0.9, speed=2.0)
        work = WorkInput(mechWork=0.6, windWork=0.5, recoveryCost=0.0)

        initial_energy = fs.state.energy
        dt = 0.2   # 5 Hz tick
        for _ in range(300):   # 60 s
            fs.tick(dt, work, env)

        self.assertLess(
            fs.state.energy, initial_energy,
            "Energy must decrease under sustained mechanical load",
        )

    def test_energy_drains_faster_in_tough_conditions(self):
        """Storm conditions drain energy faster than mild conditions."""
        fs_mild   = FatigueSystem(_CFG)
        fs_storm  = FatigueSystem(_CFG)

        mild_env  = EnvInput(windLoad=0.1, slopeDeg=0.0, supportQuality=1.0, speed=1.0)
        storm_env = EnvInput(windLoad=0.9, slopeDeg=20.0, supportQuality=0.5, speed=1.0,
                             dustResistance=0.8, visibility=0.2)
        work      = WorkInput(mechWork=0.4, windWork=0.4, recoveryCost=0.0)

        dt = 0.2
        for _ in range(40):  # 8 s — short enough that neither system floors at 0
            fs_mild.tick(dt, work, mild_env)
            fs_storm.tick(dt, work, storm_env)

        self.assertLess(
            fs_storm.state.energy, fs_mild.state.energy,
            "Energy must drain faster in storm conditions",
        )

    def test_recovery_events_increase_drain(self):
        """Balance-recovery events (near-falls) add extra energy cost."""
        fs_normal   = FatigueSystem(_CFG)
        fs_recovery = FatigueSystem(_CFG)

        env  = EnvInput(windLoad=0.3, slopeDeg=5.0, supportQuality=0.8, speed=1.5)
        dt   = 0.2
        for _ in range(50):  # 10 s
            fs_normal.tick(dt, WorkInput(mechWork=0.3, windWork=0.2, recoveryCost=0.0), env)
            fs_recovery.tick(dt, WorkInput(mechWork=0.3, windWork=0.2, recoveryCost=0.5), env)

        self.assertLess(
            fs_recovery.state.energy, fs_normal.state.energy,
            "Recovery cost from near-falls must drain more energy",
        )


# ---------------------------------------------------------------------------
# 2. test_recovery_in_safe_conditions
# ---------------------------------------------------------------------------

class TestRecoveryInSafeConditions(unittest.TestCase):
    """Energy must recover under ideal rest conditions."""

    def test_energy_recovers_at_rest(self):
        """Start depleted; standing still on flat ground should recover energy."""
        fs = FatigueSystem(_CFG)
        fs.force_set(0.4)   # partially exhausted
        initial = fs.state.energy

        rest_env  = EnvInput(windLoad=0.05, slopeDeg=1.0, supportQuality=0.95, speed=0.05)
        rest_work = WorkInput(mechWork=0.0, windWork=0.0, recoveryCost=0.0)

        dt = 0.2
        for _ in range(150):  # 30 s rest
            fs.tick(dt, rest_work, rest_env)

        self.assertGreater(
            fs.state.energy, initial,
            "Energy must increase during rest on flat ground with no wind",
        )

    def test_recovery_detector_is_resting_on_flat(self):
        """RecoveryDetector must signal is_resting=True on flat, calm ground."""
        rd  = RecoveryDetector(_CFG)
        env = EnvInput(windLoad=0.05, slopeDeg=1.0, supportQuality=0.95, speed=0.1)
        result = rd.evaluate(env)
        self.assertTrue(result.is_resting, "Should be resting on flat, calm surface")
        self.assertGreater(result.recovery_rate, 0.2)

    def test_recovery_detector_not_resting_in_storm(self):
        """RecoveryDetector must signal is_resting=False in storm conditions."""
        rd  = RecoveryDetector(_CFG)
        env = EnvInput(windLoad=0.8, slopeDeg=25.0, supportQuality=0.3, speed=3.0)
        result = rd.evaluate(env)
        self.assertFalse(result.is_resting, "Should not be resting in storm conditions")
        self.assertLess(result.recovery_rate, 0.1)

    def test_tremor_reduces_during_rest(self):
        """Tremor level must decrease during extended rest."""
        fs = FatigueSystem(_CFG)
        fs.force_set(0.3)  # high fatigue → some tremor
        initial_tremor = fs.state.tremor

        rest_env  = EnvInput(windLoad=0.0, slopeDeg=0.0, supportQuality=1.0, speed=0.0)
        rest_work = WorkInput(mechWork=0.0, windWork=0.0, recoveryCost=0.0)

        dt = 0.2
        for _ in range(300):  # 60 s rest
            fs.tick(dt, rest_work, rest_env)

        self.assertLessEqual(
            fs.state.tremor, initial_tremor,
            "Tremor must not increase during rest",
        )


# ---------------------------------------------------------------------------
# 3. test_motor_params_degrade_with_fatigue
# ---------------------------------------------------------------------------

class TestMotorParamsDegrade(unittest.TestCase):
    """Motor parameters must degrade as fatigue factor increases."""

    def _params_at_energy(self, energy: float) -> MotorParams:
        fs = FatigueSystem(_CFG)
        fs.force_set(energy)
        adapter = FatigueToMotorAdapter(_CFG)
        return adapter.adapt(fs.state, tick_bucket=0, world_seed=0)

    def test_torque_scale_decreases_with_fatigue(self):
        """maxTorqueScale must be lower at low energy."""
        p_rested  = self._params_at_energy(1.0)
        p_tired   = self._params_at_energy(0.1)
        self.assertGreater(p_rested.maxTorqueScale, p_tired.maxTorqueScale,
                           "Torque scale must decrease with fatigue")
        self.assertGreaterEqual(p_tired.maxTorqueScale, _CFG["fatigue"]["torque_floor"] - 0.01,
                                "Torque must not fall below configured floor")

    def test_stiffness_scale_decreases_with_fatigue(self):
        """Joint stiffness must be lower at low energy."""
        p_rested = self._params_at_energy(1.0)
        p_tired  = self._params_at_energy(0.1)
        self.assertGreater(p_rested.stiffnessScale, p_tired.stiffnessScale,
                           "Stiffness must decrease with fatigue")

    def test_step_length_decreases_with_fatigue(self):
        """Step length scale must be lower at low energy."""
        p_rested = self._params_at_energy(1.0)
        p_tired  = self._params_at_energy(0.1)
        self.assertGreater(p_rested.stepLengthScale, p_tired.stepLengthScale,
                           "Step length must shorten with fatigue")

    def test_brace_bias_increases_with_fatigue(self):
        """Arm brace bias must increase as energy drops."""
        p_rested = self._params_at_energy(1.0)
        p_tired  = self._params_at_energy(0.1)
        self.assertLess(p_rested.braceBias, p_tired.braceBias,
                        "Brace bias must increase with fatigue")

    def test_step_width_increases_with_fatigue(self):
        """Step width bias must increase as energy drops."""
        p_rested = self._params_at_energy(1.0)
        p_tired  = self._params_at_energy(0.1)
        self.assertLess(p_rested.stepWidthBias, p_tired.stepWidthBias,
                        "Step width must increase with fatigue")

    def test_reaction_delay_increases_with_noise(self):
        """Reaction delay must scale with neuromuscular noise."""
        adapter = FatigueToMotorAdapter(_CFG)
        # Low noise state
        low_noise = FatigueState(energy=0.9, neuromuscularNoise=0.05, coordination=0.95,
                                 tremor=0.0, gripReserve=0.9, thermalLoad=0.0)
        # High noise state
        high_noise = FatigueState(energy=0.3, neuromuscularNoise=0.55, coordination=0.5,
                                  tremor=0.3, gripReserve=0.6, thermalLoad=0.2)
        p_low  = adapter.adapt(low_noise,  tick_bucket=0, world_seed=0)
        p_high = adapter.adapt(high_noise, tick_bucket=0, world_seed=0)
        self.assertLess(p_low.reactionDelay, p_high.reactionDelay,
                        "Reaction delay must increase with neuromuscular noise")


# ---------------------------------------------------------------------------
# 4. test_grasp_strength_scales_with_gripReserve
# ---------------------------------------------------------------------------

class TestGraspStrengthScalesWithGrip(unittest.TestCase):
    """GripForceScale must decrease as gripReserve is depleted by holding."""

    def test_grip_force_scale_decreases_while_holding(self):
        """Holding another character drains gripReserve and reduces grip force scale."""
        fs      = FatigueSystem(_CFG)
        adapter = FatigueToMotorAdapter(_CFG)

        env  = EnvInput(windLoad=0.2, slopeDeg=5.0, supportQuality=0.85, speed=1.0)
        work = WorkInput(mechWork=0.3, windWork=0.2, recoveryCost=0.0, isHoldingOther=True)

        initial_grip_scale = adapter.adapt(fs.state, tick_bucket=0, world_seed=0).gripForceScale
        dt = 0.2
        for _ in range(50):  # 10 s of holding
            fs.tick(dt, work, env)

        final_grip_scale = adapter.adapt(fs.state).gripForceScale
        self.assertLess(
            final_grip_scale, initial_grip_scale,
            "Grip force scale must decrease after sustained grasping",
        )

    def test_grip_scale_floored_at_min(self):
        """GripForceScale must never drop below grip_scale_min."""
        adapter = FatigueToMotorAdapter(_CFG)
        # Near-zero grip reserve
        exhausted = FatigueState(energy=0.1, neuromuscularNoise=0.5, coordination=0.5,
                                 tremor=0.3, gripReserve=0.0, thermalLoad=0.0)
        p = adapter.adapt(exhausted, tick_bucket=0, world_seed=0)
        self.assertGreaterEqual(
            p.gripForceScale,
            _CFG["fatigue"]["grip_scale_min"] - 0.01,
            "Grip force scale must not fall below configured minimum",
        )

    def test_grip_reserve_recovers_at_rest(self):
        """GripReserve must recover when resting and not holding anyone."""
        fs  = FatigueSystem(_CFG)
        fs.force_set(0.5)
        # Drain grip manually
        env  = EnvInput(windLoad=0.2, slopeDeg=3.0, supportQuality=0.9, speed=1.0)
        work = WorkInput(mechWork=0.3, windWork=0.2, recoveryCost=0.0, isHoldingOther=True)
        dt   = 0.2
        for _ in range(20):  # 4 s holding
            fs.tick(dt, work, env)
        drained_grip = fs.state.gripReserve

        # Now rest
        rest_env  = EnvInput(windLoad=0.0, slopeDeg=0.0, supportQuality=1.0, speed=0.0)
        rest_work = WorkInput(mechWork=0.0, windWork=0.0, recoveryCost=0.0, isHoldingOther=False)
        for _ in range(100):  # 20 s rest
            fs.tick(dt, rest_work, rest_env)

        self.assertGreater(
            fs.state.gripReserve, drained_grip,
            "GripReserve must recover when resting and not holding anyone",
        )


# ---------------------------------------------------------------------------
# 5. test_network_authoritative_fatigue
# ---------------------------------------------------------------------------

class TestNetworkAuthoritativeFatigue(unittest.TestCase):
    """FatigueReplicator encode/decode must be lossless within quantisation."""

    def test_encode_decode_roundtrip(self):
        """Encode then decode must yield values within 1/255 of originals."""
        state = FatigueState(
            energy       = 0.72,
            tremor       = 0.18,
            coordination = 0.85,
            neuromuscularNoise = 0.25,
            gripReserve  = 0.6,
            thermalLoad  = 0.1,
        )
        raw      = FatigueReplicator.encode(state)
        self.assertEqual(len(raw), 4, "Encoded payload must be exactly 4 bytes")

        snapshot = FatigueReplicator.decode(raw)
        tolerance = 1.0 / 255.0 + 1e-6
        self.assertAlmostEqual(snapshot.energy,       state.energy,       delta=tolerance)
        self.assertAlmostEqual(snapshot.tremor,        state.tremor,       delta=tolerance)
        self.assertAlmostEqual(snapshot.coordination,  state.coordination, delta=tolerance)

    def test_corrupted_packet_returns_neutral(self):
        """A packet with a bad checksum must return a neutral snapshot."""
        state = FatigueState(energy=0.5, tremor=0.3, coordination=0.7)
        raw   = bytearray(FatigueReplicator.encode(state))
        raw[3] ^= 0xFF   # corrupt checksum byte
        snapshot = FatigueReplicator.decode(bytes(raw))
        # Neutral snapshot: energy=1.0 (default)
        self.assertAlmostEqual(snapshot.energy, 1.0, places=3,
                               msg="Corrupted packet must produce neutral snapshot")

    def test_should_send_rate(self):
        """should_send must respect the configured replication interval."""
        repl = FatigueReplicator(_CFG)
        sends = sum(1 for t in range(100) if repl.should_send(t * 0.1))
        # At 2 Hz over 10 s we expect ~20 sends (+/-1 for boundary)
        self.assertGreaterEqual(sends, 18)
        self.assertLessEqual(sends, 22)

    def test_client_smoothing_converges(self):
        """apply_server_snapshot must smoothly converge to server value."""
        repl     = FatigueReplicator(_CFG)
        target   = FatigueSnapshot(energy=0.4, tremor=0.3, coordination=0.7)
        dt       = 0.1

        # Run enough ticks to converge
        for _ in range(60):
            repl.apply_server_snapshot(target, dt)

        s = repl.client_state
        self.assertAlmostEqual(s.energy,       0.4, delta=0.05)
        self.assertAlmostEqual(s.tremor,        0.3, delta=0.05)
        self.assertAlmostEqual(s.coordination,  0.7, delta=0.05)


# ---------------------------------------------------------------------------
# 6. test_determinism_replay_fatigue_hash
# ---------------------------------------------------------------------------

class TestDeterminismReplayFatigueHash(unittest.TestCase):
    """Two identical tick sequences must produce identical fatigue state."""

    @staticmethod
    def _run_sequence(ticks: int = 200) -> FatigueState:
        """Run a fixed tick sequence and return the final state."""
        fs   = FatigueSystem(_CFG)
        env  = EnvInput(windLoad=0.4, slopeDeg=8.0, supportQuality=0.75,
                        speed=1.5, dustResistance=0.3, visibility=0.6)
        work = WorkInput(mechWork=0.5, windWork=0.35, recoveryCost=0.1)
        dt   = 0.2
        for i in range(ticks):
            # Rest every 50 ticks
            if i % 50 == 49:
                fs.tick(dt, WorkInput(), EnvInput(speed=0.0, supportQuality=1.0))
            else:
                fs.tick(dt, work, env)
        return fs.state

    def test_same_inputs_same_state(self):
        """Two runs with identical inputs must yield identical FatigueState."""
        s1 = self._run_sequence()
        s2 = self._run_sequence()

        self.assertAlmostEqual(s1.energy,             s2.energy,             places=10)
        self.assertAlmostEqual(s1.neuromuscularNoise,  s2.neuromuscularNoise,  places=10)
        self.assertAlmostEqual(s1.coordination,        s2.coordination,        places=10)
        self.assertAlmostEqual(s1.tremor,              s2.tremor,              places=10)
        self.assertAlmostEqual(s1.gripReserve,         s2.gripReserve,         places=10)

    def test_adapter_deterministic_noise(self):
        """FatigueToMotorAdapter.adapt must produce identical foot noise for same inputs."""
        state   = FatigueState(energy=0.5, neuromuscularNoise=0.3, coordination=0.7,
                               tremor=0.15, gripReserve=0.6, thermalLoad=0.1)
        adapter = FatigueToMotorAdapter(_CFG)
        p1 = adapter.adapt(state, tick_bucket=42, world_seed=99)
        p2 = adapter.adapt(state, tick_bucket=42, world_seed=99)
        self.assertEqual(p1.footPlacementNoise, p2.footPlacementNoise,
                         "Foot placement noise must be deterministic for same tick_bucket+world_seed")

    def test_adapter_different_ticks_different_noise(self):
        """Different tick_bucket values produce different foot placement noise."""
        state   = FatigueState(energy=0.5, neuromuscularNoise=0.5, coordination=0.5,
                               tremor=0.3, gripReserve=0.5, thermalLoad=0.0)
        adapter = FatigueToMotorAdapter(_CFG)
        noises  = [adapter.adapt(state, tick_bucket=i, world_seed=0).footPlacementNoise
                   for i in range(10)]
        # Not all the same
        self.assertGreater(
            len(set(round(n, 8) for n in noises)), 1,
            "Different tick buckets should produce different foot placement noise values",
        )


if __name__ == "__main__":
    unittest.main()
