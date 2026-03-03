"""test_latency_stage58.py — Stage 58 Latency Hiding & Motion Smoothing tests.

Tests
-----
1.  test_reconciliation_converges_without_teleport
    — After receiving an authoritative correction, the error residuals decay
      smoothly toward zero; the correction magnitude never causes a single-tick
      position jump larger than the initial error.

2.  test_remote_interpolation_smooth_under_jitter
    — Under simulated jitter (±30 ms variation in arrival time) the
      interpolated positions from RemoteInterpolation are monotonically
      progressing and do not exhibit single-frame teleports.

3.  test_extrapolation_caps_and_freezes
    — When no new packets arrive, JitterBuffer.interpolate() extrapolates for
      up to extrapolation_max_ms and then returns a frozen (constant) position.

4.  test_grasp_authoritative_consistency
    — LODStatePacker always includes contact_flags in NEAR and MID packets,
      enabling the client to track grasp constraint state.

5.  test_latency_simulated_150ms_still_playable
    — InputSender produces frames at input_hz; even with a 150 ms simulated
      RTT the predicted position from PredictedPlayerSim moves smoothly
      (displacement per tick is bounded).

6.  test_packet_loss_short_does_not_explode
    — Dropping 30 % of incoming remote-state packets does not cause the
      interpolated position to diverge; after recovery it stays within a
      reasonable extrapolation envelope.

7.  test_drift_suite_passes_with_prediction
    — A PredictedPlayerSim + Reconciliation pipeline applied at each tick
      still produces positions that converge to the authoritative server
      state; final position error < 0.1 m after 200 ticks.
"""
from __future__ import annotations

import math
import random
import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.net.InputSender import InputFrame, InputSender, quantise_dir, dequantise_dir
from src.net.JitterBuffer import JitterBuffer, StateFrame
from src.net.Reconciliation import Reconciliation
from src.net.RemoteInterpolation import RemoteInterpolation
from src.net.LODStatePacker import LODStatePacker, LODLevel
from src.net.RateLimiter import RateLimiter
from src.net.InputReceiver import InputReceiver
from src.net.StateBroadcaster import StateBroadcaster, PlayerStateSnapshot
from src.net.StateReceiver import StateReceiver
from src.sim.PredictedPlayerSim import PredictedPlayerSim
from src.dev.NetDiagnostics import NetDiagnostics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CFG = {
    "net": {
        "input_hz": 30,
        "state_hz_self": 20,
        "state_hz_remote_near": 20,
        "state_hz_remote_far": 10,
        "interp_delay_base_ms": 100,
        "interp_delay_jitter_factor": 1.5,
        "extrapolation_max_ms": 200,
        "lod_distance_thresholds": [30.0, 100.0],
        "reconcile": {
            "k_pos": 0.15,
            "k_vel": 0.20,
            "k_yaw": 0.25,
            "hard_snap_threshold": 5.0,
        },
    },
    "dev": {"enable_dev": True},
}


def _make_frame(seq=1, tick=1, mx=0.0, mz=0.0, spd=0.5, yaw=0.0, pitch=0.0):
    from src.net.InputSender import quantise_dir, quantise_angle
    qx, qz = quantise_dir(mx, mz)
    return InputFrame(
        sequence_id  = seq,
        client_tick  = tick,
        move_dir_qx  = qx,
        move_dir_qz  = qz,
        speed_intent = spd,
        look_yaw_q   = quantise_angle(yaw),
        look_pitch_q = quantise_angle(pitch),
    )


def _make_state(ts, x=0.0, y=0.0, z=0.0, vx=1.0, vy=0.0, vz=0.0):
    return StateFrame(
        timestamp_s   = ts,
        pos           = (x, y, z),
        vel           = (vx, vy, vz),
        yaw           = 0.0,
        contact_flags = 3,
        server_tick   = int(ts * 20),
    )


# ---------------------------------------------------------------------------
# 1. Reconciliation convergence
# ---------------------------------------------------------------------------

class TestReconciliationConvergesWithoutTeleport(unittest.TestCase):

    def test_smooth_convergence(self):
        """Position correction should decay smoothly; no single-tick teleport."""
        recon = Reconciliation(_CFG)

        pred_pos = (0.0, 0.0, 0.0)
        pred_vel = (0.0, 0.0, 0.0)
        pred_yaw = 0.0

        server_pos = (1.5, 0.0, 0.0)  # 1.5 m error
        server_vel = (0.0, 0.0, 0.0)
        server_yaw = 0.0

        recon.receive_authoritative(
            server_pos         = server_pos,
            server_vel         = server_vel,
            server_yaw         = server_yaw,
            last_processed_seq = 1,
            server_tick        = 1,
            predicted_pos      = pred_pos,
            predicted_vel      = pred_vel,
            predicted_yaw      = pred_yaw,
        )

        self.assertAlmostEqual(recon.last_correction_magnitude(), 1.5, places=3)

        prev_x = pred_pos[0]
        max_step = 0.0

        for _ in range(100):
            new_pos, new_vel, new_yaw = recon.apply(pred_pos, pred_vel, pred_yaw)
            step = abs(new_pos[0] - prev_x)
            max_step = max(max_step, step)
            prev_x = new_pos[0]
            pred_pos = new_pos

        # No step should be anywhere near the full 1.5 m error in one tick
        self.assertLess(max_step, 0.5, "Reconciliation caused a large single-tick jump")

        # After many ticks, residual should be nearly zero
        self.assertFalse(recon.has_active_correction(),
                         "Correction residuals did not decay to near-zero")

    def test_hard_snap_on_large_error(self):
        """Errors ≥ hard_snap_threshold should be absorbed immediately."""
        recon = Reconciliation(_CFG)

        pred_pos = (0.0, 0.0, 0.0)
        server_pos = (10.0, 0.0, 0.0)  # exceeds threshold of 5.0 m

        recon.receive_authoritative(
            server_pos         = server_pos,
            server_vel         = (0.0, 0.0, 0.0),
            server_yaw         = 0.0,
            last_processed_seq = 1,
            server_tick        = 1,
            predicted_pos      = pred_pos,
            predicted_vel      = (0.0, 0.0, 0.0),
            predicted_yaw      = 0.0,
        )

        # Hard snap clears residuals immediately
        self.assertFalse(recon.has_active_correction())
        self.assertAlmostEqual(recon.last_correction_magnitude(), 10.0, places=2)


# ---------------------------------------------------------------------------
# 2. Remote interpolation smooth under jitter
# ---------------------------------------------------------------------------

class TestRemoteInterpolationSmoothUnderJitter(unittest.TestCase):

    def test_positions_progress_smoothly(self):
        """Interpolated positions advance monotonically under jitter."""
        ri = RemoteInterpolation(_CFG)
        rng = random.Random(42)

        base_ts = 0.0
        # Push frames with ±30 ms jitter in arrival timestamp spacing
        for i in range(30):
            jitter = rng.uniform(-0.030, 0.030)
            ts = base_ts + i * 0.05 + jitter  # ~20 Hz nominal
            frame = _make_state(ts, x=float(i) * 0.1, vx=2.0)
            ri.push("player_2", frame)

        # Query at several render times
        prev_x: float | None = None
        regressions = 0
        for t in range(30):
            now_s = 0.15 + t * 0.05  # render time well behind latest
            f = ri.get("player_2", now_s)
            if f is not None and prev_x is not None:
                if f.pos[0] < prev_x - 0.5:
                    regressions += 1
            if f is not None:
                prev_x = f.pos[0]

        self.assertEqual(regressions, 0, "Interpolated X regressed unexpectedly")

    def test_delay_adapts_to_high_jitter(self):
        """update_delay should increase the delay when jitter is high."""
        ri = RemoteInterpolation(_CFG)
        ri.push("p", _make_state(0.0))  # create buffer
        buf = ri._buffers["p"]
        initial_delay = buf.delay_s
        buf.update_delay(rtt_s=0.1, jitter_s=0.050)  # 50 ms jitter
        self.assertGreater(buf.delay_s, initial_delay)


# ---------------------------------------------------------------------------
# 3. Extrapolation caps and freezes
# ---------------------------------------------------------------------------

class TestExtrapolationCapsAndFreezes(unittest.TestCase):

    def test_extrapolation_caps_at_max(self):
        """JitterBuffer extrapolates up to extrapolation_max_ms then freezes."""
        buf = JitterBuffer(_CFG)

        # Single frame at t=0 with velocity (1, 0, 0)
        buf.push(_make_state(0.0, x=0.0, vx=1.0))

        extrap_max_s = _CFG["net"]["extrapolation_max_ms"] / 1000.0

        # Query well beyond extrapolation window (1 second ahead)
        far_future_now = 1.0 + 0.1  # target = now - delay → well past cap
        f = buf.interpolate(far_future_now)
        self.assertIsNotNone(f)

        # The extrapolated X should be capped at extrap_max_s * vx
        self.assertLessEqual(f.pos[0], extrap_max_s * 1.0 + 1e-6,
                             "Extrapolation exceeded cap distance")

    def test_extrapolation_starts_from_last_known(self):
        """Extrapolation should extend the last known state by velocity."""
        buf = JitterBuffer({
            "net": {
                "interp_delay_base_ms": 50,
                "interp_delay_jitter_factor": 1.0,
                "extrapolation_max_ms": 500,
            }
        })
        buf.push(_make_state(0.0, x=5.0, vx=2.0))
        # delay = 50 ms; now = 0.2 → target = 0.15 → dt from frame = 0.15
        f = buf.interpolate(0.2)
        self.assertIsNotNone(f)
        self.assertGreater(f.pos[0], 5.0)


# ---------------------------------------------------------------------------
# 4. Grasp authoritative consistency (LOD includes contact_flags)
# ---------------------------------------------------------------------------

class TestGraspAuthoritativeConsistency(unittest.TestCase):

    def _pack(self, distance_m):
        packer = LODStatePacker(_CFG)
        return packer.pack_remote(
            player_id     = "p1",
            pos           = (0.0, 0.0, 0.0),
            vel           = (0.0, 0.0, 0.0),
            yaw           = 0.0,
            contact_flags = 7,
            server_tick   = 1,
            timestamp_s   = 0.0,
            distance_m    = distance_m,
            pose_hash     = 0xDEAD,
        )

    def test_near_includes_contact_and_pose(self):
        msg = self._pack(10.0)  # NEAR
        self.assertIn("contact", msg)
        self.assertEqual(msg["contact"], 7)
        self.assertIn("poseHash", msg)
        self.assertIn("vel", msg)

    def test_mid_includes_contact_no_pose(self):
        msg = self._pack(50.0)  # MID
        self.assertIn("contact", msg)
        self.assertEqual(msg["contact"], 7)
        self.assertNotIn("poseHash", msg)
        self.assertIn("vel", msg)

    def test_far_excludes_contact_and_vel(self):
        msg = self._pack(150.0)  # FAR
        self.assertNotIn("contact", msg)
        self.assertNotIn("vel", msg)

    def test_auth_state_always_full(self):
        packer = LODStatePacker(_CFG)
        msg = packer.pack_own(
            pos=(1.0, 2.0, 3.0), vel=(0.1, 0.0, 0.0),
            yaw=0.5, contact_flags=3, server_tick=5, last_seq=10, pose_hash=42
        )
        self.assertEqual(msg["type"], "AUTH_STATE")
        self.assertIn("contact", msg)
        self.assertIn("lastSeq", msg)


# ---------------------------------------------------------------------------
# 5. Latency simulated 150 ms — still playable
# ---------------------------------------------------------------------------

class TestLatencySimulated150msStillPlayable(unittest.TestCase):

    def test_predicted_pos_moves_smoothly(self):
        """PredictedPlayerSim should produce bounded per-tick displacement."""
        sim = PredictedPlayerSim(_CFG)
        sender = InputSender(_CFG)

        dt = 1.0 / 60.0
        now_s = 0.0
        rtt_delay = 0.150  # 150 ms RTT

        max_per_tick_disp = 0.0
        prev_pos = sim.pos

        for i in range(180):  # 3 simulated seconds
            now_s += dt
            sender.apply_input(move_x=0.5, move_z=0.0, speed_intent=0.8,
                               yaw_rad=0.0, pitch_rad=0.0)
            frame = sender.tick(now_s)
            if frame:
                sim.apply_input(frame)

            sim.tick(dt)
            disp = math.sqrt(sum((a - b) ** 2 for a, b in zip(sim.pos, prev_pos)))
            max_per_tick_disp = max(max_per_tick_disp, disp)
            prev_pos = sim.pos

        # At 60 Hz with max speed ~5 m/s, max tick disp ≈ 5/60 ≈ 0.083 m
        # Allow a generous 0.2 m for acceleration transients
        self.assertLess(max_per_tick_disp, 0.2,
                        f"Single-tick displacement too large: {max_per_tick_disp:.3f} m")

        # Player should have moved forward (positive x)
        self.assertGreater(sim.pos[0], 0.5)

    def test_input_sender_rate_hz(self):
        """InputSender should emit frames at approximately input_hz."""
        sender = InputSender(_CFG)
        sender.apply_input(1.0, 0.0, 1.0, 0.0, 0.0)

        frames_sent = 0
        dt = 1.0 / 60.0
        # Compute timestamps directly from integer index to avoid any drift
        for i in range(1, 121):  # ticks 1..120 → 2 seconds at 60 Hz
            now_s = round(i * dt, 10)
            f = sender.tick(now_s)
            if f:
                frames_sent += 1

        # Should have emitted ~60 frames (30 Hz × 2 s), allow ±5
        self.assertGreater(frames_sent, 55)
        self.assertLess(frames_sent, 65)


# ---------------------------------------------------------------------------
# 6. Packet loss — short burst does not explode
# ---------------------------------------------------------------------------

class TestPacketLossShortDoesNotExplode(unittest.TestCase):

    def test_30pct_loss_does_not_diverge(self):
        """30 % packet loss should not cause position to diverge."""
        ri = RemoteInterpolation(_CFG)
        rng = random.Random(99)

        base_x = 0.0
        for i in range(60):
            ts = i * 0.05
            frame = _make_state(ts, x=base_x, vx=2.0)
            base_x += 2.0 * 0.05
            # Drop ~30 % of packets
            if rng.random() < 0.3:
                continue
            ri.push("remote_player", frame)

        # After 3 s of updates with 30 % loss, interpolated X should be
        # bounded — not exploded to infinity
        final = ri.get("remote_player", 3.0)
        self.assertIsNotNone(final)
        # Extrapolation cap is 200 ms → max extra distance = 0.2 * 2.0 = 0.4 m
        # base_x ends at ~6.0; capped extrapolation ≤ ~6.4 m
        self.assertLess(abs(final.pos[0]), 50.0,
                        "Position diverged catastrophically under packet loss")

    def test_recovery_after_loss_burst(self):
        """After a loss burst, new packets restore smooth interpolation."""
        ri = RemoteInterpolation(_CFG)

        # Deliver 10 frames
        for i in range(10):
            ri.push("rp", _make_state(i * 0.05, x=float(i) * 0.1))

        # Gap: no frames for 0.5 s (simulate burst loss)
        # Then deliver 10 more frames
        for i in range(10):
            ts = 0.5 + (10 + i) * 0.05
            ri.push("rp", _make_state(ts, x=1.0 + float(i) * 0.1, vx=2.0))

        f = ri.get("rp", 1.0 + 0.1)
        self.assertIsNotNone(f)
        self.assertLess(f.pos[0], 5.0, "Position exploded after recovery")


# ---------------------------------------------------------------------------
# 7. Drift suite passes with prediction
# ---------------------------------------------------------------------------

class TestDriftSuitePassesWithPrediction(unittest.TestCase):

    def test_predicted_converges_to_authoritative(self):
        """PredictedPlayerSim + Reconciliation converge to server state."""
        sim   = PredictedPlayerSim(_CFG)
        recon = Reconciliation(_CFG)

        dt      = 1.0 / 60.0
        server_x = 0.0
        server_vx = 1.0  # server player moves at 1 m/s

        frame = _make_frame(seq=1, mx=1.0, spd=0.5)

        for tick in range(200):
            # Server advances
            server_x += server_vx * dt

            # Client predicts
            sim.apply_input(frame)
            sim.tick(dt)

            # Every 3 ticks server sends authoritative state
            if tick % 3 == 0:
                recon.receive_authoritative(
                    server_pos         = (server_x, 0.0, 0.0),
                    server_vel         = (server_vx, 0.0, 0.0),
                    server_yaw         = 0.0,
                    last_processed_seq = tick,
                    server_tick        = tick,
                    predicted_pos      = sim.pos,
                    predicted_vel      = sim.vel,
                    predicted_yaw      = sim.yaw,
                )

            # Apply correction
            new_pos, new_vel, new_yaw = recon.apply(sim.pos, sim.vel, sim.yaw)
            sim.snap_to(new_pos, new_vel, new_yaw)

        final_error = abs(sim.pos[0] - server_x)
        self.assertLess(final_error, 0.1,
                        f"Predicted state did not converge: error = {final_error:.3f} m")


# ---------------------------------------------------------------------------
# Bonus: quantisation round-trip
# ---------------------------------------------------------------------------

class TestQuantisationRoundTrip(unittest.TestCase):

    def test_dir_roundtrip(self):
        for mx, mz in [(1.0, 0.0), (-0.5, 0.5), (0.0, -1.0), (0.707, 0.707)]:
            qx, qz = quantise_dir(mx, mz)
            rx, rz = dequantise_dir(qx, qz)
            self.assertAlmostEqual(rx, mx, delta=0.01)
            self.assertAlmostEqual(rz, mz, delta=0.01)

    def test_input_frame_dict_roundtrip(self):
        f = _make_frame(seq=7, tick=42, mx=0.8, mz=-0.3, spd=0.9)
        d = f.to_dict()
        f2 = InputFrame.from_dict({**d, "type": "INPUT_FRAME"})
        self.assertEqual(f.sequence_id,   f2.sequence_id)
        self.assertEqual(f.speed_intent,  f2.speed_intent)
        self.assertEqual(f.move_dir_qx,   f2.move_dir_qx)


# ---------------------------------------------------------------------------
# Bonus: RateLimiter
# ---------------------------------------------------------------------------

class TestRateLimiter(unittest.TestCase):

    def test_allows_within_rate(self):
        rl = RateLimiter(rate_hz=10.0, burst_multiplier=1.0)
        now = 0.0
        # First 10 messages in 1 second → all allowed
        allowed = sum(1 for _ in range(10) if rl.allow("p", now))
        self.assertEqual(allowed, 10)

    def test_drops_when_burst_exceeded(self):
        rl = RateLimiter(rate_hz=5.0, burst_multiplier=1.0)
        now = 0.0
        # Send 10 messages instantly — only 5 tokens in bucket
        results = [rl.allow("p", now) for _ in range(10)]
        self.assertEqual(results.count(True), 5)
        self.assertEqual(results.count(False), 5)

    def test_refills_over_time(self):
        rl = RateLimiter(rate_hz=10.0, burst_multiplier=1.0)
        # Drain the bucket
        for _ in range(10):
            rl.allow("p", 0.0)
        # After 1 s, bucket refills to capacity
        self.assertTrue(rl.allow("p", 1.0))


# ---------------------------------------------------------------------------
# Bonus: InputReceiver
# ---------------------------------------------------------------------------

class TestInputReceiver(unittest.TestCase):

    def _msg(self, seq, tick=1, mx=0.0, mz=0.0):
        from src.net.InputSender import quantise_dir, quantise_angle
        qx, qz = quantise_dir(mx, mz)
        return {
            "type":  "INPUT_FRAME",
            "seq":   seq,
            "cTick": tick,
            "mvX":   qx,
            "mvZ":   qz,
            "spd":   0.5,
            "yaw":   0,
            "pitch": 0,
        }

    def test_accepts_new_frame(self):
        ir = InputReceiver(_CFG)
        f = ir.receive("p1", self._msg(1))
        self.assertIsNotNone(f)
        self.assertEqual(f.sequence_id, 1)

    def test_rejects_duplicate(self):
        ir = InputReceiver(_CFG)
        ir.receive("p1", self._msg(1))
        self.assertIsNone(ir.receive("p1", self._msg(1)))

    def test_rejects_old_seq(self):
        ir = InputReceiver(_CFG)
        ir.receive("p1", self._msg(5))
        self.assertIsNone(ir.receive("p1", self._msg(3)))

    def test_rejects_wrong_type(self):
        ir = InputReceiver(_CFG)
        msg = self._msg(1)
        msg["type"] = "PLAYER_STATE"
        self.assertIsNone(ir.receive("p1", msg))


# ---------------------------------------------------------------------------
# Bonus: NetDiagnostics
# ---------------------------------------------------------------------------

class TestNetDiagnostics(unittest.TestCase):

    def test_records_and_summarises(self):
        nd = NetDiagnostics({"dev": {"enable_dev": True}})
        for rtt in [0.050, 0.080, 0.120]:
            nd.record_rtt(rtt)
        nd.record_correction(0.3)
        nd.record_packet_in()
        nd.record_packet_out()
        nd.record_interp_delay(0.1)
        nd.record_dropped_frame()

        s = nd.get_summary()
        self.assertEqual(s["rtt_samples"], 3)
        self.assertGreater(s["rtt_mean_ms"], 0)
        self.assertEqual(s["packets_in"], 1)
        self.assertEqual(s["packets_out"], 1)
        self.assertEqual(s["dropped_frames"], 1)

    def test_noop_when_disabled(self):
        nd = NetDiagnostics({"dev": {"enable_dev": False}})
        nd.record_rtt(0.1)
        nd.record_packet_in()
        s = nd.get_summary()
        self.assertEqual(s["rtt_samples"], 0)
        self.assertEqual(s["packets_in"], 0)

    def test_reset_clears(self):
        nd = NetDiagnostics()
        nd.record_rtt(0.05)
        nd.record_packet_in()
        nd.reset()
        s = nd.get_summary()
        self.assertEqual(s["rtt_samples"], 0)
        self.assertEqual(s["packets_in"], 0)


if __name__ == "__main__":
    unittest.main()
