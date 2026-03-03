"""test_crash_recovery_stage59.py — Stage 59 Crash Recovery & Auto Snapshot Rollback tests.

Tests
-----
1.  test_snapshot_write_atomic
    — SnapshotWriter produces .dat, .meta, .checkpoint; checksum matches.

2.  test_recover_from_partial_snapshot
    — Crash after .tmp write (no rename/checkpoint) → loader falls back to
      previous valid snapshot.

3.  test_restart_restores_world_state_hash
    — Snapshot is written, "server restarts", hash of loaded payload equals
      hash of original payload.

4.  test_epoch_increment_on_reset
    — All snapshots have incompatible format_version → WorldBootManager
      returns fresh_world=True with world_epoch incremented.

5.  test_reconnect_after_server_restart
    — Snapshot present with epoch=7; boot restores epoch=7 so clients can
      detect the restart and re-sync.

6.  test_crash_during_instability_event
    — Latest snapshot is corrupt; loader falls back to previous.

7.  test_disk_rotation_policy
    — SnapshotRotator removes oldest entries when keep limits exceeded.

8.  test_crash_recovery_suite_all_pass
    — CrashRecoveryTestSuite.run_all() — all CI scenarios pass.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.save.SnapshotWriter import SnapshotWriter
from src.save.SnapshotValidator import SnapshotValidator
from src.save.SnapshotLoader import SnapshotLoader
from src.save.SnapshotRotator import SnapshotRotator
from src.world.WorldBootManager import WorldBootManager
from src.world.WorldRollbackManager import WorldRollbackManager
from src.ops.CrashWatchdog import CrashWatchdog
from src.ci.CrashRecoveryTestSuite import CrashRecoveryTestSuite

_FMT = 1  # current format version


def _meta(epoch: int = 0, version: int = _FMT) -> dict:
    return {"format_version": version, "world_epoch": epoch}


class TestSnapshotWriteAtomic(unittest.TestCase):
    """1. test_snapshot_write_atomic"""

    def test_snapshot_write_atomic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = SnapshotWriter(tmpdir)
            payload = b"hello world state"
            entry = writer.write("baseline_00001", payload, _meta(epoch=1))

            # All three files exist
            self.assertTrue(entry.dat_path.exists(), "missing .dat")
            self.assertTrue(entry.meta_path.exists(), "missing .meta")
            self.assertTrue(entry.checkpoint.exists(), "missing .checkpoint")

            # Payload content is intact
            self.assertEqual(entry.dat_path.read_bytes(), payload)

            # Checksum in meta is correct
            meta = json.loads(entry.meta_path.read_bytes())
            expected_cs = hashlib.sha256(payload).hexdigest()
            self.assertEqual(meta["checksum"], expected_cs)

            # Validator accepts it
            validator = SnapshotValidator(current_format_version=_FMT)
            vr = validator.validate(entry)
            self.assertTrue(vr.ok, vr.reason)


class TestRecoverFromPartialSnapshot(unittest.TestCase):
    """2. test_recover_from_partial_snapshot"""

    def test_recover_from_partial_snapshot(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            snap_dir = Path(tmpdir) / "snaps"
            snap_dir.mkdir()
            writer = SnapshotWriter(str(snap_dir))

            payload_good = b"good_world"
            writer.write("baseline_00001", payload_good, _meta(epoch=1))

            # Simulate crash: only .tmp written, no rename, no checkpoint
            (snap_dir / "baseline_00002.tmp").write_bytes(b"partial_write")

            boot = WorldBootManager(str(snap_dir), format_version=_FMT)
            result = boot.boot()

            self.assertFalse(result.fresh_world)
            self.assertEqual(result.payload, payload_good)


class TestRestartRestoresWorldStateHash(unittest.TestCase):
    """3. test_restart_restores_world_state_hash"""

    def test_restart_restores_world_state_hash(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            snap_dir = Path(tmpdir) / "snaps"
            snap_dir.mkdir()

            payload = b"world_state_full_binary_blob"
            pre_crash_hash = hashlib.sha256(payload).hexdigest()

            writer = SnapshotWriter(str(snap_dir))
            writer.write("baseline_00001", payload, _meta(epoch=2))

            # "Restart": create a new loader pointing to the same dir
            loader = SnapshotLoader(str(snap_dir), format_version=_FMT)
            lr = loader.load_latest()

            self.assertIsNotNone(lr.payload)
            post_restart_hash = hashlib.sha256(lr.payload).hexdigest()
            self.assertEqual(pre_crash_hash, post_restart_hash)


class TestEpochIncrementOnReset(unittest.TestCase):
    """4. test_epoch_increment_on_reset"""

    def test_epoch_increment_on_reset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            snap_dir = Path(tmpdir) / "snaps"
            snap_dir.mkdir()

            # Write snapshot with incompatible version
            writer = SnapshotWriter(str(snap_dir))
            writer.write(
                "baseline_00001",
                b"old_format_world",
                {"format_version": 99, "world_epoch": 4},  # incompatible
            )

            boot = WorldBootManager(
                str(snap_dir),
                format_version=_FMT,   # current = 1, stored = 99
                reset_on_corrupt=True,
            )
            result = boot.boot()

            self.assertTrue(result.fresh_world)
            self.assertEqual(result.world_epoch, 5, "epoch should be old_epoch + 1")
            self.assertEqual(result.reset_reason, "format_version_incompatible")


class TestReconnectAfterServerRestart(unittest.TestCase):
    """5. test_reconnect_after_server_restart"""

    def test_reconnect_after_server_restart(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            snap_dir = Path(tmpdir) / "snaps"
            snap_dir.mkdir()

            writer = SnapshotWriter(str(snap_dir))
            writer.write("baseline_00001", b"world_v7", _meta(epoch=7))

            boot = WorldBootManager(str(snap_dir), format_version=_FMT)
            result = boot.boot()

            self.assertFalse(result.fresh_world)
            self.assertEqual(result.world_epoch, 7)


class TestCrashDuringInstabilityEvent(unittest.TestCase):
    """6. test_crash_during_instability_event"""

    def test_crash_during_instability_event(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            snap_dir = Path(tmpdir) / "snaps"
            snap_dir.mkdir()

            writer = SnapshotWriter(str(snap_dir))
            payload_v1 = b"stable_world_v1"
            writer.write("baseline_00001", payload_v1, _meta(epoch=1))
            entry2 = writer.write("baseline_00002", b"unstable_world_v2", _meta(epoch=2))

            # Corrupt newest snapshot
            entry2.dat_path.write_bytes(b"CORRUPT")

            boot = WorldBootManager(str(snap_dir), format_version=_FMT)
            result = boot.boot()

            self.assertFalse(result.fresh_world)
            self.assertEqual(result.payload, payload_v1)


class TestDiskRotationPolicy(unittest.TestCase):
    """7. test_disk_rotation_policy"""

    def test_disk_rotation_policy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            snap_dir = Path(tmpdir) / "snaps"
            snap_dir.mkdir()

            writer = SnapshotWriter(str(snap_dir))
            for i in range(1, 8):  # 7 baselines
                writer.write(f"baseline_{i:05d}", b"payload", _meta(epoch=i))

            rotator = SnapshotRotator(str(snap_dir), keep_baselines=3)
            deleted = rotator.rotate()

            # Should have deleted 4 (7 - 3)
            self.assertEqual(len(deleted), 4, f"deleted: {deleted}")

            # 3 checkpoint files should remain
            remaining = list((snap_dir).glob("*.checkpoint"))
            self.assertEqual(len(remaining), 3)


class TestCrashRecoverySuiteAllPass(unittest.TestCase):
    """8. test_crash_recovery_suite_all_pass"""

    def test_crash_recovery_suite_all_pass(self):
        suite = CrashRecoveryTestSuite()
        results = suite.run_all()
        failures = [r for r in results if not r.passed]
        self.assertEqual(
            failures,
            [],
            "Some crash recovery scenarios failed:\n"
            + "\n".join(f"  {r.name}: {r.details}" for r in failures),
        )


class TestSnapshotValidatorMissingCheckpoint(unittest.TestCase):
    """Validator rejects snapshot missing checkpoint marker."""

    def test_missing_checkpoint_invalid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            snap_dir = Path(tmpdir) / "snaps"
            snap_dir.mkdir()
            writer = SnapshotWriter(str(snap_dir))
            entry = writer.write("baseline_00001", b"payload", _meta())
            # Remove checkpoint
            entry.checkpoint.unlink()

            validator = SnapshotValidator(current_format_version=_FMT)
            vr = validator.validate(entry)
            self.assertFalse(vr.ok)
            self.assertIn("checkpoint", vr.reason)


class TestWorldRollbackManager(unittest.TestCase):
    """WorldRollbackManager produces correct rollback result."""

    def test_rollback_increments_epoch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            snap_dir = Path(tmpdir) / "snaps"
            snap_dir.mkdir()
            writer = SnapshotWriter(str(snap_dir))
            payload = b"baseline_world_state"
            writer.write("baseline_00001", payload, _meta(epoch=5))

            mgr = WorldRollbackManager(str(snap_dir), format_version=_FMT)
            rr = mgr.rollback(current_epoch=5)

            self.assertTrue(rr.ok)
            self.assertEqual(rr.world_epoch, 6)
            self.assertEqual(rr.payload, payload)
            self.assertEqual(rr.notify_msg.get("type"), "WORLD_ROLLBACK")
            self.assertEqual(rr.notify_msg.get("worldEpoch"), 6)

    def test_rollback_no_baseline_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = WorldRollbackManager(tmpdir, format_version=_FMT)
            rr = mgr.rollback(current_epoch=0)
            self.assertFalse(rr.ok)


class TestCrashWatchdog(unittest.TestCase):
    """CrashWatchdog.is_alive reflects kick timing."""

    def test_alive_after_kick(self):
        wd = CrashWatchdog(tick_timeout_ms=1_000)
        wd.kick()
        self.assertTrue(wd.is_alive)

    def test_not_alive_after_timeout(self):
        import time
        wd = CrashWatchdog(tick_timeout_ms=1)  # 1ms timeout
        time.sleep(0.01)
        self.assertFalse(wd.is_alive)


if __name__ == "__main__":
    unittest.main()
