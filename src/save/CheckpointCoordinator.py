"""CheckpointCoordinator — Stage 60 coordinated GA + RS snapshot checkpoints.

Implements the consistent checkpoint protocol described in Stage 60, §9.3:

1. The Global Authority calls ``begin_checkpoint()`` to allocate a new
   checkpoint ID and enter the *pending* state.
2. Each Region Shard calls ``register_region_snapshot(checkpoint_id, ...)``
   to declare its snapshot for that checkpoint.
3. Once all expected regions have reported (or the timeout passes),
   ``try_finalise()`` writes the combined checkpoint and transitions to
   *complete*.

This class handles only the coordination logic; actual snapshot bytes are
written by :class:`~src.global.GlobalSnapshotWriter.GlobalSnapshotWriter`
and equivalent RS snapshot writers.

Public API
----------
CheckpointCoordinator(ga_writer, state_dir, expected_region_count,
                      timeout_s=60.0)
  .begin_checkpoint(ga_snapshot) → str
      Start a new checkpoint round; returns the checkpoint ID.
  .register_region_snapshot(checkpoint_id, region_id, region_snapshot)
      Record a region's snapshot for the given checkpoint.
  .try_finalise(checkpoint_id) → bool
      Attempt to finalise. Returns True if complete, False if still waiting.
  .is_complete(checkpoint_id) → bool
  .list_checkpoints() → list[str]
  .load_checkpoint(checkpoint_id) → dict | None
"""
from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


_CHECKPOINT_DIR = "checkpoints"


class CheckpointCoordinator:
    """Coordinates a consistent checkpoint across GA and all RS nodes."""

    def __init__(
        self,
        ga_writer,              # GlobalSnapshotWriter
        state_dir:      str   = "world_state",
        expected_regions: int = 0,
        timeout_s:      float = 60.0,
    ) -> None:
        self._ga_writer        = ga_writer
        self._dir              = Path(state_dir) / _CHECKPOINT_DIR
        self._dir.mkdir(parents=True, exist_ok=True)
        self._expected_regions = expected_regions
        self._timeout_s        = float(timeout_s)

        # In-flight state: {checkpoint_id → _PendingCheckpoint}
        self._pending: Dict[str, "_PendingCheckpoint"] = {}

    # ------------------------------------------------------------------
    # Protocol
    # ------------------------------------------------------------------

    def begin_checkpoint(self, ga_snapshot: Dict[str, Any]) -> str:
        """Allocate a new checkpoint ID and record the GA snapshot.

        Returns the checkpoint ID string that RS nodes must use when calling
        :meth:`register_region_snapshot`.
        """
        cid = uuid.uuid4().hex[:12]
        self._pending[cid] = _PendingCheckpoint(
            checkpoint_id=cid,
            ga_snapshot=ga_snapshot,
            started_at=time.monotonic(),
        )
        # Also write the GA snapshot immediately
        self._ga_writer.write(ga_snapshot, checkpoint_id=cid)
        return cid

    def register_region_snapshot(
        self,
        checkpoint_id:   str,
        region_id:       int,
        region_snapshot: Dict[str, Any],
    ) -> None:
        """Record *region_id*'s snapshot for *checkpoint_id*."""
        pending = self._pending.get(checkpoint_id)
        if pending is None:
            return
        pending.region_snapshots[region_id] = region_snapshot

    def try_finalise(self, checkpoint_id: str) -> bool:
        """Try to finalise the checkpoint.

        Returns ``True`` if either:
        * All expected regions have reported, or
        * The timeout has passed (partial checkpoint written as-is).

        On success, writes the combined checkpoint JSON and removes the
        in-flight entry.
        """
        pending = self._pending.get(checkpoint_id)
        if pending is None:
            return False  # Already finalised or unknown

        n_regions = len(pending.region_snapshots)
        timed_out = (time.monotonic() - pending.started_at) > self._timeout_s
        ready     = (
            self._expected_regions <= 0
            or n_regions >= self._expected_regions
            or timed_out
        )
        if not ready:
            return False

        # Write combined checkpoint
        combined = {
            "checkpoint_id": checkpoint_id,
            "ga":            pending.ga_snapshot,
            "regions":       {
                str(rid): snap
                for rid, snap in pending.region_snapshots.items()
            },
            "finalised_at":  time.time(),
            "partial":       timed_out and n_regions < max(1, self._expected_regions),
        }
        path = self._dir / f"checkpoint_{checkpoint_id}.json"
        _atomic_write(path, combined)
        del self._pending[checkpoint_id]
        return True

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def is_complete(self, checkpoint_id: str) -> bool:
        """Return ``True`` if the checkpoint has been finalised."""
        return (
            checkpoint_id not in self._pending
            and (self._dir / f"checkpoint_{checkpoint_id}.json").exists()
        )

    def list_checkpoints(self) -> List[str]:
        """Return sorted list of finalised checkpoint IDs."""
        ids = []
        for p in self._dir.iterdir():
            if p.name.startswith("checkpoint_") and p.suffix == ".json":
                cid = p.stem[len("checkpoint_"):]
                ids.append(cid)
        ids.sort()
        return ids

    def load_checkpoint(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """Load and return a finalised checkpoint dict, or ``None``."""
        path = self._dir / f"checkpoint_{checkpoint_id}.json"
        try:
            with path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError):
            return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _PendingCheckpoint:
    __slots__ = ("checkpoint_id", "ga_snapshot", "started_at", "region_snapshots")

    def __init__(
        self,
        checkpoint_id: str,
        ga_snapshot:   Dict[str, Any],
        started_at:    float,
    ) -> None:
        self.checkpoint_id    = checkpoint_id
        self.ga_snapshot      = ga_snapshot
        self.started_at       = started_at
        self.region_snapshots: Dict[int, Dict[str, Any]] = {}


def _atomic_write(path: Path, data: Dict[str, Any]) -> None:
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent), prefix=".tmp_", suffix=".json"
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False)
        os.replace(tmp_path, str(path))
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
