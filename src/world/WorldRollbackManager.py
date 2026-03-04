"""WorldRollbackManager — Stage 59 soft rollback with epoch increment.

Handles the case where drift or corruption is detected *at runtime*:

1. Takes the last baseline snapshot from the loader.
2. Validates it.
3. Increments ``worldEpoch`` in the loaded meta.
4. Returns a ``RollbackResult`` the server can apply to live state.
5. Builds the ``WORLD_ROLLBACK`` notification dict for broadcasting to
   connected clients.

Rollback is intentionally separated from the normal boot path and is
not used "by default" — callers must decide when to invoke it.

Public API
----------
WorldRollbackManager(snapshot_dir, format_version=1, max_depth=3)
  .rollback(current_epoch) -> RollbackResult

RollbackResult
  .ok            — True if a valid baseline was found
  .payload       — bytes of the baseline snapshot
  .meta          — metadata dict (with updated world_epoch)
  .world_epoch   — new epoch value
  .notify_msg    — dict ready to JSON-serialise as WORLD_ROLLBACK
  .reason        — failure reason when ok=False
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.save.SnapshotLoader import SnapshotLoader
from src.save.SnapshotValidator import SnapshotValidator
from src.save.SnapshotWriter import SnapshotEntry


@dataclass
class RollbackResult:
    """Outcome of a rollback attempt."""

    ok: bool = False
    payload: Optional[bytes] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    world_epoch: int = 0
    notify_msg: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""


class WorldRollbackManager:
    """Performs soft rollback to the last valid baseline snapshot.

    Parameters
    ----------
    snapshot_dir :
        Directory containing snapshot files.
    format_version :
        Expected snapshot format version.
    max_depth :
        Maximum number of candidates to try before giving up.
    """

    def __init__(
        self,
        snapshot_dir: str,
        format_version: int = 1,
        max_depth: int = 3,
    ) -> None:
        self._loader    = SnapshotLoader(snapshot_dir, format_version)
        self._validator = SnapshotValidator(current_format_version=format_version)
        self._max_depth = max(1, max_depth)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def rollback(self, current_epoch: int = 0) -> RollbackResult:
        """Attempt to roll back to the last valid baseline.

        Parameters
        ----------
        current_epoch :
            The epoch currently in use; the new epoch will be
            ``current_epoch + 1``.

        Returns
        -------
        RollbackResult
        """
        # list_entries() returns entries newest-first; filter to baselines
        # while preserving that descending-mtime order so the most recent
        # valid baseline is tried first.
        entries = self._loader.list_entries()
        baselines = [e for e in entries if e.name.startswith("baseline_")]

        tried = 0
        for entry in baselines:
            if tried >= self._max_depth:
                break
            tried += 1

            vr = self._validator.validate(entry)
            if not vr.ok:
                continue

            payload = self._loader.load_entry(entry)
            if payload is None:
                continue

            meta = self._validator.load_meta(entry) or {}
            new_epoch = current_epoch + 1
            meta["world_epoch"] = new_epoch

            notify = {
                "type":        "WORLD_ROLLBACK",
                "worldEpoch":  new_epoch,
                "baselineName": entry.name,
                "timestamp":   datetime.now(timezone.utc).isoformat(),
            }

            return RollbackResult(
                ok=True,
                payload=payload,
                meta=meta,
                world_epoch=new_epoch,
                notify_msg=notify,
                reason="",
            )

        return RollbackResult(
            ok=False,
            reason="no valid baseline snapshot found",
        )
