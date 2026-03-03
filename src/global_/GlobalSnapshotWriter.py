"""GlobalSnapshotWriter — Stage 60 Global Authority snapshot persistence.

Writes and loads the Global Authority (GA) state snapshot so the world can be
recovered after a crash or failover.

Format
------
Files are stored under ``state_dir/global/``:

    global/
        ga_snapshot.json          — current GA snapshot (atomic write)
        ga_checkpoint_{id}.json   — checkpoint-pinned snapshots

Public API
----------
GlobalSnapshotWriter(state_dir="world_state")
  .write(ga_state: dict, checkpoint_id: str | None = None) -> None
  .read() -> dict | None
  .read_checkpoint(checkpoint_id: str) -> dict | None
  .list_checkpoints() -> list[str]
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional


_CURRENT_FILE   = "ga_snapshot.json"
_CHECKPOINT_PFX = "ga_checkpoint_"


class GlobalSnapshotWriter:
    """Persists and loads Global Authority snapshots."""

    def __init__(self, state_dir: str = "world_state") -> None:
        self._dir = Path(state_dir) / "global"
        self._dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write(
        self,
        ga_state: Dict[str, Any],
        checkpoint_id: Optional[str] = None,
    ) -> None:
        """Persist *ga_state*.

        Always writes to ``ga_snapshot.json`` atomically.
        If *checkpoint_id* is given, also writes a checkpoint copy.
        """
        self._atomic_write(self._dir / _CURRENT_FILE, ga_state)
        if checkpoint_id is not None:
            name = f"{_CHECKPOINT_PFX}{checkpoint_id}.json"
            self._atomic_write(self._dir / name, ga_state)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def read(self) -> Optional[Dict[str, Any]]:
        """Load the current GA snapshot, or ``None`` if not found."""
        return self._load(self._dir / _CURRENT_FILE)

    def read_checkpoint(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """Load a specific checkpoint snapshot, or ``None``."""
        name = f"{_CHECKPOINT_PFX}{checkpoint_id}.json"
        return self._load(self._dir / name)

    def list_checkpoints(self) -> List[str]:
        """Return sorted list of stored checkpoint IDs."""
        ids = []
        for p in self._dir.iterdir():
            if p.name.startswith(_CHECKPOINT_PFX) and p.suffix == ".json":
                cid = p.stem[len(_CHECKPOINT_PFX):]
                ids.append(cid)
        ids.sort()
        return ids

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
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

    @staticmethod
    def _load(path: Path) -> Optional[Dict[str, Any]]:
        try:
            with path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError):
            return None
