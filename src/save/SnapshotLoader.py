"""SnapshotLoader — Stage 59 snapshot loading with fallback.

Scans a snapshot directory, finds all valid snapshots (newest first),
and returns the payload of the first valid one.  Falls back to older
snapshots if newer ones are corrupt or version-incompatible.

Public API
----------
SnapshotLoader(snapshot_dir, format_version=1)
  .list_entries()  -> List[SnapshotEntry]   newest-first
  .load_latest()   -> LoadResult
      Tries newest → oldest; returns the first valid payload.
  .load_entry(entry) -> bytes | None
      Load payload bytes for a specific SnapshotEntry.

LoadResult
  .payload         — raw bytes, or None if no valid snapshot found
  .meta            — metadata dict of the loaded snapshot, or {}
  .entry           — SnapshotEntry that was loaded, or None
  .all_incompatible — True if every candidate had a version mismatch
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.save.SnapshotWriter import SnapshotEntry
from src.save.SnapshotValidator import SnapshotValidator, ValidationResult


@dataclass
class LoadResult:
    """Result of a snapshot load attempt."""

    payload: Optional[bytes] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    entry: Optional[SnapshotEntry] = None
    all_incompatible: bool = False


class SnapshotLoader:
    """Loads the latest valid snapshot from *snapshot_dir*.

    Parameters
    ----------
    snapshot_dir :
        Directory containing ``.dat`` / ``.meta`` / ``.checkpoint`` files.
    format_version :
        Expected snapshot format version.
    """

    def __init__(self, snapshot_dir: str, format_version: int = 1) -> None:
        self._dir = Path(snapshot_dir)
        self._validator = SnapshotValidator(current_format_version=format_version)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def list_entries(self) -> List[SnapshotEntry]:
        """Return all checkpoint-marked entries, newest first.

        Entries are sorted by the mtime of the checkpoint marker file.
        """
        if not self._dir.exists():
            return []

        entries: List[SnapshotEntry] = []
        for ckpt in sorted(
            self._dir.glob("*.checkpoint"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        ):
            name = ckpt.stem  # strip ".checkpoint"
            entries.append(
                SnapshotEntry(
                    name=name,
                    dat_path=self._dir / f"{name}.dat",
                    meta_path=self._dir / f"{name}.meta",
                    checkpoint=ckpt,
                )
            )
        return entries

    def load_latest(self) -> LoadResult:
        """Try each entry newest-first; return first valid payload.

        If every candidate has an incompatible version,
        ``LoadResult.all_incompatible`` is set to *True*.
        """
        entries = self.list_entries()
        if not entries:
            return LoadResult()

        incompatible_count = 0
        for entry in entries:
            result = self._validator.validate(entry)
            if result.ok:
                payload = self.load_entry(entry)
                if payload is not None:
                    meta = self._validator.load_meta(entry) or {}
                    return LoadResult(payload=payload, meta=meta, entry=entry)
            elif result.incompatible_version:
                incompatible_count += 1

        all_incompat = incompatible_count == len(entries) and incompatible_count > 0
        return LoadResult(all_incompatible=all_incompat)

    def load_entry(self, entry: SnapshotEntry) -> Optional[bytes]:
        """Return raw payload bytes for *entry*, or *None* on error."""
        try:
            return entry.dat_path.read_bytes()
        except OSError:
            return None
