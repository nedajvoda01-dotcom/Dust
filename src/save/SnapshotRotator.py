"""SnapshotRotator — Stage 59 snapshot retention / rotation policy.

Enforces per-config retention limits on snapshot files:
* Keep at most ``keep_baselines`` baseline snapshots.
* Keep at most ``keep_incrementals`` incremental snapshots.
* Total snapshot dir must not exceed ``max_disk_mb`` MiB.

Naming convention (used by WorldBootManager):
* Baseline:    ``baseline_<seq>.dat`` (+ ``.meta`` + ``.checkpoint``)
* Incremental: ``incr_<seq>.dat``     (+ ``.meta`` + ``.checkpoint``)

Public API
----------
SnapshotRotator(snapshot_dir, keep_baselines=5, keep_incrementals=20,
                max_disk_mb=512)
  .rotate()  — apply retention policy; return list of deleted names
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple


_BASELINE_PREFIX    = "baseline_"
_INCREMENTAL_PREFIX = "incr_"
_EXTENSIONS         = (".dat", ".meta", ".checkpoint")


class SnapshotRotator:
    """Applies snapshot retention policy to *snapshot_dir*.

    Parameters
    ----------
    snapshot_dir :
        Directory that holds snapshot files.
    keep_baselines :
        Maximum number of baseline snapshots to keep (most recent).
    keep_incrementals :
        Maximum number of incremental snapshots to keep (most recent).
    max_disk_mb :
        Hard cap on total snapshot directory size in MiB.
    """

    def __init__(
        self,
        snapshot_dir: str,
        keep_baselines: int = 5,
        keep_incrementals: int = 20,
        max_disk_mb: float = 512.0,
    ) -> None:
        self._dir            = Path(snapshot_dir)
        self._keep_baselines = max(1, keep_baselines)
        self._keep_incr      = max(1, keep_incrementals)
        self._max_bytes      = int(max_disk_mb * 1024 * 1024)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def rotate(self) -> List[str]:
        """Apply retention policy; return names of deleted snapshot groups."""
        if not self._dir.exists():
            return []

        deleted: List[str] = []
        deleted += self._purge_group(_BASELINE_PREFIX, self._keep_baselines)
        deleted += self._purge_group(_INCREMENTAL_PREFIX, self._keep_incr)
        deleted += self._purge_over_disk_limit()
        return deleted

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sorted_groups(self, prefix: str) -> List[Tuple[float, str]]:
        """Return (mtime, name) tuples for *prefix* groups, oldest-first."""
        groups: List[Tuple[float, str]] = []
        for ckpt in self._dir.glob(f"{prefix}*.checkpoint"):
            name = ckpt.stem
            try:
                mtime = ckpt.stat().st_mtime
            except OSError:
                mtime = 0.0
            groups.append((mtime, name))
        groups.sort()
        return groups

    def _purge_group(self, prefix: str, keep: int) -> List[str]:
        """Delete oldest groups of *prefix* until at most *keep* remain."""
        groups = self._sorted_groups(prefix)
        to_delete = groups[: max(0, len(groups) - keep)]
        deleted = []
        for _, name in to_delete:
            self._delete_snapshot(name)
            deleted.append(name)
        return deleted

    def _dir_size_bytes(self) -> int:
        total = 0
        for ext in _EXTENSIONS:
            for f in self._dir.glob(f"*{ext}"):
                try:
                    total += f.stat().st_size
                except OSError:
                    pass
        return total

    def _purge_over_disk_limit(self) -> List[str]:
        """Delete oldest snapshots (either type) until under disk limit."""
        deleted: List[str] = []
        while self._dir_size_bytes() > self._max_bytes:
            # Collect all groups, oldest first
            all_groups: List[Tuple[float, str]] = []
            for ckpt in self._dir.glob("*.checkpoint"):
                name = ckpt.stem
                try:
                    mtime = ckpt.stat().st_mtime
                except OSError:
                    mtime = 0.0
                all_groups.append((mtime, name))
            if not all_groups:
                break
            all_groups.sort()
            oldest_name = all_groups[0][1]
            self._delete_snapshot(oldest_name)
            deleted.append(oldest_name)
        return deleted

    def _delete_snapshot(self, name: str) -> None:
        """Delete all files associated with snapshot *name*."""
        for ext in _EXTENSIONS:
            p = self._dir / f"{name}{ext}"
            try:
                p.unlink()
            except OSError:
                pass
