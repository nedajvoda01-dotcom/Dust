"""SnapshotWriter — Stage 59 atomic snapshot persistence.

Writes world snapshots using a safe atomic sequence:

1. Write payload to ``<name>.tmp``
2. fsync the tmp file
3. Write metadata to ``<name>.meta.tmp``
4. fsync the meta file
5. Rename ``<name>.tmp`` → ``<name>.dat``
6. Rename ``<name>.meta.tmp`` → ``<name>.meta``
7. Write ``<name>.checkpoint`` marker

Only when the checkpoint marker exists is the snapshot considered valid.

Public API
----------
SnapshotWriter(snapshot_dir)
  .write(name, payload, meta) -> SnapshotEntry
      Write a snapshot atomically; returns a SnapshotEntry.

SnapshotEntry
  .name        — base name (e.g. "baseline_00042")
  .dat_path    — Path to the payload file
  .meta_path   — Path to the metadata file
  .checkpoint  — Path to the checkpoint marker
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class SnapshotEntry:
    """Describes a fully-written snapshot on disk."""

    name: str
    dat_path: Path
    meta_path: Path
    checkpoint: Path


class SnapshotWriter:
    """Writes snapshots atomically to *snapshot_dir*.

    Parameters
    ----------
    snapshot_dir :
        Directory where snapshot files are stored.  Created on first use.
    """

    def __init__(self, snapshot_dir: str) -> None:
        self._dir = Path(snapshot_dir)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def write(
        self,
        name: str,
        payload: bytes,
        meta: Dict[str, Any],
    ) -> SnapshotEntry:
        """Write a snapshot atomically.

        Parameters
        ----------
        name :
            Base name for the snapshot files (no extension).
        payload :
            Raw binary payload (world state blob).
        meta :
            Metadata dict; ``checksum`` is added automatically.

        Returns
        -------
        SnapshotEntry with paths to the written files.
        """
        self._dir.mkdir(parents=True, exist_ok=True)

        dat_path   = self._dir / f"{name}.dat"
        meta_path  = self._dir / f"{name}.meta"
        ckpt_path  = self._dir / f"{name}.checkpoint"

        checksum = hashlib.sha256(payload).hexdigest()
        meta_out = dict(meta)
        meta_out["checksum"] = checksum
        meta_bytes = json.dumps(meta_out).encode("utf-8")

        # Step 1-2: write + fsync payload tmp
        tmp_dat = self._dir / f"{name}.tmp"
        self._write_fsync(tmp_dat, payload)

        # Step 3-4: write + fsync meta tmp
        tmp_meta = self._dir / f"{name}.meta.tmp"
        self._write_fsync(tmp_meta, meta_bytes)

        # Step 5-6: atomic rename
        tmp_dat.replace(dat_path)
        tmp_meta.replace(meta_path)

        # Step 7: checkpoint marker
        ckpt_path.write_text("ok", encoding="utf-8")

        return SnapshotEntry(
            name=name,
            dat_path=dat_path,
            meta_path=meta_path,
            checkpoint=ckpt_path,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _write_fsync(path: Path, data: bytes) -> None:
        """Write *data* to *path* and fsync the file descriptor."""
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        try:
            os.write(fd, data)
            os.fsync(fd)
        finally:
            os.close(fd)
