"""PersistentStorage — save directory management and atomic file I/O.

Provides crash-safe write operations via the atomic write-to-temp +
rename pattern, so a partial write never corrupts an existing save.

Public API
----------
PersistentStorage(save_dir=None)
  .init()                         — create the save directory
  .write_json_atomic(name, data)  — atomically write a JSON file
  .read_json(name) -> dict | None — read a JSON file or None if missing
  .write_bytes_atomic(name, data) — atomically write raw bytes
  .read_bytes(name) -> bytes | None
  .exists(name) -> bool
  .delete(name)
  .reset()                        — delete all saves (--reset)
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Optional


class PersistentStorage:
    """Manages the saves directory with crash-safe I/O helpers."""

    def __init__(self, save_dir: Optional[str] = None) -> None:
        if save_dir is None:
            root = Path(__file__).parent.parent.parent
            save_dir = str(root / "saves")
        self._dir = Path(save_dir)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def init(self) -> None:
        """Create the save directory (no-op if it already exists)."""
        self._dir.mkdir(parents=True, exist_ok=True)

    def reset(self) -> None:
        """Delete all save files and recreate the directory (--reset)."""
        if self._dir.exists():
            shutil.rmtree(self._dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # JSON helpers
    # ------------------------------------------------------------------

    def write_json_atomic(self, filename: str, data: Any) -> None:
        """Write *data* as JSON to *filename* atomically.

        Writes to a temp file first, then renames over the target so a
        crash mid-write never leaves a corrupt file.
        """
        target = self._dir / filename
        fd, tmp_path = tempfile.mkstemp(dir=self._dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
            shutil.move(tmp_path, str(target))
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def read_json(self, filename: str) -> Optional[Any]:
        """Read and return a JSON file, or None if it does not exist."""
        target = self._dir / filename
        if not target.exists():
            return None
        with open(target, "r", encoding="utf-8") as fh:
            return json.load(fh)

    # ------------------------------------------------------------------
    # Binary helpers
    # ------------------------------------------------------------------

    def write_bytes_atomic(self, filename: str, data: bytes) -> None:
        """Write raw *bytes* to *filename* atomically."""
        target = self._dir / filename
        fd, tmp_path = tempfile.mkstemp(dir=self._dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            shutil.move(tmp_path, str(target))
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def read_bytes(self, filename: str) -> Optional[bytes]:
        """Read and return raw bytes, or None if the file does not exist."""
        target = self._dir / filename
        if not target.exists():
            return None
        with open(target, "rb") as fh:
            return fh.read()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def exists(self, filename: str) -> bool:
        """Return True if *filename* exists in the save directory."""
        return (self._dir / filename).exists()

    def delete(self, filename: str) -> None:
        """Delete *filename* from the save directory (no-op if missing)."""
        target = self._dir / filename
        if target.exists():
            target.unlink()

    def path(self, filename: str) -> Path:
        """Return the full path for *filename* in the save directory."""
        return self._dir / filename
