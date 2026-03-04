"""SnapshotValidator — Stage 59 snapshot integrity & version checks.

Validates that a snapshot on disk is:
1. Structurally complete (all three files present).
2. Checksum-verified (SHA-256 of payload matches meta["checksum"]).
3. Version-compatible (meta["format_version"] matches current).

Public API
----------
SnapshotValidator(current_format_version=1)
  .validate(entry) -> ValidationResult
      Validate a SnapshotEntry; returns a ValidationResult.
  .load_meta(entry) -> dict | None
      Load and return the metadata dict, or None on failure.

ValidationResult
  .ok          — True if snapshot is valid
  .reason      — human-readable failure reason (empty string if ok)
  .incompatible_version — True if failure is due to version mismatch
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from src.save.SnapshotWriter import SnapshotEntry


@dataclass
class ValidationResult:
    """Result of validating one snapshot."""

    ok: bool = False
    reason: str = ""
    incompatible_version: bool = False


class SnapshotValidator:
    """Validates snapshot integrity and version compatibility.

    Parameters
    ----------
    current_format_version :
        The format version this server understands.
    """

    def __init__(self, current_format_version: int = 1) -> None:
        self._version = current_format_version

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def validate(self, entry: SnapshotEntry) -> ValidationResult:
        """Validate *entry*.

        Checks (in order):
        1. Checkpoint marker exists.
        2. Payload and meta files exist.
        3. Meta JSON is parseable.
        4. Format version is compatible.
        5. SHA-256 checksum matches.

        Returns
        -------
        ValidationResult
        """
        # 1. Checkpoint marker
        if not entry.checkpoint.exists():
            return ValidationResult(ok=False, reason="missing checkpoint marker")

        # 2. File presence
        if not entry.dat_path.exists():
            return ValidationResult(ok=False, reason="missing payload file")
        if not entry.meta_path.exists():
            return ValidationResult(ok=False, reason="missing meta file")

        # 3. Parse meta
        meta = self.load_meta(entry)
        if meta is None:
            return ValidationResult(ok=False, reason="corrupt meta JSON")

        # 4. Version compatibility
        stored_version = int(meta.get("format_version", 0))
        if stored_version != self._version:
            return ValidationResult(
                ok=False,
                reason=f"format version mismatch: stored={stored_version} current={self._version}",
                incompatible_version=True,
            )

        # 5. Checksum
        expected = meta.get("checksum", "")
        try:
            payload = entry.dat_path.read_bytes()
        except OSError as exc:
            return ValidationResult(ok=False, reason=f"cannot read payload: {exc}")

        actual = hashlib.sha256(payload).hexdigest()
        if actual != expected:
            return ValidationResult(
                ok=False,
                reason=f"checksum mismatch: expected={expected} actual={actual}",
            )

        return ValidationResult(ok=True, reason="")

    def load_meta(self, entry: SnapshotEntry) -> Optional[Dict[str, Any]]:
        """Load metadata from *entry.meta_path*.

        Returns the parsed dict, or *None* on any error.
        """
        try:
            return json.loads(entry.meta_path.read_bytes())
        except Exception:
            return None
