"""WorldBootManager — Stage 59 auto-recovery on server start.

Implements the boot sequence:

1. Find the latest snapshot with a valid checkpoint marker.
2. Validate it (checksum + version).
3. If valid → load payload and return it.
4. If corrupt → try the previous snapshot.
5. If version-incompatible → perform a controlled world reset
   (increment worldEpoch).
6. If no valid snapshot and ``reset_on_corrupt=True`` → fresh world.
7. If no valid snapshot and ``reset_on_corrupt=False`` → raise.

Public API
----------
WorldBootManager(snapshot_dir, format_version=1, reset_on_corrupt=True)
  .boot() -> BootResult
      Execute the boot sequence and return a BootResult.

BootResult
  .payload        — bytes of loaded snapshot, or None (fresh world)
  .meta           — snapshot metadata dict, or {}
  .world_epoch    — worldEpoch to use (incremented on reset)
  .fresh_world    — True if a fresh world was started
  .reset_reason   — human-readable reason for reset, or ""
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from src.save.SnapshotLoader import SnapshotLoader, LoadResult


@dataclass
class BootResult:
    """Outcome of the WorldBootManager boot sequence."""

    payload: Optional[bytes] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    world_epoch: int = 0
    fresh_world: bool = False
    reset_reason: str = ""


class WorldBootManager:
    """Manages world state recovery on server start.

    Parameters
    ----------
    snapshot_dir :
        Directory containing snapshot files.
    format_version :
        Expected snapshot format version for this server binary.
    reset_on_corrupt :
        When *True*, start a fresh world if no valid snapshot is found.
        When *False*, raise ``RuntimeError`` instead.
    """

    def __init__(
        self,
        snapshot_dir: str,
        format_version: int = 1,
        reset_on_corrupt: bool = True,
    ) -> None:
        self._loader         = SnapshotLoader(snapshot_dir, format_version)
        self._reset_on_corrupt = reset_on_corrupt

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def boot(self) -> BootResult:
        """Execute the boot sequence.

        Returns
        -------
        BootResult
            Describes the world state to resume (or start fresh).
        """
        result: LoadResult = self._loader.load_latest()

        if result.payload is not None:
            # Successfully loaded a valid snapshot.
            epoch = int(result.meta.get("world_epoch", 0))
            return BootResult(
                payload=result.payload,
                meta=result.meta,
                world_epoch=epoch,
                fresh_world=False,
                reset_reason="",
            )

        if result.all_incompatible:
            # Every candidate had an incompatible format version →
            # controlled world reset with epoch bump.
            current_epoch = self._current_epoch_from_entries()
            new_epoch = current_epoch + 1
            return BootResult(
                payload=None,
                meta={},
                world_epoch=new_epoch,
                fresh_world=True,
                reset_reason="format_version_incompatible",
            )

        # No valid snapshot found at all.
        if not self._reset_on_corrupt:
            raise RuntimeError(
                "WorldBootManager: no valid snapshot found and "
                "reset_on_corrupt=False"
            )

        return BootResult(
            payload=None,
            meta={},
            world_epoch=0,
            fresh_world=True,
            reset_reason="no_valid_snapshot",
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _current_epoch_from_entries(self) -> int:
        """Return the highest worldEpoch found in any snapshot meta."""
        best = 0
        for entry in self._loader.list_entries():
            meta = self._loader._validator.load_meta(entry)
            if meta:
                best = max(best, int(meta.get("world_epoch", 0)))
        return best
