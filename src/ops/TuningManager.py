"""TuningManager — Stage 61 live tuning with config epochs.

Implements deterministic live parameter changes:

* Each change is tagged with a ``tuningEpoch`` (uint32).
* Changes are validated against the allowlist before acceptance.
* Changes are applied only at a tick boundary (``serverTick % N == 0``).
* The last *K* configs are retained for rollback.
* ``tuningEpoch`` and ``tuningConfigHash`` are included in snapshots.

Forbidden parameters (worldSeed, epoch, any physics-breaking param)
are never accepted; enforcement is delegated to :class:`TuningValidator`.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

from src.ops.TuningValidator import TuningValidator


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_APPLY_TICK_BOUNDARY = 64   # apply on multiples of this tick
_DEFAULT_ROLLBACK_KEEP_K     = 10   # keep last K configs


# ---------------------------------------------------------------------------
# Epoch entry
# ---------------------------------------------------------------------------

class TuningEpochEntry:
    """One versioned tuning configuration."""

    def __init__(
        self,
        epoch:      int,
        config:     Dict[str, Any],
        applied_at: Optional[int] = None,
    ) -> None:
        self.epoch      = epoch
        self.config     = dict(config)
        self.applied_at = applied_at   # serverTick when applied (None = pending)
        self.config_hash = self._compute_hash(config)

    @staticmethod
    def _compute_hash(config: Dict[str, Any]) -> str:
        raw = json.dumps(config, sort_keys=True).encode()
        return hashlib.sha256(raw).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tuningEpoch":      self.epoch,
            "tuningConfigHash": self.config_hash,
            "config":           self.config,
            "appliedAtTick":    self.applied_at,
        }


# ---------------------------------------------------------------------------
# TuningManager
# ---------------------------------------------------------------------------

class TuningManager:
    """Manages live tuning epochs with deterministic tick-boundary application.

    Parameters
    ----------
    validator:
        :class:`~src.ops.TuningValidator.TuningValidator` instance.
    apply_tick_boundary:
        Apply pending changes when ``serverTick % apply_tick_boundary == 0``.
    rollback_keep_k:
        Number of past configs to retain for rollback.
    """

    def __init__(
        self,
        validator:            Optional[TuningValidator] = None,
        apply_tick_boundary:  int = _DEFAULT_APPLY_TICK_BOUNDARY,
        rollback_keep_k:      int = _DEFAULT_ROLLBACK_KEEP_K,
    ) -> None:
        self._validator           = validator or TuningValidator()
        self._apply_tick_boundary = max(1, apply_tick_boundary)
        self._rollback_keep_k     = max(1, rollback_keep_k)

        # Current live config (epoch 0 = baseline empty)
        self._current_epoch:  int               = 0
        self._current_config: Dict[str, Any]    = {}
        self._pending:        Optional[TuningEpochEntry] = None

        # History deque for rollback
        self._history: Deque[TuningEpochEntry] = deque(maxlen=rollback_keep_k)

        # Seed history with epoch-0 baseline
        entry = TuningEpochEntry(0, {}, applied_at=0)
        self._history.append(entry)

    # ------------------------------------------------------------------
    # Propose
    # ------------------------------------------------------------------

    def propose(
        self,
        delta: Dict[str, Any],
    ) -> Tuple[bool, List[str]]:
        """Propose a config delta for the next epoch.

        Returns
        -------
        (accepted, errors)
            *accepted* is True iff the delta passed validation.
        """
        errors = self._validator.validate(delta)
        if errors:
            return False, errors

        new_epoch  = self._current_epoch + 1
        new_config = {**self._current_config, **delta}
        self._pending = TuningEpochEntry(new_epoch, new_config)
        return True, []

    # ------------------------------------------------------------------
    # Tick hook
    # ------------------------------------------------------------------

    def on_tick(self, server_tick: int) -> Optional[TuningEpochEntry]:
        """Called every server tick.

        If a proposed change is pending and *server_tick* falls on the
        configured boundary, the change is applied and returned.
        Returns *None* otherwise.
        """
        if self._pending is None:
            return None
        if server_tick % self._apply_tick_boundary != 0:
            return None

        entry = self._pending
        entry.applied_at = server_tick
        self._current_epoch  = entry.epoch
        self._current_config = dict(entry.config)
        self._history.append(entry)
        self._pending = None
        return entry

    # ------------------------------------------------------------------
    # Rollback
    # ------------------------------------------------------------------

    def rollback(self, to_epoch: int) -> Tuple[bool, str]:
        """Roll back to a previously applied epoch.

        Returns
        -------
        (success, message)
        """
        for entry in reversed(list(self._history)):
            if entry.epoch == to_epoch and entry.applied_at is not None:
                self._current_epoch  = entry.epoch
                self._current_config = dict(entry.config)
                self._pending        = None
                return True, f"rolled back to tuningEpoch={to_epoch}"
        return False, f"epoch {to_epoch} not found in history"

    # ------------------------------------------------------------------
    # Snapshot integration
    # ------------------------------------------------------------------

    def snapshot_meta(self) -> Dict[str, Any]:
        """Return the tuning fields that must be included in snapshots."""
        return {
            "tuningEpoch":      self._current_epoch,
            "tuningConfigHash": TuningEpochEntry._compute_hash(self._current_config),
        }

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def current_epoch(self) -> int:
        return self._current_epoch

    @property
    def current_config(self) -> Dict[str, Any]:
        return dict(self._current_config)

    @property
    def has_pending(self) -> bool:
        return self._pending is not None

    def history(self) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in self._history]
