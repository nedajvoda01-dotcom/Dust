"""Telemetry — Stage 42 dev-only runtime diagnostics.

Collects per-frame counters for all budget-tracked subsystems and stores
them in a ring-buffer log.  Emits budget-alarm warnings when thresholds are
exceeded.

This module is **dev-only** — it must be disabled in production builds by
setting ``telemetry.enable_dev = false`` in CONFIG_DEFAULTS.json (or by
not instantiating :class:`Telemetry` at all).

Per-frame counters
------------------
* ``ik_iters``          — IK solver iterations this frame
* ``deform_uploads``    — GPU deformation uploads this frame
* ``resonators_active`` — active audio resonators
* ``impulses_per_sec``  — audio impulses per second (rolling)
* ``net_bytes_per_sec`` — network bytes per second (rolling)

Budget alarms
-------------
* > 90 % of limit → ``WARN``
* > 100 % of limit → ``ERROR`` and fallback tier applied

Usage
-----
tel = Telemetry(config)

# Each frame:
tel.begin_frame()
tel.record("ik_iters", 47)
tel.record("resonators_active", 12)
tel.end_frame(sim_time=current_sim_time)

# Query recent history:
log = tel.get_log(last_n_seconds=30)
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

from src.core.Config import Config
from src.core.Logger import Logger

_TAG = "Telemetry"

# Budget alarm thresholds (fraction of limit)
_WARN_FRACTION  = 0.90
_ERROR_FRACTION = 1.00


@dataclass
class FrameRecord:
    """Telemetry snapshot for one simulation frame."""
    sim_time: float
    wall_time: float
    counters: Dict[str, float] = field(default_factory=dict)


class Telemetry:
    """Dev-only per-frame diagnostics with ring-buffer storage."""

    # Budget limits (soft defaults; BudgetManager holds authoritative values)
    _LIMITS: Dict[str, float] = {
        "ik_iters":           120.0,
        "deform_uploads":       8.0,
        "resonators_active":   32.0,
        "impulses_per_sec":   120.0,
        "net_bytes_per_sec": 65536.0,
    }

    def __init__(self, config: Optional[Config] = None) -> None:
        self._enabled: bool = True
        ringbuffer_sec: float = 60.0

        if config is not None:
            enabled_val = config.get("telemetry", "enable_dev")
            if enabled_val is not None:
                self._enabled = bool(enabled_val)
            rb_val = config.get("telemetry", "ringbuffer_sec")
            if rb_val is not None:
                ringbuffer_sec = float(rb_val)
            # Override limits from budget config
            for key, cfg_path in [
                ("ik_iters",          ("budget", "motor", "max_ik_iters")),
                ("resonators_active", ("budget", "audio", "max_resonators")),
                ("deform_uploads",    ("budget", "deform", "max_gpu_uploads_per_frame")),
                ("impulses_per_sec",  ("budget", "audio", "max_impulses_per_sec")),
                ("net_bytes_per_sec", ("budget", "net", "max_bps")),
            ]:
                val = config.get(*cfg_path)
                if val is not None:
                    self._LIMITS[key] = float(val)

        self._ringbuffer_sec: float = ringbuffer_sec
        self._log: Deque[FrameRecord] = deque()

        # Mutable per-frame accumulator
        self._current: Dict[str, float] = {k: 0.0 for k in self._LIMITS}
        self._frame_start_wall: float = 0.0

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ------------------------------------------------------------------
    # Frame lifecycle
    # ------------------------------------------------------------------

    def begin_frame(self) -> None:
        """Reset all counters.  Call once at the start of each frame."""
        if not self._enabled:
            return
        for k in self._current:
            self._current[k] = 0.0
        self._frame_start_wall = time.monotonic()

    def record(self, counter: str, value: float) -> None:
        """Set (or accumulate) a counter value for the current frame.

        Unknown counters are dynamically added (dev mode) so callers can
        record subsystem-specific metrics without pre-registration.
        """
        if not self._enabled:
            return
        self._current[counter] = self._current.get(counter, 0.0) + value

    def end_frame(self, sim_time: float) -> None:
        """Finalise the frame, check alarms, append to ring-buffer log."""
        if not self._enabled:
            return

        wall_now = time.monotonic()
        snap = FrameRecord(
            sim_time=sim_time,
            wall_time=wall_now,
            counters=dict(self._current),
        )
        self._log.append(snap)
        self._evict_old(sim_time)
        self._check_alarms(snap)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_log(self, last_n_seconds: float = 60.0) -> List[FrameRecord]:
        """Return all records within the last *last_n_seconds* of sim time."""
        if not self._log:
            return []
        latest = self._log[-1].sim_time
        cutoff = latest - last_n_seconds
        return [r for r in self._log if r.sim_time >= cutoff]

    def latest(self, counter: str) -> float:
        """Return the most recent value of *counter*, or 0.0."""
        if not self._log:
            return 0.0
        return self._log[-1].counters.get(counter, 0.0)

    def log_size(self) -> int:
        """Number of frame records currently stored."""
        return len(self._log)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _evict_old(self, sim_time: float) -> None:
        """Remove records older than ringbuffer_sec."""
        cutoff = sim_time - self._ringbuffer_sec
        while self._log and self._log[0].sim_time < cutoff:
            self._log.popleft()

    def _check_alarms(self, snap: FrameRecord) -> None:
        """Emit warnings/errors for over-budget counters."""
        for key, limit in self._LIMITS.items():
            if limit <= 0:
                continue
            val = snap.counters.get(key, 0.0)
            frac = val / limit
            if frac >= _ERROR_FRACTION:
                Logger.error(
                    _TAG,
                    f"BUDGET EXCEEDED: {key}={val:.1f} limit={limit:.1f} "
                    f"({frac*100:.0f}%) at t={snap.sim_time:.2f}",
                )
            elif frac >= _WARN_FRACTION:
                Logger.warn(
                    _TAG,
                    f"Budget warning: {key}={val:.1f}/{limit:.1f} "
                    f"({frac*100:.0f}%) at t={snap.sim_time:.2f}",
                )
