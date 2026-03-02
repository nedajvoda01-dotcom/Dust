"""Telemetry — Stage 42 §5.  Dev-only runtime telemetry with ring buffers.

Collects per-frame counters and stores them in a fixed-size ring buffer.
**Never enabled in production builds**; gate usage behind
``telemetry.enable_dev``.

Metrics tracked (per frame)
---------------------------
ik_iters, deform_uploads, resonators_active, impulses_s, net_bytes_s

Budget alarms
-------------
* > 90 % of limit → WARN level log
* > 100 % of limit → ERROR level log

Ring-buffer retention
---------------------
``telemetry.ringbuffer_sec`` (default 60 s); at 60 fps the default ring
holds 3 600 frames.  Older frames are silently discarded.

Public API
----------
Telemetry(config_dict, budget_limits_dict)
  .record_frame(frame_no, metrics_dict)
  .get_recent(n_frames)           → list[FrameSnapshot]
  .check_alarms(metrics_dict)
  .clear()
  .frame_count()                  → int
  .enabled                        → bool
"""
from __future__ import annotations

import collections
import time
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

from src.core.Logger import Logger, LogLevel

_TAG = "Telemetry"

_DEFAULT_LIMITS: Dict[str, float] = {
    "ik_iters":          64.0,
    "deform_uploads":    8.0,
    "resonators_active": 24.0,
    "impulses_s":        120.0,
    "net_bytes_s":       65536.0,
}


@dataclass
class FrameSnapshot:
    """One frame's telemetry snapshot stored in the ring buffer."""
    frame_no: int
    wall_time: float
    metrics: Dict[str, float] = field(default_factory=dict)


class Telemetry:
    """Dev-only per-frame telemetry collector.

    When ``telemetry.enable_dev`` is ``False`` (default in production),
    all methods are no-ops and ``enabled`` is ``False``.

    Parameters
    ----------
    config:
        Dict that may contain a ``telemetry`` sub-dict with keys
        ``enable_dev`` (bool) and ``ringbuffer_sec`` (int/float).
    budget_limits:
        Dict mapping metric-name → limit value.
        Defaults to ``_DEFAULT_LIMITS`` if not provided.
    """

    def __init__(
        self,
        config: Optional[Dict] = None,
        budget_limits: Optional[Dict[str, float]] = None,
    ) -> None:
        cfg = config or {}
        tel_cfg = cfg.get("telemetry", {})
        self._enabled: bool = bool(tel_cfg.get("enable_dev", False))
        ringbuffer_sec: float = float(tel_cfg.get("ringbuffer_sec", 60.0))
        max_frames: int = max(1, int(ringbuffer_sec * 60))
        self._ring: Deque[FrameSnapshot] = collections.deque(maxlen=max_frames)
        self._limits: Dict[str, float] = dict(budget_limits or _DEFAULT_LIMITS)

    @property
    def enabled(self) -> bool:
        """``True`` only in dev builds with ``telemetry.enable_dev=true``."""
        return self._enabled

    def record_frame(self, frame_no: int, metrics: Dict[str, float]) -> None:
        """Push a snapshot of *metrics* for *frame_no* into the ring buffer."""
        if not self._enabled:
            return
        snap = FrameSnapshot(
            frame_no=frame_no,
            wall_time=time.monotonic(),
            metrics=dict(metrics),
        )
        self._ring.append(snap)
        self.check_alarms(metrics)

    def check_alarms(self, metrics: Dict[str, float]) -> None:
        """Log warnings for any metrics that exceed 90 % of their limit."""
        for metric, value in metrics.items():
            limit = self._limits.get(metric)
            if limit is None or limit <= 0.0:
                continue
            ratio = value / limit
            if ratio > 1.0:
                Logger.error(
                    _TAG,
                    f"budget EXCEEDED {metric}={value:.1f}/{limit:.1f} "
                    f"({ratio*100:.0f}%) — apply fallback tier",
                )
            elif ratio > 0.9:
                Logger.warn(
                    _TAG,
                    f"budget WARNING {metric}={value:.1f}/{limit:.1f} "
                    f"({ratio*100:.0f}%)",
                )

    def get_recent(self, n_frames: int = 60) -> List[FrameSnapshot]:
        """Return up to *n_frames* most-recent snapshots (oldest first)."""
        if not self._enabled:
            return []
        buf = list(self._ring)
        return buf[-n_frames:]

    def clear(self) -> None:
        """Discard all buffered snapshots."""
        self._ring.clear()

    def frame_count(self) -> int:
        """Number of snapshots currently held in the ring buffer."""
        return len(self._ring)
