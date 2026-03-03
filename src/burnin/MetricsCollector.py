"""MetricsCollector — Stage 56 burn-in metric aggregation.

Collects three categories of metrics during a headless burn-in run:

Global (planet-scale)
    totalDustMass, meanDustThickness, varianceDust, meanIceFilm,
    iceCoverage, meanCrustHardness, crustVariance, entropy,
    energyReservoirs, instabilityEventsPerDay

Local (per-bot / per-player)
    slipRate, fallEvents, meanWindLoad, recoveryTime,
    avgStepLength, avgStepWidth

Performance / budgets
    qpItersAvg, qpItersMax, raycastsPerSec, activeChunks,
    memoryGrowthKb, netBytesPerSec

Usage
-----
    collector = MetricsCollector()
    collector.begin_day(day=0)

    # ... per-tick updates ...
    collector.record_global("totalDustMass", value)
    collector.record_local("slipRate", value)
    collector.record_perf("activeChunks", value)

    collector.end_day(day=0)

    snapshot = collector.day_snapshot(day=0)
    report   = collector.summary_report()
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.core.Logger import Logger

_TAG = "MetricsCollector"

# Metric name sets for each category
_GLOBAL_METRICS = {
    "totalDustMass",
    "meanDustThickness",
    "varianceDust",
    "meanIceFilm",
    "iceCoverage",
    "meanCrustHardness",
    "crustVariance",
    "entropy",
    "instabilityEventsPerDay",
}

_LOCAL_METRICS = {
    "slipRate",
    "fallEvents",
    "meanWindLoad",
    "recoveryTime",
    "avgStepLength",
    "avgStepWidth",
}

_PERF_METRICS = {
    "qpItersAvg",
    "qpItersMax",
    "raycastsPerSec",
    "activeChunks",
    "memoryGrowthKb",
    "netBytesPerSec",
}


# ---------------------------------------------------------------------------
# Day snapshot
# ---------------------------------------------------------------------------

@dataclass
class DaySnapshot:
    """Aggregated metric snapshot for one simulated day."""
    day: int = 0
    planet_time: float = 0.0
    global_metrics: Dict[str, float]  = field(default_factory=dict)
    local_metrics:  Dict[str, float]  = field(default_factory=dict)
    perf_metrics:   Dict[str, float]  = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "day":            self.day,
            "planet_time":    self.planet_time,
            "global_metrics": dict(self.global_metrics),
            "local_metrics":  dict(self.local_metrics),
            "perf_metrics":   dict(self.perf_metrics),
        }


# ---------------------------------------------------------------------------
# MetricsCollector
# ---------------------------------------------------------------------------

class MetricsCollector:
    """Accumulates per-tick values and emits per-day averaged snapshots."""

    def __init__(self) -> None:
        self._snapshots: List[DaySnapshot] = []
        self._current_day: Optional[int] = None
        self._accum: Dict[str, List[float]] = {}

    # ------------------------------------------------------------------
    # Day lifecycle
    # ------------------------------------------------------------------

    def begin_day(self, day: int, planet_time: float = 0.0) -> None:
        """Start accumulating metrics for *day*."""
        self._current_day = day
        self._planet_time = planet_time
        self._accum = {}

    def end_day(self, day: int) -> DaySnapshot:
        """Finalise the current day and store a snapshot."""
        if self._current_day != day:
            Logger.warn(_TAG, f"end_day({day}) called but current day is {self._current_day}")

        def _mean(vals: List[float]) -> float:
            return sum(vals) / len(vals) if vals else 0.0

        snap = DaySnapshot(
            day=day,
            planet_time=getattr(self, "_planet_time", 0.0),
            global_metrics={k: _mean(v) for k, v in self._accum.items() if k in _GLOBAL_METRICS},
            local_metrics ={k: _mean(v) for k, v in self._accum.items() if k in _LOCAL_METRICS},
            perf_metrics  ={k: _mean(v) for k, v in self._accum.items() if k in _PERF_METRICS},
        )
        self._snapshots.append(snap)
        self._current_day = None
        return snap

    # ------------------------------------------------------------------
    # Per-tick recording
    # ------------------------------------------------------------------

    def record_global(self, name: str, value: float) -> None:
        """Record one sample of a global (planet) metric."""
        self._accum.setdefault(name, []).append(value)

    def record_local(self, name: str, value: float) -> None:
        """Record one sample of a local (player/bot) metric."""
        self._accum.setdefault(name, []).append(value)

    def record_perf(self, name: str, value: float) -> None:
        """Record one sample of a performance/budget metric."""
        self._accum.setdefault(name, []).append(value)

    def record(self, name: str, value: float) -> None:
        """Record under whichever category the metric belongs to."""
        if name in _GLOBAL_METRICS:
            self.record_global(name, value)
        elif name in _LOCAL_METRICS:
            self.record_local(name, value)
        elif name in _PERF_METRICS:
            self.record_perf(name, value)
        else:
            # Unknown metric: store in perf bucket by default
            self._accum.setdefault(name, []).append(value)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def day_snapshot(self, day: int) -> Optional[DaySnapshot]:
        """Return the snapshot for *day*, or None."""
        for snap in self._snapshots:
            if snap.day == day:
                return snap
        return None

    def all_snapshots(self) -> List[DaySnapshot]:
        """Return all stored snapshots."""
        return list(self._snapshots)

    def summary_report(self) -> Dict[str, Any]:
        """Return a compact summary over all days.

        For each metric, reports the mean across days plus first/last values
        (useful for detecting monotonic drift).
        """
        if not self._snapshots:
            return {"days": 0}

        def _across_days(key: str) -> List[float]:
            vals = []
            for snap in self._snapshots:
                for cat in (snap.global_metrics, snap.local_metrics, snap.perf_metrics):
                    if key in cat:
                        vals.append(cat[key])
            return vals

        all_keys = set()
        for snap in self._snapshots:
            all_keys.update(snap.global_metrics)
            all_keys.update(snap.local_metrics)
            all_keys.update(snap.perf_metrics)

        report: Dict[str, Any] = {"days": len(self._snapshots)}
        for key in sorted(all_keys):
            vals = _across_days(key)
            if not vals:
                continue
            report[key] = {
                "mean":  sum(vals) / len(vals),
                "first": vals[0],
                "last":  vals[-1],
                "min":   min(vals),
                "max":   max(vals),
            }
        return report
