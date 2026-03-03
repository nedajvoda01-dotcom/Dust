"""MetricsRegistry — Stage 61 observability metrics collection.

Collects and exports structured metrics in four categories:

Performance (per-server)
    tick_ms_avg, tick_ms_max, tick_ms_p99, sim_lag_ticks,
    heap_mb, gc_pause_ms, net_in_bps, net_out_bps,
    active_clients, active_chunks, snapshot_write_ms,
    snapshot_fail_count

World health (global invariants)
    dust_total_mass, ice_coverage_ratio, crust_hardness_mean,
    crust_hardness_var, entropy, instability_events_per_hour,
    cascade_avg_radius, energy_reservoirs

Player aggregates
    avg_slip_rate, fall_events_per_hour, avg_wind_load,
    avg_fatigue_level, injury_incidence_rate

Network quality
    rtt_avg, rtt_p95, jitter_avg, state_corrections_avg,
    resync_count, handoff_count, drift_detected_count
"""
from __future__ import annotations

import math
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Dimension keys (required on every exported metric)
# ---------------------------------------------------------------------------

REQUIRED_DIMENSIONS = ("worldId", "regionId", "serverTick")


# ---------------------------------------------------------------------------
# Metric categories
# ---------------------------------------------------------------------------

PERF_METRICS = {
    "tick_ms_avg", "tick_ms_max", "tick_ms_p99",
    "sim_lag_ticks", "heap_mb", "gc_pause_ms",
    "net_in_bps", "net_out_bps", "active_clients",
    "active_chunks", "snapshot_write_ms", "snapshot_fail_count",
}

WORLD_METRICS = {
    "dust_total_mass", "ice_coverage_ratio",
    "crust_hardness_mean", "crust_hardness_var",
    "entropy", "instability_events_per_hour",
    "cascade_avg_radius",
}

PLAYER_METRICS = {
    "avg_slip_rate", "fall_events_per_hour",
    "avg_wind_load", "avg_fatigue_level",
    "injury_incidence_rate",
}

NET_METRICS = {
    "rtt_avg", "rtt_p95", "jitter_avg",
    "state_corrections_avg", "resync_count",
    "handoff_count", "drift_detected_count",
}

ALL_METRICS = PERF_METRICS | WORLD_METRICS | PLAYER_METRICS | NET_METRICS


class MetricsRegistry:
    """Collects, aggregates, and exports metrics with required dimensions.

    Parameters
    ----------
    world_id:
        Stable world identifier included in every exported record.
    region_id:
        Region/shard identifier (empty string for single-server deployments).
    """

    def __init__(self, world_id: str = "", region_id: str = "") -> None:
        self._world_id   = world_id
        self._region_id  = region_id
        self._server_tick: int = 0
        self._values: Dict[str, float] = {}
        # For p99 / percentile metrics we keep a ring buffer of recent samples
        self._tick_ms_samples: List[float] = []

    # ------------------------------------------------------------------
    # Tick counter
    # ------------------------------------------------------------------

    def advance_tick(self, tick: Optional[int] = None) -> None:
        """Increment (or set) the server tick counter."""
        if tick is None:
            self._server_tick += 1
        else:
            self._server_tick = tick

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(self, name: str, value: float) -> None:
        """Record a single metric value.  Unknown names are accepted."""
        self._values[name] = value
        if name == "tick_ms_avg":
            self._tick_ms_samples.append(value)
            if len(self._tick_ms_samples) > 1000:
                self._tick_ms_samples = self._tick_ms_samples[-1000:]

    def record_tick_ms(self, ms: float) -> None:
        """Convenience: record one tick-time sample and recompute p99/max."""
        self._tick_ms_samples.append(ms)
        if len(self._tick_ms_samples) > 1000:
            self._tick_ms_samples = self._tick_ms_samples[-1000:]
        samples = self._tick_ms_samples
        self._values["tick_ms_avg"] = sum(samples) / len(samples)
        self._values["tick_ms_max"] = max(samples)
        self._values["tick_ms_p99"] = _percentile(sorted(samples), 99)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export(self) -> Dict[str, Any]:
        """Return a snapshot dict with all recorded values and dimensions."""
        return {
            "worldId":    self._world_id,
            "regionId":   self._region_id,
            "serverTick": self._server_tick,
            "metrics":    dict(self._values),
        }

    def get(self, name: str, default: float = 0.0) -> float:
        """Return the latest value for a metric."""
        return self._values.get(name, default)

    # ------------------------------------------------------------------
    # Dimension helpers
    # ------------------------------------------------------------------

    def dimensions(self) -> Dict[str, Any]:
        return {
            "worldId":    self._world_id,
            "regionId":   self._region_id,
            "serverTick": self._server_tick,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _percentile(sorted_data: List[float], pct: float) -> float:
    if not sorted_data:
        return 0.0
    k = (len(sorted_data) - 1) * pct / 100.0
    lo = int(k)
    hi = lo + 1
    if hi >= len(sorted_data):
        return sorted_data[-1]
    return sorted_data[lo] + (k - lo) * (sorted_data[hi] - sorted_data[lo])
