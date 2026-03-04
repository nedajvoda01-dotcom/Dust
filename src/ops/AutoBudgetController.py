"""AutoBudgetController — Stage 61 automatic LOD / budget adjustment.

Watches performance and health metrics and reduces LOD/quality parameters
when thresholds are exceeded.  **Never touches world physics.**

Controllable parameters (subset of tuning allowlist):
* ``net_update_hz``         — remote state update rate
* ``max_raycasts_per_sec``  — acoustic raycasts budget
* ``chunk_update_interval`` — how often inactive chunks are ticked

Decision logic (per tick)
-------------------------
1. If ``tick_ms_p99 > tick_ms_threshold``:
   reduce ``net_update_hz`` and ``max_raycasts_per_sec``
2. If ``net_out_bps > net_out_threshold``:
   reduce ``net_update_hz``
3. If ``world_health_score < health_threshold``:
   increase ``chunk_update_interval``

Adjustments are proposed through :class:`~src.ops.TuningManager.TuningManager`
so they go through the same epoch / tick-boundary machinery.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from src.obs.MetricsRegistry import MetricsRegistry
from src.obs.WorldHealthScorer import WorldHealthScorer, HealthInputs
from src.ops.TuningManager import TuningManager


# ---------------------------------------------------------------------------
# Default thresholds
# ---------------------------------------------------------------------------

_DEFAULT_TICK_MS_THRESHOLD    = 16.0    # ms  (p99 tick time)
_DEFAULT_NET_OUT_THRESHOLD    = 10e6    # bps (10 Mbps)
_DEFAULT_HEALTH_THRESHOLD     = 0.7     # world health score


# ---------------------------------------------------------------------------
# LOD parameter floors / ceilings  (must stay within TuningValidator ranges)
# ---------------------------------------------------------------------------

_NET_HZ_FLOOR          = 5
_NET_HZ_CEILING        = 30
_RAYCASTS_FLOOR        = 100
_RAYCASTS_CEILING      = 20_000
_CHUNK_INTERVAL_CEILING = 60


class AutoBudgetController:
    """Automatically reduces LOD/budget parameters under load.

    Parameters
    ----------
    tuning_manager:
        The live :class:`TuningManager` used to propose adjustments.
    tick_ms_threshold:
        p99 tick time (ms) above which quality is reduced.
    net_out_threshold:
        Outbound bandwidth (bps) above which net_update_hz is reduced.
    health_threshold:
        World health score below which chunk update intervals are relaxed.
    """

    def __init__(
        self,
        tuning_manager:      TuningManager,
        tick_ms_threshold:   float = _DEFAULT_TICK_MS_THRESHOLD,
        net_out_threshold:   float = _DEFAULT_NET_OUT_THRESHOLD,
        health_threshold:    float = _DEFAULT_HEALTH_THRESHOLD,
    ) -> None:
        self._tm                = tuning_manager
        self._tick_ms_threshold = tick_ms_threshold
        self._net_out_threshold = net_out_threshold
        self._health_threshold  = health_threshold

        # Cached current budget state (starts at ceilings)
        self._net_hz:          int   = _NET_HZ_CEILING
        self._raycasts:        int   = _RAYCASTS_CEILING
        self._chunk_interval:  int   = 1

    # ------------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------------

    def update(
        self,
        metrics:       MetricsRegistry,
        health_score:  float,
        server_tick:   int,
    ) -> Dict[str, Any]:
        """Evaluate metrics and propose budget adjustments if needed.

        Returns the delta dict that was proposed (empty dict = no change).
        """
        delta: Dict[str, Any] = {}

        tick_p99   = metrics.get("tick_ms_p99",   0.0)
        net_out    = metrics.get("net_out_bps",    0.0)

        # Rule 1: tick time too high → reduce net_hz and raycasts
        if tick_p99 > self._tick_ms_threshold:
            new_hz = max(_NET_HZ_FLOOR, int(self._net_hz * 0.8))
            if new_hz != self._net_hz:
                delta["net_update_hz"] = new_hz
                self._net_hz = new_hz

            new_ray = max(_RAYCASTS_FLOOR, int(self._raycasts * 0.8))
            if new_ray != self._raycasts:
                delta["max_raycasts_per_sec"] = new_ray
                self._raycasts = new_ray

        # Rule 2: net bandwidth high → reduce net_hz
        if net_out > self._net_out_threshold:
            new_hz = max(_NET_HZ_FLOOR, int(self._net_hz * 0.8))
            if new_hz != self._net_hz:
                delta["net_update_hz"] = new_hz
                self._net_hz = new_hz

        # Rule 3: health score low → relax chunk tick intervals
        if health_score < self._health_threshold:
            new_interval = min(_CHUNK_INTERVAL_CEILING, self._chunk_interval + 2)
            if new_interval != self._chunk_interval:
                delta["chunk_update_interval"] = new_interval
                self._chunk_interval = new_interval

        # Propose through TuningManager (epoch machinery)
        if delta:
            accepted, errors = self._tm.propose(delta)
            if not accepted:
                delta = {}  # revert local state on rejection

        return delta

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def current_budgets(self) -> Dict[str, Any]:
        return {
            "net_update_hz":          self._net_hz,
            "max_raycasts_per_sec":   self._raycasts,
            "chunk_update_interval":  self._chunk_interval,
        }

    def only_changes_lod_params(self, delta: Dict[str, Any]) -> bool:
        """Return True iff every key in delta is a LOD/budget parameter."""
        lod_params = {"net_update_hz", "max_raycasts_per_sec", "chunk_update_interval"}
        return all(k in lod_params for k in delta)
