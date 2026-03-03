"""RegionAutoscaler — Stage 60 dynamic region load balancing (optional).

Monitors per-region player counts and simulation CPU budgets, then emits
``ScalingDecision`` objects that the operator layer can act on (e.g. spin up
a new RS node, migrate a region to a less-loaded node, or merge two lightly
loaded regions).

This class does *not* perform I/O or process management itself; it is a pure
decision engine so it can be unit-tested synchronously.

Degradation policy
------------------
When a region's player count exceeds ``soft_cap``:

* ``chunk_update_interval`` is increased (lower tick frequency for material /
  microclimate updates).
* ``acoustic_raycasts`` budget is reduced.
* ``remote_state_hz`` is reduced for players in the region.

When count exceeds ``hard_cap``:

* ``status`` is set to ``OVERLOADED`` and the router should reject new joins.

Public API
----------
RegionAutoscaler(config=None)
  .update(region_id, player_count, cpu_budget_ms) → ScalingDecision
  .decision_for(region_id) → ScalingDecision | None
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Status constants
# ---------------------------------------------------------------------------

STATUS_OK         = "OK"
STATUS_DEGRADED   = "DEGRADED"
STATUS_OVERLOADED = "OVERLOADED"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_SOFT_CAP  = 100
_DEFAULT_HARD_CAP  = 200
_DEFAULT_CHUNK_INTERVAL_NORMAL  = 1.0   # seconds between chunk updates
_DEFAULT_CHUNK_INTERVAL_DEGRADED = 3.0
_DEFAULT_ACOUSTIC_RAYCASTS_NORMAL   = 32
_DEFAULT_ACOUSTIC_RAYCASTS_DEGRADED = 8
_DEFAULT_REMOTE_STATE_HZ_NORMAL   = 20.0
_DEFAULT_REMOTE_STATE_HZ_DEGRADED = 10.0


# ---------------------------------------------------------------------------
# ScalingDecision
# ---------------------------------------------------------------------------

@dataclass
class ScalingDecision:
    """Recommended operating parameters for one region."""
    region_id:              int
    status:                 str    # OK / DEGRADED / OVERLOADED
    chunk_update_interval:  float  # seconds
    acoustic_raycasts:      int
    remote_state_hz:        float
    player_count:           int


# ---------------------------------------------------------------------------
# RegionAutoscaler
# ---------------------------------------------------------------------------

class RegionAutoscaler:
    """Pure-logic autoscaler / degradation controller."""

    def __init__(self, config=None) -> None:
        self._cfg = config
        self._soft_cap = self._cfg_int(
            ("scale", "max_players_per_region_soft"), _DEFAULT_SOFT_CAP
        )
        self._hard_cap = self._cfg_int(
            ("scale", "max_players_per_region_hard"), _DEFAULT_HARD_CAP
        )
        self._decisions: Dict[int, ScalingDecision] = {}

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------

    def update(
        self,
        region_id:     int,
        player_count:  int,
        cpu_budget_ms: float = 0.0,
    ) -> ScalingDecision:
        """Compute and store a new ``ScalingDecision`` for *region_id*.

        Parameters
        ----------
        region_id
            The region being evaluated.
        player_count
            Current number of players in the region.
        cpu_budget_ms
            Optional hint: recent tick duration in milliseconds (unused in
            the current policy but reserved for future use).
        """
        if player_count >= self._hard_cap:
            status = STATUS_OVERLOADED
        elif player_count >= self._soft_cap:
            status = STATUS_DEGRADED
        else:
            status = STATUS_OK

        if status == STATUS_OK:
            decision = ScalingDecision(
                region_id=region_id,
                status=status,
                chunk_update_interval=_DEFAULT_CHUNK_INTERVAL_NORMAL,
                acoustic_raycasts=_DEFAULT_ACOUSTIC_RAYCASTS_NORMAL,
                remote_state_hz=_DEFAULT_REMOTE_STATE_HZ_NORMAL,
                player_count=player_count,
            )
        else:
            decision = ScalingDecision(
                region_id=region_id,
                status=status,
                chunk_update_interval=_DEFAULT_CHUNK_INTERVAL_DEGRADED,
                acoustic_raycasts=_DEFAULT_ACOUSTIC_RAYCASTS_DEGRADED,
                remote_state_hz=_DEFAULT_REMOTE_STATE_HZ_DEGRADED,
                player_count=player_count,
            )

        self._decisions[region_id] = decision
        return decision

    def decision_for(self, region_id: int) -> Optional[ScalingDecision]:
        """Return the last computed decision for *region_id*, or ``None``."""
        return self._decisions.get(region_id)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _cfg_int(self, path: tuple, default: int) -> int:
        if self._cfg is None:
            return default
        return int(self._cfg.get(*path, default=default))
