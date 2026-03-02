"""BudgetManager — Stage 42 §4.  Per-subsystem CPU / memory budgets.

Tracks usage for each simulation subsystem and enforces quality-LOD
fallback when a budget is exceeded.

Subsystem budgets
-----------------
motor   IK / constraint solver iterations
deform  Active deformation chunks + GPU uploads/s
audio   Active resonators + modal modes + impulses/s
social  Active grasps + constraint-solve iterations
macro   Active macro phenomena + mega patch segments
net     Bytes/s per client + stamp batches/s

Budget enforcement
------------------
``BudgetManager.record(subsystem, metric, value)`` accumulates usage and
returns the current :class:`FallbackTier` for that subsystem.

Fallback tiers (0 = full quality → 3 = minimal / off):
  Tier 0 FULL      — full quality
  Tier 1 REDUCED   — fewer iterations / modes
  Tier 2 PROXY     — coarse proxy / material-only
  Tier 3 DISABLED  — disabled / drop source

Public API
----------
BudgetManager(config_dict)
  .record(subsystem, metric, value) → FallbackTier
  .tier(subsystem)                  → FallbackTier
  .reset_frame()                    → None
  .usage_summary()                  → dict
  .limits                           → dict

FallbackTier  — IntEnum 0..3
"""
from __future__ import annotations

from enum import IntEnum
from typing import Dict, Optional

from src.core.Logger import Logger

_TAG = "BudgetManager"


class FallbackTier(IntEnum):
    """Quality tier: 0 = full, 3 = minimal/off."""
    FULL     = 0
    REDUCED  = 1
    PROXY    = 2
    DISABLED = 3


# ---------------------------------------------------------------------------
# Default budget limits
# ---------------------------------------------------------------------------

_DEFAULT_LIMITS: Dict[str, Dict[str, float]] = {
    "motor": {
        "ik_iters":              64.0,
        "constraint_iters":      32.0,
        "ragdoll_bodies":        8.0,
    },
    "deform": {
        "active_chunks":         128.0,
        "stamps_per_second":     200.0,
        "gpu_uploads_per_frame": 8.0,
    },
    "audio": {
        "active_resonators":     24.0,
        "modes_per_resonator":   16.0,
        "impulses_per_second":   120.0,
    },
    "social": {
        "active_grasps":         16.0,
        "grasp_iters":           32.0,
    },
    "macro": {
        "active_phenomena":      4.0,
        "mega_segments":         512.0,
    },
    "net": {
        "bytes_per_second":      65536.0,
        "stamp_batches_per_sec": 20.0,
    },
}

# Usage fraction thresholds that trigger each tier
# >= 100% → DISABLED, >= 75% → PROXY, >= 50% → REDUCED, < 50% → FULL
_TIER_THRESHOLDS = (1.0, 0.75, 0.5)


class _SubsystemBudget:
    """Per-subsystem usage accumulator for one frame."""

    __slots__ = ("name", "limits", "_frame_usage", "_tier")

    def __init__(self, name: str, limits: Dict[str, float]) -> None:
        self.name = name
        self.limits: Dict[str, float] = limits
        self._frame_usage: Dict[str, float] = {k: 0.0 for k in limits}
        self._tier: FallbackTier = FallbackTier.FULL

    def record(self, metric: str, value: float) -> FallbackTier:
        if metric not in self.limits:
            return self._tier
        self._frame_usage[metric] += value
        limit = self.limits[metric]
        if limit <= 0.0:
            return self._tier
        ratio = self._frame_usage[metric] / limit
        if ratio >= _TIER_THRESHOLDS[0]:
            self._tier = FallbackTier.DISABLED
        elif ratio >= _TIER_THRESHOLDS[1]:
            self._tier = max(self._tier, FallbackTier.PROXY)
        elif ratio >= _TIER_THRESHOLDS[2]:
            self._tier = max(self._tier, FallbackTier.REDUCED)
        return self._tier

    def reset(self) -> None:
        for k in self._frame_usage:
            self._frame_usage[k] = 0.0
        self._tier = FallbackTier.FULL

    def usage(self) -> Dict[str, float]:
        return dict(self._frame_usage)

    def tier(self) -> FallbackTier:
        return self._tier


class BudgetManager:
    """Per-subsystem budget enforcer with LOD fallback.

    Parameters
    ----------
    config:
        Dict with optional ``budget`` sub-dict.  Set
        ``budget.fallback_enable`` to ``false`` to disable tier changes.
    """

    def __init__(self, config: Optional[Dict] = None) -> None:
        cfg = config or {}
        budget_cfg = cfg.get("budget", {})
        self._fallback_enabled: bool = bool(
            budget_cfg.get("fallback_enable", True)
        )

        limits: Dict[str, Dict[str, float]] = {}
        for subs, defaults in _DEFAULT_LIMITS.items():
            merged = dict(defaults)
            overrides = budget_cfg.get(subs, {})
            for k, v in overrides.items():
                if k in merged:
                    merged[k] = float(v)
            limits[subs] = merged

        self._subs: Dict[str, _SubsystemBudget] = {
            name: _SubsystemBudget(name, lim)
            for name, lim in limits.items()
        }

    # ------------------------------------------------------------------
    # Per-frame API
    # ------------------------------------------------------------------

    def record(self, subsystem: str, metric: str, value: float) -> FallbackTier:
        """Accumulate *value* usage for *metric* in *subsystem*.

        Returns the current :class:`FallbackTier` after accumulation.
        Logs warnings at 90 % and errors at 100 % of the limit.
        """
        if subsystem not in self._subs:
            return FallbackTier.FULL
        sub = self._subs[subsystem]
        tier = sub.record(metric, value)

        if self._fallback_enabled:
            limit = sub.limits.get(metric, 0.0)
            if limit > 0.0:
                usage = sub.usage().get(metric, 0.0)
                ratio = usage / limit
                if ratio >= 1.0:
                    Logger.warn(
                        _TAG,
                        f"{subsystem}.{metric} budget EXCEEDED "
                        f"({usage:.1f}/{limit:.1f}) → tier={tier.name}",
                    )
                elif ratio >= 0.9:
                    Logger.warn(
                        _TAG,
                        f"{subsystem}.{metric} budget at {ratio*100:.0f}% "
                        f"({usage:.1f}/{limit:.1f})",
                    )

        return tier

    def tier(self, subsystem: str) -> FallbackTier:
        """Return the current fallback tier for *subsystem*."""
        if subsystem not in self._subs:
            return FallbackTier.FULL
        return self._subs[subsystem].tier()

    def reset_frame(self) -> None:
        """Reset per-frame accumulators and tiers. Call once per frame."""
        for sub in self._subs.values():
            sub.reset()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def usage_summary(self) -> Dict[str, Dict[str, float]]:
        """Return a snapshot of current-frame usage for all subsystems."""
        return {name: sub.usage() for name, sub in self._subs.items()}

    @property
    def limits(self) -> Dict[str, Dict[str, float]]:
        """Configured limits for all subsystems."""
        return {name: dict(sub.limits) for name, sub in self._subs.items()}

    @property
    def fallback_enabled(self) -> bool:
        """Whether fallback tier escalation is active."""
        return self._fallback_enabled
