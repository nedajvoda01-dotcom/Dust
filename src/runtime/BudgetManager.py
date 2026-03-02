"""BudgetManager — Stage 42 per-subsystem CPU/memory/network budgets.

Tracks usage counters for each subsystem and triggers :class:`FallbackLadder`
degradation when a subsystem exceeds its budget.  Recovery happens automatically
when usage drops below the budget on subsequent frames.

Budget categories
-----------------
* **motor**    — IK iterations, constraint-solve iterations, ragdoll bodies
* **deform**   — active deform chunks, stamp applies/s, GPU uploads/frame
* **audio**    — active resonators, modes/resonator, impulses/s
* **grasp**    — active world grasps, grasp solve iterations
* **macro**    — active macro phenomena, mega patch segments
* **net**      — bytes/second/client, stamp batches/second

Usage
-----
mgr = BudgetManager(config)

# Each frame, reset counters then report actual usage:
mgr.begin_frame()
mgr.report("audio", "active_resonators", current_active)
mgr.report("motor", "ik_iters", total_ik_iters_this_frame)

# BudgetManager auto-engages fallback when > 100 % budget:
tier = mgr.fallback_tier("audio")   # AudioTier int
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from src.core.Config import Config
from src.core.Logger import Logger
from src.runtime.FallbackLadders import FallbackLadder, FallbackLadders

_TAG = "BudgetMgr"

# Fraction of budget that triggers a warning
_WARN_THRESHOLD = 0.90


@dataclass
class BudgetLimit:
    """A single budget limit with optional ladder reference."""
    max_value: float
    ladder_name: Optional[str] = None   # which FallbackLadder to degrade


class BudgetManager:
    """Tracks per-subsystem usage and enforces fallback tiers."""

    # Default limits — overridden by CONFIG_DEFAULTS.json
    _DEFAULTS: Dict[str, Dict[str, BudgetLimit]] = {
        "motor": {
            "ik_iters":              BudgetLimit(120.0,  "ik"),
            "constraint_iters":      BudgetLimit(64.0,   "ik"),
            "active_ragdoll_bodies": BudgetLimit(8.0,    None),
        },
        "deform": {
            "active_chunks":         BudgetLimit(128.0,  "deform"),
            "stamp_apply_per_sec":   BudgetLimit(60.0,   "deform"),
            "gpu_uploads_per_frame": BudgetLimit(8.0,    "deform"),
        },
        "audio": {
            "active_resonators":     BudgetLimit(32.0,   "audio"),
            "modes_per_resonator":   BudgetLimit(16.0,   "audio"),
            "impulses_per_sec":      BudgetLimit(120.0,  "audio"),
        },
        "grasp": {
            "active_grasps_world":   BudgetLimit(8.0,    None),
            "grasp_solve_iters":     BudgetLimit(16.0,   None),
        },
        "macro": {
            "active_phenomena":      BudgetLimit(4.0,    None),
            "mega_patch_segments":   BudgetLimit(256.0,  None),
        },
        "net": {
            "bytes_per_sec":         BudgetLimit(65536.0, None),
            "stamp_batches_per_sec": BudgetLimit(30.0,    None),
        },
    }

    def __init__(self, config: Optional[Config] = None) -> None:
        self._limits: Dict[str, Dict[str, BudgetLimit]] = {}
        self._usage:  Dict[str, Dict[str, float]] = {}
        self._fallback_enable: bool = True

        # Copy defaults
        for cat, limits in self._DEFAULTS.items():
            self._limits[cat] = dict(limits)
            self._usage[cat]  = {k: 0.0 for k in limits}

        # Override from config
        if config is not None:
            self._apply_config(config)

    # ------------------------------------------------------------------
    # Frame lifecycle
    # ------------------------------------------------------------------

    def begin_frame(self) -> None:
        """Reset all per-frame usage counters.  Call once at frame start."""
        for cat in self._usage:
            for key in self._usage[cat]:
                self._usage[cat][key] = 0.0

    def report(self, category: str, metric: str, value: float) -> None:
        """Report the current-frame usage of a metric.

        Automatically triggers fallback ladder degradation when the budget
        is exceeded, and attempts recovery when under budget.

        Parameters
        ----------
        category: e.g. ``"audio"``
        metric:   e.g. ``"active_resonators"``
        value:    Actual usage this frame.
        """
        if category not in self._usage:
            Logger.warn(_TAG, f"Unknown budget category '{category}'")
            return
        if metric not in self._usage[category]:
            Logger.warn(_TAG, f"Unknown metric '{category}.{metric}'")
            return

        self._usage[category][metric] = value
        limit = self._limits[category][metric]

        fraction = value / limit.max_value if limit.max_value > 0 else 0.0

        if fraction >= _WARN_THRESHOLD and fraction < 1.0:
            Logger.warn(
                _TAG,
                f"{category}.{metric} at {fraction*100:.0f}% budget "
                f"({value:.1f}/{limit.max_value:.1f})",
            )

        if limit.ladder_name and self._fallback_enable:
            ladder = FallbackLadders.get(limit.ladder_name)
            if ladder is None:
                return
            if fraction >= 1.0:
                ladder.degrade()
            elif fraction < _WARN_THRESHOLD and ladder.is_degraded:
                # Recovery threshold is deliberately the same as the warn
                # threshold (90 %).  This creates intentional hysteresis:
                # once degraded, the subsystem must drop to < 90 % to
                # recover, preventing oscillation at the 100 % boundary.
                ladder.recover()

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def fallback_tier(self, ladder_name: str) -> int:
        """Return the current tier index for *ladder_name* (0 = best)."""
        ladder = FallbackLadders.get(ladder_name)
        return ladder.current if ladder is not None else 0

    def usage(self, category: str, metric: str) -> float:
        """Return the last reported value for a metric."""
        return self._usage.get(category, {}).get(metric, 0.0)

    def limit(self, category: str, metric: str) -> float:
        """Return the configured limit for a metric."""
        return self._limits.get(category, {}).get(metric, BudgetLimit(0.0)).max_value

    def all_usage(self) -> Dict[str, Dict[str, float]]:
        """Return a snapshot of all current usage values."""
        return {cat: dict(vals) for cat, vals in self._usage.items()}

    # ------------------------------------------------------------------
    # Config override
    # ------------------------------------------------------------------

    def _apply_config(self, config: Config) -> None:
        """Read budget limits from CONFIG_DEFAULTS.json ``budget`` section."""
        mapping = {
            ("motor", "ik_iters"):              ("budget", "motor", "max_ik_iters"),
            ("motor", "constraint_iters"):      ("budget", "motor", "max_constraint_iters"),
            ("audio", "active_resonators"):     ("budget", "audio", "max_resonators"),
            ("audio", "modes_per_resonator"):   ("budget", "audio", "max_modes_per_resonator"),
            ("audio", "impulses_per_sec"):      ("budget", "audio", "max_impulses_per_sec"),
            ("deform", "active_chunks"):        ("budget", "deform", "max_chunks"),
            ("deform", "stamp_apply_per_sec"):  ("budget", "deform", "max_stamp_apply_per_sec"),
            ("deform", "gpu_uploads_per_frame"):("budget", "deform", "max_gpu_uploads_per_frame"),
            ("grasp", "active_grasps_world"):   ("budget", "grasp", "max_active_grasps"),
            ("grasp", "grasp_solve_iters"):     ("budget", "grasp", "max_solve_iters"),
            ("macro", "active_phenomena"):      ("budget", "macro", "max_active_phenomena"),
            ("macro", "mega_patch_segments"):   ("budget", "macro", "max_mega_patch_segments"),
            ("net", "bytes_per_sec"):           ("budget", "net", "max_bps"),
            ("net", "stamp_batches_per_sec"):   ("budget", "net", "max_stamp_batches_per_sec"),
        }
        for (cat, metric), config_path in mapping.items():
            val = config.get(*config_path)
            if val is not None:
                self._limits[cat][metric].max_value = float(val)

        fb_enable = config.get("budget", "fallback_enable")
        if fb_enable is not None:
            self._fallback_enable = bool(fb_enable)
