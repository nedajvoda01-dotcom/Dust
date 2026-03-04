"""WorldHealthScorer — Stage 61 aggregate world health score.

Computes a scalar ``worldHealthScore`` in [0, 1] from invariant metrics:

    worldHealthScore = f(
        entropy_bounds,
        dust_conservation_error,
        instability_rate,
        energy_balance_error,
        snapshot_fail_rate,
        tick_lag,
    )

A score of 1.0 means all invariants are within nominal bounds.
A score below the configured ``alert_threshold`` triggers an alert.

The score is also used by :class:`~src.ops.AutoBudgetController` to
decide when to engage automatic LOD/budget reduction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class HealthInputs:
    """Inputs to the world health scorer."""
    entropy:               float = 0.5   # current entropy [0,1]
    entropy_lower:         float = 0.2   # acceptable lower bound
    entropy_upper:         float = 0.8   # acceptable upper bound
    dust_conservation_err: float = 0.0   # |actual - expected| / expected
    instability_rate:      float = 0.0   # events/hour
    instability_max:       float = 50.0  # threshold events/hour
    energy_balance_err:    float = 0.0   # |in - out| / max
    snapshot_fail_rate:    float = 0.0   # [0,1]
    tick_lag_ticks:        int   = 0     # current lag
    tick_lag_max:          int   = 10    # max acceptable lag


@dataclass
class HealthScore:
    """Result of one health scoring pass."""
    score:       float               = 1.0
    components:  Dict[str, float]    = field(default_factory=dict)
    alerts:      list                = field(default_factory=list)

    @property
    def is_healthy(self) -> bool:
        return self.score >= 0.6


class WorldHealthScorer:
    """Computes a World Health Score from invariant metrics.

    Parameters
    ----------
    alert_threshold:
        Score below this value triggers an alert.
    """

    def __init__(self, alert_threshold: float = 0.6) -> None:
        self._alert_threshold = alert_threshold

    def score(self, inputs: HealthInputs) -> HealthScore:
        """Compute and return the health score."""
        components: Dict[str, float] = {}
        alerts: list = []

        # 1. Entropy within bounds
        if inputs.entropy < inputs.entropy_lower:
            delta = inputs.entropy_lower - inputs.entropy
            c_entropy = max(0.0, 1.0 - delta / max(inputs.entropy_lower, 1e-9))
        elif inputs.entropy > inputs.entropy_upper:
            delta = inputs.entropy - inputs.entropy_upper
            c_entropy = max(0.0, 1.0 - delta / max(1.0 - inputs.entropy_upper, 1e-9))
        else:
            c_entropy = 1.0
        components["entropy_bounds"] = c_entropy
        if c_entropy < 1.0:
            alerts.append("entropy_out_of_bounds")

        # 2. Dust conservation
        c_dust = max(0.0, 1.0 - inputs.dust_conservation_err)
        components["dust_conservation_error"] = c_dust
        if inputs.dust_conservation_err > 0.05:
            alerts.append("dust_not_conserved")

        # 3. Instability rate
        if inputs.instability_max > 0:
            c_instab = max(0.0, 1.0 - inputs.instability_rate / inputs.instability_max)
        else:
            c_instab = 1.0
        components["instability_rate"] = c_instab
        if inputs.instability_rate > inputs.instability_max:
            alerts.append("instability_rate_high")

        # 4. Energy balance
        c_energy = max(0.0, 1.0 - inputs.energy_balance_err)
        components["energy_balance_error"] = c_energy
        if inputs.energy_balance_err > 0.1:
            alerts.append("energy_imbalance")

        # 5. Snapshot fail rate
        c_snap = max(0.0, 1.0 - inputs.snapshot_fail_rate)
        components["snapshot_fail_rate"] = c_snap
        if inputs.snapshot_fail_rate > 0.0:
            alerts.append("snapshots_failing")

        # 6. Tick lag
        if inputs.tick_lag_max > 0:
            c_lag = max(0.0, 1.0 - inputs.tick_lag_ticks / inputs.tick_lag_max)
        else:
            c_lag = 1.0
        components["tick_lag"] = c_lag
        if inputs.tick_lag_ticks >= inputs.tick_lag_max:
            alerts.append("tick_lag_exceeded")

        # Weighted geometric mean (equal weights)
        vals = list(components.values())
        if not vals:
            total = 1.0
        else:
            product = 1.0
            for v in vals:
                product *= max(v, 1e-9)
            total = product ** (1.0 / len(vals))

        result = HealthScore(score=round(total, 6), components=components, alerts=alerts)

        if result.score < self._alert_threshold:
            if "health_score_low" not in result.alerts:
                result.alerts.append("health_score_low")

        return result
