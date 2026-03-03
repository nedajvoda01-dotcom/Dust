"""BurnInScenarios — Stage 56 named scenario parameter bundles.

Each scenario is a plain data class that fully specifies a headless burn-in
run.  :class:`~src.burnin.BurnInHarness.BurnInHarness` accepts a
:class:`BurnInScenario` and drives the simulation accordingly.

Available scenarios
-------------------
* :data:`SCENARIO_FAST`               — ``burnin_fast``   (no players)
* :data:`SCENARIO_WITH_PLAYERS`       — ``burnin_with_players``
* :data:`SCENARIO_STORM_CYCLE`        — ``burnin_storm_cycle``
* :data:`SCENARIO_INSTABILITY_CYCLE`  — ``burnin_instability_cycle``
* :data:`SCENARIO_ORBIT_CYCLE`        — ``burnin_orbit_cycle``

Usage
-----
    from src.burnin.BurnInScenarios import SCENARIO_FAST
    harness = BurnInHarness(SCENARIO_FAST)
    report  = harness.run()
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Scenario dataclass
# ---------------------------------------------------------------------------

@dataclass
class BurnInScenario:
    """All parameters that govern one burn-in run."""

    name: str = "unnamed"

    # Time
    days_to_simulate: int   = 30
    time_scale: float       = 100.0    # simulation speed multiplier
    tick_hz: float          = 60.0     # fixed sim tick rate

    # Snapshot / metric intervals
    snapshot_interval_hours: float  = 6.0    # simulated hours between snapshots
    metrics_interval_minutes: float = 10.0   # simulated minutes between metric samples

    # Bots
    bot_count: int           = 0          # 0 = no bots (fast mode)
    bot_seed_base: int       = 42         # seeds for bots: base, base+1, …

    # Environment stressors
    wind_load_override: float       = 0.0    # 0 = use natural wind
    dust_injection_rate: float      = 0.0    # extra dust per tick (0 = natural)
    instability_boost: float        = 0.0    # extra instability intensity [0..1]
    solar_boost: float              = 0.0    # extra solar injection [0..1]
    orbit_long_horizon: bool        = False  # enable very long orbit simulation

    # Assertions (invariants)
    assert_eps_dust_mass: float          = 0.05
    assert_max_instability_per_day: float = 50.0
    assert_max_entropy: float            = 0.99
    budget_fail_action: str              = "fallback"   # "fallback" | "fail"

    # RNG
    world_seed: int = 42


# ---------------------------------------------------------------------------
# Pre-defined scenario instances
# ---------------------------------------------------------------------------

#: Fastest headless run — no bots, baseline stress.
SCENARIO_FAST = BurnInScenario(
    name="burnin_fast",
    days_to_simulate=30,
    time_scale=200.0,
    bot_count=0,
    snapshot_interval_hours=6.0,
    metrics_interval_minutes=60.0,
    world_seed=42,
)

#: Moderate run with 2 bots walking around.
SCENARIO_WITH_PLAYERS = BurnInScenario(
    name="burnin_with_players",
    days_to_simulate=100,
    time_scale=100.0,
    bot_count=2,
    snapshot_interval_hours=6.0,
    metrics_interval_minutes=30.0,
    world_seed=43,
)

#: Maximum wind+dust stress cycle.
SCENARIO_STORM_CYCLE = BurnInScenario(
    name="burnin_storm_cycle",
    days_to_simulate=30,
    time_scale=100.0,
    bot_count=2,
    wind_load_override=0.9,
    dust_injection_rate=0.1,
    snapshot_interval_hours=3.0,
    metrics_interval_minutes=20.0,
    world_seed=44,
)

#: Frequent critical-instability events.
SCENARIO_INSTABILITY_CYCLE = BurnInScenario(
    name="burnin_instability_cycle",
    days_to_simulate=30,
    time_scale=100.0,
    bot_count=1,
    instability_boost=0.8,
    assert_max_instability_per_day=500.0,   # stress scenario — higher limit
    snapshot_interval_hours=3.0,
    metrics_interval_minutes=20.0,
    world_seed=45,
)

#: Two suns + ring, long orbital time-horizon.
SCENARIO_ORBIT_CYCLE = BurnInScenario(
    name="burnin_orbit_cycle",
    days_to_simulate=365,
    time_scale=500.0,
    bot_count=0,
    orbit_long_horizon=True,
    solar_boost=0.1,
    snapshot_interval_hours=12.0,
    metrics_interval_minutes=120.0,
    world_seed=46,
)

#: All named scenarios in a list (used by CI runner).
ALL_SCENARIOS: List[BurnInScenario] = [
    SCENARIO_FAST,
    SCENARIO_WITH_PLAYERS,
    SCENARIO_STORM_CYCLE,
    SCENARIO_INSTABILITY_CYCLE,
    SCENARIO_ORBIT_CYCLE,
]
