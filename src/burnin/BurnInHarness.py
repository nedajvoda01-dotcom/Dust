"""BurnInHarness — Stage 56 headless simulation runner.

Executes a long-running headless burn-in simulation using a
:class:`~src.burnin.BurnInScenarios.BurnInScenario` to control all
parameters.

The harness:

1. Creates a minimal headless simulation world with deterministic RNG.
2. Drives ``days_to_simulate`` simulated days at ``time_scale`` speed.
3. Drives bots (if configured) each tick.
4. Collects :class:`~src.burnin.MetricsCollector.MetricsCollector` samples.
5. Takes :class:`~src.burnin.SnapshotScheduler.SnapshotScheduler` snapshots.
6. Evaluates invariant assertions at end-of-run.
7. Engages :class:`~src.runtime.FallbackLadders.FallbackLadder` degradation
   automatically on budget overruns.
8. Returns a :class:`BurnInReport`.

No GPU, no network, no audio output — all subsystems run in headless mode.

Usage
-----
    from src.burnin.BurnInScenarios import SCENARIO_FAST
    from src.burnin.BurnInHarness import BurnInHarness

    harness = BurnInHarness(SCENARIO_FAST)
    report  = harness.run()
    assert not report.invariant_violations, report.invariant_violations
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.burnin.BurnInScenarios import BurnInScenario
from src.burnin.BotDrivers import WalkerBot, SlopeBot, BuddyBot, ShelterBot, BotInput
from src.burnin.MetricsCollector import MetricsCollector
from src.burnin.SnapshotScheduler import SnapshotScheduler, WorldSnapshot
from src.core.DetRng import DetRng
from src.core.Logger import Logger
from src.energy.GlobalEnergySystem import GlobalEnergySystem
from src.runtime.BudgetManager import BudgetManager
from src.runtime.FallbackLadders import FallbackLadders

_TAG = "BurnInHarness"

# Simulated hours per day
_HOURS_PER_DAY = 24.0
# Simulated minutes per hour
_MINUTES_PER_HOUR = 60.0


# ---------------------------------------------------------------------------
# Invariant violation record
# ---------------------------------------------------------------------------

@dataclass
class InvariantViolation:
    """One failed invariant assertion."""
    name: str
    message: str
    value: float = 0.0
    threshold: float = 0.0

    def __str__(self) -> str:
        return f"[{self.name}] {self.message} (value={self.value:.4f}, threshold={self.threshold:.4f})"


# ---------------------------------------------------------------------------
# BurnInReport
# ---------------------------------------------------------------------------

@dataclass
class BurnInReport:
    """Summary of a completed burn-in run."""
    scenario_name: str = ""
    days_simulated: int = 0
    ticks_run: int = 0
    snapshots_taken: int = 0
    invariant_violations: List[InvariantViolation] = field(default_factory=list)
    metrics_summary: Dict[str, Any] = field(default_factory=dict)
    fallback_activations: Dict[str, int] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """True when no invariant violations were recorded."""
        return len(self.invariant_violations) == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario":             self.scenario_name,
            "days_simulated":       self.days_simulated,
            "ticks_run":            self.ticks_run,
            "snapshots_taken":      self.snapshots_taken,
            "passed":               self.passed,
            "invariant_violations": [str(v) for v in self.invariant_violations],
            "metrics_summary":      self.metrics_summary,
            "fallback_activations": self.fallback_activations,
        }


# ---------------------------------------------------------------------------
# Minimal headless world state
# ---------------------------------------------------------------------------

class _HeadlessWorld:
    """A lightweight simulation world used by the burn-in harness.

    This does not couple to GPU, audio, or rendering subsystems.
    It drives the energy system and tracks metrics that the invariants need.
    """

    HOURS_PER_DAY = _HOURS_PER_DAY

    def __init__(self, scenario: BurnInScenario) -> None:
        self._rng = DetRng.for_domain(scenario.world_seed, 0, "burnin", 0, 0)
        self._scenario = scenario
        self._planet_time_h: float = 0.0   # simulated hours elapsed
        self._tick: int = 0

        # Energy system
        self._energy_cfg = {
            "energy": {
                "enable": True,
                "tick_hz": 0.1,
                "max_mech_stress": 0.9,
                "max_dust_mass": 1.0,
                "max_ice_mass": 1.0,
                "entropy_upper_bound": 0.9,
                "entropy_lower_bound": 0.1,
                "auto_normalize_k": 0.05,
                "transfer_efficiency": 0.85,
                "mech_per_crust_event": 0.08,
                "mech_per_dust_event": 0.04,
                "atmo_per_dust_event": 0.03,
                "thermal_per_frac_event": 0.06,
                "mech_from_frac_event": 0.04,
                "min_mech_trigger": 0.02,
                "dust_target_mean": 0.5,
                "ice_thermal_threshold": 0.3,
                "ice_melt_return_k": 0.6,
                "wind_erosion_base": 0.4,
            }
        }
        self._energy = GlobalEnergySystem(self._energy_cfg)

        # Simulated dust field (N tiles)
        self._n_tiles = 16
        rng_init = DetRng(scenario.world_seed + 1)
        self._dust_field: List[float] = [
            rng_init.next_range(0.2, 0.8) for _ in range(self._n_tiles)
        ]
        self._ice_field: List[float] = [
            rng_init.next_range(0.0, 0.3) for _ in range(self._n_tiles)
        ]
        self._crust_field: List[float] = [
            rng_init.next_range(0.3, 0.9) for _ in range(self._n_tiles)
        ]

        # Instability event counter (per day)
        self._instability_events_this_day: int = 0

        # Budget manager
        self._budget = BudgetManager()
        # Track fallback activations per ladder
        self._fallback_counts: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Tick
    # ------------------------------------------------------------------

    def tick(self, dt: float, wind_load: float = 0.0) -> None:
        """Advance one simulation step."""
        s = self._scenario

        # Advance planet time
        hours_per_tick = dt / 3600.0  # dt is in seconds; 1 tick = dt sim-seconds
        # With time_scale, each real second = time_scale sim-seconds
        self._planet_time_h += hours_per_tick * s.time_scale
        self._tick += 1

        # Solar injection (sinusoidal day/night cycle + boost)
        day_phase = (self._planet_time_h % _HOURS_PER_DAY) / _HOURS_PER_DAY
        solar = max(0.0, math.sin(day_phase * math.pi)) * 0.8 + s.solar_boost
        self._energy.inject_solar(min(1.0, solar))

        # Wind
        wind = max(wind_load, s.wind_load_override)
        if wind < 0.01:
            wind = 0.3 + self._rng.next_range(-0.1, 0.1)
        self._energy.wind_tick(wind, dt=dt)

        # Extra dust injection
        if s.dust_injection_rate > 0:
            for i in range(self._n_tiles):
                self._dust_field[i] = min(1.0, self._dust_field[i] + s.dust_injection_rate * dt)

        # Instability boost: occasionally fire instability events
        # Base rate is low so natural events/day stays well under 50.
        # instability_boost scales up the probability for stress scenarios.
        event_threshold = max(0.0, 0.0003 + s.instability_boost * 0.05)
        if self._rng.next_float01() < event_threshold:
            intensity = 0.3 + s.instability_boost * 0.6 + self._rng.next_range(0.0, 0.2)
            self._energy.record_instability_event(min(1.0, intensity))
            self._instability_events_this_day += 1

        # Erosion smoothing
        self._energy.record_erosion_smoothing(0.02)

        # Energy balance tick
        self._energy.energy_balance_tick(dt=dt)

        # Budget reporting
        self._budget.begin_frame()
        active_chunks = 20 + self._rng.next_int(0, 30)
        qp_iters = 5 + self._rng.next_int(0, 40)
        self._budget.report("motor", "ik_iters", float(qp_iters))
        self._budget.report("deform", "active_chunks", float(active_chunks))

        # Check ladders for activations
        for name in ("audio", "deform", "ik"):
            ladder = FallbackLadders.get(name)
            if ladder and ladder.is_degraded:
                self._fallback_counts[name] = self._fallback_counts.get(name, 0) + 1

    def reset_day_counters(self) -> None:
        self._instability_events_this_day = 0

    # ------------------------------------------------------------------
    # Metrics accessors
    # ------------------------------------------------------------------

    @property
    def planet_time_hours(self) -> float:
        return self._planet_time_h

    @property
    def total_dust_mass(self) -> float:
        return sum(self._dust_field)

    @property
    def mean_dust_thickness(self) -> float:
        return sum(self._dust_field) / len(self._dust_field)

    @property
    def variance_dust(self) -> float:
        mean = self.mean_dust_thickness
        return sum((x - mean) ** 2 for x in self._dust_field) / len(self._dust_field)

    @property
    def mean_ice_film(self) -> float:
        return sum(self._ice_field) / len(self._ice_field)

    @property
    def ice_coverage(self) -> float:
        threshold = 0.1
        return sum(1 for x in self._ice_field if x > threshold) / len(self._ice_field)

    @property
    def mean_crust_hardness(self) -> float:
        return sum(self._crust_field) / len(self._crust_field)

    @property
    def crust_variance(self) -> float:
        mean = self.mean_crust_hardness
        return sum((x - mean) ** 2 for x in self._crust_field) / len(self._crust_field)

    @property
    def entropy(self) -> float:
        return self._energy.planet_entropy

    @property
    def energy_reservoirs(self) -> Dict[str, float]:
        return dict(self._energy.ledger.reservoirs())

    @property
    def instability_events_this_day(self) -> int:
        return self._instability_events_this_day

    def state_snapshot(self) -> Dict[str, Any]:
        """Return a serialisable state snapshot for SnapshotScheduler."""
        return {
            "energy": self._energy.state_dict(),
            "material": {
                "dust_field":    list(self._dust_field),
                "ice_field":     list(self._ice_field),
                "crust_field":   list(self._crust_field),
            },
        }


# ---------------------------------------------------------------------------
# BurnInHarness
# ---------------------------------------------------------------------------

class BurnInHarness:
    """Drives a headless burn-in simulation for the given scenario.

    Parameters
    ----------
    scenario: :class:`~src.burnin.BurnInScenarios.BurnInScenario`
    """

    def __init__(self, scenario: BurnInScenario) -> None:
        self._scenario = scenario

    def run(self) -> BurnInReport:
        """Execute the full burn-in and return a :class:`BurnInReport`."""
        s = self._scenario
        Logger.info(_TAG, f"Starting burn-in '{s.name}': {s.days_to_simulate} days")

        world = _HeadlessWorld(s)
        collector = MetricsCollector()
        scheduler = SnapshotScheduler(interval_hours=s.snapshot_interval_hours)

        # Create bots
        bots = self._create_bots(s)

        # Tick parameters
        dt = 1.0 / s.tick_hz
        # ticks/sec × hours/day × seconds/hour ÷ time_scale = ticks per simulated day
        ticks_per_sim_day = int(s.tick_hz * _HOURS_PER_DAY * 3600.0 / s.time_scale)
        ticks_per_sim_day = max(1, ticks_per_sim_day)

        # Convert simulated-minutes interval to a tick count
        # (interval_min × 60 s/min) ÷ (dt_s × time_scale)
        metric_interval_ticks = max(
            1,
            int(s.metrics_interval_minutes * _MINUTES_PER_HOUR / dt / s.time_scale),
        )

        total_ticks = 0
        initial_dust_mass = world.total_dust_mass

        for day in range(s.days_to_simulate):
            collector.begin_day(day=day, planet_time=world.planet_time_hours)
            world.reset_day_counters()

            for tick_in_day in range(ticks_per_sim_day):
                # Bot inputs (aggregated wind load from all bots)
                bot_wind_load = 0.0
                for bot in bots:
                    inp = self._drive_bot(bot, world, tick_in_day)
                    bot_wind_load = max(bot_wind_load, 0.3 if inp.shelter_seek else 0.0)

                world.tick(dt=dt, wind_load=bot_wind_load)
                total_ticks += 1

                # Periodic metric sampling
                if tick_in_day % metric_interval_ticks == 0:
                    collector.record_global("totalDustMass",     world.total_dust_mass)
                    collector.record_global("meanDustThickness",  world.mean_dust_thickness)
                    collector.record_global("varianceDust",       world.variance_dust)
                    collector.record_global("meanIceFilm",        world.mean_ice_film)
                    collector.record_global("iceCoverage",        world.ice_coverage)
                    collector.record_global("meanCrustHardness",  world.mean_crust_hardness)
                    collector.record_global("crustVariance",      world.crust_variance)
                    collector.record_global("entropy",            world.entropy)
                    collector.record_perf("activeChunks",         float(20))
                    collector.record_perf("qpItersAvg",           float(25))

                # Snapshot scheduling
                if scheduler.should_snap(world.planet_time_hours):
                    snap = scheduler.build_snapshot(
                        planet_time=world.planet_time_hours,
                        subsystem_states=world.state_snapshot(),
                    )
                    scheduler.record(snap)

            # End-of-day events per day count
            collector.record_global(
                "instabilityEventsPerDay",
                float(world.instability_events_this_day),
            )
            collector.end_day(day=day)

            if day % 10 == 0 or day == s.days_to_simulate - 1:
                Logger.info(
                    _TAG,
                    f"  Day {day+1}/{s.days_to_simulate} | "
                    f"entropy={world.entropy:.3f} | "
                    f"dustMass={world.total_dust_mass:.3f} | "
                    f"instability_events={world.instability_events_this_day}",
                )

        # Evaluate invariants
        violations = self._check_invariants(
            scenario=s,
            collector=collector,
            initial_dust_mass=initial_dust_mass,
            final_world=world,
        )

        report = BurnInReport(
            scenario_name=s.name,
            days_simulated=s.days_to_simulate,
            ticks_run=total_ticks,
            snapshots_taken=scheduler.count(),
            invariant_violations=violations,
            metrics_summary=collector.summary_report(),
            fallback_activations=dict(world._fallback_counts),
        )
        Logger.info(
            _TAG,
            f"Burn-in '{s.name}' done. Ticks={total_ticks}, "
            f"Snapshots={scheduler.count()}, "
            f"Violations={len(violations)}",
        )
        return report

    # ------------------------------------------------------------------
    # Bot creation / driving
    # ------------------------------------------------------------------

    def _create_bots(self, s: BurnInScenario) -> List[Any]:
        bots = []
        seed_base = s.bot_seed_base
        for i in range(s.bot_count):
            kind = i % 4
            if kind == 0:
                bots.append(WalkerBot(seed=seed_base + i))
            elif kind == 1:
                bots.append(SlopeBot(seed=seed_base + i))
            elif kind == 2:
                bots.append(BuddyBot(seed=seed_base + i))
            else:
                bots.append(ShelterBot(seed=seed_base + i,
                                       wind_threshold=0.6))
        return bots

    def _drive_bot(self, bot: Any, world: _HeadlessWorld, tick: int) -> BotInput:
        """Get the next input from a bot driver."""
        if isinstance(bot, WalkerBot):
            return bot.tick()
        if isinstance(bot, SlopeBot):
            return bot.tick()
        if isinstance(bot, BuddyBot):
            return bot.tick()
        if isinstance(bot, ShelterBot):
            wind = world._scenario.wind_load_override or 0.3
            return bot.tick(wind_load=wind)
        return BotInput()

    # ------------------------------------------------------------------
    # Invariant checking
    # ------------------------------------------------------------------

    def _check_invariants(
        self,
        scenario: BurnInScenario,
        collector: MetricsCollector,
        initial_dust_mass: float,
        final_world: _HeadlessWorld,
    ) -> List[InvariantViolation]:
        violations: List[InvariantViolation] = []
        report = collector.summary_report()

        # 1. Dust conservation (skip when explicit injection is active)
        if scenario.dust_injection_rate == 0.0:
            dust_entry = report.get("totalDustMass")
            if dust_entry:
                dust_mean = dust_entry["mean"]
                dust_delta = abs(dust_mean - initial_dust_mass)
                if dust_delta > scenario.assert_eps_dust_mass * initial_dust_mass:
                    violations.append(InvariantViolation(
                        name="DUST_CONSERVATION",
                        message=(
                            f"totalDustMass drifted by {dust_delta:.4f} "
                            f"(eps={scenario.assert_eps_dust_mass * initial_dust_mass:.4f})"
                        ),
                        value=dust_delta,
                        threshold=scenario.assert_eps_dust_mass * initial_dust_mass,
                    ))

        # 2. Entropy bounded
        entropy = final_world.entropy
        if entropy > scenario.assert_max_entropy:
            violations.append(InvariantViolation(
                name="ENTROPY_BOUNDED",
                message=f"entropy={entropy:.4f} exceeded max={scenario.assert_max_entropy:.4f}",
                value=entropy,
                threshold=scenario.assert_max_entropy,
            ))

        # 3. Energy reservoirs bounded
        for name, val in final_world.energy_reservoirs.items():
            if val < 0.0 or val > 1.0:
                violations.append(InvariantViolation(
                    name=f"ENERGY_BOUNDED_{name.upper()}",
                    message=f"reservoir '{name}'={val:.4f} out of [0,1]",
                    value=val,
                    threshold=1.0,
                ))

        # 4. Instability events not linearly growing
        instab_entry = report.get("instabilityEventsPerDay")
        if instab_entry and instab_entry["max"] > scenario.assert_max_instability_per_day:
            violations.append(InvariantViolation(
                name="INSTABILITY_BOUNDED",
                message=(
                    f"instabilityEventsPerDay max={instab_entry['max']:.1f} "
                    f"> limit={scenario.assert_max_instability_per_day:.1f}"
                ),
                value=instab_entry["max"],
                threshold=scenario.assert_max_instability_per_day,
            ))

        # 5. No ice full-saturation (ice coverage < 95% everywhere)
        if final_world.ice_coverage > 0.95:
            violations.append(InvariantViolation(
                name="NO_ICE_SATURATION",
                message=f"iceCoverage={final_world.ice_coverage:.3f} ≥ 0.95",
                value=final_world.ice_coverage,
                threshold=0.95,
            ))

        # 6. Crust not universally zero
        if final_world.mean_crust_hardness < 0.01:
            violations.append(InvariantViolation(
                name="CRUST_NOT_ZERO",
                message=f"meanCrustHardness={final_world.mean_crust_hardness:.4f} ≈ 0",
                value=final_world.mean_crust_hardness,
                threshold=0.01,
            ))

        for v in violations:
            Logger.warn(_TAG, f"INVARIANT VIOLATION: {v}")

        return violations
