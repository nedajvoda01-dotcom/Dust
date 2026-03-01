"""SimulationScheduler — Stage 16 system update orchestration.

Provides three classes:

BudgetedJobQueue
    Push arbitrary jobs with a cost estimate (ms); process up to a
    configurable wall-time budget and job-count cap per frame.  Jobs that
    don't fit are deferred to the next call automatically.

SchedulerLog
    Lightweight periodic logger: counts climate steps, insolation updates,
    geology ticks, geo-event ticks, and completed jobs, then flushes a
    summary every ``log_interval_s`` of sim-time.

SimulationScheduler
    Drives all simulation subsystems in the correct causal order with
    configurable update frequencies:

    1. AstroSystem          — every frame (variable dt)
    2. InsolationField      — at ``insolation_hz`` (rate-limited, paced)
    3. ClimateSystem        — fixed-step at ``climate_fixed_dt``
    4. Geology (Tectonic)   — fixed-step at ``geology_tick_seconds``
    5. GeoEventSystem       — fixed-step at ``geoevent_tick_seconds``
    6. Job queue            — up to ``job_budget_ms`` / ``job_max_per_frame``

    Spiral-of-death protection: each fixed-step system is limited to
    ``_MAX_STEPS_PER_FRAME`` sub-steps per frame; the accumulator is reset
    when the cap is hit (and a warning is logged).

    Temporal LOD:
    ``_lod_tick`` counts climate steps.  When
    ``_lod_tick % far_update_divisor == 0`` the ``is_far_lod_tick`` property
    is True — callers can use this to decide when to refresh far-field data.

Config keys read (all under the ``sched`` section):
    insolation_hz, climate_fixed_dt, geology_tick_seconds,
    geoevent_tick_seconds, near_radius, far_update_divisor,
    job_budget_ms, job_max_per_frame
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, List, Optional

from src.core.Config import Config
from src.core.Logger import Logger

_TAG = "SimSched"
_MAX_STEPS_PER_FRAME = 8  # hard spiral-of-death cap for any fixed-step system


# ---------------------------------------------------------------------------
# BudgetedJobQueue
# ---------------------------------------------------------------------------

@dataclass
class BudgetedJob:
    """A single queued job with a cost estimate used for budget accounting."""
    job_type: str
    cost_estimate: float   # estimated execution cost in milliseconds
    priority: int = 0      # lower value = higher priority
    payload: Any = None


class BudgetedJobQueue:
    """Priority queue of jobs processed within a per-frame time/count budget.

    Jobs that cannot fit in the current frame's budget remain in the queue
    and are processed on the next call to ``process_jobs()``.
    """

    def __init__(self) -> None:
        self._jobs: List[BudgetedJob] = []
        self.completed_total: int = 0

    # ------------------------------------------------------------------
    def push_job(
        self,
        job_type: str,
        cost_estimate: float,
        priority: int = 0,
        payload: Any = None,
    ) -> None:
        """Enqueue a job.

        Parameters
        ----------
        job_type:       Identifier string (e.g. ``"near_mesh"``, ``"far_tile"``).
        cost_estimate:  Expected cost in milliseconds (used for budget gating).
        priority:       Lower value = processed first (0 = highest priority).
        payload:        Arbitrary data passed through to the caller unchanged.
        """
        self._jobs.append(BudgetedJob(job_type, cost_estimate, priority, payload))

    # ------------------------------------------------------------------
    def process_jobs(self, max_ms: float, max_jobs: int) -> List[BudgetedJob]:
        """Process jobs up to budget and return completed jobs.

        Stops early when either:
        * ``max_jobs`` jobs have been completed, or
        * the accumulated *estimated* cost exceeds ``max_ms``, or
        * the wall-clock time spent in this call reaches ``max_ms``.

        Remaining jobs stay in the queue for the next call.

        Parameters
        ----------
        max_ms:   Per-call wall-time + cost-estimate budget (milliseconds).
        max_jobs: Maximum number of jobs to complete in one call.

        Returns
        -------
        List of completed :class:`BudgetedJob` instances.
        """
        # Sort by priority (stable; keeps insertion order for equal priorities)
        self._jobs.sort(key=lambda j: j.priority)

        completed: List[BudgetedJob] = []
        budget_used: float = 0.0
        t_start: float = time.monotonic()

        while self._jobs and len(completed) < max_jobs:
            elapsed_ms = (time.monotonic() - t_start) * 1000.0
            if elapsed_ms >= max_ms:
                break
            job = self._jobs[0]
            # If cost estimate alone would bust the budget *and* we've already
            # done some work, defer this job to the next frame.
            if budget_used + job.cost_estimate > max_ms and completed:
                break
            self._jobs.pop(0)
            completed.append(job)
            budget_used += job.cost_estimate

        self.completed_total += len(completed)
        return completed

    # ------------------------------------------------------------------
    def pending_count(self) -> int:
        """Number of jobs waiting in the queue."""
        return len(self._jobs)

    def clear(self) -> None:
        """Discard all pending jobs (e.g. on level unload)."""
        self._jobs.clear()


# ---------------------------------------------------------------------------
# SchedulerLog
# ---------------------------------------------------------------------------

class SchedulerLog:
    """Accumulates per-interval statistics and flushes them to the Logger.

    All counters reset after each flush.  Set ``log_interval_s`` to 0 to
    log every frame (useful for debugging).
    """

    def __init__(self, log_interval_s: float = 30.0) -> None:
        self.log_interval_s: float = log_interval_s
        self._last_log_sim_time: float = 0.0

        # Counters — reset on each flush
        self.climate_steps: int = 0
        self.insolation_updates: int = 0
        self.geology_ticks: int = 0
        self.geoevent_ticks: int = 0
        self.jobs_completed: int = 0

    # ------------------------------------------------------------------
    def maybe_flush(self, sim_time: float, job_queue: Optional[BudgetedJobQueue] = None) -> None:
        """Emit a log line if ``log_interval_s`` has elapsed since the last flush."""
        if sim_time - self._last_log_sim_time < self.log_interval_s:
            return
        if job_queue is not None:
            self.jobs_completed = job_queue.completed_total
        Logger.info(
            _TAG,
            f"[t={sim_time:.1f}s] "
            f"climate={self.climate_steps} "
            f"insol={self.insolation_updates} "
            f"geo={self.geology_ticks} "
            f"events={self.geoevent_ticks} "
            f"jobs={self.jobs_completed}",
        )
        self._last_log_sim_time = sim_time
        self.climate_steps = 0
        self.insolation_updates = 0
        self.geology_ticks = 0
        self.geoevent_ticks = 0
        self.jobs_completed = 0


# ---------------------------------------------------------------------------
# SimulationScheduler
# ---------------------------------------------------------------------------

class SimulationScheduler:
    """Orchestrates update order and frequencies for all simulation subsystems.

    All system references are optional; set them after construction via
    attribute assignment::

        sched = SimulationScheduler(config)
        sched.astro      = my_astro_system
        sched.insolation = my_insolation_field
        sched.climate    = my_climate_system
        sched.geology    = my_tectonic_system
        sched.geo_events = my_geo_event_system

    Then call ``tick(clock, player_pos)`` once per frame.
    """

    def __init__(self, config: Config, planet_radius: float = 1000.0) -> None:
        # -- Scheduler frequencies --
        self._insolation_hz: float = float(
            config.get("sched", "insolation_hz", default=2.0)
        )
        self._climate_fixed_dt: float = float(
            config.get("sched", "climate_fixed_dt", default=1.0)
        )
        self._geology_tick_s: float = float(
            config.get("sched", "geology_tick_seconds", default=5.0)
        )
        self._geoevent_tick_s: float = float(
            config.get("sched", "geoevent_tick_seconds", default=10.0)
        )

        # -- Temporal LOD --
        self._near_radius: float = float(
            config.get("sched", "near_radius", default=10000.0)
        )
        self._far_update_divisor: int = int(
            config.get("sched", "far_update_divisor", default=4)
        )

        # -- Job budget --
        self._job_budget_ms: float = float(
            config.get("sched", "job_budget_ms", default=4.0)
        )
        self._job_max_per_frame: int = int(
            config.get("sched", "job_max_per_frame", default=4)
        )

        # -- Physics helpers --
        self._planet_radius: float = planet_radius

        # -- Systems (set externally) --
        self.astro: Any = None         # AstroSystem — update(game_time)
        self.insolation: Any = None    # InsolationField — update(game_time, astro, radius)
        self.climate: Any = None       # ClimateSystem — update(dt, insolation)
        self.geology: Any = None       # TectonicPlatesSystem — update(dt)
        self.geo_events: Any = None    # GeoEventSystem — update_with_dt(dt, game_time, player_pos)

        # -- Fixed-step accumulators --
        self._insolation_accum: float = 0.0
        self._climate_accum: float = 0.0
        self._geology_accum: float = 0.0
        self._geoevent_accum: float = 0.0

        # -- Tick counters (for determinism and temporal LOD) --
        self._climate_tick: int = 0   # total climate steps executed
        self._geology_tick: int = 0
        self._geoevent_tick: int = 0
        self._insolation_tick: int = 0

        # LOD divisor tick (increments with each climate tick)
        self._lod_tick: int = 0

        # -- Job queue --
        self.job_queue: BudgetedJobQueue = BudgetedJobQueue()

        # -- Periodic logger --
        self._log: SchedulerLog = SchedulerLog()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_far_lod_tick(self) -> bool:
        """True when the current climate tick should refresh far-field data.

        External systems can use this to decide whether to run expensive
        far-field computations (every ``far_update_divisor`` climate steps).
        """
        if self._far_update_divisor <= 1:
            return True
        return (self._lod_tick % self._far_update_divisor) == 0

    @property
    def climate_tick_count(self) -> int:
        """Total number of climate fixed steps executed so far."""
        return self._climate_tick

    @property
    def geology_tick_count(self) -> int:
        """Total number of geology fixed steps executed so far."""
        return self._geology_tick

    @property
    def geoevent_tick_count(self) -> int:
        """Total number of geo-event ticks executed so far."""
        return self._geoevent_tick

    @property
    def insolation_update_count(self) -> int:
        """Total number of insolation updates triggered so far."""
        return self._insolation_tick

    # ------------------------------------------------------------------
    # Main per-frame update
    # ------------------------------------------------------------------

    def tick(self, game_dt: float, sim_time: float, player_pos: Any = None) -> None:
        """Advance all subsystems by one game frame.

        Parameters
        ----------
        game_dt:    Scaled, clamped frame delta in game-seconds
                    (``WorldClock.game_dt``).
        sim_time:   Current accumulated simulation time in game-seconds
                    (``WorldClock.sim_time``).
        player_pos: Optional world-space player position (Vec3 or None).
                    Used for near/far LOD evaluation and passed through to
                    GeoEventSystem.
        """
        if game_dt <= 0.0:
            return

        # ------------------------------------------------------------------
        # 1. AstroSystem — every frame, variable dt
        # ------------------------------------------------------------------
        if self.astro is not None:
            self.astro.update(sim_time)

        # ------------------------------------------------------------------
        # 2. InsolationField — rate-limited at insolation_hz
        # ------------------------------------------------------------------
        insolation_interval = 1.0 / self._insolation_hz
        self._insolation_accum += game_dt
        if self._insolation_accum >= insolation_interval:
            self._insolation_accum -= insolation_interval
            if self.insolation is not None and self.astro is not None:
                self.insolation.update(sim_time, self.astro, self._planet_radius)
                self._insolation_tick += 1
                self._log.insolation_updates += 1

        # ------------------------------------------------------------------
        # 3. ClimateSystem — fixed-step accumulator
        # ------------------------------------------------------------------
        self._climate_accum += game_dt
        steps = 0
        while self._climate_accum >= self._climate_fixed_dt:
            if steps >= _MAX_STEPS_PER_FRAME:
                Logger.warn(
                    _TAG,
                    f"Climate accumulator capped at {_MAX_STEPS_PER_FRAME} steps/frame; "
                    f"trimming accum={self._climate_accum:.3f}s",
                )
                self._climate_accum = 0.0
                break
            if self.climate is not None:
                self.climate.update(
                    self._climate_fixed_dt,
                    insolation=self.insolation,
                )
            self._climate_accum -= self._climate_fixed_dt
            self._climate_tick += 1
            self._lod_tick += 1
            steps += 1
            self._log.climate_steps += 1

        # ------------------------------------------------------------------
        # 4. Geology (TectonicPlatesSystem) — slow fixed intervals
        # ------------------------------------------------------------------
        self._geology_accum += game_dt
        steps = 0
        while self._geology_accum >= self._geology_tick_s:
            if steps >= _MAX_STEPS_PER_FRAME:
                Logger.warn(
                    _TAG,
                    f"Geology accumulator capped at {_MAX_STEPS_PER_FRAME} steps/frame; "
                    f"trimming accum={self._geology_accum:.3f}s",
                )
                self._geology_accum = 0.0
                break
            if self.geology is not None:
                self.geology.update(self._geology_tick_s)
            self._geology_accum -= self._geology_tick_s
            self._geology_tick += 1
            steps += 1
            self._log.geology_ticks += 1

        # ------------------------------------------------------------------
        # 5. GeoEventSystem — rare fixed intervals
        # ------------------------------------------------------------------
        self._geoevent_accum += game_dt
        steps = 0
        while self._geoevent_accum >= self._geoevent_tick_s:
            if steps >= _MAX_STEPS_PER_FRAME:
                Logger.warn(
                    _TAG,
                    f"GeoEvent accumulator capped at {_MAX_STEPS_PER_FRAME} steps/frame; "
                    f"trimming accum={self._geoevent_accum:.3f}s",
                )
                self._geoevent_accum = 0.0
                break
            if self.geo_events is not None:
                self.geo_events.update_with_dt(
                    dt=self._geoevent_tick_s,
                    game_time=sim_time,
                    player_pos=player_pos,
                )
            self._geoevent_accum -= self._geoevent_tick_s
            self._geoevent_tick += 1
            steps += 1
            self._log.geoevent_ticks += 1

        # ------------------------------------------------------------------
        # 6. Budgeted job queue — process within frame budget
        # ------------------------------------------------------------------
        self.job_queue.process_jobs(
            max_ms=self._job_budget_ms,
            max_jobs=self._job_max_per_frame,
        )

        # ------------------------------------------------------------------
        # Periodic stats log
        # ------------------------------------------------------------------
        self._log.maybe_flush(sim_time, self.job_queue)
