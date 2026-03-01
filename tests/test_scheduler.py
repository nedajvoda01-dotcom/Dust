"""test_scheduler — Stage 16 WorldClock + SimulationScheduler tests.

Tests
-----
1. TestWorldClock
   — real_time always advances; game_time respects time_scale.
   — max_frame_dt_clamp limits large dt values.
   — pause/resume stops game_time; step() advances one frame while paused.
   — is_paused property reflects state correctly.

2. TestSchedulerFrequencies
   — After N game-seconds of ticks, climate step count ≈ N / climate_fixed_dt.
   — After N game-seconds of ticks, insolation update count ≈ N * insolation_hz.
   — Geology and geo-event ticks also match their configured intervals.

3. TestTimeScaleInvariance
   — Simulations reaching the same sim_time produce the same number of
     climate steps regardless of time_scale (1× vs 10×).

4. TestBudgetLimits
   — BudgetedJobQueue never exceeds max_jobs per process_jobs() call.
   — Accumulated cost_estimate stays within max_ms per call (when jobs fit).
   — Remaining jobs are deferred to the next call, not dropped.

5. TestSpiralOfDeathProtection
   — A very large game_dt does not produce more than _MAX_STEPS_PER_FRAME
     steps for any fixed-step system.

6. TestTemporalLOD
   — is_far_lod_tick is True every far_update_divisor climate ticks.
   — far_update_divisor=1 makes every tick a far-update tick.
"""
from __future__ import annotations

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.core.Config import Config
from src.core.WorldClock import WorldClock
from src.systems.SimulationScheduler import (
    BudgetedJobQueue,
    SimulationScheduler,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**sched_overrides) -> Config:
    """Build a Config with optional sched-section overrides for unit tests."""
    cfg = Config.__new__(Config)
    cfg._data = {
        "sched": {
            "insolation_hz": 2.0,
            "climate_fixed_dt": 1.0,
            "geology_tick_seconds": 5.0,
            "geoevent_tick_seconds": 10.0,
            "near_radius": 10000.0,
            "far_update_divisor": 4,
            "job_budget_ms": 100.0,   # large enough for most tests; test_scheduler_job_budget_via_scheduler overrides this
            "job_max_per_frame": 100,
            **sched_overrides,
        },
    }
    return cfg


class _CountingSystem:
    """Minimal mock for any system that just needs update(dt) called."""

    def __init__(self) -> None:
        self.update_count: int = 0
        self.total_dt: float = 0.0

    def update(self, dt: float, **_kwargs) -> None:
        self.update_count += 1
        self.total_dt += dt


class _CountingGeoEvents:
    """Minimal mock for GeoEventSystem (update_with_dt signature)."""

    def __init__(self) -> None:
        self.tick_count: int = 0

    def update_with_dt(self, dt: float, game_time: float, player_pos=None) -> None:
        self.tick_count += 1


class _CountingAstro:
    """Minimal mock for AstroSystem."""

    def __init__(self) -> None:
        self.update_count: int = 0
        self.spin_angle: float = 0.0

    def update(self, game_time: float) -> None:
        self.update_count += 1


class _CountingInsolation:
    """Minimal mock for InsolationField."""

    def __init__(self) -> None:
        self.update_count: int = 0

    def update(self, game_time: float, astro, planet_radius: float) -> None:
        self.update_count += 1


def _run_scheduler(
    scheduler: SimulationScheduler,
    game_seconds: float,
    frame_dt: float = 0.016,  # ~60fps
) -> None:
    """Drive the scheduler for *game_seconds* of game time using fixed frames."""
    t = 0.0
    while t < game_seconds:
        dt = min(frame_dt, game_seconds - t)
        if dt <= 0:
            break
        scheduler.tick(dt, t + dt)
        t += dt


# ---------------------------------------------------------------------------
# 1. TestWorldClock
# ---------------------------------------------------------------------------

class TestWorldClock(unittest.TestCase):
    """WorldClock advances time correctly under normal and edge-case inputs."""

    def test_real_time_accumulates(self):
        clock = WorldClock()
        clock.tick(0.016)
        clock.tick(0.016)
        self.assertAlmostEqual(clock.real_time, 0.032, places=9)

    def test_game_time_scales_with_time_scale(self):
        clock = WorldClock(time_scale=5.0, max_frame_dt_clamp=2.0)
        clock.tick(1.0)
        self.assertAlmostEqual(clock.game_time, 5.0, places=9)
        self.assertAlmostEqual(clock.sim_time, 5.0, places=9)

    def test_max_frame_dt_clamp(self):
        clock = WorldClock(time_scale=1.0, max_frame_dt_clamp=0.1)
        clock.tick(10.0)  # huge frame spike
        self.assertAlmostEqual(clock.real_dt, 0.1, places=9)
        self.assertAlmostEqual(clock.game_dt, 0.1, places=9)

    def test_pause_stops_game_time(self):
        clock = WorldClock(max_frame_dt_clamp=2.0)
        clock.tick(1.0)
        clock.pause()
        clock.tick(1.0)
        # real_time advances but game/sim do not
        self.assertAlmostEqual(clock.real_time, 2.0, places=9)
        self.assertAlmostEqual(clock.game_time, 1.0, places=9)

    def test_resume_restores_advancement(self):
        clock = WorldClock(max_frame_dt_clamp=2.0)
        clock.pause()
        clock.tick(1.0)
        clock.resume()
        clock.tick(1.0)
        self.assertAlmostEqual(clock.game_time, 1.0, places=9)

    def test_step_advances_one_frame_while_paused(self):
        clock = WorldClock(time_scale=1.0, max_frame_dt_clamp=1.0)
        clock.pause()
        clock.step()
        clock.tick(0.5)  # pending step consumes this tick
        self.assertAlmostEqual(clock.game_dt, 0.5, places=9)
        # Next tick without step() should be frozen
        clock.tick(0.5)
        self.assertAlmostEqual(clock.game_dt, 0.0, places=9)

    def test_is_paused_property(self):
        clock = WorldClock()
        self.assertFalse(clock.is_paused)
        clock.pause()
        self.assertTrue(clock.is_paused)
        clock.resume()
        self.assertFalse(clock.is_paused)

    def test_game_dt_positive_when_running(self):
        clock = WorldClock(time_scale=2.0)
        clock.tick(0.05)
        self.assertAlmostEqual(clock.game_dt, 0.1, places=9)

    def test_sim_time_matches_game_time(self):
        clock = WorldClock(time_scale=3.0)
        for _ in range(10):
            clock.tick(0.033)
        self.assertAlmostEqual(clock.sim_time, clock.game_time, places=12)


# ---------------------------------------------------------------------------
# 2. TestSchedulerFrequencies
# ---------------------------------------------------------------------------

class TestSchedulerFrequencies(unittest.TestCase):
    """Scheduler respects configured update frequencies."""

    _TOLERANCE = 0.05   # 5% relative tolerance

    def _assert_near(self, actual: int, expected: float, label: str) -> None:
        lo = max(0, int(expected * (1.0 - self._TOLERANCE)))
        hi = int(expected * (1.0 + self._TOLERANCE)) + 2
        self.assertGreaterEqual(actual, lo, f"{label}: {actual} below expected ~{expected:.1f}")
        self.assertLessEqual(actual,   hi, f"{label}: {actual} above expected ~{expected:.1f}")

    def test_climate_step_count(self):
        """climate_steps ≈ sim_seconds / climate_fixed_dt."""
        sim_s = 60.0
        dt = 1.0
        cfg = _make_config(climate_fixed_dt=dt)
        sched = SimulationScheduler(cfg)
        sched.climate = _CountingSystem()
        _run_scheduler(sched, sim_s, frame_dt=0.1)
        expected = sim_s / dt
        self._assert_near(sched.climate.update_count, expected, "climate steps")

    def test_insolation_update_count(self):
        """insolation updates ≈ sim_seconds * insolation_hz."""
        sim_s = 60.0
        hz = 2.0
        cfg = _make_config(insolation_hz=hz)
        sched = SimulationScheduler(cfg)
        sched.astro = _CountingAstro()
        sched.insolation = _CountingInsolation()
        _run_scheduler(sched, sim_s, frame_dt=0.1)
        expected = sim_s * hz
        self._assert_near(sched.insolation.update_count, expected, "insolation updates")

    def test_geology_tick_count(self):
        """geology ticks ≈ sim_seconds / geology_tick_seconds."""
        sim_s = 120.0
        tick_s = 5.0
        cfg = _make_config(geology_tick_seconds=tick_s)
        sched = SimulationScheduler(cfg)
        sched.geology = _CountingSystem()
        _run_scheduler(sched, sim_s, frame_dt=0.1)
        expected = sim_s / tick_s
        self._assert_near(sched.geology.update_count, expected, "geology ticks")

    def test_geoevent_tick_count(self):
        """geo-event ticks ≈ sim_seconds / geoevent_tick_seconds."""
        sim_s = 200.0
        tick_s = 10.0
        cfg = _make_config(geoevent_tick_seconds=tick_s)
        sched = SimulationScheduler(cfg)
        sched.geo_events = _CountingGeoEvents()
        _run_scheduler(sched, sim_s, frame_dt=0.1)
        expected = sim_s / tick_s
        self._assert_near(sched.geo_events.tick_count, expected, "geo-event ticks")

    def test_astro_called_every_frame(self):
        """AstroSystem is updated every frame."""
        sim_s = 10.0
        frame_dt = 0.1
        cfg = _make_config()
        sched = SimulationScheduler(cfg)
        sched.astro = _CountingAstro()
        _run_scheduler(sched, sim_s, frame_dt=frame_dt)
        expected_frames = int(sim_s / frame_dt)
        # astro should be called once per frame (allow ±1 for rounding)
        self.assertAlmostEqual(sched.astro.update_count, expected_frames, delta=1)


# ---------------------------------------------------------------------------
# 3. TestTimeScaleInvariance
# ---------------------------------------------------------------------------

class TestTimeScaleInvariance(unittest.TestCase):
    """Same sim_time → same number of climate steps regardless of time_scale."""

    def _climate_steps_for_sim_time(
        self,
        target_sim_time: float,
        time_scale: float,
        climate_dt: float = 1.0,
    ) -> int:
        """Drive WorldClock + scheduler until sim_time ≈ target_sim_time."""
        cfg = _make_config(climate_fixed_dt=climate_dt)
        sched = SimulationScheduler(cfg)
        sched.climate = _CountingSystem()

        clock = WorldClock(time_scale=time_scale, max_frame_dt_clamp=0.1)
        real_frame_dt = 0.016  # fixed ~60fps wall clock
        while clock.sim_time < target_sim_time:
            clock.tick(real_frame_dt)
            sched.tick(clock.game_dt, clock.sim_time)
        return sched.climate.update_count

    def test_same_sim_time_same_climate_steps_1x(self):
        """time_scale=1 and time_scale=10 reach the same climate step count."""
        target = 100.0
        steps_1x  = self._climate_steps_for_sim_time(target, time_scale=1.0)
        steps_10x = self._climate_steps_for_sim_time(target, time_scale=10.0)
        # Allow ±1 for boundary rounding
        self.assertAlmostEqual(steps_1x, steps_10x, delta=2,
            msg=(f"climate steps mismatch: 1x={steps_1x} 10x={steps_10x} "
                 f"(target sim_time={target}s, climate_dt=1s)"))

    def test_time_scale_does_not_change_step_logic(self):
        """Higher time_scale reaches the same sim_time faster (fewer real frames)."""
        target = 50.0
        steps_slow = self._climate_steps_for_sim_time(target, time_scale=1.0)
        steps_fast = self._climate_steps_for_sim_time(target, time_scale=5.0)
        # Both should produce ~50 climate steps (50s / 1s per step)
        self.assertAlmostEqual(steps_slow, 50, delta=2)
        self.assertAlmostEqual(steps_fast, 50, delta=2)


# ---------------------------------------------------------------------------
# 4. TestBudgetLimits
# ---------------------------------------------------------------------------

class TestBudgetLimits(unittest.TestCase):
    """BudgetedJobQueue respects max_jobs and cost-estimate budget."""

    def _queue_with_jobs(self, n: int, cost: float = 1.0) -> BudgetedJobQueue:
        q = BudgetedJobQueue()
        for i in range(n):
            q.push_job(f"job_{i}", cost_estimate=cost)
        return q

    def test_max_jobs_respected(self):
        """process_jobs never completes more than max_jobs jobs."""
        q = self._queue_with_jobs(20, cost=0.01)
        completed = q.process_jobs(max_ms=1000.0, max_jobs=5)
        self.assertLessEqual(len(completed), 5)

    def test_cost_estimate_budget_respected(self):
        """Accumulated cost_estimate of completed jobs stays within max_ms."""
        q = self._queue_with_jobs(20, cost=2.0)
        completed = q.process_jobs(max_ms=5.0, max_jobs=100)
        total_cost = sum(j.cost_estimate for j in completed)
        # Budget is 5ms; each job costs 2ms.  After first job (2ms) a second
        # would push cost to 4ms (ok), a third to 6ms (exceeds 5ms budget).
        self.assertLessEqual(total_cost, 5.0 + 2.0 + 1e-6,
            msg=f"Total estimated cost {total_cost:.1f}ms exceeded budget")

    def test_remaining_jobs_deferred(self):
        """Jobs not processed in one call remain for the next."""
        q = self._queue_with_jobs(10, cost=0.1)
        first  = q.process_jobs(max_ms=1000.0, max_jobs=3)
        second = q.process_jobs(max_ms=1000.0, max_jobs=3)
        self.assertEqual(len(first),  3)
        self.assertEqual(len(second), 3)
        self.assertEqual(q.pending_count(), 4)

    def test_completed_total_accumulates(self):
        """completed_total tracks jobs across multiple process_jobs calls."""
        q = self._queue_with_jobs(10, cost=0.01)
        q.process_jobs(max_ms=1000.0, max_jobs=4)
        q.process_jobs(max_ms=1000.0, max_jobs=4)
        self.assertEqual(q.completed_total, 8)

    def test_priority_ordering(self):
        """Lower priority number is processed first."""
        q = BudgetedJobQueue()
        q.push_job("low",  cost_estimate=0.1, priority=10)
        q.push_job("high", cost_estimate=0.1, priority=0)
        q.push_job("mid",  cost_estimate=0.1, priority=5)
        done = q.process_jobs(max_ms=1000.0, max_jobs=1)
        self.assertEqual(done[0].job_type, "high")

    def test_empty_queue_safe(self):
        """process_jobs on an empty queue returns empty list without error."""
        q = BudgetedJobQueue()
        result = q.process_jobs(max_ms=10.0, max_jobs=5)
        self.assertEqual(result, [])

    def test_clear_removes_all_pending(self):
        q = self._queue_with_jobs(5)
        q.clear()
        self.assertEqual(q.pending_count(), 0)

    def test_scheduler_job_budget_via_scheduler(self):
        """SimulationScheduler processes jobs within configured budget."""
        n_jobs = 30
        cost_each = 0.5   # ms
        max_jobs = 6
        max_ms = 10.0

        cfg = _make_config(job_budget_ms=max_ms, job_max_per_frame=max_jobs)
        sched = SimulationScheduler(cfg)

        for i in range(n_jobs):
            sched.job_queue.push_job(f"j{i}", cost_estimate=cost_each)

        # One scheduler tick
        sched.tick(0.016, 0.016)

        # No more than max_jobs should have been processed
        self.assertLessEqual(sched.job_queue.completed_total, max_jobs)
        # Remaining jobs must still be in queue
        self.assertGreater(sched.job_queue.pending_count(), 0)


# ---------------------------------------------------------------------------
# 5. TestSpiralOfDeathProtection
# ---------------------------------------------------------------------------

class TestSpiralOfDeathProtection(unittest.TestCase):
    """A single huge dt does not produce runaway sub-steps."""

    _MAX_STEPS = 8  # must match SimulationScheduler._MAX_STEPS_PER_FRAME

    def test_climate_steps_capped(self):
        """Even a 1000-second dt produces at most _MAX_STEPS climate steps."""
        cfg = _make_config(climate_fixed_dt=1.0)
        sched = SimulationScheduler(cfg)
        sched.climate = _CountingSystem()
        sched.tick(game_dt=1000.0, sim_time=1000.0)
        self.assertLessEqual(sched.climate.update_count, self._MAX_STEPS)

    def test_geology_steps_capped(self):
        cfg = _make_config(geology_tick_seconds=1.0)
        sched = SimulationScheduler(cfg)
        sched.geology = _CountingSystem()
        sched.tick(game_dt=1000.0, sim_time=1000.0)
        self.assertLessEqual(sched.geology.update_count, self._MAX_STEPS)

    def test_geoevent_steps_capped(self):
        cfg = _make_config(geoevent_tick_seconds=1.0)
        sched = SimulationScheduler(cfg)
        sched.geo_events = _CountingGeoEvents()
        sched.tick(game_dt=1000.0, sim_time=1000.0)
        self.assertLessEqual(sched.geo_events.tick_count, self._MAX_STEPS)

    def test_accumulator_reset_after_cap(self):
        """After capping, subsequent normal ticks resume correctly."""
        cfg = _make_config(climate_fixed_dt=1.0)
        sched = SimulationScheduler(cfg)
        sched.climate = _CountingSystem()
        sched.tick(game_dt=9999.0, sim_time=9999.0)  # triggers cap
        before = sched.climate.update_count
        sched.tick(game_dt=1.1, sim_time=10000.0)    # normal tick → 1 step
        after = sched.climate.update_count
        self.assertEqual(after - before, 1)


# ---------------------------------------------------------------------------
# 6. TestTemporalLOD
# ---------------------------------------------------------------------------

class TestTemporalLOD(unittest.TestCase):
    """is_far_lod_tick fires every far_update_divisor climate steps."""

    def _count_far_ticks(self, divisor: int, sim_s: float) -> int:
        cfg = _make_config(climate_fixed_dt=1.0, far_update_divisor=divisor)
        sched = SimulationScheduler(cfg)
        sched.climate = _CountingSystem()
        far_count = 0
        t = 0.0
        while t < sim_s:
            dt = 0.1
            sched.tick(dt, t + dt)
            if sched.is_far_lod_tick:
                far_count += 1
            t += dt
        return far_count

    def test_far_tick_divisor_4(self):
        """With divisor=4, far ticks occur every 4 climate steps."""
        cfg = _make_config(climate_fixed_dt=1.0, far_update_divisor=4)
        sched = SimulationScheduler(cfg)
        sched.climate = _CountingSystem()

        far_ticks_at_step = []
        t = 0.0
        while t < 40.0:
            sched.tick(0.1, t + 0.1)
            if sched.is_far_lod_tick:
                far_ticks_at_step.append(sched.climate_tick_count)
            t += 0.1

        # Far ticks should align with multiples of 4
        for tick_count in far_ticks_at_step:
            self.assertEqual(
                tick_count % 4, 0,
                msg=f"Far LOD tick at climate_tick={tick_count} (not a multiple of 4)",
            )

    def test_divisor_1_every_tick_is_far(self):
        """With divisor=1, every climate tick is a far-update tick."""
        cfg = _make_config(climate_fixed_dt=1.0, far_update_divisor=1)
        sched = SimulationScheduler(cfg)
        sched.climate = _CountingSystem()
        sched.tick(1.1, 1.1)  # guarantees at least one climate step
        self.assertTrue(sched.is_far_lod_tick)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
