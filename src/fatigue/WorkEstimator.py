"""WorkEstimator — Stage 44 mechanical work aggregator.

Estimates per-tick joint mechanical work by aggregating normalised torque
and angular velocity proxies from physics/motor state.  The output is a
[0..1] normalised ``mechWork`` value suitable for the fatigue integrator.

A sliding window (configurable length) smooths instantaneous power spikes
into a representative average.

Public API
----------
WorkInput (re-exported for convenience)
WorkEstimator(config=None)
  .update(torque_proxy, angular_velocity_proxy, recovery_events, dt) → float
  .mech_work → float   # current smoothed value
  .wind_work → float   # current wind-work proxy (caller must set)
  .record_recovery()   # call once per balance-recovery event
  .reset_recovery_cost()
"""
from __future__ import annotations

from collections import deque
from typing import Deque, Optional


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


class WorkEstimator:
    """Estimates normalised mechanical work from physics proxies.

    Parameters
    ----------
    config :
        Optional dict; reads ``fatigue.*`` keys.
    """

    _DEFAULT_WINDOW_SEC   = 3.0   # sliding window length for power averaging
    _DEFAULT_TICK_HZ      = 5.0
    _DEFAULT_RECOVERY_COST = 0.25  # single recovery event contributes this amount

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg  = config or {}
        fcfg = cfg.get("fatigue", {}) or {}

        self._window_sec     = float(fcfg.get("work_window_sec", self._DEFAULT_WINDOW_SEC))
        self._recovery_cost  = float(fcfg.get("single_recovery_cost", self._DEFAULT_RECOVERY_COST))
        self._tick_hz        = float(fcfg.get("tick_hz", self._DEFAULT_TICK_HZ))

        # Sliding window of (dt, power_sample)
        self._window: Deque[tuple] = deque()   # (timestamp, power)
        self._window_time: float = 0.0

        self._mech_work:     float = 0.0
        self._wind_work:     float = 0.0
        self._recovery_cost_acc: float = 0.0   # accumulated this tick

        self._sim_time: float = 0.0

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    @property
    def mech_work(self) -> float:
        """Current smoothed normalised mechanical work [0..1]."""
        return self._mech_work

    @property
    def wind_work(self) -> float:
        """Wind work proxy, set via :meth:`set_wind_work`."""
        return self._wind_work

    def set_wind_work(self, value: float) -> None:
        """Update the wind work proxy from PerceptionSystem.windLoad."""
        self._wind_work = _clamp(value, 0.0, 1.0)

    def record_recovery(self) -> None:
        """Record one balance-recovery event (near-fall / stumble)."""
        self._recovery_cost_acc = _clamp(
            self._recovery_cost_acc + self._recovery_cost, 0.0, 1.0
        )

    def reset_recovery_cost(self) -> float:
        """Consume and reset accumulated recovery cost.  Returns the value."""
        v = self._recovery_cost_acc
        self._recovery_cost_acc = 0.0
        return v

    def update(
        self,
        torque_proxy:           float,
        angular_velocity_proxy: float,
        dt:                     float,
    ) -> float:
        """Compute smoothed mechanical work for this tick.

        Parameters
        ----------
        torque_proxy :
            Normalised average joint torque [0..1] (from motor state).
        angular_velocity_proxy :
            Normalised average angular velocity [0..1] (from motor state).
        dt :
            Elapsed time [s].

        Returns
        -------
        float
            Smoothed normalised mechanical work [0..1].
        """
        if dt <= 0.0:
            return self._mech_work

        self._sim_time += dt

        # Instantaneous power proxy = torque * angular_velocity
        instant_power = _clamp(torque_proxy * angular_velocity_proxy, 0.0, 1.0)

        # Push into sliding window
        self._window.append((self._sim_time, instant_power))
        self._window_time += instant_power * dt

        # Drop samples outside the window
        cutoff = self._sim_time - self._window_sec
        while self._window and self._window[0][0] < cutoff:
            old_t, old_p = self._window.popleft()
            # Approximate removal contribution
            self._window_time = max(0.0, self._window_time - old_p * (1.0 / self._tick_hz))

        # Average over window
        window_len = max(self._window_sec, 1.0)
        self._mech_work = _clamp(self._window_time / window_len, 0.0, 1.0)
        return self._mech_work
