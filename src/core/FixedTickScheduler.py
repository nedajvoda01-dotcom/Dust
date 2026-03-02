"""FixedTickScheduler — Stage 42 multi-domain fixed-tick dispatcher.

Drives deterministic update domains at their contract-defined rates:

* ``sim``        — 60 Hz  (physics / motor / IK)
* ``perception`` — 20 Hz
* ``intent``     — 10 Hz
* ``social``      — 5 Hz
* ``evolution``   — configurable slow rate

Each domain has:
* a fixed-dt accumulator
* a tick counter (used in deterministic seeds)
* a list of registered callbacks

Usage
-----
sched = FixedTickScheduler()
sched.register("sim",  my_physics_system.fixed_update)
sched.register("intent", my_intent_system.tick)

# Each frame, call with the frame's game_dt:
sched.tick(game_dt)
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List

from src.core.DeterminismContract import TICK_RATES, TICK_DT
from src.core.Logger import Logger

_TAG = "FixedTickSched"

# Hard cap on catch-up steps per call (spiral-of-death guard)
_MAX_CATCHUP_STEPS = 8


class FixedTickScheduler:
    """Dispatches fixed-rate tick callbacks for all deterministic domains."""

    def __init__(self, extra_domains: Dict[str, float] | None = None) -> None:
        """
        Parameters
        ----------
        extra_domains:
            Optional ``{name: hz}`` map to register additional tick domains
            beyond the standard contract set.
        """
        self._domains: Dict[str, float] = dict(TICK_RATES)
        if extra_domains:
            self._domains.update(extra_domains)

        # Per-domain state
        self._accum: Dict[str, float]    = {name: 0.0 for name in self._domains}
        self._tick_count: Dict[str, int] = {name: 0   for name in self._domains}
        self._callbacks: Dict[str, List[Callable[..., None]]] = {
            name: [] for name in self._domains
        }

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, domain: str, callback: Callable[..., None]) -> None:
        """Register *callback* to be called on every tick of *domain*.

        The callback signature must be ``callback(tick_index: int, dt: float)``.
        """
        if domain not in self._callbacks:
            Logger.warn(_TAG, f"Registering callback for unknown domain '{domain}'; adding dynamically.")
            self._domains[domain] = 1.0  # default 1 Hz
            self._accum[domain] = 0.0
            self._tick_count[domain] = 0
            self._callbacks[domain] = []
        self._callbacks[domain].append(callback)

    def unregister(self, domain: str, callback: Callable[..., None]) -> None:
        """Remove a previously registered callback."""
        cbs = self._callbacks.get(domain, [])
        if callback in cbs:
            cbs.remove(callback)

    # ------------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------------

    def tick(self, game_dt: float) -> None:
        """Advance all domains by *game_dt* seconds.

        Each domain fires all registered callbacks for every fixed step
        it accumulates.  Capped at :data:`_MAX_CATCHUP_STEPS` steps per
        domain per call to prevent spiral-of-death.
        """
        if game_dt <= 0.0:
            return

        for name, hz in self._domains.items():
            fixed_dt = 1.0 / hz
            self._accum[name] += game_dt
            steps = 0
            while self._accum[name] >= fixed_dt:
                if steps >= _MAX_CATCHUP_STEPS:
                    Logger.warn(
                        _TAG,
                        f"Domain '{name}' capped at {_MAX_CATCHUP_STEPS} steps; "
                        f"trimming accum={self._accum[name]:.4f}s",
                    )
                    self._accum[name] = 0.0
                    break
                self._accum[name] -= fixed_dt
                self._tick_count[name] += 1
                tick_idx = self._tick_count[name]
                for cb in list(self._callbacks[name]):
                    try:
                        cb(tick_idx, fixed_dt)
                    except Exception as exc:  # noqa: BLE001
                        cb_name = getattr(cb, "__name__", repr(cb))
                        Logger.error(_TAG, f"Callback '{cb_name}' error in domain '{name}' tick {tick_idx}: {exc}")
                steps += 1

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def tick_count(self, domain: str) -> int:
        """Return the number of fixed ticks executed for *domain* so far."""
        return self._tick_count.get(domain, 0)

    def domains(self) -> List[str]:
        """Return all registered domain names."""
        return list(self._domains.keys())

    def reset(self) -> None:
        """Reset accumulators and tick counters for all domains."""
        for name in self._domains:
            self._accum[name] = 0.0
            self._tick_count[name] = 0
