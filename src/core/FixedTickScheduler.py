"""FixedTickScheduler — Stage 42 §2.1 / §3.

Multi-rate fixed-tick dispatcher.  All simulation subsystems run at one of
the five canonical tick rates defined by
:class:`~src.core.DeterminismContract.DeterminismContract`.

Floating-point ``dt`` accumulation is allowed **only** in the scheduler
itself; subsystems receive only an integer ``tick_index``.

Spiral-of-death protection: each rate is limited to
``MAX_STEPS_PER_FRAME`` catch-up steps per :meth:`advance` call.

Public API
----------
FixedTickScheduler(sim_hz, perception_hz, intent_hz, social_hz, evolution_hz)
  .advance(real_dt)       — accumulate time, fire pending ticks
  .register(rate, cb)     — register a callback for a given tick rate
  .tick_counts            — dict of {rate_name: tick_index}
  .tick_index(rate_name)  — current tick index for one rate
  .MAX_STEPS_PER_FRAME    — spiral-of-death cap (default 8)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

from src.core.DeterminismContract import DeterminismContract
from src.core.Logger import Logger

_TAG = "FixedTickSched"

MAX_STEPS_PER_FRAME: int = 8


@dataclass
class TickStats:
    """Number of ticks fired for each rate in one :meth:`advance` call."""
    sim: int = 0
    perception: int = 0
    intent: int = 0
    social: int = 0
    evolution: int = 0


class _RateState:
    __slots__ = ("name", "dt", "accum", "tick_index", "callbacks")

    def __init__(self, name: str, hz: int) -> None:
        self.name: str = name
        self.dt: float = 1.0 / hz
        self.accum: float = 0.0
        self.tick_index: int = 0
        self.callbacks: List[Callable[[int], None]] = []


class FixedTickScheduler:
    """Multi-rate fixed-tick dispatcher for deterministic simulation.

    Registered callbacks are called with a single argument:
    ``tick_index`` (int) — the monotonically increasing tick counter for
    that rate.

    Example
    -------
    ::

        sched = FixedTickScheduler()
        sched.register("sim", my_physics_step)
        sched.register("intent", my_intent_step)

        while running:
            stats = sched.advance(frame_dt)
    """

    MAX_STEPS_PER_FRAME: int = MAX_STEPS_PER_FRAME

    def __init__(
        self,
        sim_hz: int = DeterminismContract.SIM_TICK_HZ,
        perception_hz: int = DeterminismContract.PERCEPTION_TICK_HZ,
        intent_hz: int = DeterminismContract.INTENT_TICK_HZ,
        social_hz: int = DeterminismContract.SOCIAL_TICK_HZ,
        evolution_hz: int = DeterminismContract.EVOLUTION_TICK_HZ,
    ) -> None:
        self._rates: Dict[str, _RateState] = {
            "sim":        _RateState("sim",        sim_hz),
            "perception": _RateState("perception", perception_hz),
            "intent":     _RateState("intent",     intent_hz),
            "social":     _RateState("social",     social_hz),
            "evolution":  _RateState("evolution",  evolution_hz),
        }

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, rate_name: str, callback: Callable[[int], None]) -> None:
        """Register *callback* to be called on every tick of *rate_name*."""
        if rate_name not in self._rates:
            raise ValueError(
                f"Unknown rate '{rate_name}'. Valid: {list(self._rates)}"
            )
        self._rates[rate_name].callbacks.append(callback)

    # ------------------------------------------------------------------
    # Advance
    # ------------------------------------------------------------------

    def advance(self, real_dt: float) -> TickStats:
        """Accumulate *real_dt* seconds and fire any pending fixed ticks."""
        real_dt = max(0.0, real_dt)
        stats = TickStats()

        for name, state in self._rates.items():
            state.accum += real_dt
            steps = 0
            while state.accum >= state.dt:
                state.accum -= state.dt
                for cb in state.callbacks:
                    cb(state.tick_index)
                state.tick_index += 1
                steps += 1
                if steps >= self.MAX_STEPS_PER_FRAME:
                    if state.accum > 0.0:
                        Logger.warn(
                            _TAG,
                            f"spiral-of-death cap hit for rate='{name}'; "
                            f"dropping {state.accum:.4f}s of accumulated time",
                        )
                    state.accum = 0.0
                    break
            setattr(stats, name, steps)

        return stats

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def tick_counts(self) -> Dict[str, int]:
        """Current tick index for each rate."""
        return {name: s.tick_index for name, s in self._rates.items()}

    def tick_index(self, rate_name: str) -> int:
        """Return the current tick index for the named rate."""
        return self._rates[rate_name].tick_index
