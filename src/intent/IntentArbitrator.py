"""IntentArbitrator — Stage 38 Autonomous Micro-Intent orchestrator (§2).

Converts a :class:`~src.perception.PerceptionSystem.PerceptionState` into a
:class:`~src.intent.StrategySelector.MotorIntent` that MotorStack executes
physically.

Architecture (§2)::

    PerceptionState
          ↓
    IntentArbitrator  (ticks at intent.tick_hz, §10)
          ↓
    MotorIntent
          ↓
    MotorStack

Design principles
-----------------
* **No random()** — variability uses ``hash(player_id, time_bucket)`` (§9).
* **Cost minimisation** — :class:`~src.intent.CostEvaluator.CostEvaluator`
  quantifies risk; :class:`~src.intent.StrategySelector.StrategySelector`
  finds the mode that minimises it.
* **Hysteresis** — :class:`~src.intent.ModeHysteresis.ModeHysteresis`
  prevents oscillation between adjacent modes (§11).
* **Smooth transitions** — exponential blending between intents on mode
  change.

Public API
----------
IntentArbitrator(config=None, sim_time=0.0, player_id=0)
  .update(dt, sim_time, perception_state, input_velocity=None) → MotorIntent
  .current_mode    → MotorMode
  .current_intent  → MotorIntent
  .debug_info()    → dict
"""
from __future__ import annotations

import math
from typing import Optional

from src.math.Vec3 import Vec3
from src.perception.PerceptionSystem import PerceptionState
from src.intent.CostEvaluator  import CostEvaluator, CostBreakdown
from src.intent.StrategySelector import StrategySelector, MotorMode, MotorIntent


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _lerp_vec(a: Vec3, b: Vec3, t: float) -> Vec3:
    return a * (1.0 - t) + b * t


class IntentArbitrator:
    """Orchestrates cost evaluation, strategy selection, and intent blending.

    Parameters
    ----------
    config :
        Optional dict; reads ``intent.*`` keys.
    sim_time :
        Initial simulation time [s].
    player_id :
        Stable integer identifier for this player; used for deterministic
        micro-variation (§9) — never calls ``random()``.
    """

    _DEFAULT_TICK_HZ         = 10.0
    _DEFAULT_BLEND_TAU       = 0.15   # seconds; transition smoothing

    def __init__(
        self,
        config:    Optional[dict] = None,
        sim_time:  float          = 0.0,
        player_id: int            = 0,
    ) -> None:
        self._cfg       = config or {}
        self._player_id = int(player_id)

        icfg = self._cfg.get("intent", {}) or {}
        self._tick_dt  = 1.0 / max(1.0, float(icfg.get("tick_hz", self._DEFAULT_TICK_HZ)))
        self._blend_tau = float(icfg.get("blend_tau_sec", self._DEFAULT_BLEND_TAU))

        self._t_next   = sim_time

        self._cost_eval = CostEvaluator(config)
        self._selector  = StrategySelector(config)

        self._current_mode   = MotorMode.NormalLocomotion
        self._current_intent = MotorIntent()
        self._last_cost: Optional[CostBreakdown] = None

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    @property
    def current_mode(self) -> MotorMode:
        """Most recently selected motor mode."""
        return self._current_mode

    @property
    def current_intent(self) -> MotorIntent:
        """Most recently computed MotorIntent (may be blended)."""
        return self._current_intent

    def update(
        self,
        dt:               float,
        sim_time:         float,
        perception_state: PerceptionState,
        input_velocity:   Optional[Vec3] = None,
    ) -> MotorIntent:
        """Advance the arbitrator and return the current MotorIntent.

        If the tick timer has not yet expired the cached intent is returned
        unchanged, ensuring updates at exactly ``intent.tick_hz``.

        Parameters
        ----------
        dt :
            Elapsed time since last call [s].
        sim_time :
            Current absolute simulation time [s].
        perception_state :
            Latest PerceptionState from PerceptionSystem.
        input_velocity :
            Player/controller desired velocity; ``None`` = idle.

        Returns
        -------
        MotorIntent
        """
        if sim_time < self._t_next:
            return self._current_intent

        self._t_next = sim_time + self._tick_dt

        if input_velocity is None:
            input_velocity = Vec3.zero()

        # 1. Evaluate risk cost
        cost = self._cost_eval.evaluate(perception_state)
        self._last_cost = cost

        # 2. Select optimal mode
        new_mode, target_intent = self._selector.select(
            perception_state, cost, input_velocity
        )

        # 3. Blend toward target intent
        blend_t = _clamp(self._tick_dt / max(self._blend_tau, 1e-6), 0.0, 1.0)
        blended = self._blend_intents(self._current_intent, target_intent, blend_t)

        self._current_mode   = new_mode
        self._current_intent = blended
        return self._current_intent

    def debug_info(self) -> dict:
        """Return a serialisable snapshot for logging and dev tools (§14)."""
        i = self._current_intent
        c = self._last_cost
        return {
            "mode": self._current_mode.name,
            "desiredVelocity":     (i.desiredVelocity.x,    i.desiredVelocity.y,    i.desiredVelocity.z),
            "stanceWidthBias":     i.stanceWidthBias,
            "stepLengthBias":      i.stepLengthBias,
            "bracePreference":     i.bracePreference,
            "attentionTargetDir":  (i.attentionTargetDir.x, i.attentionTargetDir.y, i.attentionTargetDir.z),
            "proximityPreference": i.proximityPreference,
            "assistWillingness":   i.assistWillingness,
            "cost_balance":        c.balance     if c else 0.0,
            "cost_slip":           c.slip        if c else 0.0,
            "cost_visibility":     c.visibility  if c else 0.0,
            "cost_wind":           c.wind        if c else 0.0,
            "cost_vibration":      c.vibration   if c else 0.0,
            "cost_proximity":      c.proximity   if c else 0.0,
            "cost_uncertainty":    c.uncertainty if c else 0.0,
            "cost_total":          c.total       if c else 0.0,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _blend_intents(a: MotorIntent, b: MotorIntent, t: float) -> MotorIntent:
        """Exponentially blend intent *a* toward *b* by factor *t*."""
        return MotorIntent(
            desiredVelocity     = _lerp_vec(a.desiredVelocity,    b.desiredVelocity,    t),
            stanceWidthBias     = _lerp(a.stanceWidthBias,     b.stanceWidthBias,     t),
            stepLengthBias      = _lerp(a.stepLengthBias,      b.stepLengthBias,      t),
            bracePreference     = _lerp(a.bracePreference,     b.bracePreference,     t),
            attentionTargetDir  = _lerp_vec(a.attentionTargetDir, b.attentionTargetDir, t),
            proximityPreference = _lerp(a.proximityPreference, b.proximityPreference, t),
            assistWillingness   = _lerp(a.assistWillingness,   b.assistWillingness,   t),
        )
