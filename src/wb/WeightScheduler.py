"""WeightScheduler — Stage 47 risk- and fatigue-aware task weight scheduling.

Computes per-tick task weights for the whole-body QP based on:
* Perception risk (globalRisk, slipRisk, windLoad) from Stage 37.
* Fatigue state (energy, torqueScale) from Stage 44.

The scheduler follows the priority rule from §6:

    As risk and fatigue increase:
    - balance and contact weights increase (safety first)
    - posture/style weights decrease (graceful degradation)

Public API
----------
TaskWeights (dataclass) — per-task weight vector
WeightScheduler(config=None)
  .schedule(risk, slip_risk, wind_load, fatigue_energy,
            fatigue_torque_scale) → TaskWeights
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * _clamp(t, 0.0, 1.0)


# ---------------------------------------------------------------------------
# TaskWeights
# ---------------------------------------------------------------------------

@dataclass
class TaskWeights:
    """Per-task weight vector for the whole-body QP.

    All values are non-negative scalars.  Higher weight → task is
    prioritised more strongly by the solver.
    """
    balance:  float = 1.0
    foot:     float = 1.0
    look:     float = 1.0
    brace:    float = 1.0
    grasp:    float = 1.0
    effort:   float = 0.01


# ---------------------------------------------------------------------------
# WeightScheduler
# ---------------------------------------------------------------------------

class WeightScheduler:
    """Computes QP task weights from risk and fatigue state.

    Parameters
    ----------
    config :
        Optional dict; reads ``qp.task_weights.*`` and ``qp.effort_weight``
        for nominal/baseline values.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg  = (config or {}).get("qp", {}) or {}
        tw   = cfg.get("task_weights", {}) or {}

        # Nominal (rested, low-risk) weights from config
        self._nom_balance = float(tw.get("balance", 2.0))
        self._nom_foot    = float(tw.get("foot",    1.0))
        self._nom_look    = float(tw.get("look",    0.5))
        self._nom_brace   = float(tw.get("brace",   0.5))
        self._nom_grasp   = float(tw.get("grasp",   1.0))
        self._effort      = float(cfg.get("effort_weight", 0.01))

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def schedule(
        self,
        risk:                 float = 0.0,
        slip_risk:            float = 0.0,
        wind_load:            float = 0.0,
        fatigue_energy:       float = 1.0,
        fatigue_torque_scale: float = 1.0,
    ) -> TaskWeights:
        """Compute task weights for the current tick.

        Parameters
        ----------
        risk :
            Global perception risk [0..1] from PerceptionState.
        slip_risk :
            Slip risk [0..1] from PerceptionState.
        wind_load :
            Wind load [0..1] from PerceptionState.
        fatigue_energy :
            Current energy [0..1]; 1 = fully rested.
        fatigue_torque_scale :
            Current torque capacity scale [0..1] from FatigueToMotorAdapter.
        """
        # Combined threat: max of risk signals
        threat = _clamp(max(risk, slip_risk * 0.8, wind_load * 0.6), 0.0, 1.0)
        # Fatigue factor: 0 = rested, 1 = exhausted
        fatigue = _clamp(1.0 - fatigue_energy, 0.0, 1.0)
        # Degradation = max of threat and fatigue
        deg = _clamp(max(threat, fatigue), 0.0, 1.0)

        # Balance scales up under threat/fatigue (more conservative)
        balance_w = _lerp(self._nom_balance, self._nom_balance * 3.0, deg)

        # Foot placement also up under slip risk specifically
        foot_w = _lerp(self._nom_foot, self._nom_foot * 2.0, slip_risk)

        # Look/posture degrades (less priority)
        look_w  = _lerp(self._nom_look, self._nom_look * 0.2, deg)

        # Brace scales up with brace signal from fatigue
        brace_w = _lerp(self._nom_brace, self._nom_brace * 2.5, fatigue)

        # Grasp stays stable (safety constraint)
        grasp_w = self._nom_grasp

        return TaskWeights(
            balance=balance_w,
            foot=foot_w,
            look=look_w,
            brace=brace_w,
            grasp=grasp_w,
            effort=self._effort,
        )
