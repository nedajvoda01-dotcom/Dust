"""WBControllerAdapter — Stage 47 whole-body controller adapter.

Replaces the MotorStack as the *tactical* decision layer.  The footstep
planner (Stage 34) remains the *strategic* layer and feeds target
positions into this adapter each tick.

Architecture (§12)
------------------
The adapter runs in two modes:

* ``shadow`` (47.1):  QP runs but its output is discarded; only metrics
  are logged for validation.
* ``active`` (47.2+): QP output is applied to the physics body.

Fallback
--------
If the QP solver fails (``converged=False`` and ``fallback_on_fail=True``),
the adapter returns the legacy motor params unchanged so the old controller
keeps running.

Public API
----------
WBInput    (dataclass) — all signals feeding the WB controller each tick
WBOutput   (dataclass) — per-tick result (torques / accelerations + status)
WBControllerAdapter(config=None)
  .tick(dt, wb_input) → WBOutput
  .set_shadow_mode(enabled)
  .debug_info()       → dict
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

from src.math.Vec3 import Vec3
from src.wb.QPFormulation import QPFormulation
from src.wb.QPSolver import QPSolver, QPResult
from src.wb.TaskLibrary import (
    N_DOF,
    build_balance_task, build_foot_task, build_look_task,
    build_brace_task, build_grasp_task, build_effort_task,
)
from src.wb.ConstraintLibrary import (
    build_joint_limit_constraints, build_torque_limit_constraints,
    build_contact_no_penetration_constraints, build_friction_cone_constraints,
    build_grasp_force_constraint,
)
from src.wb.WeightScheduler import WeightScheduler, TaskWeights


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


# ---------------------------------------------------------------------------
# WBInput
# ---------------------------------------------------------------------------

@dataclass
class WBInput:
    """All signals required by the whole-body controller for one tick.

    Fields map directly to §4 (Inputs) of the Stage 47 spec.
    """
    # --- Primary intent (43) ---
    desired_com_acc:      Vec3  = field(default_factory=Vec3.zero)
    desired_swing_acc:    Vec3  = field(default_factory=Vec3.zero)
    desired_yaw_acc:      float = 0.0

    # --- Reflex overlay (43) ---
    brace_bias:           float = 0.0   # [0..1]
    slowdown_bias:        float = 0.0   # [0..1]

    # --- Perception (37) ---
    global_risk:          float = 0.0
    slip_risk:            float = 0.0
    wind_load:            float = 0.0

    # --- Contact (34/35/45) ---
    surface_normal:       Optional[Vec3]  = None
    mu:                   float = 0.8
    normal_force:         float = 700.0

    # --- Grasp (40) ---
    grasp_active:         bool  = False
    grasp_rel_error_x:    float = 0.0
    grasp_rel_error_z:    float = 0.0
    grasp_max_force:      float = 800.0

    # --- Fatigue (44) ---
    fatigue_energy:       float = 1.0
    fatigue_torque_scale: float = 1.0


# ---------------------------------------------------------------------------
# WBOutput
# ---------------------------------------------------------------------------

@dataclass
class WBOutput:
    """Per-tick result of the whole-body QP controller.

    Attributes
    ----------
    joint_accelerations :
        Target accelerations for each DOF (length = n_dof).
    converged :
        True when the QP solver found a solution within the budget.
    used_fallback :
        True when the adapter fell back to the legacy controller.
    solver_iters :
        Number of solver iterations used.
    solution_hash :
        Determinism cross-check hash (see QPResult).
    task_weights :
        The TaskWeights used this tick (for diagnostics).
    """
    joint_accelerations: List[float]
    converged:           bool
    used_fallback:       bool       = False
    solver_iters:        int        = 0
    solution_hash:       str        = ""
    task_weights:        Optional[TaskWeights] = None


# ---------------------------------------------------------------------------
# WBControllerAdapter
# ---------------------------------------------------------------------------

class WBControllerAdapter:
    """Whole-body QP controller (Stage 47).

    Parameters
    ----------
    config :
        Optional dict; reads ``qp.*`` keys.
    n_dof :
        Number of decision variables.  Defaults to TaskLibrary.N_DOF.
    """

    def __init__(
        self,
        config:  Optional[dict] = None,
        n_dof:   int = N_DOF,
    ) -> None:
        cfg = config or {}
        qpcfg = cfg.get("qp", {}) or {}

        self._n_dof          = n_dof
        self._shadow_mode    = bool(qpcfg.get("shadow_mode", False))
        self._enabled        = bool(qpcfg.get("enable", True))
        self._fallback_on_fail = bool(qpcfg.get("fallback_on_fail", True))
        self._lod_remote_reduce = float(qpcfg.get("lod", {}).get("remote_reduce_factor", 0.5))

        self._solver    = QPSolver(config)
        self._formul    = QPFormulation(n_dof, config)
        self._scheduler = WeightScheduler(config)

        # Internal state for diagnostics
        self._last_result:   Optional[QPResult]    = None
        self._last_weights:  Optional[TaskWeights] = None

        # Warm-start cache
        self._x_prev: Optional[List[float]] = None

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_shadow_mode(self, enabled: bool) -> None:
        """Enable/disable shadow mode (§47.1).

        In shadow mode the QP runs and metrics are logged but its output
        is not applied to the physics body.
        """
        self._shadow_mode = enabled

    # ------------------------------------------------------------------
    # Tick
    # ------------------------------------------------------------------

    def tick(self, dt: float, wb_input: WBInput) -> WBOutput:
        """Run one whole-body QP tick.

        Parameters
        ----------
        dt :
            Simulation time step [s].
        wb_input :
            All signals for this tick.

        Returns
        -------
        WBOutput
            Joint accelerations and solver status.
        """
        if not self._enabled:
            return self._fallback_output()

        weights = self._scheduler.schedule(
            risk=wb_input.global_risk,
            slip_risk=wb_input.slip_risk,
            wind_load=wb_input.wind_load,
            fatigue_energy=wb_input.fatigue_energy,
            fatigue_torque_scale=wb_input.fatigue_torque_scale,
        )
        self._last_weights = weights

        self._formul.reset()

        # --- Tasks ---
        self._formul.add_task(build_balance_task(
            wb_input.desired_com_acc, weight=weights.balance, n=self._n_dof))
        self._formul.add_task(build_foot_task(
            wb_input.desired_swing_acc, weight=weights.foot, n=self._n_dof))
        self._formul.add_task(build_look_task(
            wb_input.desired_yaw_acc, weight=weights.look, n=self._n_dof))
        if wb_input.brace_bias > 0.0:
            self._formul.add_task(build_brace_task(
                wb_input.brace_bias, weight=weights.brace, n=self._n_dof))
        if wb_input.grasp_active:
            self._formul.add_task(build_grasp_task(
                wb_input.grasp_rel_error_x,
                wb_input.grasp_rel_error_z,
                weight=weights.grasp,
                n=self._n_dof,
            ))
        self._formul.add_task(build_effort_task(
            weight=weights.effort, n=self._n_dof))

        # --- Constraints ---
        # Torque limits (fatigue-aware)
        torque_cap = wb_input.fatigue_torque_scale * 10.0  # m/s² proxy
        self._formul.add_ineq(build_torque_limit_constraints(
            torque_max=torque_cap, n=self._n_dof))
        self._formul.add_ineq(build_joint_limit_constraints(
            acc_max=torque_cap, n=self._n_dof))

        # Contact / friction
        self._formul.add_ineq(build_contact_no_penetration_constraints(
            surface_normal=wb_input.surface_normal, n=self._n_dof))
        self._formul.add_ineq(build_friction_cone_constraints(
            mu=wb_input.mu,
            normal_force=wb_input.normal_force,
            n=self._n_dof,
        ))

        # Grasp force cap
        if wb_input.grasp_active:
            self._formul.add_ineq(build_grasp_force_constraint(
                max_force=wb_input.grasp_max_force, n=self._n_dof))

        problem = self._formul.build(x0=self._x_prev)
        result  = self._solver.solve(problem)
        self._last_result = result

        # Cache warm-start
        self._x_prev = list(result.x[:self._n_dof])

        if self._shadow_mode:
            # Shadow: return zeros (legacy controller keeps running)
            return WBOutput(
                joint_accelerations=[0.0] * self._n_dof,
                converged=result.converged,
                used_fallback=True,
                solver_iters=result.iters,
                solution_hash=result.solution_hash,
                task_weights=weights,
            )

        if not result.converged and self._fallback_on_fail:
            return self._fallback_output(
                solver_iters=result.iters,
                solution_hash=result.solution_hash,
                task_weights=weights,
            )

        return WBOutput(
            joint_accelerations=list(result.x[:self._n_dof]),
            converged=result.converged,
            used_fallback=False,
            solver_iters=result.iters,
            solution_hash=result.solution_hash,
            task_weights=weights,
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def debug_info(self) -> dict:
        """Return last-tick diagnostic information."""
        r = self._last_result
        w = self._last_weights
        return {
            "shadow_mode":   self._shadow_mode,
            "enabled":       self._enabled,
            "n_dof":         self._n_dof,
            "last_converged": r.converged if r else None,
            "last_iters":    r.iters if r else None,
            "last_cost":     r.cost  if r else None,
            "last_hash":     r.solution_hash if r else None,
            "weights": {
                "balance": w.balance if w else None,
                "foot":    w.foot    if w else None,
                "look":    w.look    if w else None,
                "brace":   w.brace   if w else None,
                "grasp":   w.grasp   if w else None,
            },
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _fallback_output(
        self,
        solver_iters:  int = 0,
        solution_hash: str = "",
        task_weights:  Optional[TaskWeights] = None,
    ) -> WBOutput:
        return WBOutput(
            joint_accelerations=[0.0] * self._n_dof,
            converged=False,
            used_fallback=True,
            solver_iters=solver_iters,
            solution_hash=solution_hash,
            task_weights=task_weights,
        )
