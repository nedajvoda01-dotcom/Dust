"""JointLoadEstimator — Stage 48 joint-load extraction from physics outputs.

Converts raw physics quantities (joint torques, angular velocities, contact
impulses, grasp constraint forces) into normalised :class:`LoadInput` used
by :class:`~src.injury.InjurySystem.InjurySystem`.

The estimator runs a short peak-tracking window (0.5–2 s) and emits
time-averaged and peak values at the injury tick rate.

Public API
----------
PhysicsFrame (dataclass)   — one physics step worth of raw data
JointLoadEstimator(config=None)
  .update(dt, frame)   — ingest one physics frame (runs at physics rate)
  .flush()             → LoadInput  — drain the current window
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from src.injury.InjurySystem import JointLoad, LoadInput, JOINT_NAMES


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


# ---------------------------------------------------------------------------
# PhysicsFrame — one physics step of raw data
# ---------------------------------------------------------------------------

@dataclass
class PhysicsFrame:
    """Raw physics quantities for one simulation step.

    Attributes
    ----------
    joint_tau :
        Per-joint torque magnitude, normalised [0..1] relative to nominal max.
    joint_omega :
        Per-joint angular velocity magnitude, normalised [0..1].
    impact_impulse :
        Per-joint contact impulse this step, normalised [0..1].
    grasp_force :
        Per-joint grasp constraint force peak this step, normalised [0..1].
    """
    joint_tau:      Dict[str, float] = field(default_factory=dict)
    joint_omega:    Dict[str, float] = field(default_factory=dict)
    impact_impulse: Dict[str, float] = field(default_factory=dict)
    grasp_force:    Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# JointLoadEstimator
# ---------------------------------------------------------------------------

class JointLoadEstimator:
    """Accumulates physics frames and produces :class:`LoadInput` snapshots.

    Parameters
    ----------
    config :
        Optional dict; reads ``injury.*`` keys.
    """

    _DEFAULT_WINDOW_SEC = 1.0   # peak-tracking window length [s]
    _DEFAULT_TAU_MAX    = 1.0   # normalised torque max (physics already normalised)

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg  = config or {}
        icfg = cfg.get("injury", {}) or {}

        self._window_sec = float(icfg.get("load_window_sec", self._DEFAULT_WINDOW_SEC))
        self._tau_max    = float(icfg.get("tau_max",         self._DEFAULT_TAU_MAX))

        # Accumulators per joint
        self._sum_tau:     Dict[str, float] = {n: 0.0 for n in JOINT_NAMES}
        self._peak_tau:    Dict[str, float] = {n: 0.0 for n in JOINT_NAMES}
        self._peak_imp:    Dict[str, float] = {n: 0.0 for n in JOINT_NAMES}
        self._peak_grasp:  Dict[str, float] = {n: 0.0 for n in JOINT_NAMES}
        self._sum_omega:   Dict[str, float] = {n: 0.0 for n in JOINT_NAMES}
        self._window_time: float = 0.0
        self._n_frames:    int   = 0

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def update(self, dt: float, frame: PhysicsFrame) -> None:
        """Ingest one physics frame.

        Parameters
        ----------
        dt :
            Physics step size [s].
        frame :
            Raw joint loads this step.
        """
        self._window_time += dt
        self._n_frames    += 1

        for name in JOINT_NAMES:
            tau   = _clamp(frame.joint_tau.get(name, 0.0),      0.0, 1.0)
            omega = _clamp(frame.joint_omega.get(name, 0.0),    0.0, 1.0)
            imp   = _clamp(frame.impact_impulse.get(name, 0.0), 0.0, 1.0)
            gf    = _clamp(frame.grasp_force.get(name, 0.0),    0.0, 1.0)

            self._sum_tau[name]   += tau * dt
            self._sum_omega[name] += omega * dt
            self._peak_tau[name]  = max(self._peak_tau[name],  tau)
            self._peak_imp[name]  = max(self._peak_imp[name],  imp)
            self._peak_grasp[name] = max(self._peak_grasp[name], gf)

    def flush(self) -> LoadInput:
        """Drain the current window and return a :class:`LoadInput`.

        Resets internal accumulators after draining.

        Returns
        -------
        LoadInput
        """
        wt = max(self._window_time, 1e-6)
        joints: Dict[str, JointLoad] = {}
        for name in JOINT_NAMES:
            avg_tau = self._sum_tau[name] / wt
            joints[name] = JointLoad(
                tau=avg_tau,
                tau_max=self._tau_max,
                omega=self._sum_omega[name] / wt,
                impactImpulse=self._peak_imp[name],
                graspForcePeak=self._peak_grasp[name],
            )
        self._reset()
        return LoadInput(joints=joints)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _reset(self) -> None:
        for name in JOINT_NAMES:
            self._sum_tau[name]    = 0.0
            self._sum_omega[name]  = 0.0
            self._peak_tau[name]   = 0.0
            self._peak_imp[name]   = 0.0
            self._peak_grasp[name] = 0.0
        self._window_time = 0.0
        self._n_frames    = 0
