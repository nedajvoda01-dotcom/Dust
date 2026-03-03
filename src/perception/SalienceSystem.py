"""SalienceSystem — Stage 55 Perceptual Meaning Emergence.

Aggregates physical signals into a :class:`PerceptualState` that downstream
subsystems (camera, motor, audio) use to react subtly without imposing
narrative or UI.  "Meaning" emerges from physics, not from script.

Architecture
------------
::

    RiskEstimator        ─┐
    ScaleEstimator       ─┤
    StructuralEventTracker─┼─► SalienceSystem ─► PerceptualState
    acoustic_anomaly     ─┤     (weighted sum, smoothed)
    wind_gust_variance   ─┘

All salience scalars are in [0..1].  The system applies a secondary
exponential smoothing pass on ``globalSalience`` to prevent jitter.

Public API
----------
PerceptualState (dataclass)
SalienceEnv     (dataclass)  — caller fills in; missing → neutral
SalienceSystem(config=None)
  .update(dt, env)  → PerceptualState
  .register_structural_event(magnitude)  → None
  .debug_info()     → dict
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from src.perception.RiskEstimator          import RiskEstimator
from src.perception.ScaleEstimator         import ScaleEstimator
from src.perception.StructuralEventTracker import StructuralEventTracker


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


# ---------------------------------------------------------------------------
# SalienceEnv — caller-provided snapshot
# ---------------------------------------------------------------------------

@dataclass
class SalienceEnv:
    """Environmental inputs for one salience update tick.

    All fields are optional; unset fields fall back to neutral (0).
    """
    # --- Risk inputs (from Stage 37 PerceptionState / Stage 52)
    slip_risk:             float = 0.0
    vibration_level:       float = 0.0
    instability_proximity: float = 0.0

    # --- Scale inputs (from Stage 29–33 astro / terrain)
    fov_scale_metric:   float = 0.0   # 0=enclosed, 1=vast
    sun_alignment:      float = 0.0   # eclipse proximity [0..1]
    horizon_curvature:  float = 0.0   # [0..1]

    # --- Environmental / acoustic inputs (from Stage 46)
    acoustic_anomaly:   float = 0.0   # infrasound / unusual echo [0..1]
    wind_gust_variance: float = 0.0   # gustiness variance [0..1]


# ---------------------------------------------------------------------------
# PerceptualState — output struct
# ---------------------------------------------------------------------------

@dataclass
class PerceptualState:
    """Aggregated perceptual salience outputs.

    All scalars are in [0..1].  Higher = more perceptually significant.
    """
    environmentalSalience: float = 0.0
    structuralSalience:    float = 0.0
    motionSalience:        float = 0.0
    riskSalience:          float = 0.0
    scaleSalience:         float = 0.0
    globalSalience:        float = 0.0


# ---------------------------------------------------------------------------
# SalienceSystem
# ---------------------------------------------------------------------------

class SalienceSystem:
    """Orchestrates perceptual salience sub-fields.

    Parameters
    ----------
    config :
        Optional dict; reads ``salience.*`` keys.
    """

    _DEFAULT_TICK_HZ    = 20.0
    _DEFAULT_TAU        = 0.25   # global smoothing tau [s]

    _DEFAULT_RISK_W     = 0.35
    _DEFAULT_SCALE_W    = 0.25
    _DEFAULT_STRUCT_W   = 0.20
    _DEFAULT_ENV_W      = 0.20

    def __init__(self, config: Optional[dict] = None) -> None:
        self._cfg = config or {}
        cfg = self._cfg.get("salience", {}) or {}

        self._enabled: bool = bool(cfg.get("enable", True))

        # Tick rate
        hz = float(cfg.get("tick_hz", self._DEFAULT_TICK_HZ))
        self._dt_tick: float = 1.0 / max(1.0, hz)

        # Global smoothing tau
        tau = float(cfg.get("smoothing_tau", self._DEFAULT_TAU))
        self._tau: float = max(1e-3, tau)

        # Weights for weighted sum → globalSalience
        self._risk_w:   float = float(cfg.get("risk_weight",         self._DEFAULT_RISK_W))
        self._scale_w:  float = float(cfg.get("scale_weight",        self._DEFAULT_SCALE_W))
        self._struct_w: float = float(cfg.get("structural_weight",   self._DEFAULT_STRUCT_W))
        self._env_w:    float = float(cfg.get("environmental_weight",self._DEFAULT_ENV_W))

        # Sub-estimators
        self._risk_est    = RiskEstimator(config)
        self._scale_est   = ScaleEstimator(config)
        self._struct_est  = StructuralEventTracker(config)

        # Current state
        self._state = PerceptualState()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def update(self, dt: float, env: SalienceEnv) -> PerceptualState:
        """Advance the salience system and return updated PerceptualState.

        Parameters
        ----------
        dt :
            Elapsed simulation time since last call [s].
        env :
            Environmental snapshot from the caller.
        """
        if not self._enabled:
            return self._state

        dt = max(1e-6, dt)

        # 1. Update sub-estimators
        self._risk_est.update(
            slip_risk=env.slip_risk,
            vibration_level=env.vibration_level,
            instability_proximity=env.instability_proximity,
            dt=dt,
        )

        self._scale_est.update(
            fov_scale_metric=env.fov_scale_metric,
            sun_alignment=env.sun_alignment,
            horizon_curvature=env.horizon_curvature,
            dt=dt,
        )

        self._struct_est.update(dt=dt)

        # 2. Environmental salience: acoustic anomaly + wind gust variance
        acoustic = _clamp(env.acoustic_anomaly, 0.0, 1.0)
        gust     = _clamp(env.wind_gust_variance, 0.0, 1.0)
        env_raw  = _clamp(acoustic * 0.6 + gust * 0.4, 0.0, 1.0)

        # 3. Motion salience: currently driven by vibration (proxy for motion)
        motion_raw = _clamp(env.vibration_level * 0.5 + env.slip_risk * 0.5, 0.0, 1.0)

        # 4. Apply global smoothing to environmental and motion salience
        alpha = 1.0 - math.exp(-dt / self._tau)
        env_sal    = self._state.environmentalSalience + alpha * (env_raw - self._state.environmentalSalience)
        motion_sal = self._state.motionSalience         + alpha * (motion_raw - self._state.motionSalience)

        risk_sal   = self._risk_est.risk_salience
        scale_sal  = self._scale_est.scale_salience
        struct_sal = self._struct_est.structural_salience

        # 5. Weighted sum → globalSalience
        w_sum = self._risk_w + self._scale_w + self._struct_w + self._env_w
        global_raw = _clamp(
            (
                risk_sal   * self._risk_w
                + scale_sal  * self._scale_w
                + struct_sal * self._struct_w
                + env_sal    * self._env_w
            ) / max(1e-6, w_sum),
            0.0, 1.0,
        )

        # Secondary smoothing on globalSalience
        global_sal = (
            self._state.globalSalience
            + alpha * (global_raw - self._state.globalSalience)
        )

        self._state = PerceptualState(
            environmentalSalience=env_sal,
            structuralSalience=struct_sal,
            motionSalience=motion_sal,
            riskSalience=risk_sal,
            scaleSalience=scale_sal,
            globalSalience=global_sal,
        )
        return self._state

    def register_structural_event(self, magnitude: float = 1.0) -> None:
        """Notify the system of a structural instability event.

        Parameters
        ----------
        magnitude :
            Normalised event severity [0..1].
        """
        self._struct_est.register_event(magnitude)

    def debug_info(self) -> dict:
        """Return current salience values for logging."""
        s = self._state
        return {
            "riskSalience":          round(s.riskSalience,          4),
            "scaleSalience":         round(s.scaleSalience,         4),
            "structuralSalience":    round(s.structuralSalience,    4),
            "environmentalSalience": round(s.environmentalSalience, 4),
            "motionSalience":        round(s.motionSalience,        4),
            "globalSalience":        round(s.globalSalience,        4),
        }
