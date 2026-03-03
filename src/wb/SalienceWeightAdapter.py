"""SalienceWeightAdapter — Stage 55 salience-driven motor weight modifier.

Maps a :class:`~src.perception.SalienceSystem.PerceptualState` to
adjustments for the whole-body QP :class:`~src.wb.WeightScheduler.TaskWeights`.

Behaviour (per §4.2):
* ``riskSalience``  ↑  →  balance weight ↑, shorter step bias, braceBias ↑.
* ``scaleSalience`` ↑  →  posture weight slight increase (stands taller),
                           head look dwell bias ↑.

All modifiers are bounded by ``max_motor_bias`` from config.

Public API
----------
MotorBias (dataclass)
SalienceWeightAdapter(config=None)
  .compute(perceptual_state) → MotorBias
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.perception.SalienceSystem import PerceptualState


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


# ---------------------------------------------------------------------------
# MotorBias
# ---------------------------------------------------------------------------

@dataclass
class MotorBias:
    """Salience-derived motor weight adjustments.

    Attributes
    ----------
    balance_bias :
        Additional balance task weight (additive).
    brace_bias :
        Additional brace task weight (additive).
    step_length_scale :
        Multiplier on step length (< 1 = shorter steps).
    posture_bias :
        Subtle upright-posture encouragement [0..1].
    head_dwell_bias :
        Extra head-look dwell time multiplier [0..1].
    """
    balance_bias:       float = 0.0
    brace_bias:         float = 0.0
    step_length_scale:  float = 1.0
    posture_bias:       float = 0.0
    head_dwell_bias:    float = 0.0


# ---------------------------------------------------------------------------
# SalienceWeightAdapter
# ---------------------------------------------------------------------------

class SalienceWeightAdapter:
    """Maps PerceptualState → MotorBias.

    Parameters
    ----------
    config :
        Optional dict; reads ``salience.*`` keys.
    """

    _DEFAULT_MAX_MOTOR_BIAS     = 0.5
    _DEFAULT_RISK_BALANCE       = 0.8   # additive at full risk
    _DEFAULT_RISK_BRACE         = 0.6   # additive at full risk
    _DEFAULT_RISK_STEP_SCALE    = 0.6   # minimum step-length scale at full risk
    _DEFAULT_SCALE_POSTURE      = 0.4   # posture bias at full scaleSalience
    _DEFAULT_SCALE_HEAD_DWELL   = 0.5   # head dwell at full scaleSalience

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = (config or {}).get("salience", {}) or {}
        self._max_bias: float = float(cfg.get("max_motor_bias", self._DEFAULT_MAX_MOTOR_BIAS))

        self._risk_balance:    float = float(cfg.get("motor_risk_balance",    self._DEFAULT_RISK_BALANCE))
        self._risk_brace:      float = float(cfg.get("motor_risk_brace",      self._DEFAULT_RISK_BRACE))
        self._risk_step_scale: float = float(cfg.get("motor_risk_step_scale", self._DEFAULT_RISK_STEP_SCALE))
        self._scale_posture:   float = float(cfg.get("motor_scale_posture",   self._DEFAULT_SCALE_POSTURE))
        self._scale_dwell:     float = float(cfg.get("motor_scale_dwell",     self._DEFAULT_SCALE_HEAD_DWELL))

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def compute(self, perceptual_state: PerceptualState) -> MotorBias:
        """Compute motor biases from the current perceptual state.

        Parameters
        ----------
        perceptual_state :
            Current :class:`~src.perception.SalienceSystem.PerceptualState`.
        """
        risk  = _clamp(perceptual_state.riskSalience,  0.0, 1.0)
        scale = _clamp(perceptual_state.scaleSalience, 0.0, 1.0)

        balance_bias    = _clamp(risk * self._risk_balance,           0.0, self._max_bias)
        brace_bias      = _clamp(risk * self._risk_brace,             0.0, self._max_bias)
        step_len_scale  = _clamp(1.0 - risk * (1.0 - self._risk_step_scale), 0.0, 1.0)
        posture_bias    = _clamp(scale * self._scale_posture,         0.0, self._max_bias)
        head_dwell_bias = _clamp(scale * self._scale_dwell,           0.0, self._max_bias)

        return MotorBias(
            balance_bias=balance_bias,
            brace_bias=brace_bias,
            step_length_scale=step_len_scale,
            posture_bias=posture_bias,
            head_dwell_bias=head_dwell_bias,
        )
