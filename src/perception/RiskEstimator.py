"""RiskEstimator — Stage 55 risk salience sub-field.

Estimates ``riskSalience`` [0..1] from:
* ``slipRisk``             — ground slip probability (Stage 37).
* ``vibrationLevel``       — geo-vibration intensity (Stage 37).
* ``instabilityProximity`` — normalised distance to nearest instability event
                             (Stage 52); 0 = far, 1 = at threshold.

All inputs are expected in [0..1].

Public API
----------
RiskEstimator(config=None)
  .update(slip_risk, vibration_level, instability_proximity, dt) → None
  .risk_salience  → float
"""
from __future__ import annotations

import math
from typing import Optional


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


class RiskEstimator:
    """Computes riskSalience from ground-level hazard signals.

    Parameters
    ----------
    config :
        Optional dict; reads ``salience.risk_*`` keys.
    """

    _DEFAULT_SLIP_WEIGHT     = 0.45
    _DEFAULT_VIBR_WEIGHT     = 0.30
    _DEFAULT_INSTAB_WEIGHT   = 0.25
    _DEFAULT_SMOOTHING_TAU   = 0.20  # seconds

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = (config or {}).get("salience", {}) or {}
        self._slip_w:  float = float(cfg.get("risk_slip_weight",  self._DEFAULT_SLIP_WEIGHT))
        self._vibr_w:  float = float(cfg.get("risk_vibr_weight",  self._DEFAULT_VIBR_WEIGHT))
        self._instab_w: float = float(cfg.get("risk_instab_weight", self._DEFAULT_INSTAB_WEIGHT))
        tau = float(cfg.get("smoothing_tau", self._DEFAULT_SMOOTHING_TAU))
        self._tau: float = max(1e-3, tau)

        self._risk_salience: float = 0.0

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def update(
        self,
        slip_risk: float = 0.0,
        vibration_level: float = 0.0,
        instability_proximity: float = 0.0,
        dt: float = 1.0 / 20.0,
    ) -> None:
        """Advance the risk estimator one tick.

        Parameters
        ----------
        slip_risk :
            Ground slip risk [0..1] from GroundStabilityField.
        vibration_level :
            Geo-vibration intensity [0..1] from VibrationField.
        instability_proximity :
            Proximity to nearest instability event [0..1].
        dt :
            Elapsed simulation time [s].
        """
        s = _clamp(slip_risk, 0.0, 1.0)
        v = _clamp(vibration_level, 0.0, 1.0)
        i = _clamp(instability_proximity, 0.0, 1.0)

        raw = _clamp(
            s * self._slip_w + v * self._vibr_w + i * self._instab_w,
            0.0, 1.0,
        )

        alpha = 1.0 - math.exp(-dt / self._tau)
        self._risk_salience += alpha * (raw - self._risk_salience)

    @property
    def risk_salience(self) -> float:
        """Current risk salience [0..1]."""
        return self._risk_salience
