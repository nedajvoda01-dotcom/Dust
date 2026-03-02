"""CostEvaluator — Stage 38 risk cost function (§4).

Computes a weighted TotalCost and per-component breakdown from a
:class:`PerceptionState`::

    TotalCost =
        w_balance    * balanceRisk               (≈ globalRisk)
      + w_slip       * slipRisk
      + w_visibility * (1 - visibility)
      + w_wind       * windLoad
      + w_vibration  * vibrationLevel
      + w_proximity  * crowding                  (≈ presenceNear)
      + w_uncertainty * audioUrgency

All weights are configurable via ``config["intent"]["w_*"]``.

Public API
----------
CostEvaluator(config=None)
  .evaluate(state) → CostBreakdown

CostBreakdown
  .balance, .slip, .visibility, .wind, .vibration, .proximity,
  .uncertainty, .total
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.perception.PerceptionSystem import PerceptionState


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


@dataclass
class CostBreakdown:
    """Per-component risk cost, as defined in §4 of the Stage 38 spec."""
    balance:     float = 0.0
    slip:        float = 0.0
    visibility:  float = 0.0
    wind:        float = 0.0
    vibration:   float = 0.0
    proximity:   float = 0.0
    uncertainty: float = 0.0

    @property
    def total(self) -> float:
        """Sum of all weighted components."""
        return (
            self.balance
            + self.slip
            + self.visibility
            + self.wind
            + self.vibration
            + self.proximity
            + self.uncertainty
        )


_DEFAULT_WEIGHTS = {
    "w_balance":     1.0,
    "w_slip":        1.0,
    "w_visibility":  1.0,
    "w_wind":        1.0,
    "w_vibration":   1.0,
    "w_proximity":   0.5,
    "w_uncertainty": 0.8,
}


class CostEvaluator:
    """Computes TotalCost and a per-component breakdown from a PerceptionState.

    Parameters
    ----------
    config :
        Optional dict; reads ``intent.w_*`` keys for weight overrides.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = ((config or {}).get("intent", {})) or {}
        w = _DEFAULT_WEIGHTS
        self._w_balance     = float(cfg.get("w_balance",     w["w_balance"]))
        self._w_slip        = float(cfg.get("w_slip",        w["w_slip"]))
        self._w_visibility  = float(cfg.get("w_visibility",  w["w_visibility"]))
        self._w_wind        = float(cfg.get("w_wind",        w["w_wind"]))
        self._w_vibration   = float(cfg.get("w_vibration",   w["w_vibration"]))
        self._w_proximity   = float(cfg.get("w_proximity",   w["w_proximity"]))
        self._w_uncertainty = float(cfg.get("w_uncertainty", w["w_uncertainty"]))

    def evaluate(self, state: PerceptionState) -> CostBreakdown:
        """Return a per-component and total cost for the given PerceptionState.

        Parameters
        ----------
        state :
            Current aggregated perception state from the perception system.

        Returns
        -------
        CostBreakdown
            Weighted per-component costs; ``.total`` is the sum.
        """
        return CostBreakdown(
            balance     = self._w_balance     * _clamp(state.globalRisk,               0.0, 1.0),
            slip        = self._w_slip        * _clamp(state.slipRisk,                 0.0, 1.0),
            visibility  = self._w_visibility  * _clamp(1.0 - state.visibility,         0.0, 1.0),
            wind        = self._w_wind        * _clamp(state.windLoad,                 0.0, 1.0),
            vibration   = self._w_vibration   * _clamp(state.vibrationLevel,           0.0, 1.0),
            proximity   = self._w_proximity   * _clamp(state.presenceNear,             0.0, 1.0),
            uncertainty = self._w_uncertainty * _clamp(state.audioUrgency,             0.0, 1.0),
        )
