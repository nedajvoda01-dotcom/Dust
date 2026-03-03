"""CrustFailureModel — Stage 52 brittle crust collapse logic.

Reads ``crustFailurePotential`` from InstabilityState.  When a tile exceeds
the configured ``crust_threshold`` the model fires a collapse event:

* crustHardness in SurfaceMaterialState is reduced (via adapter).
* roughness is increased locally.
* crustFailurePotential is partially discharged.

Public API
----------
CrustFailureModel(config=None)
  .process(state, tile) → CrustFailureEvent | None
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.instability.InstabilityState import InstabilityState


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


@dataclass
class CrustFailureEvent:
    """Fired when crustFailurePotential exceeds threshold.

    Attributes
    ----------
    tile           : Flat tile index.
    intensity      : Discharge energy (field value above threshold).
    crust_delta    : Suggested reduction in crust_hardness [0..1].
    roughness_gain : Suggested increase in roughness [0..1].
    """
    tile:           int
    intensity:      float
    crust_delta:    float
    roughness_gain: float


class CrustFailureModel:
    """Detects and processes crust failure events.

    Parameters
    ----------
    config :
        Optional dict; reads ``instability.*`` keys.
    """

    _DEFAULT_THRESHOLD    = 0.65
    _DEFAULT_DISCHARGE_K  = 0.8   # fraction of excess energy discharged
    _DEFAULT_CRUST_K      = 0.4   # fraction converted to crust damage
    _DEFAULT_ROUGH_K      = 0.2   # fraction converted to roughness gain

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = (config or {}).get("instability", {}) or {}
        self._threshold:   float = float(cfg.get("crust_threshold",   self._DEFAULT_THRESHOLD))
        self._discharge_k: float = float(cfg.get("crust_discharge_k", self._DEFAULT_DISCHARGE_K))
        self._crust_k:     float = float(cfg.get("energy_to_material_k", self._DEFAULT_CRUST_K))
        self._rough_k:     float = float(cfg.get("crust_rough_k",     self._DEFAULT_ROUGH_K))

    def process(
        self,
        state: InstabilityState,
        tile:  int,
    ) -> Optional[CrustFailureEvent]:
        """Test tile for crust failure and discharge if above threshold.

        Parameters
        ----------
        state : InstabilityState (modified in-place on discharge).
        tile  : Flat tile index to test.

        Returns
        -------
        CrustFailureEvent if a collapse occurred, else None.
        """
        potential = state.crustFailurePotential[tile]
        if potential <= self._threshold:
            return None

        excess = potential - self._threshold
        discharge = excess * self._discharge_k

        state.crustFailurePotential[tile] = _clamp(potential - discharge)

        return CrustFailureEvent(
            tile=tile,
            intensity=discharge,
            crust_delta=discharge * self._crust_k,
            roughness_gain=discharge * self._rough_k,
        )
