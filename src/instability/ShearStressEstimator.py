"""ShearStressEstimator — Stage 52 shear stress field updater.

Reads slope, dustLoad, and memory stressAccumulation to increment
``shearStressField``.  Also applies slow relaxation each tick.

Public API
----------
ShearStressEstimator(config=None)
  .tick(state, slope_map, dust_map, stress_map, dt) → None
"""
from __future__ import annotations

from typing import List, Optional

from src.instability.InstabilityState import InstabilityState


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


class ShearStressEstimator:
    """Updates shearStressField from slope, dust, and memory stress.

    Parameters
    ----------
    config :
        Optional dict; reads ``instability.*`` keys.
    """

    _DEFAULT_GAIN    = 0.15   # rate at which stress builds from inputs
    _DEFAULT_RELAX   = 0.02   # per-second relaxation rate

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = (config or {}).get("instability", {}) or {}
        self._gain:  float = float(cfg.get("shear_gain",  self._DEFAULT_GAIN))
        self._relax: float = float(cfg.get("shear_relax", self._DEFAULT_RELAX))

    def tick(
        self,
        state: InstabilityState,
        slope_map:  Optional[List[float]],
        dust_map:   Optional[List[float]],
        stress_map: Optional[List[float]],
        dt: float,
    ) -> None:
        """Advance shearStressField by *dt* seconds.

        Parameters
        ----------
        state      : InstabilityState to update.
        slope_map  : Per-tile slope [0..1]; None → treated as 0.
        dust_map   : Per-tile dust thickness [0..1]; None → treated as 0.
        stress_map : Per-tile memory stressAccumulation [0..1]; None → 0.
        dt         : Elapsed time in seconds.
        """
        n = state.size()
        for i in range(n):
            slope  = slope_map[i]  if slope_map  is not None else 0.0
            dust   = dust_map[i]   if dust_map   is not None else 0.0
            stress = stress_map[i] if stress_map is not None else 0.0

            # Combined driving signal: slope × dust contribution + stress bias
            driver = slope * dust * 0.7 + stress * 0.3
            gain = _clamp(driver) * self._gain * dt
            relax = state.shearStressField[i] * self._relax * dt

            state.shearStressField[i] = _clamp(
                state.shearStressField[i] + gain - relax
            )
