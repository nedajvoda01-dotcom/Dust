"""ColdBiasEstimator — Stage 49 local cold-bias proxy.

Estimates how much colder a location tends to be relative to the macro
temperature, based on:

* Shadow factor — ring / mountain shadow reduces insolation → colder.
* Topographic lowness — cold air pools in basins (optional height bias).

The result feeds into :class:`~src.microclimate.LocalClimateComposer.LocalClimateComposer`
and ultimately accelerates ice-film formation (Stage 45) and may impair
joint flexibility (Stage 48).

Public API
----------
ColdBiasEstimator(config=None)
  .estimate(insolation, height_above_basin) → float   (coldBias 0..1)
"""
from __future__ import annotations

from typing import Optional


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


class ColdBiasEstimator:
    """Computes cold-bias from insolation and topographic position.

    Parameters
    ----------
    config :
        Optional dict; reads ``micro.coldbias.*`` keys.
    """

    _DEFAULT_COLD_DELTA         = 0.3   # max temperature offset (normalised)
    _DEFAULT_HEIGHT_COLD_K      = 0.1   # cold per unit of relative height above basin

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg  = config or {}
        mcfg = cfg.get("micro", {}) or {}
        bcfg = mcfg.get("coldbias", {}) or {}

        self._cold_delta:    float = float(bcfg.get("cold_delta",    self._DEFAULT_COLD_DELTA))
        self._height_cold_k: float = float(bcfg.get("height_cold_k", self._DEFAULT_HEIGHT_COLD_K))

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def estimate(
        self,
        insolation:         float,
        height_above_basin: float = 0.0,
    ) -> float:
        """Compute coldBias proxy.

        Parameters
        ----------
        insolation :
            Local insolation [0 = full shadow, 1 = full sun].
        height_above_basin :
            How high the position is above the local basin floor [metres].
            Positive = above basin (warmer); negative or 0 = in basin (colder).

        Returns
        -------
        float
            coldBias in [0, 1].  1.0 = maximally cold.
        """
        # Shadow component: full shadow → max cold bias
        shadow_cold = _clamp(1.0 - insolation)

        # Basin / valley cold-air pooling: lower is colder
        # height_above_basin > 0 means on a slope (warmer), reduce bias
        height_warm = _clamp(height_above_basin * self._height_cold_k)
        basin_cold = _clamp(1.0 - height_warm)

        # Combine: shadow dominates; basin adds secondary effect
        raw = shadow_cold * 0.7 + basin_cold * 0.3
        return _clamp(raw)
