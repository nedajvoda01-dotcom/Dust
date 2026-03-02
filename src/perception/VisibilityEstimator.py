"""VisibilityEstimator — Stage 37 visibility perception field.

Produces two scalars:

* ``visibility``  (0..1) — how much the character can see (1 = clear)
* ``contrast``    (0..1) — contrast / ability to distinguish features

Inputs:
* ``dust_density``   — 0..1
* ``fog``            — 0..1
* ``night_factor``   — 0..1 (1 = fully dark)

Public API
----------
VisibilityEstimator(config=None)
  .update(dust_density, fog, night_factor, dt) → None
  .visibility  → float
  .contrast    → float
"""
from __future__ import annotations

import math
from typing import Optional


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


class VisibilityEstimator:
    """Perception sub-field: visibility and contrast.

    Parameters
    ----------
    config :
        Optional dict; reads ``perception.visibility.*`` keys.
    """

    _DEFAULT_VIS_WEIGHT     = 1.0
    _DEFAULT_SMOOTHING_TAU  = 0.5   # seconds — visibility changes slowly

    def __init__(self, config: Optional[dict] = None) -> None:
        pcfg = ((config or {}).get("perception", {}) or {}).get("visibility", {}) or {}
        self._weight: float = float(pcfg.get("weight", self._DEFAULT_VIS_WEIGHT))
        tau = float(
            ((config or {}).get("perception", {}) or {}).get(
                "smoothing_tau_sec", self._DEFAULT_SMOOTHING_TAU
            )
        )
        self._tau: float = max(1e-3, tau)

        self._visibility: float = 1.0
        self._contrast:   float = 1.0

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def update(
        self,
        dust_density: float = 0.0,
        fog:          float = 0.0,
        night_factor: float = 0.0,
        dt:           float = 1.0 / 5.0,
    ) -> None:
        """Recompute visibility this tick.

        Parameters
        ----------
        dust_density :
            0..1 suspended dust density.
        fog :
            0..1 fog opacity.
        night_factor :
            0..1 darkness level (1 = full night).
        dt :
            Elapsed simulation time [s].
        """
        dust_density = _clamp(dust_density, 0.0, 1.0)
        fog          = _clamp(fog,          0.0, 1.0)
        night_factor = _clamp(night_factor, 0.0, 1.0)

        # Visibility: multiplicative combination of obscurants
        raw_vis = (1.0 - dust_density * 0.9) * (1.0 - fog * 0.8) * (1.0 - night_factor * 0.6)
        raw_vis = _clamp(raw_vis * self._weight, 0.0, 1.0)

        # Contrast suffers from dust scatter and night equally
        raw_contrast = _clamp(
            (1.0 - dust_density * 0.7) * (1.0 - night_factor * 0.5),
            0.0, 1.0,
        )

        alpha = 1.0 - math.exp(-dt / self._tau)
        self._visibility = self._visibility + alpha * (raw_vis      - self._visibility)
        self._contrast   = self._contrast   + alpha * (raw_contrast - self._contrast)

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------

    @property
    def visibility(self) -> float:
        """Visibility fraction [0..1]; 1 = perfectly clear."""
        return self._visibility

    @property
    def contrast(self) -> float:
        """Visual contrast fraction [0..1]."""
        return self._contrast
