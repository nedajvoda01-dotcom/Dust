"""ScaleEstimator — Stage 55 scale salience sub-field.

Estimates ``scaleSalience`` [0..1] from:
* ``fov_scale_metric``  — normalised measure of how large the scene feels
                          (e.g. wide crater / open plain; 0=enclosed, 1=vast).
* ``sun_alignment``     — how close the binary star system is to eclipse
                          configuration [0..1] (from Stage 29–31 astro).
* ``horizon_curvature`` — visible horizon curvature accentuating scale
                          [0..1] (0 = flat / obstructed, 1 = full curve).

All inputs are expected in [0..1].

Public API
----------
ScaleEstimator(config=None)
  .update(fov_scale_metric, sun_alignment, horizon_curvature, dt) → None
  .scale_salience  → float
"""
from __future__ import annotations

import math
from typing import Optional


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


class ScaleEstimator:
    """Computes scaleSalience from visual-scale signals.

    Parameters
    ----------
    config :
        Optional dict; reads ``salience.scale_*`` keys.
    """

    _DEFAULT_FOV_WEIGHT       = 0.40
    _DEFAULT_SUN_WEIGHT       = 0.35
    _DEFAULT_HORIZON_WEIGHT   = 0.25
    _DEFAULT_SMOOTHING_TAU    = 0.40  # seconds (slower — scale is ambient)

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = (config or {}).get("salience", {}) or {}
        self._fov_w:     float = float(cfg.get("scale_fov_weight",     self._DEFAULT_FOV_WEIGHT))
        self._sun_w:     float = float(cfg.get("scale_sun_weight",     self._DEFAULT_SUN_WEIGHT))
        self._horizon_w: float = float(cfg.get("scale_horizon_weight", self._DEFAULT_HORIZON_WEIGHT))
        tau = float(cfg.get("smoothing_tau", self._DEFAULT_SMOOTHING_TAU))
        self._tau: float = max(1e-3, tau)

        self._scale_salience: float = 0.0

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def update(
        self,
        fov_scale_metric: float = 0.0,
        sun_alignment: float = 0.0,
        horizon_curvature: float = 0.0,
        dt: float = 1.0 / 20.0,
    ) -> None:
        """Advance the scale estimator one tick.

        Parameters
        ----------
        fov_scale_metric :
            Normalised scene vastness [0..1].
        sun_alignment :
            Binary star eclipse proximity [0..1].
        horizon_curvature :
            Visible horizon curvature [0..1].
        dt :
            Elapsed simulation time [s].
        """
        f = _clamp(fov_scale_metric, 0.0, 1.0)
        s = _clamp(sun_alignment, 0.0, 1.0)
        h = _clamp(horizon_curvature, 0.0, 1.0)

        raw = _clamp(
            f * self._fov_w + s * self._sun_w + h * self._horizon_w,
            0.0, 1.0,
        )

        alpha = 1.0 - math.exp(-dt / self._tau)
        self._scale_salience += alpha * (raw - self._scale_salience)

    @property
    def scale_salience(self) -> float:
        """Current scale salience [0..1]."""
        return self._scale_salience
