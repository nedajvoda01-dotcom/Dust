"""SpatialEmitter — Stage 36 distance-based attenuation and low-pass.

Applies two perceptual effects based on the distance from the listener to an
audio source:

1. **Inverse-square attenuation** — amplitude ∝ 1 / max(1, distance²).
2. **Distance low-pass** — the further away, the more high-frequency content
   is attenuated (simulates air absorption).

The filter is a simple one-pole IIR approximation identical to
``BiquadFilter`` (lp mode) in ``ProceduralAudioSystem``.

Public API
----------
SpatialEmitter(ref_distance=1.0, max_distance=500.0)
  .attenuate(sample, distance) → float
  .process(sample, distance)   → float   (attenuation + LP filter)
"""
from __future__ import annotations

import math
from typing import Optional


class SpatialEmitter:
    """Compute distance-based gain and filter for a mono audio sample.

    Parameters
    ----------
    ref_distance :
        Distance at which gain equals 1.0 [m or normalised units].
    max_distance :
        Beyond this distance the signal is completely inaudible.
    """

    # Low-pass cutoff as a function of distance:
    #   cutoff_norm = cutoff_near * exp(-distance / distance_lp_scale)
    _CUTOFF_NEAR      = 0.40    # normalised (≈ Nyquist/2 proxy)
    _DISTANCE_LP_SCALE = 100.0  # characteristic scale for LP roll-off

    def __init__(
        self,
        ref_distance: float = 1.0,
        max_distance: float = 500.0,
    ) -> None:
        self._ref_dist = max(1e-3, ref_distance)
        self._max_dist = max_distance
        self._lp_state: float = 0.0   # one-pole LP state

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def attenuate(self, sample: float, distance: float) -> float:
        """Apply inverse-square gain only (no filter)."""
        gain = self._gain(distance)
        return sample * gain

    def process(self, sample: float, distance: float) -> float:
        """Apply inverse-square gain AND distance low-pass filter."""
        gain   = self._gain(distance)
        sample = sample * gain
        alpha  = self._lp_alpha(distance)
        self._lp_state = self._lp_state + alpha * (sample - self._lp_state)
        return self._lp_state

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _gain(self, distance: float) -> float:
        if distance >= self._max_dist:
            return 0.0
        d = max(self._ref_dist, distance)
        return (self._ref_dist / d) ** 2

    def _lp_alpha(self, distance: float) -> float:
        """Low-pass coefficient: approaches 0 (fully muted) at long distances."""
        return self._CUTOFF_NEAR * math.exp(
            -max(0.0, distance) / self._DISTANCE_LP_SCALE
        )
