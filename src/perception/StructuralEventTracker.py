"""StructuralEventTracker — Stage 55 structural salience sub-field.

Tracks recent instability events (microfracture cascades, local collapses)
and decays them over time to produce ``structuralSalience`` [0..1].

Each call to :meth:`register_event` injects a salience impulse.  Between
events the salience decays exponentially so that a calm period drives it
toward zero.

Public API
----------
StructuralEventTracker(config=None)
  .register_event(magnitude)       → None
  .update(dt)                      → None
  .structural_salience             → float
"""
from __future__ import annotations

import math
from typing import Optional


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


class StructuralEventTracker:
    """Tracks instability events and decays structural salience.

    Parameters
    ----------
    config :
        Optional dict; reads ``salience.*`` keys.
    """

    _DEFAULT_DECAY_TAU  = 3.0   # seconds — structural events linger
    _DEFAULT_IMPULSE_K  = 0.80  # impulse strength per event (clamped to 1)

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = (config or {}).get("salience", {}) or {}
        tau = float(cfg.get("structural_decay_tau", self._DEFAULT_DECAY_TAU))
        self._tau:       float = max(1e-3, tau)
        self._impulse_k: float = float(
            cfg.get("structural_impulse_k", self._DEFAULT_IMPULSE_K)
        )

        self._structural_salience: float = 0.0

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def register_event(self, magnitude: float = 1.0) -> None:
        """Inject a salience impulse for a structural event.

        Parameters
        ----------
        magnitude :
            Normalised event magnitude [0..1].  1.0 = maximum event.
        """
        impulse = _clamp(magnitude, 0.0, 1.0) * self._impulse_k
        self._structural_salience = _clamp(
            self._structural_salience + impulse, 0.0, 1.0
        )

    def update(self, dt: float = 1.0 / 20.0) -> None:
        """Decay structural salience toward zero.

        Parameters
        ----------
        dt :
            Elapsed simulation time [s].
        """
        decay = math.exp(-dt / self._tau)
        self._structural_salience *= decay

    @property
    def structural_salience(self) -> float:
        """Current structural salience [0..1]."""
        return self._structural_salience
