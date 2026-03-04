"""VentDetector — Stage 67 vent activation detector.

Inspects a :class:`SubsurfaceFieldGrid` tile and decides whether a vent
event should be triggered based on field thresholds.

Activation condition (§2.2):
  magmaPressureProxy >= vent_threshold
  AND crustWeakness  >= vent_threshold
  AND subsurfaceStress >= vent_threshold * 0.8

Public API
----------
VentDetector(config=None)
  .should_activate(tile) → bool
  .activation_intensity(tile) → float  [0, 1]
"""
from __future__ import annotations

from typing import Optional

from src.subsurface.SubsurfaceFieldGrid import SubsurfaceTile


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


class VentDetector:
    """Decide whether a subsurface tile should produce a vent event.

    Parameters
    ----------
    config :
        Optional dict; reads ``subsurface67.*`` keys.
    """

    _DEFAULT_THRESHOLD = 0.6

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = (config or {}).get("subsurface67", {}) or {}
        self._threshold = float(cfg.get("vent_threshold", self._DEFAULT_THRESHOLD))

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def should_activate(self, tile: SubsurfaceTile) -> bool:
        """Return ``True`` if this tile should spawn a vent event."""
        return (
            tile.magmaPressureProxy  >= self._threshold
            and tile.crustWeakness   >= self._threshold
            and tile.subsurfaceStress >= self._threshold * 0.8
        )

    def activation_intensity(self, tile: SubsurfaceTile) -> float:
        """Return vent intensity [0, 1] based on excess above threshold.

        Returns 0.0 if the activation condition is not met.
        """
        if not self.should_activate(tile):
            return 0.0
        excess = (
            (tile.magmaPressureProxy  - self._threshold)
            + (tile.crustWeakness     - self._threshold)
            + (tile.subsurfaceStress  - self._threshold * 0.8)
        )
        return _clamp(excess / (3.0 * (1.0 - self._threshold) + 1e-9))
