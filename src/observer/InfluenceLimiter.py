"""InfluenceLimiter — Stage 53 per-tile player influence cap.

Ensures the player cannot accumulate unlimited influence on any single tile
or destabilise an otherwise stable zone.

Model
-----
Each tile tracks how much player influence has already been applied this
decay window.  When a new contribution arrives the limiter clips it so that
the running total for that tile does not exceed ``max_influence_per_tile``.

The accumulator decays with time constant ``decay_tau``::

    accumulated[tile] -= accumulated[tile] / decay_tau * dt

This prevents unbounded growth from simply standing still.

Config keys (under ``observer.*``)
-----------------------------------
max_influence_per_tile : float — hard cap per tile   (default 0.05)
decay_tau              : float — decay time constant in seconds (default 30.0)

Public API
----------
InfluenceLimiter(config=None)
  .clip(tile_idx, requested_delta) -> float
      Return the clipped delta that fits within the remaining budget.
  .record(tile_idx, applied_delta) -> None
      Record that *applied_delta* was actually used on *tile_idx*.
  .tick(dt) -> None
      Advance decay on all accumulators.
  .accumulated(tile_idx) -> float
      Current accumulated influence on *tile_idx*.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


class InfluenceLimiter:
    """Per-tile player influence cap with exponential decay.

    Parameters
    ----------
    config :
        Optional dict; reads ``observer.*`` keys.
    """

    _DEFAULTS = {
        "max_influence_per_tile": 0.05,
        "decay_tau":              30.0,
    }

    def __init__(self, config=None) -> None:
        cfg = dict(self._DEFAULTS)
        if isinstance(config, dict):
            src = config.get("observer", config)
            for k in self._DEFAULTS:
                if k in src:
                    cfg[k] = src[k]

        self._max:       float = float(cfg["max_influence_per_tile"])
        self._decay_tau: float = max(float(cfg["decay_tau"]), 1e-3)

        # sparse accumulator: only tiles that have received influence
        self._acc: Dict[int, float] = defaultdict(float)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def clip(self, tile_idx: int, requested_delta: float) -> float:
        """Return the largest delta ≤ *requested_delta* that fits the budget.

        Parameters
        ----------
        tile_idx        : Flat tile index.
        requested_delta : Desired contribution (must be ≥ 0).

        Returns
        -------
        Clipped delta in [0, requested_delta].
        """
        if requested_delta <= 0.0:
            return 0.0
        remaining = max(0.0, self._max - self._acc[tile_idx])
        return min(requested_delta, remaining)

    def record(self, tile_idx: int, applied_delta: float) -> None:
        """Record that *applied_delta* was applied to *tile_idx*.

        Parameters
        ----------
        tile_idx      : Flat tile index.
        applied_delta : Amount actually applied (≥ 0).
        """
        if applied_delta > 0.0:
            self._acc[tile_idx] = _clamp(
                self._acc[tile_idx] + applied_delta, lo=0.0, hi=self._max
            )

    def tick(self, dt: float) -> None:
        """Decay all accumulators.

        Parameters
        ----------
        dt : Elapsed time in seconds.
        """
        if dt <= 0.0 or not self._acc:
            return
        decay_rate = dt / self._decay_tau
        dead: list = []
        for tile, val in self._acc.items():
            new_val = val * (1.0 - decay_rate)
            if new_val < 1e-6:
                dead.append(tile)
            else:
                self._acc[tile] = new_val
        for tile in dead:
            del self._acc[tile]

    def accumulated(self, tile_idx: int) -> float:
        """Return the current accumulated influence on *tile_idx*."""
        return self._acc.get(tile_idx, 0.0)
