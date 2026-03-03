"""ErosionBiasSystem — Stage 51 directional erosion memory.

When a region has high slope, frequent movement, and high wind channelling,
an erosion bias accumulates.  This accelerates slopeCreep (Stage 50) and
material mass redistribution.

The erosion bias is the memory of "this place gets worn away faster".

Model
-----
Per tick, for tiles where conditions are met::

    if slope > slope_threshold AND (movement OR wind_channel > wind_threshold):
        erosionBias += k_erosion_gain * slope * wind_channel * dt

Passive decay::

    erosionBias -= k_erosion_decay * dt

Effect:
  * high erosionBias → MemoryToEvolutionAdapter boosts slopeCreep
  * high erosionBias → dustReservoir redistribution accelerated

Config keys (under ``memory.*``)
----------------------------------
k_erosion_gain     : float — bias gain rate                     (default 0.05)
k_erosion_decay    : float — passive decay rate                 (default 0.002)
slope_threshold    : float — minimum slope to trigger erosion   (default 0.3)
wind_threshold     : float — minimum wind for wind-driven erosion (default 0.4)
max_tiles_per_tick : int   — budget                             (default 256)

Public API
----------
ErosionBiasSystem(config=None)
  .tick(memory_state, slope_map, wind_map, movement_map, dt) -> None
"""
from __future__ import annotations

from typing import List, Optional

from src.memory.WorldMemoryState import WorldMemoryState


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


class ErosionBiasSystem:
    """Advances ``erosionBiasField`` in :class:`WorldMemoryState`.

    Parameters
    ----------
    config : dict or None
        See module docstring for keys.
    """

    _DEFAULTS = {
        "k_erosion_gain":     0.05,
        "k_erosion_decay":    0.002,
        "slope_threshold":    0.3,
        "wind_threshold":     0.4,
        "max_tiles_per_tick": 256,
    }

    def __init__(self, config=None) -> None:
        cfg = dict(self._DEFAULTS)
        if isinstance(config, dict):
            src = config.get("memory", config)
            for k in self._DEFAULTS:
                if k in src:
                    cfg[k] = src[k]

        self._k_gain:      float = float(cfg["k_erosion_gain"])
        self._k_decay:     float = float(cfg["k_erosion_decay"])
        self._slope_thr:   float = float(cfg["slope_threshold"])
        self._wind_thr:    float = float(cfg["wind_threshold"])
        self._budget:      int   = int(cfg["max_tiles_per_tick"])
        self._cursor:      int   = 0

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def tick(
        self,
        memory_state: WorldMemoryState,
        slope_map:    Optional[List[float]],
        wind_map:     Optional[List[float]],
        movement_map: Optional[List[float]],
        dt:           float,
    ) -> None:
        """Advance erosion bias for a budget slice of tiles.

        Parameters
        ----------
        memory_state  : WorldMemoryState to update in-place.
        slope_map     : Per-tile slope magnitude [0, 1].
        wind_map      : Per-tile wind channel [0, 1].
        movement_map  : Per-tile recent movement pressure [0, 1].
        dt            : Time step (seconds).
        """
        if dt <= 0.0:
            return

        n = memory_state.size()
        eb = memory_state.erosionBiasField

        count = min(self._budget, n)
        for _ in range(count):
            idx = self._cursor % n
            self._cursor = (self._cursor + 1) % n

            slope    = slope_map[idx]    if slope_map    else 0.0
            wind     = wind_map[idx]     if wind_map     else 0.0
            movement = movement_map[idx] if movement_map else 0.0

            # Passive decay
            decay = self._k_decay
            eb[idx] = _clamp(eb[idx] - decay * dt)

            # Gain only when slope is sufficient and there is wind or movement
            if slope < self._slope_thr:
                continue
            if wind < self._wind_thr and movement <= 0.0:
                continue

            gain = self._k_gain * slope * max(wind, movement) * dt
            eb[idx] = _clamp(eb[idx] + gain)
