"""StressAccumulator — Stage 51 geo-mechanical stress accumulation.

Accumulates stress in the WorldMemoryState.stressAccumulationField from
player contact forces (steps, falls, grasp-yanks, repeated slips).

Model
-----
Per contact event at a tile::

    stress += k_stress * contact_force * dt

When stress > stress_threshold the tile becomes mechanically weakened —
this is communicated to external systems via :meth:`overstressed_tiles`.

Decay (memory fades)::

    stress -= k_stress_relax * dt

Additional decay modifiers:
  * high wind_speed   → faster decay
  * seasonal (phase)  → minor modulation
  * steep slope       → faster decay (mass creep disperses stress)

Config keys (under ``memory.*``)
----------------------------------
k_stress          : float — stress gain per unit force per second   (default 0.1)
k_stress_relax    : float — passive stress relaxation rate          (default 0.005)
stress_threshold  : float — level at which tile is "overstressed"   (default 0.6)
max_tiles_per_tick: int   — budget                                  (default 256)

Public API
----------
StressAccumulator(config=None)
  .apply_contact(memory_state, tile_idx, contact_force, dt) -> None
  .tick(memory_state, wind_map, slope_map, dt) -> List[int]
      Returns list of overstressed tile indices.
"""
from __future__ import annotations

from typing import List, Optional

from src.memory.WorldMemoryState import WorldMemoryState


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


class StressAccumulator:
    """Advances ``stressAccumulationField`` in :class:`WorldMemoryState`.

    Parameters
    ----------
    config : dict or None
        See module docstring for keys.
    """

    _DEFAULTS = {
        "k_stress":           0.1,
        "k_stress_relax":     0.005,
        "stress_threshold":   0.6,
        "max_tiles_per_tick": 256,
    }

    def __init__(self, config=None) -> None:
        cfg = dict(self._DEFAULTS)
        if isinstance(config, dict):
            src = config.get("memory", config)
            for k in self._DEFAULTS:
                if k in src:
                    cfg[k] = src[k]

        self._k_stress:     float = float(cfg["k_stress"])
        self._k_relax:      float = float(cfg["k_stress_relax"])
        self._threshold:    float = float(cfg["stress_threshold"])
        self._budget:       int   = int(cfg["max_tiles_per_tick"])
        self._cursor:       int   = 0

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def apply_contact(
        self,
        memory_state: WorldMemoryState,
        tile_idx: int,
        contact_force: float,
        dt: float,
    ) -> None:
        """Add stress to *tile_idx* from a single contact event.

        Parameters
        ----------
        memory_state  : WorldMemoryState updated in-place.
        tile_idx      : Flat tile index.
        contact_force : Normalised contact force [0, 1].
        dt            : Time step (seconds).
        """
        if dt <= 0.0 or contact_force <= 0.0:
            return
        stress = memory_state.stressAccumulationField
        delta = self._k_stress * contact_force * dt
        stress[tile_idx] = _clamp(stress[tile_idx] + delta)

    def tick(
        self,
        memory_state: WorldMemoryState,
        wind_map: Optional[List[float]],
        slope_map: Optional[List[float]],
        dt: float,
    ) -> List[int]:
        """Advance stress decay for a budget slice of tiles.

        Parameters
        ----------
        memory_state : WorldMemoryState to update in-place.
        wind_map     : Per-tile wind speed [0, 1] — faster decay when windy.
        slope_map    : Per-tile slope magnitude [0, 1] — faster decay on steep.
        dt           : Time step (seconds).

        Returns
        -------
        List of tile indices that are currently overstressed (> threshold).
        """
        if dt <= 0.0:
            return []

        n = memory_state.size()
        stress = memory_state.stressAccumulationField
        overstressed: List[int] = []

        count = min(self._budget, n)
        for _ in range(count):
            idx = self._cursor % n
            self._cursor = (self._cursor + 1) % n

            wind  = wind_map[idx]  if wind_map  else 0.0
            slope = slope_map[idx] if slope_map else 0.0

            # Decay rate increases with wind and slope
            decay = self._k_relax * (1.0 + wind + slope * 0.5)
            stress[idx] = _clamp(stress[idx] - decay * dt)

            if stress[idx] >= self._threshold:
                overstressed.append(idx)

        return overstressed
