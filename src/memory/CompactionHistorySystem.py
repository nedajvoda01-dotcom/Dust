"""CompactionHistorySystem — Stage 51 repeated-pressure trail memory.

Tracks how often each tile has been under foot or contact pressure.
High compaction history means a "trail" has formed — the material there
is more densely packed and dust has been displaced.

Model
-----
Per contact at a tile::

    compactionHistory += k_compact_gain * contact_force * dt

Passive decay (trail disappears over time)::

    compactionHistory -= k_compact_decay * dt

Decay is amplified by:
  * windChannel (high wind erodes the trail faster)
  * seasonal variation

Effect exported via :meth:`compaction_gain_for_tile`:
  * higher value → faster snowCompaction gain in MaterialState
  * higher value → lower dustThickness (dust blown away)

Config keys (under ``memory.*``)
----------------------------------
k_compact_gain    : float — compaction gain per unit force per second (default 0.08)
k_compact_decay   : float — passive decay rate                        (default 0.003)
max_tiles_per_tick: int   — budget                                    (default 256)

Public API
----------
CompactionHistorySystem(config=None)
  .apply_pressure(memory_state, tile_idx, contact_force, dt) -> None
  .tick(memory_state, wind_map, dt) -> None
  .compaction_gain_for_tile(memory_state, tile_idx) -> float
      Returns a [0, 1] multiplier for material compaction gain.
"""
from __future__ import annotations

from typing import List, Optional

from src.memory.WorldMemoryState import WorldMemoryState


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


class CompactionHistorySystem:
    """Advances ``compactionHistoryField`` in :class:`WorldMemoryState`.

    Parameters
    ----------
    config : dict or None
        See module docstring for keys.
    """

    _DEFAULTS = {
        "k_compact_gain":     0.08,
        "k_compact_decay":    0.003,
        "max_tiles_per_tick": 256,
    }

    def __init__(self, config=None) -> None:
        cfg = dict(self._DEFAULTS)
        if isinstance(config, dict):
            src = config.get("memory", config)
            for k in self._DEFAULTS:
                if k in src:
                    cfg[k] = src[k]

        self._k_gain:   float = float(cfg["k_compact_gain"])
        self._k_decay:  float = float(cfg["k_compact_decay"])
        self._budget:   int   = int(cfg["max_tiles_per_tick"])
        self._cursor:   int   = 0

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def apply_pressure(
        self,
        memory_state: WorldMemoryState,
        tile_idx: int,
        contact_force: float,
        dt: float,
    ) -> None:
        """Record a pressure event at *tile_idx*.

        Parameters
        ----------
        memory_state  : WorldMemoryState updated in-place.
        tile_idx      : Flat tile index.
        contact_force : Normalised contact force [0, 1].
        dt            : Time step (seconds).
        """
        if dt <= 0.0 or contact_force <= 0.0:
            return
        ch = memory_state.compactionHistoryField
        delta = self._k_gain * contact_force * dt
        ch[tile_idx] = _clamp(ch[tile_idx] + delta)

    def tick(
        self,
        memory_state: WorldMemoryState,
        wind_map: Optional[List[float]],
        dt: float,
    ) -> None:
        """Advance compaction decay for a budget slice of tiles.

        Parameters
        ----------
        memory_state : WorldMemoryState to update in-place.
        wind_map     : Per-tile wind channel [0, 1] — accelerates decay.
        dt           : Time step (seconds).
        """
        if dt <= 0.0:
            return

        n = memory_state.size()
        ch = memory_state.compactionHistoryField

        count = min(self._budget, n)
        for _ in range(count):
            idx = self._cursor % n
            self._cursor = (self._cursor + 1) % n

            wind = wind_map[idx] if wind_map else 0.0
            decay = self._k_decay * (1.0 + wind)
            ch[idx] = _clamp(ch[idx] - decay * dt)

    def compaction_gain_for_tile(
        self,
        memory_state: WorldMemoryState,
        tile_idx: int,
    ) -> float:
        """Return the compaction memory value [0, 1] for *tile_idx*.

        This value is used by MemoryToMaterialAdapter to boost the
        snowCompaction gain rate and reduce dustThickness in MaterialState.
        """
        return _clamp(memory_state.compactionHistoryField[tile_idx])
