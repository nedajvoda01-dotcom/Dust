"""CascadeProcessor — Stage 52 deterministic BFS instability cascade.

When an instability event fires in a tile it raises the shear stress (or
crust failure potential) of its 4-connected neighbours.  If any neighbour
then exceeds its own threshold it is added to the cascade queue.

Design goals
------------
* Fully deterministic (BFS, fixed neighbour order N/S/E/W by tile id).
* Budget-limited: max_tiles_per_tick caps the total tiles processed.
* No random — threshold crossings drive propagation.

Public API
----------
CascadeProcessor(config=None)
  .run(state, seed_tiles, field_name, increment, threshold)
      → List[int]   (tiles actually processed, in BFS order)
"""
from __future__ import annotations

from collections import deque
from typing import List, Optional

from src.instability.InstabilityState import InstabilityState


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


class CascadeProcessor:
    """Deterministic BFS cascade propagator.

    Parameters
    ----------
    config :
        Optional dict; reads ``instability.*`` keys.
    """

    _DEFAULT_MAX_TILES    = 64
    _DEFAULT_CASCADE_INCR = 0.15   # stress added to each neighbour
    _DEFAULT_RADIUS       = 3      # max BFS hops

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = (config or {}).get("instability", {}) or {}
        self._max_tiles:  int   = int(  cfg.get("max_tiles_per_tick",  self._DEFAULT_MAX_TILES))
        self._incr:       float = float(cfg.get("cascade_stress_incr", self._DEFAULT_CASCADE_INCR))
        self._radius:     int   = int(  cfg.get("cascade_radius_max",  self._DEFAULT_RADIUS))

    def run(
        self,
        state:      InstabilityState,
        seed_tiles: List[int],
        field_name: str,
        increment:  float,
        threshold:  float,
    ) -> List[int]:
        """Run deterministic BFS cascade starting from *seed_tiles*.

        For each processed tile the named field is raised by *increment*.
        If the result exceeds *threshold* the tile's neighbours are
        enqueued (respecting budget and radius limits).

        Parameters
        ----------
        state      : InstabilityState to modify in-place.
        seed_tiles : Initial trigger tiles.
        field_name : Which field to propagate (e.g. 'shearStressField').
        increment  : Amount added to each cascade tile's field.
        threshold  : If field after increment > threshold → propagate further.

        Returns
        -------
        List of all tile indices actually processed (BFS order).
        """
        field = getattr(state, field_name)
        visited = set(seed_tiles)
        queue: deque = deque()

        # Seed the BFS with (tile, depth)
        for t in sorted(seed_tiles):   # sorted for determinism
            queue.append((t, 0))

        processed: List[int] = []

        while queue and len(processed) < self._max_tiles:
            tile, depth = queue.popleft()
            if len(processed) >= self._max_tiles:
                break

            # Apply increment
            field[tile] = _clamp(field[tile] + increment)
            processed.append(tile)

            # Propagate if still above threshold and within radius
            if depth < self._radius and field[tile] > threshold:
                for nb in state.neighbors(tile):   # fixed order (deterministic)
                    if nb not in visited:
                        visited.add(nb)
                        queue.append((nb, depth + 1))

        return processed
