"""OcclusionCache — Stage 46 terrain occlusion cache with TTL.

Caches per-emitter occlusion values so that a terrain raycast is not
needed every frame.  Entries expire after ``ttl_sec`` seconds.  The
cache is updated in priority order (highest-energy emitters first) up
to ``max_raycasts_per_sec`` refreshes per second.

The actual terrain intersection test is injected as a callable
``raycast_fn(from_pos, to_pos) → bool`` that returns *True* when the
line of sight is blocked.

Public API
----------
OcclusionCache(config=None, raycast_fn=None)
  .get_occlusion(emitter_id, from_pos, to_pos, energy) → float
  .tick(dt)                                             → None
"""
from __future__ import annotations

import math
from typing import Callable, Dict, Optional, Tuple

from src.math.Vec3 import Vec3


# Type alias for the injected raycast function
RaycastFn = Callable[[Vec3, Vec3], bool]


class OcclusionCache:
    """TTL-based terrain occlusion cache.

    Parameters
    ----------
    config :
        Optional dict; reads ``audio_world.*`` keys.
    raycast_fn :
        Callable ``(from_pos: Vec3, to_pos: Vec3) -> bool`` that returns
        ``True`` when the path is blocked.  If *None*, a stub that always
        returns ``False`` (no occlusion) is used.
    """

    _DEFAULT_TTL_SEC         = 1.0
    _DEFAULT_MAX_CASTS_SEC   = 100

    def __init__(
        self,
        config:      Optional[dict] = None,
        raycast_fn:  Optional[RaycastFn] = None,
    ) -> None:
        awcfg = (config or {}).get("audio_world", {}) or {}
        self._ttl: float = float(awcfg.get("occlusion_cache_ttl_sec", self._DEFAULT_TTL_SEC))
        self._max_casts_per_sec: int = int(
            awcfg.get("max_raycasts_per_sec", self._DEFAULT_MAX_CASTS_SEC)
        )
        self._raycast_fn: RaycastFn = raycast_fn or (lambda a, b: False)

        # {emitter_id: (occlusion_value, age_sec, energy)}
        self._cache: Dict[int, Tuple[float, float, float]] = {}
        # Budget counter: raycasts issued this second
        self._casts_this_sec: int   = 0
        self._budget_timer:   float = 0.0

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def get_occlusion(
        self,
        emitter_id: int,
        from_pos:   Vec3,
        to_pos:     Vec3,
        energy:     float = 0.0,
    ) -> float:
        """Return cached occlusion value [0..1] for *emitter_id*.

        Performs a new raycast if the cached value has expired and
        budget allows.  Returns 0.0 (no occlusion) when budget is
        exhausted.

        Parameters
        ----------
        emitter_id :
            Unique emitter identifier.
        from_pos :
            Listener world position.
        to_pos :
            Emitter world position.
        energy :
            Current emitter energy (higher → higher refresh priority).
        """
        entry = self._cache.get(emitter_id)

        if entry is not None:
            occ_val, age, _old_energy = entry
            if age < self._ttl:
                # Update the stored energy for priority tracking
                self._cache[emitter_id] = (occ_val, age, energy)
                return occ_val

        # Cache miss or expired — try to issue a new cast
        if self._casts_this_sec < self._max_casts_per_sec:
            self._casts_this_sec += 1
            blocked = self._raycast_fn(from_pos, to_pos)
            occ_val = 1.0 if blocked else 0.0
            self._cache[emitter_id] = (occ_val, 0.0, energy)
            return occ_val

        # Budget exhausted: return last known value or 0
        if entry is not None:
            return entry[0]
        return 0.0

    def tick(self, dt: float) -> None:
        """Advance all cache entry ages and reset budget counter."""
        self._cache = {
            eid: (occ, age + dt, e)
            for eid, (occ, age, e) in self._cache.items()
        }
        self._budget_timer += dt
        if self._budget_timer >= 1.0:
            self._casts_this_sec = 0
            self._budget_timer -= 1.0
