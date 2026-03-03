"""GhostBorderCache — Stage 60 ghost-border chunk cache.

Each Region Shard (RS) keeps a shallow read-only copy of the chunks that
belong to *neighbouring* regions near the shared boundary.  These ghost
chunks are used for:

* Collision detection near the border (so players don't fall through).
* Acoustic ray-casts that cross region boundaries.
* Visibility / LOD transitions.

Ghost chunks are **never** simulated by the holding RS; they are pushed by
the authoritative RS periodically and stored here verbatim.

Staleness
---------
Each chunk entry carries a ``received_at`` monotonic timestamp.  Entries
older than ``max_age_s`` are considered stale and ``get_chunk`` will return
``None``, forcing a refresh request.

Public API
----------
GhostBorderCache(max_age_s=10.0)
  .put(region_id, chunk_id, data)      — store / refresh a ghost chunk
  .get(region_id, chunk_id) → dict | None — retrieve a chunk (None if stale)
  .evict(region_id)                    — drop all ghost data for a region
  .stale_chunks(region_id) → list[int] — chunk IDs that need refreshing
  .chunk_count → int
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple


_DEFAULT_MAX_AGE_S = 10.0


class GhostBorderCache:
    """Read-only cache of neighbouring-region boundary chunks."""

    def __init__(self, max_age_s: float = _DEFAULT_MAX_AGE_S) -> None:
        self._max_age_s = float(max_age_s)
        # {(region_id, chunk_id) → (data, received_at)}
        self._cache: Dict[Tuple[int, int], Tuple[Dict[str, Any], float]] = {}

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def put(
        self,
        region_id: int,
        chunk_id:  int,
        data:      Dict[str, Any],
    ) -> None:
        """Store or refresh the ghost chunk *(region_id, chunk_id)*."""
        self._cache[(region_id, chunk_id)] = (dict(data), time.monotonic())

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(
        self,
        region_id: int,
        chunk_id:  int,
    ) -> Optional[Dict[str, Any]]:
        """Return the cached chunk, or ``None`` if absent or stale."""
        entry = self._cache.get((region_id, chunk_id))
        if entry is None:
            return None
        data, received_at = entry
        if time.monotonic() - received_at > self._max_age_s:
            return None
        return dict(data)

    # ------------------------------------------------------------------
    # Management
    # ------------------------------------------------------------------

    def evict(self, region_id: int) -> None:
        """Remove all cached chunks for *region_id*."""
        keys = [k for k in self._cache if k[0] == region_id]
        for k in keys:
            del self._cache[k]

    def stale_chunks(self, region_id: int) -> List[int]:
        """Return chunk IDs for *region_id* that are stale and need refresh."""
        now = time.monotonic()
        return [
            cid
            for (rid, cid), (_, received_at) in self._cache.items()
            if rid == region_id and now - received_at > self._max_age_s
        ]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def chunk_count(self) -> int:
        return len(self._cache)
