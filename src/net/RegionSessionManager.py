"""RegionSessionManager — Stage 60 per-region WebSocket session tracking.

Tracks which WebSocket sessions (clients) are currently connected to which
Region Shard, and provides the metadata needed for interest management,
handoff initiation, and graceful reconnection.

A "session" in this context is identified by a ``session_id`` string (which
may be the player ID or a dedicated connection token).  Each session records:

* The region ID it is currently attached to.
* The player's last known position (lat, lon).
* The time of last activity.

This class is intentionally single-threaded and synchronous so it can be
tested without asyncio.

Public API
----------
RegionSessionManager()
  .register(session_id, region_id, lat, lon)
  .update_position(session_id, lat, lon)
  .unregister(session_id)
  .sessions_in_region(region_id) → list[str]
  .region_for_session(session_id) → int | None
  .position_for_session(session_id) → (float, float) | None
  .session_count → int
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class _SessionEntry:
    session_id: str
    region_id:  int
    lat:        float
    lon:        float
    last_seen:  float = field(default_factory=time.monotonic)


class RegionSessionManager:
    """Tracks WebSocket sessions across Region Shards."""

    def __init__(self) -> None:
        self._sessions: Dict[str, _SessionEntry] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def register(
        self,
        session_id: str,
        region_id:  int,
        lat:        float,
        lon:        float,
    ) -> None:
        """Record a new or reconnected session."""
        self._sessions[session_id] = _SessionEntry(
            session_id=session_id,
            region_id=region_id,
            lat=float(lat),
            lon=float(lon),
        )

    def update_position(
        self,
        session_id: str,
        lat:        float,
        lon:        float,
    ) -> None:
        """Update the last-known position of *session_id*."""
        entry = self._sessions.get(session_id)
        if entry is not None:
            entry.lat = float(lat)
            entry.lon = float(lon)
            entry.last_seen = time.monotonic()

    def unregister(self, session_id: str) -> None:
        """Remove a disconnected session."""
        self._sessions.pop(session_id, None)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def sessions_in_region(self, region_id: int) -> List[str]:
        """Return session IDs currently attached to *region_id*."""
        return [
            sid for sid, e in self._sessions.items()
            if e.region_id == region_id
        ]

    def region_for_session(self, session_id: str) -> Optional[int]:
        """Return the region ID for *session_id*, or ``None``."""
        entry = self._sessions.get(session_id)
        return entry.region_id if entry is not None else None

    def position_for_session(
        self, session_id: str
    ) -> Optional[Tuple[float, float]]:
        """Return *(lat, lon)* for *session_id*, or ``None``."""
        entry = self._sessions.get(session_id)
        return (entry.lat, entry.lon) if entry is not None else None

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def session_count(self) -> int:
        return len(self._sessions)
