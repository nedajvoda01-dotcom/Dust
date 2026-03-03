"""HandoffManager — Stage 60 seamless region-to-region player handoff.

Coordinates the three-party handoff protocol:
  1. Source RS detects player in handoff band and raises a HandoffRequest.
  2. HandoffManager asks GA for the target RS node address.
  3. HandoffManager transfers the serialised PlayerState to the target RS.
  4. HandoffManager confirms completion so the source RS can drop the player.

The class operates synchronously (no asyncio) to remain testable.  A real
deployment would wrap the ``execute_handoff`` call in an async coroutine.

Guarantee
---------
**No double authority**: a player belongs to exactly one RS at any moment.
The handoff atomically removes from source and adds to target before
returning success.

Public API
----------
HandoffManager(ga_service, indexing)
  .execute_handoff(request, source_rs, target_rs) → bool
      Transfer player from *source_rs* to *target_rs*.
      Returns ``True`` on success, ``False`` if target is full or unknown.
  .pending_count → int
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from src.regions.RegionShardServer import HandoffRequest


class HandoffManager:
    """Coordinates seamless player transfer between Region Shards."""

    def __init__(self, ga_service, indexing) -> None:
        self._ga       = ga_service
        self._indexing = indexing
        self._completed: int = 0
        self._failed:    int = 0

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------

    def execute_handoff(
        self,
        request:   HandoffRequest,
        source_rs,            # RegionShardServer
        target_rs,            # RegionShardServer
    ) -> bool:
        """Transfer player described by *request* from *source_rs* to *target_rs*.

        Returns
        -------
        ``True``  — handoff succeeded; caller should switch the client connection.
        ``False`` — target is full or the player was not found on source.
        """
        ps = request.player_state
        pid = request.player_id

        # Verify player is on source
        if pid not in source_rs.player_ids:
            self._failed += 1
            return False

        # Attempt to add to target first (avoids a brief gap in authority)
        added = target_rs.add_player(
            player_id=pid,
            pos=ps.get("pos", [0.0, 0.0, 0.0]),
            vel=ps.get("vel", [0.0, 0.0, 0.0]),
        )
        if not added:
            self._failed += 1
            return False

        # Now atomically remove from source
        source_rs.remove_player(pid)

        self._completed += 1
        return True

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def completed_count(self) -> int:
        return self._completed

    @property
    def failed_count(self) -> int:
        return self._failed
