"""GatewayRouter — Stage 60 client-to-RS routing gateway.

The Gateway is the single entry-point URL that all clients connect to.
It routes each client to the correct Region Shard (RS) based on the player's
current planet position (lat/lon) and the live region → RS mapping held by
the Global Authority (GA).

Responsibilities
----------------
* Translate a (lat, lon) position to a regionId via RegionIndexing.
* Look up the RS node address from GA.
* Track which RS a given session is currently routed to.
* On reconnect, find the RS that last held the session.
* Emit routing decisions so the transport layer can open/close connections.

This class is *not* an asyncio server; it is a pure routing table that the
actual network transport queries.

Public API
----------
GatewayRouter(ga_service, indexing)
  .route(session_id, lat, lon) → str | None
      Return the RS node address for *(lat, lon)*, or ``None`` if the region
      has no live RS.  Also records the routing decision.
  .reroute(session_id, new_region_id) → str | None
      Forcibly redirect *session_id* to *new_region_id*'s RS after a handoff.
  .current_rs(session_id) → str | None
      Return the RS address a session is currently routed to.
  .remove_session(session_id) → None
      Clean up when a client disconnects.
  .session_count → int
"""
from __future__ import annotations

from typing import Dict, Optional


class GatewayRouter:
    """Routes client sessions to the appropriate Region Shard."""

    def __init__(self, ga_service, indexing) -> None:
        self._ga       = ga_service
        self._indexing = indexing
        # {session_id → (region_id, node_addr)}
        self._sessions: Dict[str, tuple] = {}

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def route(
        self,
        session_id: str,
        lat:        float,
        lon:        float,
    ) -> Optional[str]:
        """Route *session_id* to the RS responsible for *(lat, lon)*.

        Returns the RS node address, or ``None`` if the region is offline.
        """
        region_id = self._indexing.region_id(lat, lon)
        node_addr = self._ga.region_node(region_id)
        if node_addr is None:
            return None
        self._sessions[session_id] = (region_id, node_addr)
        return node_addr

    def reroute(
        self,
        session_id:    str,
        new_region_id: int,
    ) -> Optional[str]:
        """Redirect *session_id* to *new_region_id* after a handoff.

        Returns the new RS address, or ``None`` if the target region is
        offline.
        """
        node_addr = self._ga.region_node(new_region_id)
        if node_addr is None:
            return None
        self._sessions[session_id] = (new_region_id, node_addr)
        return node_addr

    def current_rs(self, session_id: str) -> Optional[str]:
        """Return the RS address *session_id* is currently routed to."""
        entry = self._sessions.get(session_id)
        return entry[1] if entry is not None else None

    def remove_session(self, session_id: str) -> None:
        """Clean up routing state for a disconnected session."""
        self._sessions.pop(session_id, None)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def session_count(self) -> int:
        return len(self._sessions)
