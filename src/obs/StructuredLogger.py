"""StructuredLogger — Stage 61 structured JSON-lines logger.

Emits log records with required correlation fields:

    worldId, worldEpoch, regionId, tuningEpoch,
    serverTick, planetTime, playerId, correlationId

Canonical event types
---------------------
* join / leave
* handoff_start / handoff_complete
* snapshot_success / snapshot_fail
* rollback / reset
* protocol_mismatch
* drift_detected
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional


CANONICAL_EVENTS = (
    "join",
    "leave",
    "handoff_start",
    "handoff_complete",
    "snapshot_success",
    "snapshot_fail",
    "rollback",
    "reset",
    "protocol_mismatch",
    "drift_detected",
)


class StructuredLogger:
    """Emits structured JSON-lines log records.

    Parameters
    ----------
    world_id:
        Stable world identifier injected into every record.
    region_id:
        Shard/region identifier.
    output:
        File-like object (``write`` + ``flush``).  Defaults to an in-memory
        list accessible via :attr:`records`.
    """

    def __init__(
        self,
        world_id:  str          = "",
        region_id: str          = "",
        output                  = None,
    ) -> None:
        self._world_id      = world_id
        self._region_id     = region_id
        self._world_epoch:  int = 0
        self._tuning_epoch: int = 0
        self._server_tick:  int = 0
        self._planet_time:  float = 0.0

        # In-memory buffer used when output is None
        self.records: list = []
        self._output = output

    # ------------------------------------------------------------------
    # Context setters
    # ------------------------------------------------------------------

    def set_context(
        self,
        *,
        world_epoch:  Optional[int]   = None,
        tuning_epoch: Optional[int]   = None,
        server_tick:  Optional[int]   = None,
        planet_time:  Optional[float] = None,
    ) -> None:
        """Update the ambient context injected into every record."""
        if world_epoch  is not None: self._world_epoch  = world_epoch
        if tuning_epoch is not None: self._tuning_epoch = tuning_epoch
        if server_tick  is not None: self._server_tick  = server_tick
        if planet_time  is not None: self._planet_time  = planet_time

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log(
        self,
        event:          str,
        level:          str                       = "INFO",
        player_id:      Optional[str]             = None,
        correlation_id: Optional[str]             = None,
        data:           Optional[Dict[str, Any]]  = None,
    ) -> Dict[str, Any]:
        """Emit a structured log record and return it."""
        record: Dict[str, Any] = {
            "ts":            datetime.now(timezone.utc).isoformat(),
            "level":         level,
            "event":         event,
            "worldId":       self._world_id,
            "worldEpoch":    self._world_epoch,
            "regionId":      self._region_id,
            "tuningEpoch":   self._tuning_epoch,
            "serverTick":    self._server_tick,
            "planetTime":    self._planet_time,
        }
        if player_id      is not None: record["playerId"]      = player_id
        if correlation_id is not None: record["correlationId"] = correlation_id
        if data           is not None: record["data"]          = data

        if self._output is not None:
            try:
                self._output.write(json.dumps(record) + "\n")
                self._output.flush()
            except OSError:
                pass
        else:
            self.records.append(record)

        return record
