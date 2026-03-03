"""ProtocolVersioning — Stage 57 protocol version constants and message helpers.

Every WebSocket connection now opens with a WELCOME message (server → client)
that carries the protocol and snapshot-format versions.  Clients that
support Stage 57+ reply with HELLO; older clients (pre-Stage-57) receive the
same WELCOME but may ignore it and proceed via WORLD_SYNC as before.

Protocol lifecycle
------------------
1. WS connect
2. Server → ``WELCOME {protocolVersion, snapshotFormatVersion, worldId,
                        worldSeed, worldEpoch, assignedPlayerId, buildId}``
3. Client → ``HELLO {protocolVersion, playerId}``   (Stage 57 clients only)
4. If protocolVersion mismatch: Server → ``UPGRADE_REQUIRED {currentProtocolVersion}``
5. Server → ``WORLD_SYNC`` + ``WORLD_BASELINE`` (as before)

Public API
----------
PROTOCOL_VERSION        → int   current protocol revision
SNAPSHOT_FORMAT_VERSION → int   current world-snapshot format revision

make_welcome(world_id, world_seed, world_epoch, assigned_player_id, build_id)
    → dict   ready to JSON-serialise

make_upgrade_required()
    → dict   ready to JSON-serialise

check_compatible(client_protocol_version: int) → bool
"""
from __future__ import annotations

from typing import Any, Dict

# ─────────────────────────────────────────────────────────────────────────────
# Version constants
# ─────────────────────────────────────────────────────────────────────────────

PROTOCOL_VERSION:        int = 1
"""Current WebSocket protocol revision.

Increment when the message schema changes in a backward-incompatible way.
"""

SNAPSHOT_FORMAT_VERSION: int = 1
"""Current world-snapshot format revision.

Increment when the structure of persisted world state changes so that old
snapshots cannot be loaded by new code.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Message factories
# ─────────────────────────────────────────────────────────────────────────────

def make_welcome(
    world_id:           str,
    world_seed:         int,
    world_epoch:        int,
    assigned_player_id: str,
    build_id:           str,
) -> Dict[str, Any]:
    """Return a ``WELCOME`` message dict.

    Sent by the server as the **first** WebSocket message to every new client.

    Parameters
    ----------
    world_id:
        Current world UUID.
    world_seed:
        Current world seed (deterministic RNG root).
    world_epoch:
        Monotonically increasing reset counter.
    assigned_player_id:
        Stable server-issued player UUID for this connection.
    build_id:
        Static-asset content hash for cache-busting.
    """
    return {
        "type":                  "WELCOME",
        "protocolVersion":       PROTOCOL_VERSION,
        "snapshotFormatVersion": SNAPSHOT_FORMAT_VERSION,
        "worldId":               world_id,
        "worldSeed":             world_seed,
        "worldEpoch":            world_epoch,
        "assignedPlayerId":      assigned_player_id,
        "buildId":               build_id,
    }


def make_upgrade_required() -> Dict[str, Any]:
    """Return an ``UPGRADE_REQUIRED`` message dict.

    Sent by the server when a client announces an incompatible protocol
    version in its ``HELLO`` message.  The client should reload the page
    to fetch the latest static assets.
    """
    return {
        "type":                   "UPGRADE_REQUIRED",
        "currentProtocolVersion": PROTOCOL_VERSION,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Compatibility check
# ─────────────────────────────────────────────────────────────────────────────

def check_compatible(client_protocol_version: int) -> bool:
    """Return *True* when *client_protocol_version* is compatible with the server.

    Policy: major version must match exactly (no cross-version compatibility).
    """
    return int(client_protocol_version) == PROTOCOL_VERSION
