"""PlayerAppearanceReplicator — Stage 43 server-side appearance sync.

The server assigns each player a stable ``player_seed`` (derived from
``hash(worldSeed, playerId)``) and builds a
:class:`~src.character.SuitKitDescriptor.SuitKitDescriptor` once at spawn.
This descriptor is sent to all connected clients via
``announce_appearance``; joining clients receive it via
``send_appearance_to_client``.

The server sends the **descriptor**, not the seed, so clients do not depend
on a shared hash function — they just decode the 14-byte descriptor struct.

Public API
----------
PlayerAppearanceReplicator(world_seed, config=None)
  .register_player(player_id, module_counts=None) → SuitKitDescriptor
  .get_descriptor(player_id)  → SuitKitDescriptor | None
  .remove_player(player_id)
  .announce_appearance(player_id) → bytes     (wire payload)
  .send_appearance_to_client(player_id) → bytes
  .receive_appearance(payload) → (player_id_str, SuitKitDescriptor)
"""
from __future__ import annotations

import hashlib
import struct
from typing import Dict, Optional

from src.character.SuitKitDescriptor import SuitKitDescriptor


def _player_seed(world_seed: int, player_id: str) -> int:
    """Deterministic integer seed for a player."""
    raw = f"{world_seed}|{player_id}".encode("utf-8")
    digest = hashlib.sha256(raw).digest()
    return int.from_bytes(digest[:8], "big")


# Wire header: 4-byte big-endian player_id hash + 14-byte descriptor
_HDR_FMT = ">I"
_HDR_SIZE = 4
_TOTAL_PAYLOAD = _HDR_SIZE + 15   # 19 bytes per player announcement (4 header + 15 descriptor)


class PlayerAppearanceReplicator:
    """Manages player appearance descriptors and replicates them on the wire.

    Parameters
    ----------
    world_seed :
        Stable integer world seed; combined with ``player_id`` to derive
        per-player seeds.
    config :
        Optional dict; reads ``net.appearance_send_on_join``.
    """

    def __init__(
        self,
        world_seed: int,
        config: Optional[dict] = None,
    ) -> None:
        self._world_seed = int(world_seed)
        cfg = config or {}
        ncfg = cfg.get("net", {}) or {}
        self._send_on_join: bool = bool(ncfg.get("appearance_send_on_join", True))

        self._descriptors: Dict[str, SuitKitDescriptor] = {}

    # ------------------------------------------------------------------
    # Player lifecycle
    # ------------------------------------------------------------------

    def register_player(
        self,
        player_id:     str,
        module_counts: dict | None = None,
    ) -> SuitKitDescriptor:
        """Assign an appearance to *player_id* (called at spawn or join).

        If the player was already registered the existing descriptor is
        returned unchanged, ensuring one consistent appearance per session.
        """
        if player_id in self._descriptors:
            return self._descriptors[player_id]

        seed = _player_seed(self._world_seed, player_id)
        desc = SuitKitDescriptor.from_seed(seed, module_counts)
        self._descriptors[player_id] = desc
        return desc

    def get_descriptor(self, player_id: str) -> Optional[SuitKitDescriptor]:
        """Return the descriptor for *player_id*, or None if unknown."""
        return self._descriptors.get(player_id)

    def remove_player(self, player_id: str) -> None:
        """Unregister *player_id*."""
        self._descriptors.pop(player_id, None)

    # ------------------------------------------------------------------
    # Wire serialisation
    # ------------------------------------------------------------------

    def announce_appearance(self, player_id: str) -> bytes:
        """Build the announcement payload for *player_id*.

        Called by the server when a new player joins; the result is
        broadcast to all current clients.
        """
        return self._encode(player_id)

    def send_appearance_to_client(self, player_id: str) -> bytes:
        """Build a catch-up payload for a newly joining client.

        The server calls this for every currently registered player so the
        new client immediately knows everyone's appearance.
        """
        return self._encode(player_id)

    @staticmethod
    def receive_appearance(payload: bytes) -> tuple:
        """Decode an appearance payload received by a client.

        Returns
        -------
        (player_id_hash_str, SuitKitDescriptor)
            ``player_id_hash_str`` is the 4-byte hash as a hex string.
        """
        if len(payload) != _TOTAL_PAYLOAD:
            raise ValueError(
                f"Appearance payload must be {_TOTAL_PAYLOAD} bytes; "
                f"got {len(payload)}"
            )
        pid_hash = struct.unpack_from(_HDR_FMT, payload, 0)[0]
        desc     = SuitKitDescriptor.unpack(payload[_HDR_SIZE:])
        return (format(pid_hash, "08x"), desc)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _encode(self, player_id: str) -> bytes:
        desc = self._descriptors.get(player_id)
        if desc is None:
            raise KeyError(f"Player '{player_id}' not registered")
        # 4-byte truncated hash of the player_id string (for client-side lookup)
        pid_hash = int.from_bytes(
            hashlib.sha256(player_id.encode()).digest()[:4], "big"
        )
        hdr  = struct.pack(_HDR_FMT, pid_hash)
        return hdr + desc.pack()
