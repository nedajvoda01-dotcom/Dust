"""VolumetricSeedSync — Stage 65 hybrid network synchronisation for volumetric seeds.

Implements spec §9 (hybrid approach):
* Server replicates macro weather + seeds + source events (vent activations,
  collapse events, fog basin locations).
* Client simulates density locally from those seeds deterministically.
* Authoritative mass effects (deposition) are handled separately via
  MassExchangeAPI on the server (not replicated here).

Packet format
-------------
Header : magic(2B) + epoch(4B) + num_seeds(2B) = 8 bytes
Per-seed record : layer_type_id(1B) + anchor_x(4B) + anchor_y(4B)
                  + seed(4B) + source_strength(1B) = 14 bytes

Public API
----------
VolumetricSeedSync(config=None)
  .encode_seeds(seed_list)  → bytes
  .decode_seeds(data)       → list[VolumetricSeed]
  .advance_epoch()          → None

VolumetricSeed (dataclass)
  .layer_type     str
  .anchor_x       float
  .anchor_y       float
  .seed           int
  .source_strength float [0..1]
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import List, Optional

from src.vol.DensityGrid import VolumeLayerType


# ---------------------------------------------------------------------------
# Packet layout
# ---------------------------------------------------------------------------

_MAGIC           = b"V5"
_HEADER_STRUCT   = struct.Struct("!2sIH")   # magic, epoch, num_seeds
_HEADER_SIZE     = _HEADER_STRUCT.size       # 8 bytes
_SEED_STRUCT     = struct.Struct("!Bff IB")  # layer_id, ax, ay, seed, strength_u8
_SEED_SIZE       = _SEED_STRUCT.size         # 14 bytes

_LAYER_TO_ID: dict = {
    VolumeLayerType.DUST:       0,
    VolumeLayerType.FOG:        1,
    VolumeLayerType.STEAM:      2,
    VolumeLayerType.SNOW_DRIFT: 3,
}
_ID_TO_LAYER: dict = {v: k for k, v in _LAYER_TO_ID.items()}


# ---------------------------------------------------------------------------
# VolumetricSeed
# ---------------------------------------------------------------------------

@dataclass
class VolumetricSeed:
    """One seed record for a volumetric domain source."""
    layer_type:      str
    anchor_x:        float
    anchor_y:        float
    seed:            int
    source_strength: float = 1.0


# ---------------------------------------------------------------------------
# VolumetricSeedSync
# ---------------------------------------------------------------------------

class VolumetricSeedSync:
    """Encode/decode volumetric seed packets for server → client streaming.

    Parameters
    ----------
    config :
        Optional dict; currently unused but accepted for API consistency.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        self._epoch: int = 0

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def advance_epoch(self) -> None:
        """Increment epoch counter (call when macro weather parameters change)."""
        self._epoch = (self._epoch + 1) & 0xFFFFFFFF

    def encode_seeds(self, seed_list: List[VolumetricSeed]) -> bytes:
        """Serialise a list of volumetric seeds for transmission.

        Parameters
        ----------
        seed_list :
            Seeds to encode.

        Returns
        -------
        bytes
            Encoded packet.
        """
        header = _HEADER_STRUCT.pack(_MAGIC, self._epoch, len(seed_list))
        parts = [header]
        for s in seed_list:
            layer_id   = _LAYER_TO_ID.get(s.layer_type, 0)
            strength_u8 = max(0, min(255, int(round(s.source_strength * 255))))
            parts.append(_SEED_STRUCT.pack(
                layer_id,
                float(s.anchor_x),
                float(s.anchor_y),
                int(s.seed) & 0xFFFFFFFF,
                strength_u8,
            ))
        return b"".join(parts)

    def decode_seeds(self, data: bytes) -> List[VolumetricSeed]:
        """Deserialise a seed packet.

        Parameters
        ----------
        data :
            Bytes from :meth:`encode_seeds`.

        Returns
        -------
        list of :class:`VolumetricSeed`
        """
        if len(data) < _HEADER_SIZE:
            raise ValueError("Packet too short to contain header")
        magic, epoch, num_seeds = _HEADER_STRUCT.unpack_from(data, 0)
        if magic != _MAGIC:
            raise ValueError(f"Bad magic: {magic!r}")

        result = []
        offset = _HEADER_SIZE
        for _ in range(num_seeds):
            if offset + _SEED_SIZE > len(data):
                break
            layer_id, ax, ay, seed_val, strength_u8 = _SEED_STRUCT.unpack_from(data, offset)
            offset += _SEED_SIZE
            result.append(VolumetricSeed(
                layer_type      = _ID_TO_LAYER.get(layer_id, VolumeLayerType.DUST),
                anchor_x        = float(ax),
                anchor_y        = float(ay),
                seed            = int(seed_val),
                source_strength = strength_u8 / 255.0,
            ))
        return result
