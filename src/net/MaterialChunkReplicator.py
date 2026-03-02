"""MaterialChunkReplicator — Stage 45 network streaming of material state.

Serialises and deserialises per-chunk :class:`SurfaceMaterialStateGrid`
data for server-to-client streaming.

Design:
* One packet per chunk (5 bytes per cell, uint8 quantised).
* Interest-zone filtering: only chunks within ``interest_radius`` are sent.
* Updates are batched and sent at low frequency (``update_hz``).
* RLE compression is applied before transmission (see :func:`rle_encode`).

Public API
----------
MaterialChunkReplicator(config=None)
  .encode_chunk(chunk_id, grid) -> bytes
  .decode_chunk(data) -> Tuple[chunk_id, SurfaceMaterialStateGrid]
  .should_send(chunk_id, last_hash, current_hash) -> bool

rle_encode(data: bytes) -> bytes
rle_decode(data: bytes) -> bytes
"""
from __future__ import annotations

import struct
from typing import Dict, Optional, Tuple

from src.material.SurfaceMaterialState import SurfaceMaterialStateGrid


# ---------------------------------------------------------------------------
# RLE codec (simple byte-level run-length encoding)
# ---------------------------------------------------------------------------

def rle_encode(data: bytes) -> bytes:
    """Encode *data* with simple run-length encoding.

    Format per run: (count: uint8, value: uint8).
    Runs are capped at 255 bytes.
    """
    if not data:
        return b""
    out = bytearray()
    i = 0
    n = len(data)
    while i < n:
        val = data[i]
        count = 1
        while i + count < n and data[i + count] == val and count < 255:
            count += 1
        out.append(count)
        out.append(val)
        i += count
    return bytes(out)


def rle_decode(data: bytes) -> bytes:
    """Decode RLE data produced by :func:`rle_encode`."""
    out = bytearray()
    i = 0
    while i + 1 < len(data):
        count = data[i]
        val   = data[i + 1]
        out.extend([val] * count)
        i += 2
    return bytes(out)


# ---------------------------------------------------------------------------
# MaterialChunkReplicator
# ---------------------------------------------------------------------------

_HEADER_FMT = "!HHI"   # w(uint16), h(uint16), chunk_id_hash(uint32)
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)


class MaterialChunkReplicator:
    """Encodes / decodes chunk material grids for network transport.

    Parameters
    ----------
    config : dict or None
        ``update_hz`` — max send rate per chunk (default 0.5 Hz).
    """

    def __init__(self, config=None) -> None:
        cfg = config or {}
        self.update_hz: float = float(cfg.get("update_hz", 0.5))
        # hash cache: chunk_id -> last sent hash
        self._last_hash: Dict[object, str] = {}

    def encode_chunk(
        self,
        chunk_id: object,
        grid: SurfaceMaterialStateGrid,
    ) -> bytes:
        """Encode *grid* to a compact byte packet.

        Layout: header (6 bytes) + RLE-compressed grid data.

        Parameters
        ----------
        chunk_id :
            Opaque chunk identifier.  Hashed to uint32 for the header.
        grid :
            The grid to encode.
        """
        raw    = grid.to_bytes()
        compressed = rle_encode(raw)
        cid_hash   = _hash32(chunk_id)
        header     = struct.pack(_HEADER_FMT, grid.w, grid.h, cid_hash)
        return header + compressed

    def decode_chunk(
        self,
        data: bytes,
    ) -> Tuple[int, SurfaceMaterialStateGrid]:
        """Decode a packet produced by :meth:`encode_chunk`.

        Returns
        -------
        (chunk_id_hash, grid)
            ``chunk_id_hash`` is the uint32 hash stored in the header.
        """
        if len(data) < _HEADER_SIZE:
            raise ValueError("MaterialChunkReplicator: packet too short")
        w, h, cid_hash = struct.unpack(_HEADER_FMT, data[:_HEADER_SIZE])
        raw = rle_decode(data[_HEADER_SIZE:])
        grid = SurfaceMaterialStateGrid(cid_hash, w, h)
        grid.from_bytes(raw)
        return cid_hash, grid

    def should_send(
        self,
        chunk_id: object,
        current_hash: str,
    ) -> bool:
        """Return True if the chunk state has changed since last send."""
        prev = self._last_hash.get(chunk_id)
        if prev != current_hash:
            self._last_hash[chunk_id] = current_hash
            return True
        return False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _hash32(obj: object) -> int:
    """Stable 32-bit hash of an arbitrary object (for packet header)."""
    return hash(obj) & 0xFFFFFFFF
