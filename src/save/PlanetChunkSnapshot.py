"""PlanetChunkSnapshot — Stage 63 save / restore for PlanetChunkGrid.

Persists :class:`~src.material.PlanetChunkState.PlanetChunkGrid` objects in
a compact binary snapshot format.

Format
------
Magic:       b"PCS1"  (4 bytes)
Chunk count: uint32
Per chunk:
  chunk_id_hash: uint32
  w:             uint16
  h:             uint16
  data_len:      uint32
  data:          <data_len bytes>  (raw 11-bytes-per-cell grid payload)

Public API
----------
PlanetChunkSnapshot()
  .save(grids: Dict) -> bytes
  .load(data: bytes) -> Dict[int, PlanetChunkGrid]
"""
from __future__ import annotations

import hashlib
import struct
from typing import Dict

from src.material.PlanetChunkState import PlanetChunkGrid


_MAGIC   = b"PCS1"
_HDR_FMT = "!4sI"          # magic(4) + chunk_count(4)
_HDR_SZ  = struct.calcsize(_HDR_FMT)
_CHK_FMT = "!IHHI"         # cid_hash(4) + w(2) + h(2) + data_len(4)


def _hash32(chunk_id: object) -> int:
    """Stable 32-bit hash of an arbitrary chunk_id."""
    raw = str(chunk_id).encode("utf-8")
    return int(hashlib.md5(raw).hexdigest()[:8], 16) & 0xFFFFFFFF


class PlanetChunkSnapshot:
    """Serialises and deserialises a collection of PlanetChunkGrid objects.

    Usage::

        snap = PlanetChunkSnapshot()
        blob = snap.save(grids)       # bytes
        restored = snap.load(blob)    # Dict[int, PlanetChunkGrid]
    """

    def save(
        self,
        grids: Dict[object, PlanetChunkGrid],
    ) -> bytes:
        """Serialise *grids* to bytes.

        Parameters
        ----------
        grids :
            Mapping of chunk_id → PlanetChunkGrid.
        """
        chunks_out = bytearray()
        count = 0
        for chunk_id, grid in grids.items():
            cid_hash = _hash32(chunk_id)
            raw = grid.to_bytes()
            # Per-chunk header: cid_hash(4) + w(2) + h(2) + data_len(4)
            header = struct.pack(_CHK_FMT, cid_hash, grid.w, grid.h, len(raw))
            chunks_out += header + raw
            count += 1
        header_main = struct.pack(_HDR_FMT, _MAGIC, count)
        return bytes(header_main) + bytes(chunks_out)

    def load(
        self,
        data: bytes,
    ) -> Dict[int, PlanetChunkGrid]:
        """Restore grids from bytes produced by :meth:`save`.

        Returns
        -------
        dict
            Mapping of chunk_id_hash (int) → PlanetChunkGrid.
        """
        magic, count = struct.unpack_from(_HDR_FMT, data, 0)
        if magic != _MAGIC:
            raise ValueError(f"Bad magic: {magic!r} (expected {_MAGIC!r})")

        offset = _HDR_SZ
        result: Dict[int, PlanetChunkGrid] = {}
        chunk_hdr_fmt = _CHK_FMT
        chunk_hdr_sz  = struct.calcsize(chunk_hdr_fmt)

        for _ in range(count):
            cid_hash, w, h, data_len = struct.unpack_from(
                chunk_hdr_fmt, data, offset
            )
            offset += chunk_hdr_sz
            raw = data[offset: offset + data_len]
            offset += data_len

            grid = PlanetChunkGrid(chunk_id=cid_hash, w=w, h=h)
            grid.from_bytes(raw)
            result[cid_hash] = grid

        return result
