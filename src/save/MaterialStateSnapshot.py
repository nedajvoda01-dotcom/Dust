"""MaterialStateSnapshot — Stage 45 save / restore for material state grids.

Persists :class:`~src.material.SurfaceMaterialState.SurfaceMaterialStateGrid`
objects in a compact binary snapshot format suitable for inclusion in the
world baseline snapshot (Stages 24/42).

Format
------
Magic: b"MSS1"  (4 bytes)
Chunk count: uint32
Per chunk:
  chunk_id_hash: uint32
  w: uint16
  h: uint16
  rle_len: uint32
  rle_data: <rle_len bytes>

Public API
----------
MaterialStateSnapshot()
  .save(grids: Dict) -> bytes
  .load(data: bytes) -> Dict[int, SurfaceMaterialStateGrid]
"""
from __future__ import annotations

import struct
from typing import Dict

from src.material.SurfaceMaterialState import SurfaceMaterialStateGrid
from src.net.MaterialChunkReplicator import rle_encode, rle_decode, _hash32


_MAGIC    = b"MSS1"
_HDR_FMT  = "!4sI"          # magic + chunk_count


class MaterialStateSnapshot:
    """Serialises and deserialises a collection of material state grids.

    Usage::

        snap = MaterialStateSnapshot()
        blob = snap.save(grids)        # bytes
        restored = snap.load(blob)     # Dict[int, SurfaceMaterialStateGrid]
    """

    def save(
        self,
        grids: Dict[object, SurfaceMaterialStateGrid],
    ) -> bytes:
        """Serialise *grids* to bytes.

        Parameters
        ----------
        grids :
            Mapping of chunk_id → grid.
        """
        chunks_out = bytearray()
        count = 0
        for chunk_id, grid in grids.items():
            raw = grid.to_bytes()
            compressed = rle_encode(raw)
            cid_hash = _hash32(chunk_id)
            # chunk_id_hash(4) + w(2) + h(2) + rle_len(4) + data
            chunks_out += struct.pack("!IHH", cid_hash, grid.w, grid.h)
            chunks_out += struct.pack("!I", len(compressed))
            chunks_out += compressed
            count += 1

        header = _MAGIC + struct.pack("!I", count)
        return header + bytes(chunks_out)

    def load(
        self,
        data: bytes,
    ) -> Dict[int, SurfaceMaterialStateGrid]:
        """Deserialise a blob produced by :meth:`save`.

        Returns a dict mapping chunk_id_hash (int) → grid.
        """
        if data[:4] != _MAGIC:
            raise ValueError("MaterialStateSnapshot: bad magic bytes")

        count = struct.unpack("!I", data[4:8])[0]
        pos = 8
        result: Dict[int, SurfaceMaterialStateGrid] = {}

        for _ in range(count):
            cid_hash, w, h = struct.unpack("!IHH", data[pos: pos + 8])
            pos += 8
            rle_len = struct.unpack("!I", data[pos: pos + 4])[0]
            pos += 4
            compressed = data[pos: pos + rle_len]
            pos += rle_len
            raw = rle_decode(compressed)
            grid = SurfaceMaterialStateGrid(cid_hash, w, h)
            grid.from_bytes(raw)
            result[cid_hash] = grid

        return result
