"""WorldMemorySnapshot — Stage 51 binary save / restore for WorldMemoryState.

Format
------
Magic:        b"WMS1"  (4 bytes)
Header:       sim_time(8, double) + width(2, uint16) + height(2, uint16)
Fields:       4 × (width × height) bytes — one byte per tile per field,
              in order: stress, compaction, erosionBias, acousticImprint

Total size for a 32 × 16 grid:
  4 + (8 + 2 + 2) + 4 × 512 = 4 + 12 + 2048 = 2064 bytes

Public API
----------
WorldMemorySnapshot()
  .save(memory_state, sim_time=0.0) -> bytes
  .load(data: bytes) -> (WorldMemoryState, dict)
      dict keys: sim_time
"""
from __future__ import annotations

import struct
from typing import Dict, Tuple

from src.memory.WorldMemoryState import WorldMemoryState

_MAGIC    = b"WMS1"
_HDR_FMT  = "!dHH"    # network (big-endian): sim_time(d) + width(H) + height(H)
_HDR_SIZE = struct.calcsize(_HDR_FMT)
_PREAMBLE = 4 + _HDR_SIZE   # magic + header


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


class WorldMemorySnapshot:
    """Binary serialiser for :class:`WorldMemoryState`.

    Usage::

        snap = WorldMemorySnapshot()
        blob = snap.save(memory_state, sim_time=42.0)
        state2, meta = snap.load(blob)
        # meta["sim_time"]
    """

    def save(
        self,
        memory_state: WorldMemoryState,
        sim_time: float = 0.0,
    ) -> bytes:
        """Serialise *memory_state* to bytes.

        Parameters
        ----------
        memory_state : WorldMemoryState to persist.
        sim_time     : Current simulation time to embed in the snapshot.
        """
        header = struct.pack(
            _HDR_FMT,
            sim_time,
            memory_state.width,
            memory_state.height,
        )

        n = memory_state.size()
        fields = bytearray(4 * n)
        for i in range(n):
            fields[i]         = int(_clamp(memory_state.stressAccumulationField[i]) * 255)
            fields[n + i]     = int(_clamp(memory_state.compactionHistoryField[i])  * 255)
            fields[2 * n + i] = int(_clamp(memory_state.erosionBiasField[i])        * 255)
            fields[3 * n + i] = int(_clamp(memory_state.acousticImprintField[i])    * 255)

        return _MAGIC + header + bytes(fields)

    def load(
        self,
        data: bytes,
    ) -> Tuple[WorldMemoryState, Dict]:
        """Deserialise a blob produced by :meth:`save`.

        Returns
        -------
        (memory_state, meta)
            ``memory_state`` is a newly-created :class:`WorldMemoryState`.
            ``meta`` is a dict with key ``sim_time``.
        """
        if data[:4] != _MAGIC:
            raise ValueError("WorldMemorySnapshot: bad magic bytes")

        sim_time, width, height = struct.unpack_from(_HDR_FMT, data, offset=4)

        state = WorldMemoryState(width=int(width), height=int(height))
        n = int(width) * int(height)
        offset = _PREAMBLE

        for i in range(n):
            state.stressAccumulationField[i] = data[offset + i]           / 255.0
            state.compactionHistoryField[i]  = data[offset + n + i]       / 255.0
            state.erosionBiasField[i]        = data[offset + 2 * n + i]   / 255.0
            state.acousticImprintField[i]    = data[offset + 3 * n + i]   / 255.0

        meta = {"sim_time": float(sim_time)}
        return state, meta
