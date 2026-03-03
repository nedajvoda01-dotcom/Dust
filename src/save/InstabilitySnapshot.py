"""InstabilitySnapshot — Stage 52 binary save / restore for InstabilityState.

Format
------
Magic:   b"INS1"  (4 bytes)
Header:  sim_time(8, double) + width(2, uint16) + height(2, uint16)
Fields:  5 × (width × height) bytes — one byte per tile per field,
         in order: shearStress, crustFailure, massOverhang,
                   dustLoad, thermalGradient

Total size for a 32 × 16 grid:
  4 + (8 + 2 + 2) + 5 × 512 = 4 + 12 + 2560 = 2576 bytes

Public API
----------
InstabilitySnapshot()
  .save(state, sim_time=0.0) -> bytes
  .load(data: bytes) -> (InstabilityState, dict)
      dict keys: sim_time
"""
from __future__ import annotations

import struct
from typing import Dict, Tuple

from src.instability.InstabilityState import InstabilityState

_MAGIC    = b"INS1"
_HDR_FMT  = "!dHH"   # sim_time(d) + width(H) + height(H)
_HDR_SIZE = struct.calcsize(_HDR_FMT)
_PREAMBLE = 4 + _HDR_SIZE


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


class InstabilitySnapshot:
    """Binary serialiser for :class:`InstabilityState`.

    Usage::

        snap = InstabilitySnapshot()
        blob = snap.save(state, sim_time=42.0)
        state2, meta = snap.load(blob)
        # meta["sim_time"]
    """

    _FIELD_ORDER = (
        "shearStressField",
        "crustFailurePotential",
        "massOverhangField",
        "dustLoadField",
        "thermalGradientField",
    )

    def save(
        self,
        state:    InstabilityState,
        sim_time: float = 0.0,
    ) -> bytes:
        """Serialise *state* to bytes.

        Parameters
        ----------
        state    : InstabilityState to persist.
        sim_time : Simulation time to embed in the snapshot.
        """
        header = struct.pack(_HDR_FMT, sim_time, state.width, state.height)
        n = state.size()
        num_fields = len(self._FIELD_ORDER)
        fields = bytearray(num_fields * n)

        for f_idx, name in enumerate(self._FIELD_ORDER):
            lst = getattr(state, name)
            base = f_idx * n
            for i in range(n):
                fields[base + i] = int(_clamp(lst[i]) * 255)

        return _MAGIC + header + bytes(fields)

    def load(
        self,
        data: bytes,
    ) -> Tuple[InstabilityState, Dict]:
        """Deserialise a blob produced by :meth:`save`.

        Returns
        -------
        (state, meta)
            ``state`` is a newly-created :class:`InstabilityState`.
            ``meta`` is a dict with key ``sim_time``.
        """
        if data[:4] != _MAGIC:
            raise ValueError("InstabilitySnapshot: bad magic bytes")

        (sim_time, width, height) = struct.unpack_from(_HDR_FMT, data, offset=4)

        state = InstabilityState(width=int(width), height=int(height))
        n = int(width) * int(height)
        offset = _PREAMBLE

        for f_idx, name in enumerate(self._FIELD_ORDER):
            lst = getattr(state, name)
            base = offset + f_idx * n
            for i in range(n):
                lst[i] = data[base + i] / 255.0

        meta = {"sim_time": float(sim_time)}
        return state, meta
