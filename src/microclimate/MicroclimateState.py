"""MicroclimateState — Stage 49 per-chunk microclimate data structure.

Compact representation of local microclimate offsets computed from geometry.
All fields are in ``[0, 1]`` and stored as ``uint8`` values (0–255) internally
to minimise memory when many chunks are cached.

Public API
----------
MicroclimateState(windShelter, windChannel, dustTrap, coldBias,
                  thermalInertia, echoPotential)
  .pack()   → bytes  (6 bytes, one uint8 per field)
  .unpack(b) → MicroclimateState  (class method)
"""
from __future__ import annotations

import struct
from dataclasses import dataclass


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


@dataclass
class MicroclimateState:
    """Per-chunk microclimate proxy values.

    Attributes
    ----------
    windShelter : float
        How sheltered from wind the chunk is (0 = fully exposed, 1 = sheltered).
    windChannel : float
        How much the chunk geometry funnels / accelerates wind (0 = none, 1 = strong).
    dustTrap : float
        Tendency for dust to settle / accumulate (0 = dispersed, 1 = trapped).
    coldBias : float
        Tendency toward lower local temperature (0 = warm, 1 = cold).
    thermalInertia : float
        How slowly temperature changes (0 = fast/exposed, 1 = slow/enclosed).
    echoPotential : float
        Acoustic reflectivity of the enclosure (0 = open, 1 = cave-like).
    """

    windShelter:    float = 0.0
    windChannel:    float = 0.0
    dustTrap:       float = 0.0
    coldBias:       float = 0.0
    thermalInertia: float = 0.0
    echoPotential:  float = 0.0

    _PACK_FMT = "BBBBBB"  # 6 × uint8

    def pack(self) -> bytes:
        """Serialise to 6 bytes (one uint8 per field, 0–255)."""
        def _encode(v: float) -> int:
            return int(_clamp(v) * 255.0 + 0.5)

        return struct.pack(
            self._PACK_FMT,
            _encode(self.windShelter),
            _encode(self.windChannel),
            _encode(self.dustTrap),
            _encode(self.coldBias),
            _encode(self.thermalInertia),
            _encode(self.echoPotential),
        )

    @classmethod
    def unpack(cls, b: bytes) -> "MicroclimateState":
        """Deserialise from 6 bytes produced by :meth:`pack`."""
        if len(b) < 6:
            raise ValueError(f"MicroclimateState.unpack: need 6 bytes, got {len(b)}")
        vals = struct.unpack(cls._PACK_FMT, b[:6])
        return cls(
            windShelter=vals[0] / 255.0,
            windChannel=vals[1] / 255.0,
            dustTrap=vals[2] / 255.0,
            coldBias=vals[3] / 255.0,
            thermalInertia=vals[4] / 255.0,
            echoPotential=vals[5] / 255.0,
        )
