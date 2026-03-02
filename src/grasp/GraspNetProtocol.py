"""GraspNetProtocol — Stage 40 §12.

Lightweight network codec for grasp-constraint replication.

Encodes :class:`~src.grasp.GraspConstraintBinder.ConstraintEvent` objects
into compact byte strings for transmission and decodes them back.

Message format (fixed, little-endian)
--------------------------------------
``GRASP_CREATE`` (kind=0x01):
  1B kind | 4B constraint_id | 4B player_a | 4B player_b | 1B grasp_type |
  4B rest_length_mm | 4B max_force_dN | 4B break_force_dN | 1B damping_u8

``GRASP_BREAK`` (kind=0x02):
  1B kind | 4B constraint_id

``GRASP_UPDATE`` (kind=0x03):
  1B kind | 4B constraint_id | 4B force_dN

Total sizes: create=27 B, break=5 B, update=9 B.

Public API
----------
GraspNetProtocol
  .encode_event(event) → bytes
  .decode_event(data)  → ConstraintEvent
"""
from __future__ import annotations

import struct
from typing import Optional

from src.math.Vec3 import Vec3
from src.grasp.GraspConstraintBinder import (
    GraspConstraint, GraspType, ConstraintEvent,
)


# Kind bytes
_KIND_CREATE = 0x01
_KIND_BREAK  = 0x02
_KIND_UPDATE = 0x03

# Struct formats (little-endian)
_FMT_CREATE = "<BIIIBiII B"  # see module docstring for fields
_FMT_BREAK  = "<BI"
_FMT_UPDATE = "<BIi"

# Conversion helpers
_FORCE_SCALE = 10.0     # dN (deci-Newtons) ↔ N
_LEN_SCALE   = 1000.0   # mm ↔ m


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


class GraspNetProtocol:
    """Encode and decode grasp-constraint network events (§12).

    This class is stateless and all methods are static / classmethods.
    """

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    @staticmethod
    def encode_event(event: ConstraintEvent) -> bytes:
        """Serialise a :class:`ConstraintEvent` to bytes.

        Raises
        ------
        ValueError
            If the event kind is unrecognised.
        """
        c = event.constraint
        if event.kind == "create":
            grasp_type_id = list(GraspType).index(c.grasp_type)
            rest_len_mm   = int(_clamp(c.rest_length * _LEN_SCALE, 0.0, 2**31 - 1))
            max_force_dN  = int(_clamp(c.max_force   * _FORCE_SCALE, 0.0, 2**32 - 1))
            break_force_dN = int(_clamp(c.break_force * _FORCE_SCALE, 0.0, 2**32 - 1))
            damping_u8    = int(_clamp(c.damping, 0.0, 1.0) * 255)
            return struct.pack(
                "<BIIIBiIIB",
                _KIND_CREATE,
                c.id,
                c.player_a,
                c.player_b,
                grasp_type_id,
                rest_len_mm,
                max_force_dN,
                break_force_dN,
                damping_u8,
            )

        elif event.kind == "break":
            return struct.pack("<BI", _KIND_BREAK, c.id)

        elif event.kind == "update":
            force_dN = int(_clamp(c.current_force * _FORCE_SCALE, 0.0, 2**31 - 1))
            return struct.pack("<BIi", _KIND_UPDATE, c.id, force_dN)

        raise ValueError(f"Unknown event kind: {event.kind!r}")

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    @staticmethod
    def decode_event(data: bytes) -> ConstraintEvent:
        """Deserialise bytes to a :class:`ConstraintEvent`.

        Raises
        ------
        ValueError
            If the data is too short or the kind byte is unknown.
        """
        if not data:
            raise ValueError("Empty data")

        kind_byte = data[0]

        if kind_byte == _KIND_CREATE:
            if len(data) < 27:
                raise ValueError(f"CREATE packet too short: {len(data)} < 27")
            kind_b, cid, pa, pb, gt_id, rest_mm, mf_dN, bf_dN, damp_u8 = struct.unpack(
                "<BIIIBiIIB", data[:27]
            )
            grasp_types = list(GraspType)
            if gt_id >= len(grasp_types):
                raise ValueError(f"Unknown GraspType id: {gt_id}")
            c = GraspConstraint(
                id=cid,
                player_a=pa,
                player_b=pb,
                anchor_a=Vec3.zero(),
                anchor_b=Vec3.zero(),
                grasp_type=grasp_types[gt_id],
                max_force=mf_dN / _FORCE_SCALE,
                break_force=bf_dN / _FORCE_SCALE,
                damping=damp_u8 / 255.0,
                rest_length=rest_mm / _LEN_SCALE,
                created_at_tick=0,
            )
            return ConstraintEvent(kind="create", constraint=c)

        elif kind_byte == _KIND_BREAK:
            if len(data) < 5:
                raise ValueError(f"BREAK packet too short: {len(data)} < 5")
            _, cid = struct.unpack("<BI", data[:5])
            c = GraspConstraint(
                id=cid, player_a=0, player_b=0,
                anchor_a=Vec3.zero(), anchor_b=Vec3.zero(),
                grasp_type=GraspType.HAND_TO_HAND,
                max_force=0.0, break_force=0.0, damping=0.0,
                rest_length=0.0, created_at_tick=0,
            )
            return ConstraintEvent(kind="break", constraint=c)

        elif kind_byte == _KIND_UPDATE:
            if len(data) < 9:
                raise ValueError(f"UPDATE packet too short: {len(data)} < 9")
            _, cid, force_dN = struct.unpack("<BIi", data[:9])
            c = GraspConstraint(
                id=cid, player_a=0, player_b=0,
                anchor_a=Vec3.zero(), anchor_b=Vec3.zero(),
                grasp_type=GraspType.HAND_TO_HAND,
                max_force=0.0, break_force=0.0, damping=0.0,
                rest_length=0.0, created_at_tick=0,
                current_force=force_dN / _FORCE_SCALE,
            )
            return ConstraintEvent(kind="update", constraint=c)

        raise ValueError(f"Unknown kind byte: 0x{kind_byte:02X}")
