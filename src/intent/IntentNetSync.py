"""IntentNetSync — Stage 38 network synchronisation for MotorIntent (§12).

Provides compact serialisation / deserialisation of
:class:`~src.intent.StrategySelector.MotorIntent` and
:class:`~src.intent.StrategySelector.MotorMode` for replication over the
multiplayer network layer.

The encoding is intentionally compact (< 40 bytes per packet) and
uses only deterministic operations — no random numbers.

Wire format (38 bytes total)
----------------------------
Offset  Size  Field
------  ----  -----
0       1     mode index (uint8)
1       4     desiredVelocity.x  (float32, little-endian)
5       4     desiredVelocity.y
9       4     desiredVelocity.z
13      1     stanceWidthBias    (uint8, scaled 0–255)
14      1     stepLengthBias     (uint8, scaled 0–255)
15      1     bracePreference    (uint8, scaled 0–255)
16      4     attentionTargetDir.x (float32)
20      4     attentionTargetDir.y
24      4     attentionTargetDir.z
28      1     proximityPreference  (uint8, scaled 0–255, max 2.0)
29      1     assistWillingness    (uint8, scaled 0–255)
30      8     reserved / padding (zero-filled)

Public API
----------
IntentNetSync.encode(mode, intent) → bytes
IntentNetSync.decode(data)         → (MotorMode, MotorIntent)
"""
from __future__ import annotations

import struct
from typing import Tuple

from src.math.Vec3 import Vec3
from src.intent.StrategySelector import MotorMode, MotorIntent


_PACK_VEC_FMT = "<fff"
_HEADER_FMT   = "<B"
_SCALE_U8     = 255.0
_RESERVED     = b"\x00" * 8
_WIRE_SIZE    = 38


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


def _u8(v: float, max_val: float = 1.0) -> int:
    return round(_clamp(v / max_val, 0.0, 1.0) * _SCALE_U8)


def _from_u8(b: int, max_val: float = 1.0) -> float:
    return (b / _SCALE_U8) * max_val


class IntentNetSync:
    """Stateless codec for MotorIntent wire packets."""

    @staticmethod
    def encode(mode: MotorMode, intent: MotorIntent) -> bytes:
        """Serialise *mode* + *intent* to a compact 38-byte packet.

        Parameters
        ----------
        mode :
            Current MotorMode enum value.
        intent :
            Current MotorIntent to encode.

        Returns
        -------
        bytes
            38-byte wire packet.
        """
        dv  = intent.desiredVelocity
        atd = intent.attentionTargetDir

        buf = struct.pack(
            _HEADER_FMT, mode.value
        )
        buf += struct.pack(_PACK_VEC_FMT, dv.x, dv.y, dv.z)
        buf += bytes([
            _u8(intent.stanceWidthBias),
            _u8(intent.stepLengthBias),
            _u8(intent.bracePreference),
        ])
        buf += struct.pack(_PACK_VEC_FMT, atd.x, atd.y, atd.z)
        buf += bytes([
            _u8(intent.proximityPreference, max_val=2.0),
            _u8(intent.assistWillingness),
        ])
        buf += _RESERVED
        assert len(buf) == _WIRE_SIZE, f"Expected {_WIRE_SIZE} bytes, got {len(buf)}"
        return buf

    @staticmethod
    def decode(data: bytes) -> Tuple[MotorMode, MotorIntent]:
        """Deserialise a 38-byte packet back to (MotorMode, MotorIntent).

        Parameters
        ----------
        data :
            Exactly 38 bytes as produced by :meth:`encode`.

        Returns
        -------
        (MotorMode, MotorIntent)

        Raises
        ------
        ValueError
            If ``data`` is not exactly ``_WIRE_SIZE`` bytes.
        """
        if len(data) != _WIRE_SIZE:
            raise ValueError(
                f"IntentNetSync.decode: expected {_WIRE_SIZE} bytes, got {len(data)}"
            )

        mode_idx = struct.unpack_from(_HEADER_FMT, data, 0)[0]
        mode = MotorMode(mode_idx)

        dvx, dvy, dvz = struct.unpack_from(_PACK_VEC_FMT, data, 1)
        stance  = _from_u8(data[13])
        step    = _from_u8(data[14])
        brace   = _from_u8(data[15])
        adx, ady, adz = struct.unpack_from(_PACK_VEC_FMT, data, 16)
        prox   = _from_u8(data[28], max_val=2.0)
        assist = _from_u8(data[29])

        intent = MotorIntent(
            desiredVelocity     = Vec3(dvx, dvy, dvz),
            stanceWidthBias     = stance,
            stepLengthBias      = step,
            bracePreference     = brace,
            attentionTargetDir  = Vec3(adx, ady, adz),
            proximityPreference = prox,
            assistWillingness   = assist,
        )
        return mode, intent
