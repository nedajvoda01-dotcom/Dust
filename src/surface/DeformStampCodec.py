"""DeformStampCodec — Stage 35 quantised stamp encoding for network transport.

A "deform stamp" is a compact description of a contact load that any client
can replay deterministically to reproduce the same H/M field changes.

DeformStamp
    Parameters of a single deformation event (centre, radius, depth, push).

DeformStampBatch
    A list of stamps for one network packet (2–5 Hz broadcast).

DeformStampCodec
    encode_batch(stamps) → bytes
    decode_batch(data)   → List[DeformStamp]

Encoding format (little-endian):
    Header (4 bytes):
        uint16  version = 1
        uint16  count
    Per stamp (14 bytes):
        int16   lat_e2   (latitude  × 1e2, clamped to ±9000, resolution 0.01°)
        int16   lon_e2   (longitude × 1e2, clamped to ±18000, resolution 0.01°)
        uint16  radius_m (radius in metres, clamped to 65535)
        int16   depth_mm (H depth in mm, ±32767)
        int8    push_dir_x8 (push direction X × 127, ±127)
        int8    push_dir_y8 (push direction Y × 127, ±127)
        uint8   push_amount_u8 (push amount × 255)
        uint8   material_u8 (MaterialClass ordinal)
        uint32  tick_index
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import List

from src.physics.MaterialYieldModel import MaterialClass

# Packet version
_VERSION = 1

# Per-stamp struct format: h h H h b b B B I
_STAMP_FMT  = "<hhHhbbBBI"
_STAMP_SIZE = struct.calcsize(_STAMP_FMT)   # should be 14 bytes
_HEADER_FMT = "<HH"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)  # 4 bytes

# Minimum absolute depth (mm) to apply a stamp; filters noise
MIN_DEPTH_MM = 1


@dataclass
class DeformStamp:
    """Compact network-transportable deformation event.

    Attributes
    ----------
    lat, lon:
        Geographic centre of the stamp (degrees).
    radius_m:
        Affected radius [m] (clamped to uint16 range).
    depth_m:
        Vertical displacement depth [m] (negative = indent).
    push_dir_x, push_dir_y:
        Unit vector of mass-push direction in local tangent plane.
    push_amount:
        Magnitude of mass push [0, 1].
    material:
        MaterialClass at this location.
    tick_index:
        Source tick index for ordering.
    """
    lat:          float
    lon:          float
    radius_m:     float
    depth_m:      float
    push_dir_x:   float
    push_dir_y:   float
    push_amount:  float
    material:     MaterialClass = MaterialClass.DUST
    tick_index:   int = 0


@dataclass
class DeformStampBatch:
    """A batch of stamps for one network packet."""
    stamps: List[DeformStamp] = field(default_factory=list)

    def is_empty(self) -> bool:
        return len(self.stamps) == 0


class DeformStampCodec:
    """Encode and decode DeformStampBatch objects to/from bytes."""

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------

    @staticmethod
    def encode_batch(batch: DeformStampBatch) -> bytes:
        """Serialise *batch* to bytes.

        Stamps with |depth| below MIN_DEPTH_MM are silently dropped to
        reduce traffic for very weak contacts.
        """
        filtered = [
            s for s in batch.stamps
            if abs(round(s.depth_m * 1000.0)) >= MIN_DEPTH_MM
        ]

        header = struct.pack(_HEADER_FMT, _VERSION, len(filtered))
        parts  = [header]

        for s in filtered:
            lat_e2   = int(round(_clamp(s.lat,  -90.0,  90.0) * 1e2))
            lon_e2   = int(round(_clamp(s.lon, -180.0, 180.0) * 1e2))
            rad_m    = int(round(_clamp(s.radius_m, 0.0, 65535.0)))
            depth_mm = int(round(_clamp(s.depth_m * 1000.0, -32767.0, 32767.0)))
            pdx      = int(round(_clamp(s.push_dir_x, -1.0, 1.0) * 127.0))
            pdy      = int(round(_clamp(s.push_dir_y, -1.0, 1.0) * 127.0))
            pa       = int(round(_clamp(s.push_amount,  0.0,  1.0) * 255.0))
            mat_u8   = s.material.value & 0xFF

            parts.append(struct.pack(
                _STAMP_FMT,
                lat_e2, lon_e2, rad_m, depth_mm,
                pdx, pdy, pa, mat_u8,
                s.tick_index & 0xFFFFFFFF,
            ))

        return b"".join(parts)

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    @staticmethod
    def decode_batch(data: bytes) -> List[DeformStamp]:
        """Deserialise bytes into a list of DeformStamps.

        Returns an empty list on malformed input.
        """
        if len(data) < _HEADER_SIZE:
            return []

        version, count = struct.unpack_from(_HEADER_FMT, data, 0)
        if version != _VERSION:
            return []

        stamps: List[DeformStamp] = []
        offset = _HEADER_SIZE

        for _ in range(count):
            if offset + _STAMP_SIZE > len(data):
                break
            (
                lat_e2, lon_e2, rad_m, depth_mm,
                pdx, pdy, pa, mat_u8,
                tick_idx,
            ) = struct.unpack_from(_STAMP_FMT, data, offset)
            offset += _STAMP_SIZE

            # Resolve material; default DUST if ordinal unknown
            try:
                mat = MaterialClass(mat_u8)
            except ValueError:
                mat = MaterialClass.DUST

            stamps.append(DeformStamp(
                lat         = lat_e2 / 1e2,
                lon         = lon_e2 / 1e2,
                radius_m    = float(rad_m),
                depth_m     = depth_mm / 1000.0,
                push_dir_x  = pdx / 127.0,
                push_dir_y  = pdy / 127.0,
                push_amount = pa  / 255.0,
                material    = mat,
                tick_index  = tick_idx,
            ))

        return stamps


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)
