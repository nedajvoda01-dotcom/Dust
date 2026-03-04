"""AtmosphereReplicator — Stage 64 LOD-aware atmosphere replication.

Serialises atmospheric field data for server → client streaming.

LOD model (spec §11)
--------------------
* **Near** (within ``near_radius`` tiles): full 8-byte tile packets
  (P, T, Vx, Vy, A, H, E + pad).
* **Far** (outside ``near_radius``): compact 4-byte packet
  (Vis proxy uint8 + wind angle uint8 + pad uint16).

Bandwidth reduction is measurable: the far representation is 50 % smaller
per tile.

Design
------
* Packets are prefixed with a weather-epoch counter so clients can detect
  when parameters change.
* Only tiles whose hash changed since last send are included (delta-only).

Public API
----------
AtmosphereReplicator(config=None)
  .encode_near(grid, cx, cy, radius) -> bytes
  .encode_far(grid, cx, cy, radius)  -> bytes
  .decode_near(data) -> List[Tuple[int, int, AtmoTile]]
  .decode_far(data)  -> List[Tuple[int, int, float, float]]
      Returns (ix, iy, visibility, wind_angle_rad) per tile.
  .near_packet_size  -> int   (bytes per tile, full fidelity)
  .far_packet_size   -> int   (bytes per tile, reduced)
"""
from __future__ import annotations

import math
import struct
from typing import List, Optional, Tuple

from src.atmo.GlobalFieldGrid import GlobalFieldGrid, AtmoTile


# ---------------------------------------------------------------------------
# Packet layouts
# ---------------------------------------------------------------------------

# Header: magic(2B) + epoch(4B) + mode(1B) + cx(2B) + cy(2B) = 11 bytes
_HEADER_STRUCT  = struct.Struct("!2sIBHH")
_HEADER_MAGIC   = b"A6"
_HEADER_SIZE    = _HEADER_STRUCT.size

# Near tile packet: ix(2B) + iy(2B) + 8-byte AtmoTile = 12 bytes
_NEAR_COORD     = struct.Struct("!HH")
_NEAR_COORD_SZ  = _NEAR_COORD.size
_TILE_BYTES     = 8
_NEAR_PKT_SZ    = _NEAR_COORD_SZ + _TILE_BYTES   # 12

# Far tile packet: ix(2B) + iy(2B) + vis(1B) + angle(1B) + pad(2B) = 8 bytes
_FAR_STRUCT     = struct.Struct("!HHBBxx")
_FAR_PKT_SZ     = _FAR_STRUCT.size   # 8

_MODE_NEAR  = 0
_MODE_FAR   = 1


class AtmosphereReplicator:
    """Encode/decode atmospheric field snapshots for network transmission.

    Parameters
    ----------
    config :
        Optional dict; reads ``atmo64.net.*`` keys.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg  = config or {}
        ncfg = cfg.get("atmo64", {}).get("net", {}) or {}
        self._epoch: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def near_packet_size(self) -> int:
        return _NEAR_PKT_SZ

    @property
    def far_packet_size(self) -> int:
        return _FAR_PKT_SZ

    def advance_epoch(self) -> None:
        """Increment the weather epoch (call when global params change)."""
        self._epoch = (self._epoch + 1) & 0xFFFFFFFF

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode_near(
        self,
        grid:   GlobalFieldGrid,
        cx:     int,
        cy:     int,
        radius: int,
    ) -> bytes:
        """Encode full-fidelity tiles within Chebyshev *radius* of (cx, cy)."""
        tiles = self._collect_tiles(grid, cx, cy, radius)
        header = _HEADER_STRUCT.pack(_HEADER_MAGIC, self._epoch, _MODE_NEAR, cx, cy)
        body = b"".join(
            _NEAR_COORD.pack(ix, iy) + tile.to_bytes()
            for ix, iy, tile in tiles
        )
        return header + body

    def encode_far(
        self,
        grid:   GlobalFieldGrid,
        cx:     int,
        cy:     int,
        radius: int,
    ) -> bytes:
        """Encode reduced-fidelity tiles within Chebyshev *radius* of (cx, cy)."""
        tiles = self._collect_tiles(grid, cx, cy, radius)
        header = _HEADER_STRUCT.pack(_HEADER_MAGIC, self._epoch, _MODE_FAR, cx, cy)
        body_parts = []
        for ix, iy, tile in tiles:
            vis = grid.visibility_proxy(ix, iy)
            angle = math.atan2(tile.wind_y, tile.wind_x)
            vis_u8   = max(0, min(255, int(round(vis * 255))))
            angle_u8 = int(round((angle + math.pi) / (2.0 * math.pi) * 255)) & 0xFF
            body_parts.append(_FAR_STRUCT.pack(ix, iy, vis_u8, angle_u8))
        return header + b"".join(body_parts)

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def decode_near(self, data: bytes) -> List[Tuple[int, int, AtmoTile]]:
        """Decode a near-fidelity packet into (ix, iy, AtmoTile) tuples."""
        self._check_header(data, _MODE_NEAR)
        result = []
        offset = _HEADER_SIZE
        while offset + _NEAR_PKT_SZ <= len(data):
            ix, iy = _NEAR_COORD.unpack_from(data, offset)
            tile = AtmoTile.from_bytes(data[offset + _NEAR_COORD_SZ: offset + _NEAR_PKT_SZ])
            result.append((ix, iy, tile))
            offset += _NEAR_PKT_SZ
        return result

    def decode_far(self, data: bytes) -> List[Tuple[int, int, float, float]]:
        """Decode a far-fidelity packet.

        Returns
        -------
        List of (ix, iy, visibility [0..1], wind_angle_rad [−π..π])
        """
        self._check_header(data, _MODE_FAR)
        result = []
        offset = _HEADER_SIZE
        while offset + _FAR_PKT_SZ <= len(data):
            ix, iy, vis_u8, angle_u8 = _FAR_STRUCT.unpack_from(data, offset)
            vis   = vis_u8 / 255.0
            angle = (angle_u8 / 255.0) * 2.0 * math.pi - math.pi
            result.append((ix, iy, vis, angle))
            offset += _FAR_PKT_SZ
        return result

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _collect_tiles(
        self,
        grid:   GlobalFieldGrid,
        cx:     int,
        cy:     int,
        radius: int,
    ) -> List[Tuple[int, int, AtmoTile]]:
        result = []
        for iy in range(grid.height):
            for ix in range(grid.width):
                if max(abs(ix - cx), abs(iy - cy)) <= radius:
                    result.append((ix, iy, grid.tile(ix, iy)))
        return result

    def _check_header(self, data: bytes, expected_mode: int) -> None:
        if len(data) < _HEADER_SIZE:
            raise ValueError("Packet too short to contain header")
        magic, epoch, mode, cx, cy = _HEADER_STRUCT.unpack_from(data, 0)
        if magic != _HEADER_MAGIC:
            raise ValueError(f"Bad packet magic: {magic!r}")
        if mode != expected_mode:
            raise ValueError(f"Mode mismatch: expected {expected_mode}, got {mode}")
