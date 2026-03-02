"""SurfaceMaterialState — Stage 45 per-chunk surface material state.

Stores five quantised scalar fields (uint8 each) for one surface chunk:

* dustThickness   (0..1) — loose granular layer thickness
* snowCompaction  (0..1) — 0 = fluffy, 1 = hard packed / naст
* iceFilm         (0..1) — thin ice coating
* crustHardness   (0..1) — sintered/baked crust
* roughness       (0..1) — micro-roughness (1 = raw, 0 = wind-polished)

Each field is stored as uint8 (0..255) for cheap storage and network
transport.  The public API works in normalised [0, 1] floats.

Hashing, byte serialisation, and RLE helpers live here so that other
modules (snapshot, replicator) share the same codec.

Public API
----------
SurfaceMaterialState(chunk_id)
  .dust_thickness  -> float (property, r/w)
  .snow_compaction -> float
  .ice_film        -> float
  .crust_hardness  -> float
  .roughness       -> float
  .to_bytes() -> bytes      # 5 uint8 bytes
  .from_bytes(data)         # in-place decode
  .state_hash() -> str      # hex MD5 of raw bytes
  .copy() -> SurfaceMaterialState

SurfaceMaterialStateGrid(chunk_id, w, h)
  — grid of SurfaceMaterialState cells
  .cell(ix, iy) -> SurfaceMaterialState
  .to_bytes() -> bytes
  .from_bytes(data)
  .grid_hash() -> str
"""
from __future__ import annotations

import hashlib
import struct
from typing import List


_UINT8_MAX = 255


def _f2u(v: float) -> int:
    """Float [0,1] → uint8."""
    return max(0, min(_UINT8_MAX, int(round(v * _UINT8_MAX))))


def _u2f(v: int) -> float:
    """uint8 → float [0,1]."""
    return v / _UINT8_MAX


# ---------------------------------------------------------------------------
# SurfaceMaterialState
# ---------------------------------------------------------------------------

class SurfaceMaterialState:
    """Compact surface material state for one grid cell / chunk.

    Parameters
    ----------
    chunk_id :
        Opaque identifier (for logging / hashing).
    """

    __slots__ = (
        "chunk_id",
        "_dust",
        "_snow",
        "_ice",
        "_crust",
        "_rough",
    )

    def __init__(
        self,
        chunk_id: object = None,
        *,
        dust_thickness: float = 0.2,
        snow_compaction: float = 0.0,
        ice_film: float = 0.0,
        crust_hardness: float = 0.0,
        roughness: float = 0.5,
    ) -> None:
        self.chunk_id = chunk_id
        self._dust  = _f2u(dust_thickness)
        self._snow  = _f2u(snow_compaction)
        self._ice   = _f2u(ice_film)
        self._crust = _f2u(crust_hardness)
        self._rough = _f2u(roughness)

    # ------------------------------------------------------------------
    # Properties (normalised float [0,1])
    # ------------------------------------------------------------------

    @property
    def dust_thickness(self) -> float:
        return _u2f(self._dust)

    @dust_thickness.setter
    def dust_thickness(self, v: float) -> None:
        self._dust = _f2u(v)

    @property
    def snow_compaction(self) -> float:
        return _u2f(self._snow)

    @snow_compaction.setter
    def snow_compaction(self, v: float) -> None:
        self._snow = _f2u(v)

    @property
    def ice_film(self) -> float:
        return _u2f(self._ice)

    @ice_film.setter
    def ice_film(self, v: float) -> None:
        self._ice = _f2u(v)

    @property
    def crust_hardness(self) -> float:
        return _u2f(self._crust)

    @crust_hardness.setter
    def crust_hardness(self, v: float) -> None:
        self._crust = _f2u(v)

    @property
    def roughness(self) -> float:
        return _u2f(self._rough)

    @roughness.setter
    def roughness(self, v: float) -> None:
        self._rough = _f2u(v)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_bytes(self) -> bytes:
        """Pack 5 fields as 5 uint8 bytes (dust, snow, ice, crust, rough)."""
        return struct.pack("5B", self._dust, self._snow, self._ice,
                           self._crust, self._rough)

    def from_bytes(self, data: bytes) -> None:
        """Decode 5 uint8 bytes into fields (in-place)."""
        self._dust, self._snow, self._ice, self._crust, self._rough = \
            struct.unpack("5B", data[:5])

    def state_hash(self) -> str:
        """Stable hex-MD5 of the raw byte representation."""
        return hashlib.md5(self.to_bytes()).hexdigest()

    def copy(self) -> "SurfaceMaterialState":
        """Return a deep copy of this state."""
        s = SurfaceMaterialState(self.chunk_id)
        s._dust  = self._dust
        s._snow  = self._snow
        s._ice   = self._ice
        s._crust = self._crust
        s._rough = self._rough
        return s

    def __repr__(self) -> str:
        return (
            f"SurfaceMaterialState("
            f"dust={self.dust_thickness:.3f}, "
            f"snow={self.snow_compaction:.3f}, "
            f"ice={self.ice_film:.3f}, "
            f"crust={self.crust_hardness:.3f}, "
            f"rough={self.roughness:.3f})"
        )


# ---------------------------------------------------------------------------
# SurfaceMaterialStateGrid — W × H grid of cells
# ---------------------------------------------------------------------------

class SurfaceMaterialStateGrid:
    """A rectangular grid of :class:`SurfaceMaterialState` cells.

    Parameters
    ----------
    chunk_id :
        Identifier for the chunk this grid belongs to.
    w, h :
        Grid width and height in cells.
    """

    def __init__(self, chunk_id: object, w: int, h: int) -> None:
        self.chunk_id = chunk_id
        self.w = w
        self.h = h
        self._cells: List[SurfaceMaterialState] = [
            SurfaceMaterialState(chunk_id) for _ in range(w * h)
        ]

    def cell(self, ix: int, iy: int) -> SurfaceMaterialState:
        """Return the cell at grid coordinates (ix, iy)."""
        return self._cells[iy * self.w + ix]

    def to_bytes(self) -> bytes:
        """Serialise all cells as a flat byte array (5 bytes per cell)."""
        return b"".join(c.to_bytes() for c in self._cells)

    def from_bytes(self, data: bytes) -> None:
        """Decode a flat byte array produced by :meth:`to_bytes`."""
        stride = 5
        for i, cell in enumerate(self._cells):
            cell.from_bytes(data[i * stride: i * stride + stride])

    def grid_hash(self) -> str:
        """Stable MD5 of the full grid byte representation."""
        return hashlib.md5(self.to_bytes()).hexdigest()
