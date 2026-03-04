"""PlanetChunkState — Stage 63 full terrain chunk material state.

Each active terrain chunk stores eleven quantised scalar fields per cell
(uint8, representing [0, 1] normalised floats):

Field            Symbol        Mass-carrying
-----------      ----------    ------------
solidRockDepth   solid         yes
crustHardness    crust         yes
dustThickness    dust          yes
snowMass         snow          yes
snowCompaction   —             no  (phase indicator 0=fluffy … 1=hard)
iceFilmThickness ice           yes
debrisMass       debris        yes
surfaceRoughness —             no  (geometric)
temperatureProxy —             no  (thermal proxy)
moistureProxy    —             no  (moisture proxy)
stressField      —             no  (mechanical stress)

Mass conservation
-----------------
``total_mass()`` sums the six mass-carrying fields:
  solidRockDepth + crustHardness + dustThickness + snowMass +
  iceFilmThickness + debrisMass

All writes go through :class:`~src.material.MassExchangeAPI.MassExchangeAPI`
to ensure conservation and non-negativity.

Serialisation
-------------
Each cell serialises to 11 uint8 bytes (fixed stride for grid codec).

Public API
----------
PlanetChunkState(chunk_id=None, **fields)
  .<field>           — float property r/w, [0, 1]
  .total_mass()      — float, sum of mass-carrying fields
  .to_bytes()        — bytes (11 bytes)
  .from_bytes(data)  — in-place decode
  .state_hash()      — hex MD5
  .copy()            — deep copy

PlanetChunkGrid(chunk_id, w, h)
  .cell(ix, iy) -> PlanetChunkState
  .to_bytes()
  .from_bytes(data)
  .grid_hash()
"""
from __future__ import annotations

import hashlib
import struct
from typing import List


_UINT8_MAX = 255
_STRIDE    = 11   # bytes per cell


def _f2u(v: float) -> int:
    """Float [0,1] → uint8."""
    return max(0, min(_UINT8_MAX, int(round(v * _UINT8_MAX))))


def _u2f(v: int) -> float:
    """uint8 → float [0,1]."""
    return v / _UINT8_MAX


# ---------------------------------------------------------------------------
# PlanetChunkState
# ---------------------------------------------------------------------------

class PlanetChunkState:
    """Compact material state for one terrain chunk cell (Stage 63).

    All fields stored as uint8 (0..255).  Public API exposes normalised
    floats in [0, 1].

    Parameters
    ----------
    chunk_id :
        Opaque identifier (logging / hashing).
    solidRockDepth, crustHardness, dustThickness, snowMass, snowCompaction,
    iceFilmThickness, debrisMass, surfaceRoughness, temperatureProxy,
    moistureProxy, stressField : float
        Initial field values in [0, 1].
    """

    __slots__ = (
        "chunk_id",
        "_solid",   # solidRockDepth
        "_crust",   # crustHardness
        "_dust",    # dustThickness
        "_snow",    # snowMass
        "_comp",    # snowCompaction
        "_ice",     # iceFilmThickness
        "_debris",  # debrisMass
        "_rough",   # surfaceRoughness
        "_temp",    # temperatureProxy
        "_moist",   # moistureProxy
        "_stress",  # stressField
    )

    def __init__(
        self,
        chunk_id: object = None,
        *,
        solidRockDepth:   float = 0.5,
        crustHardness:    float = 0.2,
        dustThickness:    float = 0.2,
        snowMass:         float = 0.0,
        snowCompaction:   float = 0.0,
        iceFilmThickness: float = 0.0,
        debrisMass:       float = 0.0,
        surfaceRoughness: float = 0.5,
        temperatureProxy: float = 0.3,
        moistureProxy:    float = 0.0,
        stressField:      float = 0.0,
    ) -> None:
        self.chunk_id = chunk_id
        self._solid  = _f2u(solidRockDepth)
        self._crust  = _f2u(crustHardness)
        self._dust   = _f2u(dustThickness)
        self._snow   = _f2u(snowMass)
        self._comp   = _f2u(snowCompaction)
        self._ice    = _f2u(iceFilmThickness)
        self._debris = _f2u(debrisMass)
        self._rough  = _f2u(surfaceRoughness)
        self._temp   = _f2u(temperatureProxy)
        self._moist  = _f2u(moistureProxy)
        self._stress = _f2u(stressField)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def solidRockDepth(self) -> float:
        return _u2f(self._solid)

    @solidRockDepth.setter
    def solidRockDepth(self, v: float) -> None:
        self._solid = _f2u(v)

    @property
    def crustHardness(self) -> float:
        return _u2f(self._crust)

    @crustHardness.setter
    def crustHardness(self, v: float) -> None:
        self._crust = _f2u(v)

    @property
    def dustThickness(self) -> float:
        return _u2f(self._dust)

    @dustThickness.setter
    def dustThickness(self, v: float) -> None:
        self._dust = _f2u(v)

    @property
    def snowMass(self) -> float:
        return _u2f(self._snow)

    @snowMass.setter
    def snowMass(self, v: float) -> None:
        self._snow = _f2u(v)

    @property
    def snowCompaction(self) -> float:
        return _u2f(self._comp)

    @snowCompaction.setter
    def snowCompaction(self, v: float) -> None:
        self._comp = _f2u(v)

    @property
    def iceFilmThickness(self) -> float:
        return _u2f(self._ice)

    @iceFilmThickness.setter
    def iceFilmThickness(self, v: float) -> None:
        self._ice = _f2u(v)

    @property
    def debrisMass(self) -> float:
        return _u2f(self._debris)

    @debrisMass.setter
    def debrisMass(self, v: float) -> None:
        self._debris = _f2u(v)

    @property
    def surfaceRoughness(self) -> float:
        return _u2f(self._rough)

    @surfaceRoughness.setter
    def surfaceRoughness(self, v: float) -> None:
        self._rough = _f2u(v)

    @property
    def temperatureProxy(self) -> float:
        return _u2f(self._temp)

    @temperatureProxy.setter
    def temperatureProxy(self, v: float) -> None:
        self._temp = _f2u(v)

    @property
    def moistureProxy(self) -> float:
        return _u2f(self._moist)

    @moistureProxy.setter
    def moistureProxy(self, v: float) -> None:
        self._moist = _f2u(v)

    @property
    def stressField(self) -> float:
        return _u2f(self._stress)

    @stressField.setter
    def stressField(self, v: float) -> None:
        self._stress = _f2u(v)

    # ------------------------------------------------------------------
    # Mass conservation
    # ------------------------------------------------------------------

    def total_mass(self) -> float:
        """Sum of all mass-carrying fields.

        Returns
        -------
        float
            solidRockDepth + crustHardness + dustThickness + snowMass +
            iceFilmThickness + debrisMass  (range [0, 6]).
        """
        return (
            _u2f(self._solid)
            + _u2f(self._crust)
            + _u2f(self._dust)
            + _u2f(self._snow)
            + _u2f(self._ice)
            + _u2f(self._debris)
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_bytes(self) -> bytes:
        """Pack all 11 fields as 11 uint8 bytes."""
        return struct.pack(
            "11B",
            self._solid, self._crust, self._dust,
            self._snow,  self._comp,  self._ice,
            self._debris, self._rough, self._temp,
            self._moist, self._stress,
        )

    def from_bytes(self, data: bytes) -> None:
        """Decode 11 uint8 bytes into fields (in-place)."""
        (
            self._solid, self._crust, self._dust,
            self._snow,  self._comp,  self._ice,
            self._debris, self._rough, self._temp,
            self._moist, self._stress,
        ) = struct.unpack("11B", data[:_STRIDE])

    def state_hash(self) -> str:
        """Stable hex-MD5 of the raw byte representation."""
        return hashlib.md5(self.to_bytes()).hexdigest()

    def copy(self) -> "PlanetChunkState":
        """Return a deep copy."""
        s = PlanetChunkState(self.chunk_id)
        s._solid  = self._solid
        s._crust  = self._crust
        s._dust   = self._dust
        s._snow   = self._snow
        s._comp   = self._comp
        s._ice    = self._ice
        s._debris = self._debris
        s._rough  = self._rough
        s._temp   = self._temp
        s._moist  = self._moist
        s._stress = self._stress
        return s

    def __repr__(self) -> str:
        return (
            f"PlanetChunkState("
            f"solid={self.solidRockDepth:.3f}, "
            f"crust={self.crustHardness:.3f}, "
            f"dust={self.dustThickness:.3f}, "
            f"snow={self.snowMass:.3f}, "
            f"comp={self.snowCompaction:.3f}, "
            f"ice={self.iceFilmThickness:.3f}, "
            f"debris={self.debrisMass:.3f}, "
            f"rough={self.surfaceRoughness:.3f}, "
            f"temp={self.temperatureProxy:.3f}, "
            f"moist={self.moistureProxy:.3f}, "
            f"stress={self.stressField:.3f})"
        )


# ---------------------------------------------------------------------------
# PlanetChunkGrid — W × H grid of PlanetChunkState cells
# ---------------------------------------------------------------------------

class PlanetChunkGrid:
    """Rectangular grid of :class:`PlanetChunkState` cells.

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
        self._cells: List[PlanetChunkState] = [
            PlanetChunkState(chunk_id) for _ in range(w * h)
        ]

    def cell(self, ix: int, iy: int) -> PlanetChunkState:
        """Return cell at grid coordinates (ix, iy)."""
        return self._cells[iy * self.w + ix]

    def to_bytes(self) -> bytes:
        """Serialise all cells as a flat byte array (_STRIDE bytes per cell)."""
        return b"".join(c.to_bytes() for c in self._cells)

    def from_bytes(self, data: bytes) -> None:
        """Decode a flat byte array produced by :meth:`to_bytes`."""
        for i, cell in enumerate(self._cells):
            cell.from_bytes(data[i * _STRIDE: i * _STRIDE + _STRIDE])

    def grid_hash(self) -> str:
        """Stable MD5 of the full grid byte representation."""
        return hashlib.md5(self.to_bytes()).hexdigest()

    def total_mass(self) -> float:
        """Sum of total_mass() across all cells."""
        return sum(c.total_mass() for c in self._cells)
