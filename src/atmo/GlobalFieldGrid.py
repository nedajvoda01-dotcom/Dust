"""GlobalFieldGrid — Stage 64 coarse planetary atmosphere field store.

Stores seven normalised proxy fields per tile on a flat W×H grid:

  P   pressureProxy          [0..1]
  T   temperatureProxy       [0..1]
  Vx  windX                  [−1..1]  (wind vector X component)
  Vy  windY                  [−1..1]  (wind vector Y component)
  A   aerosolDustDensity     [0..1]
  H   volatileHumidityProxy  [0..1]
  E   electroActivity        [0..1]

Derived fields (computed on-demand, never stored):
  visibilityProxy — 1 − clamp(A + fog_potential)
  frontIntensity  — gradient magnitude of P/T blend
  gustiness       — proportional to windSpeed and E
  stormPotential  — f(P_grad, windSpeed, A)
  fogPotential    — f(H, cold_bias, P)

Public API
----------
GlobalFieldGrid(width, height, seed=0)
  .tile(ix, iy)               → AtmoTile (proxy dataclass)
  .set_tile(ix, iy, tile)     → None
  .width, .height             → int
  .total_aerosol()            → float  (grid-wide sum of A values)
  .visibility_proxy(ix, iy)   → float  [0..1]
  .front_intensity(ix, iy)    → float  [0..1]
  .gust_potential(ix, iy)     → float  [0..1]
  .storm_potential(ix, iy)    → float  [0..1]
  .fog_potential(ix, iy)      → float  [0..1]
  .to_bytes()                 → bytes
  .from_bytes(data)           → None  (in-place)
  .grid_hash()                → str   (hex MD5)
"""
from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


def _f2b(v: float) -> int:
    """Encode float [0..1] to uint8."""
    return max(0, min(255, int(round(v * 255.0))))


def _b2f(b: int) -> float:
    """Decode uint8 to float [0..1]."""
    return b / 255.0


def _f2b_signed(v: float) -> int:
    """Encode float [−1..1] to uint8 via centre-zero mapping."""
    return max(0, min(255, int(round((v + 1.0) * 0.5 * 255.0))))


def _b2f_signed(b: int) -> float:
    """Decode uint8 [0..255] to float [−1..1]."""
    return b / 255.0 * 2.0 - 1.0


_TILE_STRUCT = struct.Struct("!8B")   # P, T, Vx, Vy, A, H, E, _pad
_TILE_BYTES  = _TILE_STRUCT.size      # 8 bytes per tile


# ---------------------------------------------------------------------------
# AtmoTile
# ---------------------------------------------------------------------------

@dataclass
class AtmoTile:
    """Seven normalised atmospheric fields for one coarse grid tile.

    Wind components (wind_x, wind_y) use the full [−1..1] range.
    All other fields are [0..1].
    """
    pressure:    float = 0.5
    temperature: float = 0.5
    wind_x:      float = 0.0
    wind_y:      float = 0.0
    aerosol:     float = 0.0
    humidity:    float = 0.2
    electro:     float = 0.0

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def wind_speed(self) -> float:
        return math.sqrt(self.wind_x ** 2 + self.wind_y ** 2)

    def clamp(self) -> "AtmoTile":
        """Return a new tile with all fields clamped to valid ranges."""
        return AtmoTile(
            pressure    = _clamp(self.pressure),
            temperature = _clamp(self.temperature),
            wind_x      = _clamp(self.wind_x, -1.0, 1.0),
            wind_y      = _clamp(self.wind_y, -1.0, 1.0),
            aerosol     = _clamp(self.aerosol),
            humidity    = _clamp(self.humidity),
            electro     = _clamp(self.electro),
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_bytes(self) -> bytes:
        return _TILE_STRUCT.pack(
            _f2b(self.pressure),
            _f2b(self.temperature),
            _f2b_signed(self.wind_x),
            _f2b_signed(self.wind_y),
            _f2b(self.aerosol),
            _f2b(self.humidity),
            _f2b(self.electro),
            0,   # padding / reserved
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "AtmoTile":
        P, T, Vx, Vy, A, H, E, _ = _TILE_STRUCT.unpack(data)
        return cls(
            pressure    = _b2f(P),
            temperature = _b2f(T),
            wind_x      = _b2f_signed(Vx),
            wind_y      = _b2f_signed(Vy),
            aerosol     = _b2f(A),
            humidity    = _b2f(H),
            electro     = _b2f(E),
        )


# ---------------------------------------------------------------------------
# GlobalFieldGrid
# ---------------------------------------------------------------------------

class GlobalFieldGrid:
    """Flat W×H grid of AtmoTile objects.

    Parameters
    ----------
    width, height :
        Grid dimensions (coarse tiles).
    seed :
        Deterministic seed (not used directly here; passed to upstream
        dynamics for reproducibility).
    """

    def __init__(self, width: int, height: int, seed: int = 0) -> None:
        self._w = width
        self._h = height
        self._seed = seed
        self._tiles: List[AtmoTile] = [AtmoTile() for _ in range(width * height)]

    # ------------------------------------------------------------------
    # Dimensions
    # ------------------------------------------------------------------

    @property
    def width(self) -> int:
        return self._w

    @property
    def height(self) -> int:
        return self._h

    # ------------------------------------------------------------------
    # Tile access
    # ------------------------------------------------------------------

    def _idx(self, ix: int, iy: int) -> int:
        return iy * self._w + ix

    def tile(self, ix: int, iy: int) -> AtmoTile:
        return self._tiles[self._idx(ix, iy)]

    def set_tile(self, ix: int, iy: int, tile: AtmoTile) -> None:
        self._tiles[self._idx(ix, iy)] = tile.clamp()

    # ------------------------------------------------------------------
    # Aggregate / derived queries
    # ------------------------------------------------------------------

    def total_aerosol(self) -> float:
        """Sum of all tile aerosol values (proxy for global aerosol mass)."""
        return sum(t.aerosol for t in self._tiles)

    def _pressure_gradient(self, ix: int, iy: int) -> float:
        """Central-difference pressure gradient magnitude at (ix, iy)."""
        left  = self._tiles[self._idx(max(ix - 1, 0),           iy)].pressure
        right = self._tiles[self._idx(min(ix + 1, self._w - 1), iy)].pressure
        up    = self._tiles[self._idx(ix, max(iy - 1, 0))].pressure
        down  = self._tiles[self._idx(ix, min(iy + 1, self._h - 1))].pressure
        gx = (right - left) * 0.5
        gy = (down  - up)   * 0.5
        return math.sqrt(gx * gx + gy * gy)

    def _temp_gradient(self, ix: int, iy: int) -> float:
        """Central-difference temperature gradient magnitude."""
        left  = self._tiles[self._idx(max(ix - 1, 0),           iy)].temperature
        right = self._tiles[self._idx(min(ix + 1, self._w - 1), iy)].temperature
        up    = self._tiles[self._idx(ix, max(iy - 1, 0))].temperature
        down  = self._tiles[self._idx(ix, min(iy + 1, self._h - 1))].temperature
        gx = (right - left) * 0.5
        gy = (down  - up)   * 0.5
        return math.sqrt(gx * gx + gy * gy)

    def visibility_proxy(self, ix: int, iy: int) -> float:
        """Visibility proxy at tile (ix, iy): 1 − clamp(A + fogPotential)."""
        t = self._tiles[self._idx(ix, iy)]
        fp = self.fog_potential(ix, iy)
        return _clamp(1.0 - _clamp(t.aerosol + fp))

    def front_intensity(self, ix: int, iy: int) -> float:
        """Front intensity: gradient blend of P and T fields."""
        pg = self._pressure_gradient(ix, iy)
        tg = self._temp_gradient(ix, iy)
        return _clamp(0.5 * pg + 0.5 * tg)

    def gust_potential(self, ix: int, iy: int) -> float:
        """Gustiness proxy: windSpeed × (1 + E)."""
        t = self._tiles[self._idx(ix, iy)]
        return _clamp(t.wind_speed * (1.0 + t.electro))

    def storm_potential(self, ix: int, iy: int) -> float:
        """Storm potential: blend of P-gradient, windSpeed, and aerosol."""
        t   = self._tiles[self._idx(ix, iy)]
        pg  = self._pressure_gradient(ix, iy)
        return _clamp(pg * 0.4 + t.wind_speed * 0.4 + t.aerosol * 0.2)

    def fog_potential(self, ix: int, iy: int) -> float:
        """Fog potential: humidity × cold_bias proxy × pressure condition."""
        t = self._tiles[self._idx(ix, iy)]
        cold_bias  = _clamp(1.0 - t.temperature)    # colder → more fog
        p_condition = _clamp(1.0 - abs(t.pressure - 0.4))  # low-ish pressure
        return _clamp(t.humidity * cold_bias * p_condition)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_bytes(self) -> bytes:
        header = struct.pack("!HH", self._w, self._h)
        return header + b"".join(t.to_bytes() for t in self._tiles)

    def from_bytes(self, data: bytes) -> None:
        w, h = struct.unpack_from("!HH", data, 0)
        if w != self._w or h != self._h:
            raise ValueError(f"Grid size mismatch: expected {self._w}×{self._h}, got {w}×{h}")
        offset = 4
        for i in range(self._w * self._h):
            self._tiles[i] = AtmoTile.from_bytes(data[offset: offset + _TILE_BYTES])
            offset += _TILE_BYTES

    def grid_hash(self) -> str:
        return hashlib.md5(self.to_bytes()).hexdigest()
