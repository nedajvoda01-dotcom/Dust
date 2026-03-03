"""InstabilityState — Stage 52 coarse-resolution instability fields.

Stores five normalised scalar fields [0, 1] on a low-resolution tile grid.
Fields accumulate slowly from Material (45), Memory (51), and Climate (29)
and discharge when they cross critical thresholds.

Fields
------
shearStressField        : accumulated shear stress (slope × dust × movement)
crustFailurePotential   : brittle-failure energy (stress + thermal cycles)
massOverhangField       : overhanging mass bias (slope creep + erosion bias)
dustLoadField           : loose dust available for avalanching
thermalGradientField    : rapid insolation change driving micro-fracture

Public API
----------
InstabilityState(width, height)
  .tile(x, y)    -> int   flat index
  .neighbors(i)  -> List[int]   4-connected neighbor tile indices
  .size()        -> int
  .clamp_all()   -> None
  .state_hash()  -> str
"""
from __future__ import annotations

import hashlib
from typing import List


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


_FIELDS = (
    "shearStressField",
    "crustFailurePotential",
    "massOverhangField",
    "dustLoadField",
    "thermalGradientField",
)


class InstabilityState:
    """Coarse-grid instability fields for Stage 52.

    Parameters
    ----------
    width  : int — tile grid width  (longitude direction)
    height : int — tile grid height (latitude direction)
    """

    __slots__ = (
        "width", "height",
        "shearStressField",
        "crustFailurePotential",
        "massOverhangField",
        "dustLoadField",
        "thermalGradientField",
    )

    def __init__(self, width: int = 32, height: int = 16) -> None:
        self.width:  int = width
        self.height: int = height
        n = width * height

        self.shearStressField:      List[float] = [0.0] * n
        self.crustFailurePotential: List[float] = [0.0] * n
        self.massOverhangField:     List[float] = [0.0] * n
        self.dustLoadField:         List[float] = [0.0] * n
        self.thermalGradientField:  List[float] = [0.0] * n

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def tile(self, x: int, y: int) -> int:
        """Flat tile index (wraps longitude, clamps latitude)."""
        x = x % self.width
        y = max(0, min(y, self.height - 1))
        return y * self.width + x

    def neighbors(self, idx: int) -> List[int]:
        """Return indices of the 4-connected neighbours of tile *idx*.

        Longitude wraps; latitude clamps.
        """
        y, x = divmod(idx, self.width)
        result: List[int] = []
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx = (x + dx) % self.width
            ny = max(0, min(y + dy, self.height - 1))
            result.append(ny * self.width + nx)
        return result

    def size(self) -> int:
        return self.width * self.height

    def clamp_all(self) -> None:
        """Clamp every field element to [0, 1]."""
        for name in _FIELDS:
            lst = getattr(self, name)
            for i in range(len(lst)):
                if lst[i] < 0.0:
                    lst[i] = 0.0
                elif lst[i] > 1.0:
                    lst[i] = 1.0

    # ------------------------------------------------------------------
    # State hash
    # ------------------------------------------------------------------

    def _field_hash(self, lst: List[float]) -> int:
        packed = bytes(int(_clamp(v) * 255) for v in lst)
        return int.from_bytes(hashlib.md5(packed).digest()[:4], "little")

    def state_hash(self) -> str:
        """Short hex digest covering all tile fields."""
        combined = (
            self._field_hash(self.shearStressField)
            ^ self._field_hash(self.crustFailurePotential)
            ^ self._field_hash(self.massOverhangField)
            ^ self._field_hash(self.dustLoadField)
            ^ self._field_hash(self.thermalGradientField)
        )
        return format(combined & 0xFFFF_FFFF, "08x")

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialise to a compact dict (8-bit quantisation)."""
        def _q(lst: List[float]) -> List[int]:
            return [int(_clamp(v) * 255) for v in lst]

        return {
            "type":   "INSTABILITY_STATE_52",
            "width":  self.width,
            "height": self.height,
            "state_hash": self.state_hash(),
            "fields": {name: _q(getattr(self, name)) for name in _FIELDS},
        }

    def from_dict(self, d: dict) -> None:
        """Restore state from a dict produced by :meth:`to_dict`."""
        if d.get("type") != "INSTABILITY_STATE_52":
            raise ValueError("InstabilityState.from_dict: unexpected type")
        w = int(d["width"])
        h = int(d["height"])
        if w != self.width or h != self.height:
            raise ValueError(
                f"InstabilityState: size mismatch "
                f"({w}×{h} vs {self.width}×{self.height})"
            )
        fields = d["fields"]
        for name in _FIELDS:
            setattr(self, name, [v / 255.0 for v in fields[name]])
