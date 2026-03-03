"""WorldMemoryState — Stage 51 coarse-resolution emergent world memory fields.

The world "remembers" not through scripts or logs, but through accumulated
physical fields that evolve over time.  All fields are normalised floats in
[0, 1] stored on a low-resolution tile grid (coarser than MaterialState).

Fields
------
stressAccumulationField  : geo-mechanical stress built up by impacts/steps
compactionHistoryField   : how frequently a tile has been under pressure
erosionBiasField         : directional erosion bias from wind + movement
acousticImprintField     : acoustic property shift from repeated disturbance

All fields decay toward zero over time (memory is not permanent).

Public API
----------
WorldMemoryState(width, height)
  .tile(x, y) -> int          flat index (wraps x, clamps y)
  .size() -> int
  .clamp_all() -> None
  .state_hash() -> str
  .to_dict() -> dict
  .from_dict(d) -> None
"""
from __future__ import annotations

import hashlib
from typing import List


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


_FIELDS = (
    "stressAccumulationField",
    "compactionHistoryField",
    "erosionBiasField",
    "acousticImprintField",
)


class WorldMemoryState:
    """Coarse-grid emergent memory fields for Stage 51.

    Parameters
    ----------
    width  : int — tile grid width  (longitude direction)
    height : int — tile grid height (latitude direction)
    """

    __slots__ = (
        "width", "height",
        "stressAccumulationField",
        "compactionHistoryField",
        "erosionBiasField",
        "acousticImprintField",
    )

    def __init__(self, width: int = 32, height: int = 16) -> None:
        self.width:  int = width
        self.height: int = height
        n = width * height

        self.stressAccumulationField: List[float] = [0.0] * n
        self.compactionHistoryField:  List[float] = [0.0] * n
        self.erosionBiasField:        List[float] = [0.0] * n
        self.acousticImprintField:    List[float] = [0.0] * n

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def tile(self, x: int, y: int) -> int:
        """Flat tile index (wraps longitude, clamps latitude)."""
        x = x % self.width
        y = max(0, min(y, self.height - 1))
        return y * self.width + x

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
            self._field_hash(self.stressAccumulationField)
            ^ self._field_hash(self.compactionHistoryField)
            ^ self._field_hash(self.erosionBiasField)
            ^ self._field_hash(self.acousticImprintField)
        )
        return format(combined & 0xFFFF_FFFF, "08x")

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialise to a compact dict (8-bit quantisation)."""
        def _q(lst: List[float]) -> List[int]:
            return [int(_clamp(v) * 255) for v in lst]

        return {
            "type":   "WORLD_MEMORY_STATE_51",
            "width":  self.width,
            "height": self.height,
            "state_hash": self.state_hash(),
            "fields": {
                "stressAccumulationField": _q(self.stressAccumulationField),
                "compactionHistoryField":  _q(self.compactionHistoryField),
                "erosionBiasField":        _q(self.erosionBiasField),
                "acousticImprintField":    _q(self.acousticImprintField),
            },
        }

    def from_dict(self, d: dict) -> None:
        """Restore state from a dict produced by :meth:`to_dict`."""
        if d.get("type") != "WORLD_MEMORY_STATE_51":
            raise ValueError("WorldMemoryState.from_dict: unexpected type")
        w = int(d["width"])
        h = int(d["height"])
        if w != self.width or h != self.height:
            raise ValueError(
                f"WorldMemoryState: size mismatch "
                f"({w}×{h} vs {self.width}×{self.height})"
            )
        fields = d["fields"]
        for name in _FIELDS:
            raw = fields[name]
            setattr(self, name, [v / 255.0 for v in raw])
