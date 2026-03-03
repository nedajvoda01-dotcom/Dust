"""PlanetEvolutionState — Stage 50 coarse-resolution planetary evolution fields.

Stores the slow-evolving planetary state on a low-resolution tile grid
(default 64 × 32).  All per-tile fields are normalised floats in [0, 1].

Fields
------
dustReservoirMap     : dust available in the global wind-borne reservoir
crustStabilityMap    : structural stability of the surface crust
slopeCreepMap        : accumulated downhill mass flux (slow creep)
iceBeltDistribution  : probability / thickness of stable ice at each tile

Plus one scalar:
seasonalInsolationPhase : current orbital phase driving seasonal effects [0, 2π)

Public API
----------
PlanetEvolutionState(width, height)
  .tile(x, y) -> int          (flat index)
  .clamp_all() -> None
  .state_hash() -> str
  .to_dict(world_seed, planet_time) -> dict
  .from_dict(d) -> None
"""
from __future__ import annotations

import hashlib
import math
from typing import List


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


_FIELDS = (
    "dustReservoirMap",
    "crustStabilityMap",
    "slopeCreepMap",
    "iceBeltDistribution",
)

_FIELD_DEFAULTS = {
    "dustReservoirMap":    0.3,
    "crustStabilityMap":   0.7,
    "slopeCreepMap":       0.0,
    "iceBeltDistribution": 0.0,
}


class PlanetEvolutionState:
    """Coarse-grid planetary evolution fields for Stage 50.

    Parameters
    ----------
    width  : int  — tile grid width  (longitude direction)
    height : int  — tile grid height (latitude direction)
    """

    __slots__ = (
        "width", "height",
        "dustReservoirMap",
        "crustStabilityMap",
        "slopeCreepMap",
        "iceBeltDistribution",
        "seasonalInsolationPhase",
    )

    def __init__(self, width: int = 64, height: int = 32) -> None:
        self.width:  int = width
        self.height: int = height
        n = width * height

        self.dustReservoirMap:    List[float] = [_FIELD_DEFAULTS["dustReservoirMap"]]    * n
        self.crustStabilityMap:   List[float] = [_FIELD_DEFAULTS["crustStabilityMap"]]   * n
        self.slopeCreepMap:       List[float] = [_FIELD_DEFAULTS["slopeCreepMap"]]       * n
        self.iceBeltDistribution: List[float] = [_FIELD_DEFAULTS["iceBeltDistribution"]] * n
        self.seasonalInsolationPhase: float = 0.0

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
    # State hash (for determinism checks and net sync)
    # ------------------------------------------------------------------

    def _field_hash(self, lst: List[float]) -> int:
        packed = bytes(int(_clamp(v) * 255) for v in lst)
        return int.from_bytes(hashlib.md5(packed).digest()[:4], "little")

    def state_hash(self) -> str:
        """Short hex digest covering all tile fields."""
        combined = (
            self._field_hash(self.dustReservoirMap)
            ^ self._field_hash(self.crustStabilityMap)
            ^ self._field_hash(self.slopeCreepMap)
            ^ self._field_hash(self.iceBeltDistribution)
        )
        return format(combined & 0xFFFF_FFFF, "08x")

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self, world_seed: int = 0, planet_time: float = 0.0) -> dict:
        """Serialise to a compact dict (8-bit quantisation)."""
        def _q(lst: List[float]) -> List[int]:
            return [int(_clamp(v) * 255) for v in lst]

        return {
            "type":       "EVOLUTION_STATE_50",
            "world_seed": world_seed,
            "planet_time": planet_time,
            "width":      self.width,
            "height":     self.height,
            "state_hash": self.state_hash(),
            "seasonal_insolation_phase": self.seasonalInsolationPhase,
            "fields": {
                "dustReservoirMap":    _q(self.dustReservoirMap),
                "crustStabilityMap":   _q(self.crustStabilityMap),
                "slopeCreepMap":       _q(self.slopeCreepMap),
                "iceBeltDistribution": _q(self.iceBeltDistribution),
            },
        }

    def from_dict(self, d: dict) -> None:
        """Restore state from a dict produced by :meth:`to_dict`."""
        if d.get("type") != "EVOLUTION_STATE_50":
            raise ValueError("PlanetEvolutionState.from_dict: unexpected type")
        w = int(d["width"])
        h = int(d["height"])
        if w != self.width or h != self.height:
            raise ValueError(
                f"PlanetEvolutionState: size mismatch "
                f"({w}×{h} vs {self.width}×{self.height})"
            )
        self.seasonalInsolationPhase = float(
            d.get("seasonal_insolation_phase", 0.0)
        )
        fields = d["fields"]
        for name in _FIELDS:
            raw = fields[name]
            setattr(self, name, [v / 255.0 for v in raw])
