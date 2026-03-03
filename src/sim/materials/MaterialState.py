"""MaterialState — physical state record for one unit of material.

Each unit carries five scalar fields that modulate the effective physical
properties derived from its :class:`MaterialFamily`:

* porosity     [0, 1]  — void fraction (0 = dense, 1 = all air)
* compaction   [0, 1]  — degree of compaction (0 = loose, 1 = fully packed)
* temp         [0, 1]  — normalised temperature
* melt_frac    [0, 1]  — fraction in liquid/melt state
* wetness      [0, 1]  — water saturation

Public API
----------
MaterialState(family_name, porosity, compaction, temp, melt_frac, wetness)
  .family_name  → str
  .porosity     → float
  .compaction   → float
  .temp         → float
  .melt_frac    → float
  .wetness      → float
  .to_dict()    → dict
  MaterialState.from_dict(d) → MaterialState
"""
from __future__ import annotations

from typing import Any, Dict


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


class MaterialState:
    """Physical state of one material unit.

    Parameters
    ----------
    family_name :
        Key into :data:`~src.sim.materials.Material.FAMILIES`.
    """

    __slots__ = (
        "family_name",
        "porosity",
        "compaction",
        "temp",
        "melt_frac",
        "wetness",
    )

    def __init__(
        self,
        family_name: str  = "rock",
        porosity:    float = 0.05,
        compaction:  float = 0.80,
        temp:        float = 0.20,
        melt_frac:   float = 0.00,
        wetness:     float = 0.00,
    ) -> None:
        self.family_name = str(family_name)
        self.porosity    = _clamp01(porosity)
        self.compaction  = _clamp01(compaction)
        self.temp        = _clamp01(temp)
        self.melt_frac   = _clamp01(melt_frac)
        self.wetness     = _clamp01(wetness)

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "family_name": self.family_name,
            "porosity":    self.porosity,
            "compaction":  self.compaction,
            "temp":        self.temp,
            "melt_frac":   self.melt_frac,
            "wetness":     self.wetness,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MaterialState":
        return cls(
            family_name=str(d.get("family_name", "rock")),
            porosity   =float(d.get("porosity",   0.05)),
            compaction =float(d.get("compaction", 0.80)),
            temp       =float(d.get("temp",       0.20)),
            melt_frac  =float(d.get("melt_frac",  0.00)),
            wetness    =float(d.get("wetness",    0.00)),
        )
