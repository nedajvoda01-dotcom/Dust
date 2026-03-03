"""Material — material family definitions.

Material families define *what kind of substance* a unit of material is.
Physical behaviour is determined by the combination of family + MaterialState.

Public API
----------
FAMILIES : dict[str, MaterialFamily]
  Access by name: FAMILIES["rock"], FAMILIES["regolith"], FAMILIES["ice"]

MaterialFamily.name          → str
MaterialFamily.base_density  → float  (kg/m³ simulation equivalent)
MaterialFamily.base_friction → float  (μ, 0–1)
MaterialFamily.base_strength → float  (fracture threshold, 0–1)
MaterialFamily.color_rgb     → tuple  (r, g, b)  [0–1]
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class MaterialFamily:
    """Immutable description of a material family."""

    name:          str
    base_density:  float        # kg/m³ equivalent
    base_friction: float        # coefficient of friction [0, 1]
    base_strength: float        # fracture threshold [0, 1]
    color_rgb:     Tuple[float, float, float]  # representative colour


# ---------------------------------------------------------------------------
# Built-in families
# ---------------------------------------------------------------------------

_FAMILIES_LIST = [
    MaterialFamily(
        name          = "rock",
        base_density  = 2700.0,
        base_friction = 0.70,
        base_strength = 0.90,
        color_rgb     = (0.40, 0.35, 0.30),
    ),
    MaterialFamily(
        name          = "regolith",
        base_density  = 1400.0,
        base_friction = 0.40,
        base_strength = 0.15,
        color_rgb     = (0.55, 0.48, 0.38),
    ),
    MaterialFamily(
        name          = "ice",
        base_density  = 917.0,
        base_friction = 0.10,
        base_strength = 0.30,
        color_rgb     = (0.78, 0.88, 0.96),
    ),
]

FAMILIES: Dict[str, MaterialFamily] = {f.name: f for f in _FAMILIES_LIST}
