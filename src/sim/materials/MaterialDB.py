"""MaterialDB — registry of material families and default states.

The DB is the single source of truth for what materials exist and what
their default initial state is.  It is seeded at world creation and
carried in World3D.

Public API
----------
MaterialDB(seed=42)
  .families          → dict[str, MaterialFamily]
  .default_state(family_name) → MaterialState
  .to_dict()         → dict
  MaterialDB.from_dict(d) → MaterialDB
"""
from __future__ import annotations

from typing import Any, Dict

from src.sim.materials.Material import FAMILIES, MaterialFamily
from src.sim.materials.MaterialState import MaterialState


class MaterialDB:
    """Registry mapping family names to their default :class:`MaterialState`.

    Parameters
    ----------
    seed :
        World seed (currently reserved for future noise-seeded defaults).
    """

    def __init__(self, seed: int = 42) -> None:
        self._seed     = int(seed)
        self._families: Dict[str, MaterialFamily] = dict(FAMILIES)

        # Default initial states per family
        self._defaults: Dict[str, MaterialState] = {
            "rock":     MaterialState("rock",     porosity=0.03, compaction=0.90,
                                      temp=0.20, melt_frac=0.00, wetness=0.00),
            "regolith": MaterialState("regolith", porosity=0.35, compaction=0.20,
                                      temp=0.25, melt_frac=0.00, wetness=0.00),
            "ice":      MaterialState("ice",      porosity=0.10, compaction=0.60,
                                      temp=0.05, melt_frac=0.00, wetness=0.05),
        }

    # ------------------------------------------------------------------
    @property
    def families(self) -> Dict[str, MaterialFamily]:
        return self._families

    def default_state(self, family_name: str) -> MaterialState:
        """Return a copy of the default :class:`MaterialState` for *family_name*.

        Falls back to "rock" defaults for unknown families.
        """
        template = self._defaults.get(family_name, self._defaults["rock"])
        return MaterialState.from_dict(template.to_dict())

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "seed":     self._seed,
            "families": list(self._families.keys()),
            "defaults": {k: v.to_dict() for k, v in self._defaults.items()},
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MaterialDB":
        db = cls(seed=int(d.get("seed", 42)))
        for name, state_d in d.get("defaults", {}).items():
            db._defaults[name] = MaterialState.from_dict(state_d)
        return db
