"""SuitMassBinder — Stage 43 suit-to-physics parameter binding.

Applies the physical micro-variations stored in a
:class:`~src.character.SuitKitAssembler.SuitKit` to a
:class:`~src.character.BodyArchetypeDB.BodyArchetype` baseline, producing
a :class:`PhysicsParams` struct that the motor stack uses.

Constraints (§6.5)
-------------------
* Total mass variation ±5–8 % of archetype + base suit mass.
* COM shift ≤ ``suit.com_shift_max`` normalised units.
* Inertia scale ∈ [0.95, 1.05].
* Joint limits and torque caps are **never modified**.

Public API
----------
SuitMassBinder(config=None)
  .bind(archetype, suit_kit) → PhysicsParams
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from src.character.BodyArchetypeDB   import BodyArchetype
from src.character.SuitKitAssembler  import SuitKit


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


# ---------------------------------------------------------------------------
# PhysicsParams
# ---------------------------------------------------------------------------

@dataclass
class PhysicsParams:
    """Physics parameters derived from archetype + suit.

    These are the *effective* values the motor stack should use.
    """
    total_mass_kg:   float = 75.0
    com_shift_back:  float = 0.0   # normalised; 0 = neutral, 1 = full shift
    com_shift_up:    float = 0.0   # normalised
    inertia_scale:   float = 1.0
    # Joint limits and torque caps are passed through unchanged from archetype
    archetype_id:    str   = "default"


# ---------------------------------------------------------------------------
# SuitMassBinder
# ---------------------------------------------------------------------------

class SuitMassBinder:
    """Binds SuitKit physical micro-variations to the physics engine.

    Parameters
    ----------
    config :
        Optional dict; reads ``suit.*`` sub-keys.
    """

    # Hard bounds that can never be exceeded regardless of suit params
    _MASS_VAR_MAX    = 0.08   # ±8%
    _COM_SHIFT_MAX   = 0.04   # normalised
    _INERTIA_MIN     = 0.95
    _INERTIA_MAX     = 1.05

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg  = config or {}
        scfg = cfg.get("suit", {}) or {}
        self._mass_var_pct = float(scfg.get("mass_variation_pct", 0.06))
        self._com_max      = float(scfg.get("com_shift_max",      0.03))

    def bind(
        self,
        archetype: BodyArchetype,
        suit_kit:  SuitKit,
    ) -> PhysicsParams:
        """Compute effective physics params for *archetype* + *suit_kit*.

        Parameters
        ----------
        archetype :
            Canonical body archetype (fixed joint limits, torque caps).
        suit_kit :
            Assembled suit kit (carries micro-variation fields).
        """
        base_mass  = archetype.total_mass_kg + suit_kit.extra_mass_kg
        max_delta  = base_mass * min(self._mass_var_pct, self._MASS_VAR_MAX)
        total_mass = _clamp(base_mass, base_mass - max_delta,
                            base_mass + max_delta)

        com_back = _clamp(suit_kit.com_shift_norm[0],
                          0.0, min(self._com_max, self._COM_SHIFT_MAX))
        com_up   = _clamp(suit_kit.com_shift_norm[1],
                          0.0, min(self._com_max, self._COM_SHIFT_MAX))

        inertia  = _clamp(suit_kit.inertia_scale,
                          self._INERTIA_MIN, self._INERTIA_MAX)

        return PhysicsParams(
            total_mass_kg  = total_mass,
            com_shift_back = com_back,
            com_shift_up   = com_up,
            inertia_scale  = inertia,
            archetype_id   = archetype.archetype_id,
        )
