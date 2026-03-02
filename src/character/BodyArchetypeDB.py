"""BodyArchetypeDB — Stage 43 physical body archetype definitions.

Provides the single, near-fixed physical archetype that all players share.
Small cosmetic variations (suit mass, COM) are applied by
:mod:`src.physics.SuitMassBinder`; the skeleton joints and torque caps
defined here are **never varied** between players to preserve motor parity.

Design
------
* One ``BodyArchetype`` record in the DB is the canonical character.
* Joint limits are in degrees.
* Torque caps are normalised (1.0 = reference human adult torque at normal G).
* Segment masses are in kg (reference Earth-like body; gravity is applied
  externally by the physics engine).

Public API
----------
BodyArchetypeDB()
  .get(archetype_id)  → BodyArchetype
  .default_id         → str
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class SegmentPhysics:
    """Physical parameters for one body segment."""
    mass_kg:    float = 0.0
    inertia_k:  float = 1.0   # inertia scale relative to reference


@dataclass
class JointLimit:
    """Symmetric ± angular limit for a joint in degrees."""
    flex_deg:  float = 90.0
    abduct_deg:float = 45.0
    rot_deg:   float = 30.0


@dataclass
class BodyArchetype:
    """Complete physical description of the character skeleton.

    These values are fixed per-archetype and must not be varied per-player
    (doing so would break motor parity — see §6.5 of the spec).
    """
    archetype_id: str = "default"

    # Segment masses (kg)
    torso_mass:   float = 35.0
    head_mass:    float = 5.0
    upper_arm_mass: float = 2.5
    lower_arm_mass: float = 1.5
    hand_mass:    float = 0.5
    upper_leg_mass: float = 7.0
    lower_leg_mass: float = 4.0
    foot_mass:    float = 1.2

    # Torque caps (normalised; 1.0 = reference)
    torque_cap_spine:   float = 1.0
    torque_cap_knee:    float = 1.0
    torque_cap_ankle:   float = 1.0
    torque_cap_shoulder:float = 1.0
    torque_cap_elbow:   float = 1.0

    # Joint limits
    neck_limit:     JointLimit = field(default_factory=lambda: JointLimit(70, 45, 50))
    spine_limit:    JointLimit = field(default_factory=lambda: JointLimit(40, 35, 30))
    shoulder_limit: JointLimit = field(default_factory=lambda: JointLimit(160, 90, 90))
    elbow_limit:    JointLimit = field(default_factory=lambda: JointLimit(140, 10, 5))
    hip_limit:      JointLimit = field(default_factory=lambda: JointLimit(120, 45, 40))
    knee_limit:     JointLimit = field(default_factory=lambda: JointLimit(130, 10, 5))
    ankle_limit:    JointLimit = field(default_factory=lambda: JointLimit(60, 25, 20))

    @property
    def total_mass_kg(self) -> float:
        """Sum of all segment masses (bare body, no suit)."""
        return (
            self.torso_mass + self.head_mass
            + 2 * (self.upper_arm_mass + self.lower_arm_mass + self.hand_mass)
            + 2 * (self.upper_leg_mass + self.lower_leg_mass + self.foot_mass)
        )


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

class BodyArchetypeDB:
    """Registry of body archetypes.

    Only one archetype ("default") is defined for Stage 43.  Additional
    archetypes (e.g. heavier suit variants) can be added later without
    changing the public API.
    """

    def __init__(self) -> None:
        self._db: Dict[str, BodyArchetype] = {
            "default": BodyArchetype(),
        }

    @property
    def default_id(self) -> str:
        """Key of the default archetype."""
        return "default"

    def get(self, archetype_id: str) -> BodyArchetype:
        """Return the archetype for *archetype_id*.

        Falls back to "default" if *archetype_id* is unknown.
        """
        return self._db.get(archetype_id, self._db["default"])
