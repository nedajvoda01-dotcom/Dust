"""PlayerIntent — Stage 43 player-input contract.

Defines the two primary targets that WASD + mouse produce, plus the
reflex overlay that Perception may contribute.  All MotorStack and
LookRigController logic reads from this struct every tick.

Contract
--------
``PrimaryMoveTarget``
    Formed from WASD raw axes.
    * ``moveDir_local`` — unit Vec3 in camera-relative horizontal plane;
      (0,0,0) when no key is held.
    * ``speedIntent``   — [0..1] fraction of normal walk speed.

``PrimaryLookTarget``
    Formed from mouse delta.
    * ``lookDir_world`` — unit Vec3 in world space pointing where the
      player wants to look (camera forward projected onto the sphere
      surface plane).

``ReflexOverlay``
    Short-lived additive biases from Perception (Stage 37).  These
    **never replace** the primary targets; they are blended with a
    small, decaying weight ``r``.
    * ``lookBias_dir``      — direction of reflex gaze bias (world).
    * ``lookBias_strength`` — current blend weight r ∈ [0..rMax].
    * ``braceBias``         — [0..1] encourages ArmSupportController.
    * ``slowdownBias``      — [0..1] fractional speed reduction.

``PlayerIntent``
    Aggregated struct combining all of the above.  Callers read:
    * ``FinalLookDir``  — property: lerp(primary, reflex, r).
    * ``FinalMoveDir``  — property: primary with slowdown applied.
    * ``FinalSpeed``    — property: speedIntent × (1 − slowdownBias).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from src.math.Vec3 import Vec3


# ---------------------------------------------------------------------------
# Sub-structs
# ---------------------------------------------------------------------------

@dataclass
class PrimaryMoveTarget:
    """WASD-derived movement intention."""
    moveDir_local: Vec3  = field(default_factory=Vec3.zero)
    speedIntent:   float = 0.0   # [0..1]


@dataclass
class PrimaryLookTarget:
    """Mouse-derived look intention (world space)."""
    lookDir_world: Vec3  = field(default_factory=lambda: Vec3(0.0, 0.0, -1.0))


@dataclass
class ReflexOverlay:
    """Short-lived biases from Perception.

    These supplement but never override primary targets.
    ``r`` decays exponentially toward 0.
    """
    lookBias_dir:      Vec3  = field(default_factory=Vec3.zero)
    lookBias_strength: float = 0.0   # current r ∈ [0..rMax]
    braceBias:         float = 0.0   # [0..1]
    slowdownBias:      float = 0.0   # [0..1]


# ---------------------------------------------------------------------------
# PlayerIntent
# ---------------------------------------------------------------------------

@dataclass
class PlayerIntent:
    """Combined player intention for one simulation tick.

    Assembled by :class:`~src.input.InputSystem.InputSystem` and
    :class:`~src.reflex.ReflexOverlaySystem.ReflexOverlaySystem`.

    Read-only computed properties expose the *final* values that
    downstream systems (LookRigController, CharacterPhysicalController)
    should use.
    """
    move:   PrimaryMoveTarget = field(default_factory=PrimaryMoveTarget)
    look:   PrimaryLookTarget = field(default_factory=PrimaryLookTarget)
    reflex: ReflexOverlay     = field(default_factory=ReflexOverlay)

    # ------------------------------------------------------------------
    # Computed finals
    # ------------------------------------------------------------------

    @property
    def FinalLookDir(self) -> Vec3:
        """Blend primary look with reflex bias by weight r.

        ``FinalLookDir = normalize( lerp(primary, reflex, r) )``
        where r = ``reflex.lookBias_strength`` ∈ [0, rMax].
        """
        r = max(0.0, min(1.0, self.reflex.lookBias_strength))
        if r < 1e-8:
            return self.look.lookDir_world
        primary = self.look.lookDir_world
        bias    = self.reflex.lookBias_dir
        blended = primary * (1.0 - r) + bias * r
        length  = blended.length()
        if length < 1e-9:
            return primary
        return blended * (1.0 / length)

    @property
    def FinalMoveDir(self) -> Vec3:
        """Primary move direction is not modified by reflex."""
        return self.move.moveDir_local

    @property
    def FinalSpeed(self) -> float:
        """speedIntent reduced by slowdownBias.

        ``FinalSpeed = speedIntent × (1 − slowdownBias)``
        """
        sb = max(0.0, min(1.0, self.reflex.slowdownBias))
        return self.move.speedIntent * (1.0 - sb)
