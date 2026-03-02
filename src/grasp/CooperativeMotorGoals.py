"""CooperativeMotorGoals — Stage 40 §9 / §10.

After a :class:`~src.grasp.GraspConstraintBinder.GraspConstraint` is
established, both characters receive modified motor goals that guide their
behaviour during the grasp interaction.

This module emits :class:`GraspMotorGoals` for each character in the
constraint.  These goals are merged into the
:class:`~src.intent.StrategySelector.MotorIntent` pipeline by the caller.

Public API
----------
GraspRole (enum)
GraspMotorGoals (dataclass)

CooperativeMotorGoals(config=None)
  .compute(role, body_state, partner_body_state,
           constraint, dt) → GraspMotorGoals
"""
from __future__ import annotations

import enum
import math
from dataclasses import dataclass
from typing import Optional

from src.math.Vec3 import Vec3
from src.grasp.GraspConstraintBinder import GraspConstraint


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


# ---------------------------------------------------------------------------
# GraspRole
# ---------------------------------------------------------------------------

class GraspRole(enum.Enum):
    """Which side of the constraint is this character on?"""
    HELPER  = "helper"   # player B — stable, providing support
    ASSISTED = "assisted" # player A — in distress, receiving help


# ---------------------------------------------------------------------------
# GraspMotorGoals
# ---------------------------------------------------------------------------

@dataclass
class GraspMotorGoals:
    """Motor goal modifiers for one character in a grasp constraint (§9).

    All values are additive/multiplicative modifiers on top of the base
    MotorIntent produced by IntentArbitrator.

    Attributes
    ----------
    pull_bias :
        [0..1] additional force directed toward partner's anchor point.
        Helper side only; grows when helper is stable (§10).
    brace_bias :
        [0..1] additional stance-width / brace request.
    recover_bias :
        [0..1] urge to restore foot contact (ASSISTED side only).
    damp_velocity_scale :
        Multiplier on velocity to reduce sudden jerks (damping).
    role :
        Which role this goal set was computed for.
    """
    pull_bias:           float
    brace_bias:          float
    recover_bias:        float
    damp_velocity_scale: float
    role:                GraspRole


# ---------------------------------------------------------------------------
# CooperativeMotorGoals
# ---------------------------------------------------------------------------

class CooperativeMotorGoals:
    """Computes per-role motor goal modifiers for a grasping pair (§9 / §10).

    Parameters
    ----------
    config :
        Optional dict; reads ``grasp.*`` keys.
    """

    _DEFAULT_PULL_BIAS_MAX = 0.6
    _DEFAULT_PULL_BIAS_TAU = 2.0   # seconds — ramp-up time constant

    def __init__(self, config: Optional[dict] = None) -> None:
        gcfg = (config or {}).get("grasp", {}) or {}
        self._pull_bias_max: float = float(gcfg.get("pull_bias_max", self._DEFAULT_PULL_BIAS_MAX))
        self._pull_bias_tau: float = float(gcfg.get("pull_bias_tau", self._DEFAULT_PULL_BIAS_TAU))
        self._pull_bias_acc: float = 0.0   # accumulated pull bias over time

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def compute(
        self,
        role:               GraspRole,
        body_state:         "BodyState",  # noqa: F821 — imported at runtime
        partner_state:      "BodyState",
        constraint:         GraspConstraint,
        dt:                 float,
    ) -> GraspMotorGoals:
        """Compute motor goals for one character during a grasp.

        Parameters
        ----------
        role :
            HELPER or ASSISTED.
        body_state :
            Physics state of this character.
        partner_state :
            Physics state of the other character.
        constraint :
            The active grasp constraint.
        dt :
            Time step [s].
        """
        if role == GraspRole.HELPER:
            return self._helper_goals(body_state, partner_state, constraint, dt)
        else:
            return self._assisted_goals(body_state, partner_state, constraint, dt)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _helper_goals(
        self,
        body:     "BodyState",
        partner:  "BodyState",
        c:        GraspConstraint,
        dt:       float,
    ) -> GraspMotorGoals:
        """Goals for the helper (B): brace, optionally pull (§10)."""
        helper_stable = body.support_quality > 0.5 and body.global_risk < 0.4

        # Pull bias ramps up slowly when helper is stable and partner still falls
        partner_still_falling = partner.global_risk > 0.3

        if helper_stable and partner_still_falling:
            target_pull = self._pull_bias_max * body.support_quality * (1.0 - body.global_risk)
        else:
            target_pull = 0.0

        # Exponential approach toward target
        alpha = _clamp(dt / max(self._pull_bias_tau, 1e-6), 0.0, 1.0)
        self._pull_bias_acc += (target_pull - self._pull_bias_acc) * alpha
        self._pull_bias_acc  = _clamp(self._pull_bias_acc, 0.0, self._pull_bias_max)

        # Limit pull if helper's own slip risk rises
        pull = self._pull_bias_acc * _clamp(1.0 - body.global_risk * 2.0, 0.0, 1.0)

        brace = _clamp(0.4 + body.global_risk * 0.4, 0.0, 1.0)
        damp  = _clamp(1.0 - pull * 0.3, 0.5, 1.0)   # slight velocity damping when pulling

        return GraspMotorGoals(
            pull_bias=pull,
            brace_bias=brace,
            recover_bias=0.0,
            damp_velocity_scale=damp,
            role=GraspRole.HELPER,
        )

    def _assisted_goals(
        self,
        body:     "BodyState",
        partner:  "BodyState",
        c:        GraspConstraint,
        dt:       float,
    ) -> GraspMotorGoals:
        """Goals for the assisted player (A): recover foot contact (§9)."""
        recover = _clamp(1.0 - body.support_quality, 0.0, 1.0)
        brace   = _clamp(body.global_risk * 0.5, 0.0, 1.0)
        damp    = _clamp(1.0 - body.velocity.length() * 0.1, 0.3, 1.0)

        return GraspMotorGoals(
            pull_bias=0.0,
            brace_bias=brace,
            recover_bias=recover,
            damp_velocity_scale=damp,
            role=GraspRole.ASSISTED,
        )
