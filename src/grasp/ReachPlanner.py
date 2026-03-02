"""ReachPlanner — Stage 40 §6.

Plans the hand trajectory toward a grasp target using whole-body IK
constraints, ensuring the character's COM stays within the support
polygon.

The planner does **not** teleport the hand; it computes a target endpoint
and a feasibility flag.  The MotorStack drives the hand incrementally
toward the target each tick.

Public API
----------
ReachPlanner(config=None)
  .plan(self_pos, self_vel, target_pos, self_risk,
        support_quality) → ReachPlan
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from src.math.Vec3 import Vec3


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


# ---------------------------------------------------------------------------
# ReachPlan
# ---------------------------------------------------------------------------

@dataclass
class ReachPlan:
    """Output of :class:`ReachPlanner`.

    Attributes
    ----------
    feasible :
        True when the reach can be performed without losing balance.
    hand_target :
        World-space target for the reaching hand.
    reach_fraction :
        [0..1] fraction of full extension currently safe to apply.
        0 = do not reach; 1 = full extension safe.
    com_margin :
        Estimated COM-to-support-edge clearance [0..1]; lower = less safe.
    """
    feasible:       bool
    hand_target:    Vec3
    reach_fraction: float
    com_margin:     float


# ---------------------------------------------------------------------------
# ReachPlanner
# ---------------------------------------------------------------------------

class ReachPlanner:
    """Plans a whole-body-IK reach toward a grasp target (§6).

    Parameters
    ----------
    config :
        Optional dict; reads ``grasp.*`` keys.
    """

    _DEFAULT_REACH_RADIUS   = 2.0    # metres — arm + lean reach
    _DEFAULT_COM_MARGIN_MIN = 0.15   # abort reach if COM margin falls below

    def __init__(self, config: Optional[dict] = None) -> None:
        gcfg = (config or {}).get("grasp", {}) or {}
        self._reach_radius:    float = float(gcfg.get("reach_radius",    self._DEFAULT_REACH_RADIUS))
        self._com_margin_min:  float = float(gcfg.get("com_margin_min",  self._DEFAULT_COM_MARGIN_MIN))

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def plan(
        self,
        self_pos:        Vec3,
        self_vel:        Vec3,
        target_pos:      Vec3,
        self_risk:       float,
        support_quality: float = 1.0,
    ) -> ReachPlan:
        """Compute a reach plan toward *target_pos*.

        Parameters
        ----------
        self_pos :
            World position of the reaching character.
        self_vel :
            Current velocity of the reaching character.
        target_pos :
            World position of the grasp target (hand/forearm/harness).
        self_risk :
            Current global risk for this character [0..1].
        support_quality :
            Foot-support quality [0..1]; 1 = solid ground, 0 = no support.
        """
        diff = target_pos - self_pos
        dist = diff.length()

        # COM margin: degrades with risk and velocity magnitude
        speed = self_vel.length()
        com_margin = _clamp(
            support_quality * (1.0 - self_risk) - speed * 0.05,
            0.0, 1.0
        )

        # Can we safely reach at all?
        if dist > self._reach_radius or com_margin < self._com_margin_min:
            return ReachPlan(
                feasible=False,
                hand_target=self_pos,
                reach_fraction=0.0,
                com_margin=com_margin,
            )

        # Fraction of full extension that is safe
        # — further away and lower COM margin → smaller fraction
        dist_factor    = _clamp(1.0 - dist / self._reach_radius, 0.0, 1.0)
        margin_factor  = _clamp((com_margin - self._com_margin_min) / (1.0 - self._com_margin_min), 0.0, 1.0)
        reach_fraction = _clamp(dist_factor * 0.4 + margin_factor * 0.6, 0.0, 1.0)

        # Limit hand target to within safe reach envelope
        if dist > 1e-6:
            safe_dist   = min(dist, self._reach_radius * reach_fraction)
            hand_target = self_pos + diff * (safe_dist / dist)
        else:
            hand_target = target_pos

        return ReachPlan(
            feasible=True,
            hand_target=hand_target,
            reach_fraction=reach_fraction,
            com_margin=com_margin,
        )
