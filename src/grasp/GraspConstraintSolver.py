"""GraspConstraintSolver — Stage 40 §8 / §9.

Integrates a confirmed :class:`~src.grasp.GraspConstraintBinder.GraspConstraint`
into the physics simulation by computing the constraint force each tick and
applying it to both bodies.

This module runs on **both** server and client.  On the server it produces
the authoritative force value; on clients it provides smooth prediction.

Public API
----------
BodyState (dataclass)
GraspConstraintSolver(config=None)
  .solve(constraint, body_a, body_b, dt) → ConstraintForceResult
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from src.math.Vec3 import Vec3
from src.grasp.GraspConstraintBinder import GraspConstraint


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


# ---------------------------------------------------------------------------
# BodyState — minimal physics state for the solver
# ---------------------------------------------------------------------------

@dataclass
class BodyState:
    """Minimal physics representation needed by the grasp solver.

    Attributes
    ----------
    position :
        World position of the body's reference point (e.g. pelvis).
    velocity :
        World velocity [m/s].
    mass :
        Body mass [kg]; used for impulse → velocity conversion.
    support_quality :
        Foot-support quality [0..1]; affects pull-bias capping.
    global_risk :
        Current risk estimate [0..1].
    """
    position:        Vec3
    velocity:        Vec3
    mass:            float = 70.0
    support_quality: float = 1.0
    global_risk:     float = 0.0


# ---------------------------------------------------------------------------
# ConstraintForceResult
# ---------------------------------------------------------------------------

@dataclass
class ConstraintForceResult:
    """Per-tick output from :class:`GraspConstraintSolver`.

    Attributes
    ----------
    force_magnitude :
        Constraint tension/compression magnitude [N or normalised].
    impulse_a :
        Velocity impulse to apply to body A (the one needing help).
    impulse_b :
        Velocity impulse to apply to body B (the helper).
    should_break :
        True when force exceeded the break threshold.
    """
    force_magnitude: float
    impulse_a:       Vec3
    impulse_b:       Vec3
    should_break:    bool


# ---------------------------------------------------------------------------
# GraspConstraintSolver
# ---------------------------------------------------------------------------

class GraspConstraintSolver:
    """Computes per-tick constraint forces for a GraspConstraint (§8).

    Parameters
    ----------
    config :
        Optional dict; reads ``grasp.*`` keys.
    """

    _DEFAULT_MAX_ITERATIONS = 4   # solve iterations per tick (§15)

    def __init__(self, config: Optional[dict] = None) -> None:
        gcfg = (config or {}).get("grasp", {}) or {}
        self._max_iter: int = int(gcfg.get("max_constraint_solve_iterations", self._DEFAULT_MAX_ITERATIONS))

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def solve(
        self,
        constraint: GraspConstraint,
        body_a:     BodyState,
        body_b:     BodyState,
        dt:         float,
    ) -> ConstraintForceResult:
        """Compute one tick of constraint impulse for the two bodies.

        Uses a simple position-based spring model with damping.
        """
        # World anchor positions (simplified: anchors treated as offsets from body CoM)
        anchor_world_a = body_a.position + constraint.anchor_a
        anchor_world_b = body_b.position + constraint.anchor_b

        delta = anchor_world_b - anchor_world_a
        dist  = delta.length()

        # Constraint violation = current length − rest length
        violation = dist - constraint.rest_length

        if dist < 1e-6:
            return ConstraintForceResult(
                force_magnitude=0.0,
                impulse_a=Vec3.zero(),
                impulse_b=Vec3.zero(),
                should_break=False,
            )

        direction = delta * (1.0 / dist)

        # Spring force (Hooke's law, simple)
        stiffness = constraint.max_force / max(constraint.rest_length, 0.5)
        spring_force = stiffness * violation

        # Damping — relative velocity along constraint axis
        rel_vel = body_b.velocity - body_a.velocity
        vel_along = rel_vel.dot(direction)
        damp_force = constraint.damping * vel_along

        total_force = _clamp(spring_force + damp_force, -constraint.max_force, constraint.max_force)
        force_mag   = abs(total_force)

        # Impulse = F * dt / mass  (simple first-order)
        impulse_mag_a = (total_force * dt) / max(body_a.mass, 1.0)
        impulse_mag_b = -(total_force * dt) / max(body_b.mass, 1.0)

        impulse_a = direction * impulse_mag_a
        impulse_b = direction * impulse_mag_b

        should_break = force_mag > constraint.break_force

        return ConstraintForceResult(
            force_magnitude=force_mag,
            impulse_a=impulse_a,
            impulse_b=impulse_b,
            should_break=should_break,
        )
