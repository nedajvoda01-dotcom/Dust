"""ConstraintLibrary — Stage 47 whole-body constraint builders.

Each function returns a :class:`~src.wb.QPFormulation.ConstraintBlock`
that encodes a physical constraint in the QP decision space.

Decision-variable layout (matching TaskLibrary)
------------------------------------------------
x[0] : COM acceleration x
x[1] : COM acceleration z
x[2] : swing-foot acc x
x[3] : swing-foot acc z
x[4] : upper-body yaw acc
x[5] : effort slack

Public API
----------
build_joint_limit_constraints(acc_max, n)           → ConstraintBlock
build_torque_limit_constraints(torque_max, n)        → ConstraintBlock
build_contact_no_penetration_constraints(normal, n)  → ConstraintBlock
build_friction_cone_constraints(mu, normal_force, n) → ConstraintBlock
build_grasp_force_constraint(max_force, n)           → ConstraintBlock
"""
from __future__ import annotations

from typing import List, Optional

from src.wb.QPFormulation import ConstraintBlock
from src.math.Vec3 import Vec3
from src.wb.TaskLibrary import N_DOF, _COM_X, _COM_Z, _FOOT_X, _FOOT_Z, _YAW


def _zeros_row(n: int) -> List[float]:
    return [0.0] * n


# ---------------------------------------------------------------------------
# Joint / acceleration limits
# ---------------------------------------------------------------------------

def build_joint_limit_constraints(
    acc_max: float = 10.0,
    n: int = N_DOF,
) -> ConstraintBlock:
    """Box constraints: -acc_max ≤ x_i ≤ acc_max for all DOFs.

    Encoded as 2n inequality rows:
        x_i ≤  acc_max   →  [0…1…0] x ≤  acc_max
       -x_i ≤  acc_max   → [0…-1…0] x ≤  acc_max
    """
    rows: List[List[float]] = []
    vals: List[float]       = []
    for i in range(n):
        # upper bound
        row_pos = _zeros_row(n)
        row_pos[i] = 1.0
        rows.append(row_pos)
        vals.append(acc_max)
        # lower bound
        row_neg = _zeros_row(n)
        row_neg[i] = -1.0
        rows.append(row_neg)
        vals.append(acc_max)
    return ConstraintBlock(A_rows=rows, b_vals=vals, name="joint_limits")


# ---------------------------------------------------------------------------
# Torque / effort limits (fatigue-aware)
# ---------------------------------------------------------------------------

def build_torque_limit_constraints(
    torque_max: float = 1.0,
    n: int = N_DOF,
) -> ConstraintBlock:
    """Limit peak control magnitude (torque proxy) across all DOFs.

    Caller passes ``torque_max = tau_nominal * fatigueTorqueScale`` so
    fatigue automatically tightens this constraint.

    Encoded as a single row-sum upper bound:
        sum(x_i) ≤ n * torque_max
    """
    # One row per DOF bounding its individual contribution
    rows: List[List[float]] = []
    vals: List[float]       = []
    for i in range(n):
        row_pos = _zeros_row(n)
        row_pos[i] = 1.0
        rows.append(row_pos)
        vals.append(torque_max)
        row_neg = _zeros_row(n)
        row_neg[i] = -1.0
        rows.append(row_neg)
        vals.append(torque_max)
    return ConstraintBlock(A_rows=rows, b_vals=vals, name="torque_limits")


# ---------------------------------------------------------------------------
# Contact no-penetration
# ---------------------------------------------------------------------------

def build_contact_no_penetration_constraints(
    surface_normal: Optional[Vec3] = None,
    n: int = N_DOF,
) -> ConstraintBlock:
    """Prevent COM from accelerating into the contact surface.

    The normal (pointing away from the surface) must have a non-negative
    component in the COM acceleration:

        -n_x * x[COM_X] - n_z * x[COM_Z] ≤ 0

    Parameters
    ----------
    surface_normal :
        Unit surface normal in the body frame (horizontal components used).
        Defaults to Vec3(0, 1, 0) (flat ground pointing up).
    n :
        Total DOF count.
    """
    if surface_normal is None:
        surface_normal = Vec3(0.0, 1.0, 0.0)

    row = _zeros_row(n)
    row[_COM_X] = -surface_normal.x
    row[_COM_Z] = -surface_normal.z
    return ConstraintBlock(A_rows=[row], b_vals=[0.0], name="no_penetration")


# ---------------------------------------------------------------------------
# Friction cone
# ---------------------------------------------------------------------------

def build_friction_cone_constraints(
    mu: float = 0.8,
    normal_force: float = 700.0,
    n: int = N_DOF,
) -> ConstraintBlock:
    """Tangential foot force must stay inside the friction cone.

    Simplified 2D Coulomb cone:

        |tangential_acc| ≤ mu * normal_force / mass_proxy

    Encoded as two rows (positive/negative tangential direction):
        x[FOOT_X] ≤  mu * normal_force
       -x[FOOT_X] ≤  mu * normal_force
        x[FOOT_Z] ≤  mu * normal_force
       -x[FOOT_Z] ≤  mu * normal_force

    Parameters
    ----------
    mu :
        Coulomb friction coefficient [0..1].
    normal_force :
        Normal force at the contact [N].  Caller provides this from the
        physics engine or a body-weight proxy.
    n :
        Total DOF count.
    """
    limit = mu * max(normal_force, 0.0)
    rows: List[List[float]] = []
    vals: List[float]       = []
    for idx in [_FOOT_X, _FOOT_Z]:
        if idx < n:
            row_pos = _zeros_row(n)
            row_pos[idx] = 1.0
            rows.append(row_pos)
            vals.append(limit)
            row_neg = _zeros_row(n)
            row_neg[idx] = -1.0
            rows.append(row_neg)
            vals.append(limit)
    return ConstraintBlock(A_rows=rows, b_vals=vals, name="friction_cone")


# ---------------------------------------------------------------------------
# Grasp force limit
# ---------------------------------------------------------------------------

def build_grasp_force_constraint(
    max_force: float = 800.0,
    n: int = N_DOF,
) -> ConstraintBlock:
    """Limit the grasp-induced COM acceleration to max_force proxy.

    Encoded as:
        x[COM_X] ≤  max_force
       -x[COM_X] ≤  max_force
        x[COM_Z] ≤  max_force
       -x[COM_Z] ≤  max_force

    Parameters
    ----------
    max_force :
        Maximum grasp force [N]; from GraspConstraint.max_force.
    n :
        Total DOF count.
    """
    rows: List[List[float]] = []
    vals: List[float]       = []
    for idx in [_COM_X, _COM_Z]:
        if idx < n:
            row_pos = _zeros_row(n)
            row_pos[idx] = 1.0
            rows.append(row_pos)
            vals.append(max_force)
            row_neg = _zeros_row(n)
            row_neg[idx] = -1.0
            rows.append(row_neg)
            vals.append(max_force)
    return ConstraintBlock(A_rows=rows, b_vals=vals, name="grasp_force")
