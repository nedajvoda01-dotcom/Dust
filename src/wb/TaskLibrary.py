"""TaskLibrary — Stage 47 whole-body task cost builders.

Each function returns a :class:`~src.wb.QPFormulation.TaskContribution`
that encodes the task's quadratic cost in the QP decision space.

Decision-variable layout (n = 6 in the minimal model)
------------------------------------------------------
x[0] : COM acceleration x  (horizontal, body frame)
x[1] : COM acceleration z  (horizontal, body frame)
x[2] : swing-foot acc x
x[3] : swing-foot acc z
x[4] : upper-body yaw acc  (head/torso blend)
x[5] : effort slack        (minimised directly)

This layout is the *minimal* 6-DOF model.  The :class:`WBControllerAdapter`
can expand to more DOFs by calling with a larger n.

Public API
----------
N_DOF = 6  (minimal DOF count)
build_balance_task(desired_com_acc, weight) → TaskContribution
build_foot_task(desired_swing_acc, weight)  → TaskContribution
build_look_task(desired_yaw_acc, weight)    → TaskContribution
build_brace_task(brace_bias, weight)        → TaskContribution
build_grasp_task(rel_pos_error, weight)     → TaskContribution
build_effort_task(weight)                   → TaskContribution
"""
from __future__ import annotations

from src.wb.QPFormulation import TaskContribution
from src.math.Vec3 import Vec3


# Minimal DOF count used by the default task library
N_DOF = 6

# Variable index aliases (documentation only)
_COM_X  = 0
_COM_Z  = 1
_FOOT_X = 2
_FOOT_Z = 3
_YAW    = 4
_EFFORT = 5


def _identity_task(n: int, indices: list, desired: list, weight: float, name: str) -> TaskContribution:
    """Build a task that tracks desired values for selected DOF indices.

    Equivalent to:  Ht = I_sel,  ft = -desired  (for ||x_sel - desired||^2)
    """
    Ht = [[0.0] * n for _ in range(n)]
    ft = [0.0] * n
    for idx, des in zip(indices, desired):
        if 0 <= idx < n:
            Ht[idx][idx] = 1.0
            ft[idx] = -des
    return TaskContribution(Ht=Ht, ft=ft, weight=weight, name=name)


# ---------------------------------------------------------------------------
# Balance task
# ---------------------------------------------------------------------------

def build_balance_task(
    desired_com_acc: Vec3,
    weight: float = 1.0,
    n: int = N_DOF,
) -> TaskContribution:
    """Track desired COM horizontal acceleration.

    Parameters
    ----------
    desired_com_acc :
        Target COM acceleration in the body horizontal plane (x, z used).
    weight :
        Task weight (higher → solver prioritises this task).
    n :
        Total DOF count.
    """
    return _identity_task(
        n=n,
        indices=[_COM_X, _COM_Z],
        desired=[desired_com_acc.x, desired_com_acc.z],
        weight=weight,
        name="balance",
    )


# ---------------------------------------------------------------------------
# Foot placement task
# ---------------------------------------------------------------------------

def build_foot_task(
    desired_swing_acc: Vec3,
    weight: float = 1.0,
    n: int = N_DOF,
) -> TaskContribution:
    """Track desired swing-foot horizontal acceleration.

    Parameters
    ----------
    desired_swing_acc :
        Target acceleration for the swing foot (x, z used).
    weight :
        Task weight.
    n :
        Total DOF count.
    """
    return _identity_task(
        n=n,
        indices=[_FOOT_X, _FOOT_Z],
        desired=[desired_swing_acc.x, desired_swing_acc.z],
        weight=weight,
        name="foot",
    )


# ---------------------------------------------------------------------------
# Look task
# ---------------------------------------------------------------------------

def build_look_task(
    desired_yaw_acc: float,
    weight: float = 1.0,
    n: int = N_DOF,
) -> TaskContribution:
    """Track desired upper-body yaw acceleration (head+torso blend).

    Parameters
    ----------
    desired_yaw_acc :
        Target yaw angular acceleration [rad/s²].
    weight :
        Task weight.
    n :
        Total DOF count.
    """
    return _identity_task(
        n=n,
        indices=[_YAW],
        desired=[desired_yaw_acc],
        weight=weight,
        name="look",
    )


# ---------------------------------------------------------------------------
# Brace task
# ---------------------------------------------------------------------------

def build_brace_task(
    brace_bias: float,
    weight: float = 1.0,
    n: int = N_DOF,
) -> TaskContribution:
    """Encourage arm-support posture when brace_bias is high.

    This task biases the COM toward a protective lean by requesting a
    small deceleration proportional to ``brace_bias``.

    Parameters
    ----------
    brace_bias :
        Brace urgency [0..1] from ReflexOverlay or PerceptionState.
    weight :
        Task weight.
    n :
        Total DOF count.
    """
    # Brace = pull COM backward / downward (negative x acceleration)
    desired_x = -brace_bias * 0.5   # m/s² equivalent
    return _identity_task(
        n=n,
        indices=[_COM_X],
        desired=[desired_x],
        weight=weight,
        name="brace",
    )


# ---------------------------------------------------------------------------
# Grasp task
# ---------------------------------------------------------------------------

def build_grasp_task(
    rel_pos_error_x: float,
    rel_pos_error_z: float,
    weight: float = 1.0,
    n: int = N_DOF,
) -> TaskContribution:
    """Track relative anchor position error between grasping players.

    Parameters
    ----------
    rel_pos_error_x :
        Anchor separation error in x [m].
    rel_pos_error_z :
        Anchor separation error in z [m].
    weight :
        Task weight.
    n :
        Total DOF count.
    """
    # Desired COM acceleration to close the anchor separation
    return _identity_task(
        n=n,
        indices=[_COM_X, _COM_Z],
        desired=[rel_pos_error_x, rel_pos_error_z],
        weight=weight,
        name="grasp",
    )


# ---------------------------------------------------------------------------
# Effort minimisation task
# ---------------------------------------------------------------------------

def build_effort_task(
    weight: float = 0.01,
    n: int = N_DOF,
) -> TaskContribution:
    """Minimise total effort (minimise ||x||²).

    This is the Tikhonov / regularisation term that keeps solutions
    small when no other tasks demand large accelerations.

    Parameters
    ----------
    weight :
        Task weight (should be small relative to other tasks).
    n :
        Total DOF count.
    """
    Ht = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    ft = [0.0] * n
    return TaskContribution(Ht=Ht, ft=ft, weight=weight, name="effort")
