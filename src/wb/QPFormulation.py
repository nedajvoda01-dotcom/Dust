"""QPFormulation — Stage 47 whole-body QP problem builder.

Assembles the :class:`~src.wb.QPSolver.QPProblem` from task and
constraint contributions each simulation tick.

Architecture
------------
Each *task* adds a (Ht, ft) pair (quadratic cost with weight wt) to H
and f.  Each *constraint* adds rows to A/b or Aeq/beq.

The combined cost is:

    H   = Σ_t  wt * Ht
    f   = Σ_t  wt * ft

Public API
----------
TaskContribution   (dataclass) — weighted task cost terms
ConstraintBlock    (dataclass) — inequality or equality constraint rows
QPFormulation(n, config=None)
  .reset()
  .add_task(task_contribution)
  .add_ineq(constraint_block)
  .add_eq(constraint_block)
  .build() → QPProblem
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from src.wb.QPSolver import QPProblem


# ---------------------------------------------------------------------------
# TaskContribution
# ---------------------------------------------------------------------------

@dataclass
class TaskContribution:
    """A single task's contribution to the QP cost.

    Adds:  weight * (0.5 x^T Ht x + ft^T x)

    Attributes
    ----------
    Ht :
        Task Hessian (n×n, row-major list of rows).
    ft :
        Task linear cost (length n).
    weight :
        Scalar weight ≥ 0.
    name :
        Human-readable task name for diagnostics.
    """
    Ht:     List[List[float]]
    ft:     List[float]
    weight: float
    name:   str = ""


# ---------------------------------------------------------------------------
# ConstraintBlock
# ---------------------------------------------------------------------------

@dataclass
class ConstraintBlock:
    """A set of linear constraints (rows).

    For inequalities:  A_rows x ≤ b_vals
    For equalities:    A_rows x  = b_vals

    Attributes
    ----------
    A_rows :
        Coefficient matrix rows (each row has length n).
    b_vals :
        RHS values (one per row).
    name :
        Human-readable name for diagnostics.
    """
    A_rows: List[List[float]]
    b_vals: List[float]
    name:   str = ""


# ---------------------------------------------------------------------------
# QPFormulation
# ---------------------------------------------------------------------------

class QPFormulation:
    """Assembles a :class:`QPProblem` from incremental task and constraint additions.

    Parameters
    ----------
    n :
        Number of decision variables.
    config :
        Optional dict; reads ``qp.*`` keys (currently unused here but
        kept for future regularisation options).
    """

    def __init__(self, n: int, config: Optional[dict] = None) -> None:
        self._n = n
        self._tasks:       List[TaskContribution] = []
        self._ineq_blocks: List[ConstraintBlock]  = []
        self._eq_blocks:   List[ConstraintBlock]  = []

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all tasks and constraints (call at start of each tick)."""
        self._tasks.clear()
        self._ineq_blocks.clear()
        self._eq_blocks.clear()

    def add_task(self, task: TaskContribution) -> None:
        """Register a task cost contribution."""
        if task.weight > 0.0:
            self._tasks.append(task)

    def add_ineq(self, block: ConstraintBlock) -> None:
        """Register an inequality constraint block (A x ≤ b)."""
        if block.A_rows:
            self._ineq_blocks.append(block)

    def add_eq(self, block: ConstraintBlock) -> None:
        """Register an equality constraint block (A x = b)."""
        if block.A_rows:
            self._eq_blocks.append(block)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, x0: Optional[List[float]] = None) -> QPProblem:
        """Assemble and return the complete :class:`QPProblem`.

        Parameters
        ----------
        x0 :
            Optional warm-start vector (length n).

        Returns
        -------
        QPProblem
            Ready to pass to :class:`~src.wb.QPSolver.QPSolver`.
        """
        n = self._n

        # Accumulate weighted task costs
        H = [[0.0] * n for _ in range(n)]
        f = [0.0] * n

        for task in self._tasks:
            w = task.weight
            for i in range(n):
                for j in range(n):
                    H[i][j] += w * task.Ht[i][j]
            for j in range(n):
                f[j] += w * task.ft[j]

        # Add small Tikhonov regularisation for positive-definiteness
        for i in range(n):
            H[i][i] += 1e-6

        # Flatten constraint blocks
        A   = [row for blk in self._ineq_blocks for row in blk.A_rows]
        b   = [v   for blk in self._ineq_blocks for v   in blk.b_vals]
        Aeq = [row for blk in self._eq_blocks   for row in blk.A_rows]
        beq = [v   for blk in self._eq_blocks   for v   in blk.b_vals]

        return QPProblem(n=n, H=H, f=f, A=A, b=b, Aeq=Aeq, beq=beq, x0=x0)
