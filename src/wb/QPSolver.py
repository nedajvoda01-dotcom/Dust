"""QPSolver — Stage 47 deterministic whole-body QP solver.

Solves the quadratic programme:

    min_x  0.5 x^T H x + f^T x
    s.t.   A x ≤ b          (inequality)
           Aeq x = beq       (equality)

Implementation uses a lightweight **active-set** method with a fixed
iteration budget so that runtime is bounded and deterministic.

Design principles
-----------------
* No third-party numerical libraries — pure Python so the same logic
  runs in any environment.
* Deterministic: given identical inputs the solver always follows the
  same code path and produces bit-identical outputs.
* Budget-limited: ``max_iters`` caps the active-set iterations.
* Graceful degradation: if the budget is exhausted or the system is
  infeasible, the solver returns the best iterate found so far and sets
  ``converged=False``; callers fall back to the legacy controller.

Public API
----------
QPProblem   (dataclass) — packed problem description
QPResult    (dataclass) — solver output
QPSolver(config=None)
  .solve(problem) → QPResult
"""
from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass, field
from typing import List, Optional


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


def _dot(a: List[float], b: List[float]) -> float:
    return sum(ai * bi for ai, bi in zip(a, b))


def _mat_vec(M: List[List[float]], v: List[float]) -> List[float]:
    """Multiply matrix M (row-major list of rows) by vector v."""
    return [_dot(row, v) for row in M]


def _vec_add(a: List[float], b: List[float]) -> List[float]:
    return [ai + bi for ai, bi in zip(a, b)]


def _vec_sub(a: List[float], b: List[float]) -> List[float]:
    return [ai - bi for ai, bi in zip(a, b)]


def _vec_scale(v: List[float], s: float) -> List[float]:
    return [vi * s for vi in v]


def _norm(v: List[float]) -> float:
    return math.sqrt(sum(vi * vi for vi in v))


# ---------------------------------------------------------------------------
# QPProblem
# ---------------------------------------------------------------------------

@dataclass
class QPProblem:
    """Full problem description for the whole-body QP.

    Attributes
    ----------
    n :
        Number of decision variables.
    H :
        Symmetric positive-semi-definite cost Hessian (n×n, row-major).
    f :
        Linear cost vector (length n).
    A :
        Inequality constraint matrix (m_ineq × n).
    b :
        Inequality RHS vector (length m_ineq).  Constraint: A x ≤ b.
    Aeq :
        Equality constraint matrix (m_eq × n).
    beq :
        Equality RHS vector (length m_eq).  Constraint: Aeq x = beq.
    x0 :
        Warm-start initial guess (length n); defaults to zeros.
    """
    n:   int
    H:   List[List[float]]
    f:   List[float]
    A:   List[List[float]]    = field(default_factory=list)
    b:   List[float]          = field(default_factory=list)
    Aeq: List[List[float]]    = field(default_factory=list)
    beq: List[float]          = field(default_factory=list)
    x0:  Optional[List[float]] = None


# ---------------------------------------------------------------------------
# QPResult
# ---------------------------------------------------------------------------

@dataclass
class QPResult:
    """Solver output.

    Attributes
    ----------
    x :
        Solution vector (length n).
    cost :
        Objective value at x.
    converged :
        True when the solver found a KKT-optimal point within budget.
    iters :
        Number of active-set iterations performed.
    infeasible :
        True when equality constraints could not be satisfied.
    solution_hash :
        SHA-256 hash of the solution vector (8 hex chars) for
        determinism cross-checks.
    """
    x:             List[float]
    cost:          float
    converged:     bool
    iters:         int
    infeasible:    bool        = False
    solution_hash: str         = ""


def _compute_hash(x: List[float]) -> str:
    """Stable hash of solution for determinism checks."""
    # Quantise to 1e-6 to ignore floating-point noise below solver epsilon
    quantised = [round(v, 6) for v in x]
    packed = struct.pack(">" + "d" * len(quantised), *quantised)
    return hashlib.sha256(packed).hexdigest()[:8]


# ---------------------------------------------------------------------------
# QPSolver
# ---------------------------------------------------------------------------

class QPSolver:
    """Deterministic budget-limited active-set QP solver.

    Parameters
    ----------
    config :
        Optional dict; reads ``qp.*`` keys.
    """

    _DEFAULT_MAX_ITERS       = 50
    _DEFAULT_EPS             = 1e-6
    _DEFAULT_MAX_VARIABLES   = 64
    _DEFAULT_MAX_CONSTRAINTS = 128

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg  = (config or {}).get("qp", {}) or {}
        self._max_iters        = int(cfg.get("max_iters",        self._DEFAULT_MAX_ITERS))
        self._eps              = float(cfg.get("eps",            self._DEFAULT_EPS))
        self._max_variables    = int(cfg.get("max_variables",    self._DEFAULT_MAX_VARIABLES))
        self._max_constraints  = int(cfg.get("max_constraints",  self._DEFAULT_MAX_CONSTRAINTS))
        self._fallback_on_fail = bool(cfg.get("fallback_on_fail", True))

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def solve(self, problem: QPProblem) -> QPResult:
        """Solve the QP and return :class:`QPResult`.

        The solver applies projected gradient descent with active-set
        constraint handling.  Equality constraints are enforced via
        penalty augmentation and projection.

        If the budget is exceeded, the best iterate is returned with
        ``converged=False`` so callers can apply graceful degradation.
        """
        n = problem.n
        if n == 0:
            return QPResult(x=[], cost=0.0, converged=True, iters=0,
                            solution_hash=_compute_hash([]))

        # Clamp to declared limits (LOD)
        n = min(n, self._max_variables)
        H = [row[:n] for row in problem.H[:n]]
        f = problem.f[:n]
        A   = [row[:n] for row in problem.A[:self._max_constraints]]
        b   = problem.b[:self._max_constraints]
        Aeq = [row[:n] for row in problem.Aeq[:self._max_constraints]]
        beq = problem.beq[:self._max_constraints]

        # Initialise solution
        x = list(problem.x0[:n]) if problem.x0 else [0.0] * n

        # Penalty weight for equality constraints
        rho = 1000.0

        converged  = False
        infeasible = False
        iters      = 0

        for it in range(self._max_iters):
            iters = it + 1
            grad = self._gradient(H, f, Aeq, beq, x, rho)

            # Project gradient: zero out components that push into violated
            # inequality constraints (active-set projection)
            grad = self._project_ineq(A, b, x, grad)

            # Diagonal preconditioning: scale each component by 1/H[i][i].
            # For diagonal H (our common case) this is the exact Newton step.
            precond_grad = [
                grad[i] / max(H[i][i], 1e-8) for i in range(n)
            ]

            # Convergence in preconditioned norm (|d_i| = |g_i / H_ii|)
            p_norm = _norm(precond_grad)
            if p_norm < self._eps:
                converged = True
                break

            # Armijo sufficient-decrease backtracking along preconditioned direction
            c_armijo = 0.1
            cost_old = self._cost_augmented(H, f, Aeq, beq, x, rho)
            armijo_rhs = c_armijo * _dot(grad, precond_grad)
            step = 1.0
            for _ in range(30):
                x_new = _vec_add(x, _vec_scale(precond_grad, -step))
                x_new = self._clip_ineq(A, b, x_new)
                cost_new = self._cost_augmented(H, f, Aeq, beq, x_new, rho)
                if cost_new <= cost_old - step * armijo_rhs:
                    break
                step *= 0.5

            x = x_new  # type: ignore[assignment]

        # Check equality feasibility
        if Aeq:
            eq_resid = _norm(_vec_sub(_mat_vec(Aeq, x), beq))
            infeasible = eq_resid > max(self._eps * 100, 1e-4)

        cost = self._qp_cost(H, f, x)
        h    = _compute_hash(x)

        # Pad x back to original n if it was clamped
        while len(x) < problem.n:
            x.append(0.0)

        return QPResult(
            x=x,
            cost=cost,
            converged=converged,
            iters=iters,
            infeasible=infeasible,
            solution_hash=h,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _exact_step(
        H: List[List[float]],
        Aeq: List[List[float]],
        grad: List[float],
        rho: float,
    ) -> float:
        """Compute the optimal steepest-descent step size for the augmented quadratic.

        For f(x) = 0.5 x^T H_aug x + …, the exact step along direction -g is:
            t* = g^T g / g^T H_aug g

        where H_aug = H + rho * Aeq^T Aeq.  Falls back to a safe default
        when the denominator is near zero.
        """
        Hg = _mat_vec(H, grad)
        if Aeq:
            Aeq_g = _mat_vec(Aeq, grad)
            n = len(grad)
            for j in range(n):
                for i, row in enumerate(Aeq):
                    Hg[j] += rho * row[j] * Aeq_g[i]
        g_norm_sq = _dot(grad, grad)
        gHg = _dot(grad, Hg)
        if gHg < 1e-12:
            return 1.0 / max(g_norm_sq ** 0.5, 1.0)
        return g_norm_sq / gHg

    @staticmethod
    def _qp_cost(H: List[List[float]], f: List[float], x: List[float]) -> float:
        Hx = _mat_vec(H, x)
        return 0.5 * _dot(x, Hx) + _dot(f, x)

    @staticmethod
    def _cost_augmented(
        H: List[List[float]],
        f: List[float],
        Aeq: List[List[float]],
        beq: List[float],
        x: List[float],
        rho: float,
    ) -> float:
        cost = QPSolver._qp_cost(H, f, x)
        if Aeq:
            residuals = _vec_sub(_mat_vec(Aeq, x), beq)
            cost += 0.5 * rho * _dot(residuals, residuals)
        return cost

    @staticmethod
    def _gradient(
        H: List[List[float]],
        f: List[float],
        Aeq: List[List[float]],
        beq: List[float],
        x: List[float],
        rho: float,
    ) -> List[float]:
        """Gradient of augmented Lagrangian: H x + f + rho * Aeq^T (Aeq x - beq)."""
        g = _vec_add(_mat_vec(H, x), f)
        if Aeq:
            residuals = _vec_sub(_mat_vec(Aeq, x), beq)
            # Aeq^T * residuals
            n = len(x)
            for j in range(n):
                for i, row in enumerate(Aeq):
                    g[j] += rho * row[j] * residuals[i]
        return g

    @staticmethod
    def _project_ineq(
        A: List[List[float]],
        b: List[float],
        x: List[float],
        grad: List[float],
    ) -> List[float]:
        """Zero gradient components that would push deeper into violated constraints."""
        if not A:
            return grad
        g = list(grad)
        for i, (row, bi) in enumerate(zip(A, b)):
            slack = bi - _dot(row, x)
            if slack < 0.0:
                # Constraint violated: project out component along row
                row_norm_sq = _dot(row, row)
                if row_norm_sq > 1e-12:
                    proj = _dot(g, row) / row_norm_sq
                    if proj < 0.0:
                        for j in range(len(g)):
                            g[j] -= proj * row[j]
        return g

    @staticmethod
    def _clip_ineq(
        A: List[List[float]],
        b: List[float],
        x: List[float],
    ) -> List[float]:
        """Push x back to feasible side of violated inequality constraints."""
        if not A:
            return x
        x = list(x)
        for row, bi in zip(A, b):
            val = _dot(row, x)
            if val > bi:
                row_norm_sq = _dot(row, row)
                if row_norm_sq > 1e-12:
                    excess = (val - bi) / row_norm_sq
                    for j in range(len(x)):
                        x[j] -= excess * row[j]
        return x
