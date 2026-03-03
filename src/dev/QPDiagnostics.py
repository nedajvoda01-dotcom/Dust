"""QPDiagnostics — Stage 47 QP solver developer diagnostics.

Provides toggle, logging, and visualisation helpers for the whole-body
QP controller in dev/debug builds.

In production (``enable_dev=False``) all methods are no-ops so there
is zero runtime cost.

Public API
----------
QPDiagnostics(config=None)
  .toggle_qp(enabled)
  .record_tick(result, wb_input, weights)
  .print_convergence()
  .get_summary()        → dict
  .reset()
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.wb.QPSolver import QPResult
    from src.wb.WBControllerAdapter import WBInput
    from src.wb.WeightScheduler import TaskWeights


@dataclass
class _TickRecord:
    """One tick of diagnostics data."""
    converged:      bool
    iters:          int
    cost:           float
    solution_hash:  str
    infeasible:     bool
    used_fallback:  bool
    brace_bias:     float
    global_risk:    float
    fatigue_energy: float
    balance_weight: float
    foot_weight:    float


class QPDiagnostics:
    """Developer-mode diagnostics for the whole-body QP controller.

    Parameters
    ----------
    config :
        Optional dict; reads ``dev.enable_dev`` and ``dev.qp_diag_buffer_size``.
    """

    _DEFAULT_BUFFER = 256

    def __init__(self, config: Optional[dict] = None) -> None:
        devcfg = (config or {}).get("dev", {}) or {}
        self._enabled = bool(devcfg.get("enable_dev", True))
        buf_size = int(devcfg.get("qp_diag_buffer_size", self._DEFAULT_BUFFER))
        self._buf_size = buf_size
        self._records:  List[_TickRecord] = []
        self._qp_enabled_override: Optional[bool] = None

    # ------------------------------------------------------------------
    # Toggle
    # ------------------------------------------------------------------

    def toggle_qp(self, enabled: bool) -> None:
        """Force-override the QP enable flag for testing (dev only)."""
        if self._enabled:
            self._qp_enabled_override = enabled

    @property
    def qp_override(self) -> Optional[bool]:
        """Current QP enable override (None = no override)."""
        return self._qp_enabled_override

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_tick(
        self,
        result:       "QPResult",
        wb_input:     "WBInput",
        weights:      "TaskWeights",
        used_fallback: bool = False,
    ) -> None:
        """Record one tick of solver data.

        Does nothing when dev mode is disabled.
        """
        if not self._enabled:
            return
        rec = _TickRecord(
            converged=result.converged,
            iters=result.iters,
            cost=result.cost,
            solution_hash=result.solution_hash,
            infeasible=result.infeasible,
            used_fallback=used_fallback,
            brace_bias=wb_input.brace_bias,
            global_risk=wb_input.global_risk,
            fatigue_energy=wb_input.fatigue_energy,
            balance_weight=weights.balance,
            foot_weight=weights.foot,
        )
        self._records.append(rec)
        # Ring-buffer: drop oldest when full
        if len(self._records) > self._buf_size:
            self._records.pop(0)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def print_convergence(self) -> None:
        """Print a brief convergence summary to stdout (dev only)."""
        if not self._enabled or not self._records:
            return
        total    = len(self._records)
        conv     = sum(1 for r in self._records if r.converged)
        fallback = sum(1 for r in self._records if r.used_fallback)
        avg_iter = sum(r.iters for r in self._records) / total
        print(
            f"[QPDiag] ticks={total}  converged={conv}/{total} "
            f"fallback={fallback}  avg_iters={avg_iter:.1f}"
        )

    def get_summary(self) -> dict:
        """Return a dict summary of recorded ticks."""
        if not self._records:
            return {"ticks": 0}
        total    = len(self._records)
        conv     = sum(1 for r in self._records if r.converged)
        fallback = sum(1 for r in self._records if r.used_fallback)
        avg_iter = sum(r.iters for r in self._records) / total
        avg_cost = sum(r.cost  for r in self._records) / total
        infeas   = sum(1 for r in self._records if r.infeasible)
        return {
            "ticks":          total,
            "converged":      conv,
            "fallback":       fallback,
            "infeasible":     infeas,
            "avg_iters":      avg_iter,
            "avg_cost":       avg_cost,
            "convergence_rate": conv / total,
        }

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all recorded ticks."""
        self._records.clear()
        self._qp_enabled_override = None
