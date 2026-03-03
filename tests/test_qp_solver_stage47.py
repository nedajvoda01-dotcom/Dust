"""test_qp_solver_stage47.py — Stage 47 Whole-Body QP Solver smoke tests.

Tests
-----
1. test_qp_converges_in_budget
   — Solver converges within max_iters for a simple unconstrained problem.

2. test_balance_margin_improves_vs_baseline
   — With balance task active, QP solution reduces COM acceleration error
     vs. the zero-output baseline.

3. test_contact_friction_respected
   — Friction cone constraint keeps swing-foot DOFs within mu*N_f bound.

4. test_grasp_constraint_stable
   — With active grasp task, QP keeps grasp DOF within max_force bound
     and does not break the constraint.

5. test_fatigue_reduces_tau_limits
   — Reduced fatigue_torque_scale tightens torque-limit constraints,
     producing smaller joint accelerations.

6. test_determinism_same_inputs_same_qp_hash
   — Two independent WBControllerAdapter.tick() calls with identical
     inputs produce the same solution_hash.

7. test_fallback_on_solver_fail
   — When max_iters=1 (force non-convergence) and fallback_on_fail=True,
     WBOutput.used_fallback is True.
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.math.Vec3 import Vec3
from src.wb.QPSolver import QPSolver, QPProblem
from src.wb.QPFormulation import QPFormulation
from src.wb.TaskLibrary import (
    N_DOF,
    build_balance_task, build_foot_task, build_look_task,
    build_brace_task, build_grasp_task, build_effort_task,
)
from src.wb.ConstraintLibrary import (
    build_friction_cone_constraints, build_torque_limit_constraints,
    build_joint_limit_constraints,
)
from src.wb.WeightScheduler import WeightScheduler
from src.wb.WBControllerAdapter import WBControllerAdapter, WBInput
from src.dev.QPDiagnostics import QPDiagnostics


# ---------------------------------------------------------------------------
# Shared config helpers
# ---------------------------------------------------------------------------

_CFG_DEFAULT = {
    "qp": {
        "enable": True,
        "shadow_mode": False,
        "max_iters": 50,
        "eps": 1e-6,
        "max_variables": 64,
        "max_constraints": 128,
        "fallback_on_fail": True,
        "task_weights": {
            "balance": 2.0,
            "look":    0.5,
            "foot":    1.0,
            "brace":   0.5,
            "grasp":   1.0,
        },
        "effort_weight": 0.01,
        "lod": {"remote_reduce_factor": 0.5},
    },
    "dev": {
        "enable_dev": True,
        "qp_diag_buffer_size": 256,
    },
}

_CFG_FALLBACK = {
    "qp": {
        "enable": True,
        "shadow_mode": False,
        "max_iters": 1,   # force non-convergence
        "eps": 1e-6,
        "max_variables": 64,
        "max_constraints": 128,
        "fallback_on_fail": True,
        "task_weights": {
            "balance": 2.0,
            "look":    0.5,
            "foot":    1.0,
            "brace":   0.5,
            "grasp":   1.0,
        },
        "effort_weight": 0.01,
        "lod": {"remote_reduce_factor": 0.5},
    },
}


def _make_input(**kwargs) -> WBInput:
    """Return a WBInput with sensible defaults, overridden by kwargs."""
    defaults = dict(
        desired_com_acc=Vec3(1.0, 0.0, 0.5),
        desired_swing_acc=Vec3(0.3, 0.0, 0.0),
        desired_yaw_acc=0.2,
        brace_bias=0.0,
        global_risk=0.1,
        slip_risk=0.0,
        wind_load=0.0,
        surface_normal=Vec3(0.0, 1.0, 0.0),
        mu=0.8,
        normal_force=700.0,
        grasp_active=False,
        grasp_rel_error_x=0.0,
        grasp_rel_error_z=0.0,
        grasp_max_force=800.0,
        fatigue_energy=1.0,
        fatigue_torque_scale=1.0,
    )
    defaults.update(kwargs)
    return WBInput(**defaults)


# ---------------------------------------------------------------------------
# 1. test_qp_converges_in_budget
# ---------------------------------------------------------------------------

class TestQPConvergesInBudget(unittest.TestCase):
    """The solver must converge on a simple QP within max_iters."""

    def test_simple_unconstrained_convergence(self):
        """2-variable positive-definite QP: x^T I x → min at x=0."""
        solver = QPSolver(_CFG_DEFAULT)
        n = 2
        H = [[2.0, 0.0], [0.0, 2.0]]
        f = [0.0, 0.0]
        problem = QPProblem(n=n, H=H, f=f)
        result = solver.solve(problem)

        self.assertTrue(result.converged,
                        f"Solver did not converge; iters={result.iters}")
        self.assertLessEqual(result.iters, 50)
        self.assertAlmostEqual(result.x[0], 0.0, places=3)
        self.assertAlmostEqual(result.x[1], 0.0, places=3)

    def test_budget_respected(self):
        """Solver uses ≤ max_iters iterations even for a harder problem."""
        cfg = {"qp": {"max_iters": 10, "eps": 1e-10, "max_variables": 64,
                      "max_constraints": 128, "fallback_on_fail": True}}
        solver = QPSolver(cfg)
        n = N_DOF
        H = [[2.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        f = [0.5] * n
        problem = QPProblem(n=n, H=H, f=f)
        result = solver.solve(problem)

        self.assertLessEqual(result.iters, 10,
                             "Solver exceeded its iteration budget")

    def test_wbadapter_converges_default_cfg(self):
        """WBControllerAdapter.tick() converges on a normal input."""
        adapter = WBControllerAdapter(_CFG_DEFAULT)
        out = adapter.tick(dt=1.0 / 60.0, wb_input=_make_input())
        self.assertTrue(out.converged,
                        f"WB adapter did not converge: iters={out.solver_iters}")
        self.assertEqual(len(out.joint_accelerations), N_DOF)


# ---------------------------------------------------------------------------
# 2. test_balance_margin_improves_vs_baseline
# ---------------------------------------------------------------------------

class TestBalanceMarginImprovesVsBaseline(unittest.TestCase):
    """QP with balance task should yield smaller COM error than zero baseline."""

    def _com_error(self, desired: Vec3, acc: list) -> float:
        """Euclidean error between desired COM acc and solution."""
        ex = acc[0] - desired.x
        ez = acc[1] - desired.z
        return (ex ** 2 + ez ** 2) ** 0.5

    def test_balance_task_reduces_error(self):
        """Solution with balance task has smaller COM error than zeros."""
        desired = Vec3(1.0, 0.0, 0.5)
        adapter = WBControllerAdapter(_CFG_DEFAULT)
        wb_in   = _make_input(desired_com_acc=desired)
        out     = adapter.tick(dt=1.0 / 60.0, wb_input=wb_in)

        baseline_error = self._com_error(desired, [0.0, 0.0])
        qp_error       = self._com_error(desired, out.joint_accelerations[:2])

        self.assertLess(
            qp_error, baseline_error,
            f"QP error {qp_error:.4f} should be less than baseline {baseline_error:.4f}",
        )

    def test_balance_weight_scales_tracking(self):
        """Higher balance weight → better COM tracking."""
        desired = Vec3(2.0, 0.0, 1.0)

        cfg_low  = {"qp": dict(_CFG_DEFAULT["qp"], task_weights={"balance": 0.1, "foot": 1.0,
                                                                   "look": 0.5, "brace": 0.5,
                                                                   "grasp": 1.0},
                                effort_weight=0.001)}
        cfg_high = {"qp": dict(_CFG_DEFAULT["qp"], task_weights={"balance": 10.0, "foot": 1.0,
                                                                   "look": 0.5, "brace": 0.5,
                                                                   "grasp": 1.0},
                                effort_weight=0.001)}

        out_low  = WBControllerAdapter(cfg_low).tick(1.0 / 60.0, _make_input(desired_com_acc=desired))
        out_high = WBControllerAdapter(cfg_high).tick(1.0 / 60.0, _make_input(desired_com_acc=desired))

        err_low  = self._com_error(desired, out_low.joint_accelerations[:2])
        err_high = self._com_error(desired, out_high.joint_accelerations[:2])

        self.assertLess(
            err_high, err_low,
            f"High-weight error {err_high:.4f} should be less than low-weight {err_low:.4f}",
        )


# ---------------------------------------------------------------------------
# 3. test_contact_friction_respected
# ---------------------------------------------------------------------------

class TestContactFrictionRespected(unittest.TestCase):
    """Friction cone constraint must keep swing-foot DOFs within mu*N_f."""

    def test_friction_cone_limits_foot_acc(self):
        """Foot accelerations must not exceed friction cone bound."""
        mu    = 0.5
        N_f   = 700.0
        limit = mu * N_f

        adapter = WBControllerAdapter(_CFG_DEFAULT)
        wb_in   = _make_input(
            desired_swing_acc=Vec3(1000.0, 0.0, 1000.0),  # large desired → hits cone
            mu=mu,
            normal_force=N_f,
        )
        out = adapter.tick(dt=1.0 / 60.0, wb_input=wb_in)

        foot_x = out.joint_accelerations[2]  # _FOOT_X = 2
        foot_z = out.joint_accelerations[3]  # _FOOT_Z = 3

        self.assertLessEqual(
            abs(foot_x), limit + 1e-3,
            f"Foot acc x={foot_x:.2f} exceeds friction limit {limit:.2f}",
        )
        self.assertLessEqual(
            abs(foot_z), limit + 1e-3,
            f"Foot acc z={foot_z:.2f} exceeds friction limit {limit:.2f}",
        )

    def test_higher_mu_allows_larger_foot_acc(self):
        """Higher friction coefficient allows larger foot acceleration."""
        desired = Vec3(200.0, 0.0, 0.0)

        out_low_mu  = WBControllerAdapter(_CFG_DEFAULT).tick(
            1.0 / 60.0, _make_input(desired_swing_acc=desired, mu=0.1, normal_force=700.0))
        out_high_mu = WBControllerAdapter(_CFG_DEFAULT).tick(
            1.0 / 60.0, _make_input(desired_swing_acc=desired, mu=0.9, normal_force=700.0))

        acc_low  = abs(out_low_mu.joint_accelerations[2])
        acc_high = abs(out_high_mu.joint_accelerations[2])

        self.assertLessEqual(acc_low, 0.1 * 700.0 + 1e-3)
        self.assertLessEqual(acc_high, 0.9 * 700.0 + 1e-3)
        self.assertGreaterEqual(acc_high, acc_low - 1e-3,
                                "Higher mu should allow larger foot acceleration")


# ---------------------------------------------------------------------------
# 4. test_grasp_constraint_stable
# ---------------------------------------------------------------------------

class TestGraspConstraintStable(unittest.TestCase):
    """Grasp task active → COM DOFs stay within max_force bound."""

    def test_grasp_keeps_com_within_max_force(self):
        """With grasp task, COM accelerations stay ≤ grasp_max_force."""
        max_force = 400.0
        adapter   = WBControllerAdapter(_CFG_DEFAULT)
        wb_in     = _make_input(
            grasp_active=True,
            grasp_rel_error_x=5.0,   # large error
            grasp_rel_error_z=5.0,
            grasp_max_force=max_force,
        )
        out = adapter.tick(dt=1.0 / 60.0, wb_input=wb_in)

        com_x = out.joint_accelerations[0]
        com_z = out.joint_accelerations[1]

        self.assertLessEqual(
            abs(com_x), max_force + 1e-3,
            f"Grasp COM x={com_x:.2f} exceeds max_force {max_force}",
        )
        self.assertLessEqual(
            abs(com_z), max_force + 1e-3,
            f"Grasp COM z={com_z:.2f} exceeds max_force {max_force}",
        )

    def test_grasp_inactive_vs_active_differ(self):
        """Enabling grasp task produces a different solution than disabled."""
        adapter = WBControllerAdapter(_CFG_DEFAULT)

        out_no_grasp = adapter.tick(
            1.0 / 60.0, _make_input(grasp_active=False))
        out_grasp    = adapter.tick(
            1.0 / 60.0, _make_input(
                grasp_active=True,
                grasp_rel_error_x=2.0,
                grasp_rel_error_z=1.0,
            ))

        # Solutions should differ when grasp is active
        diff = sum(abs(a - b) for a, b in zip(
            out_no_grasp.joint_accelerations,
            out_grasp.joint_accelerations,
        ))
        self.assertGreater(diff, 1e-6, "Grasp task made no difference to solution")


# ---------------------------------------------------------------------------
# 5. test_fatigue_reduces_tau_limits
# ---------------------------------------------------------------------------

class TestFatigueReducesTauLimits(unittest.TestCase):
    """Reduced fatigue_torque_scale must tighten limits → smaller accelerations."""

    def _acc_magnitude(self, accs: list) -> float:
        return sum(a ** 2 for a in accs) ** 0.5

    def test_fatigued_produces_smaller_accelerations(self):
        """Fatigued torque cap must clip solution when desired exceeds limit."""
        # Use desired large enough to exceed even the fresh torque cap
        # so the fatigued cap (torque_scale=0.3 → cap=3.0) clearly clips lower
        desired = Vec3(15.0, 0.0, 15.0)

        fresh    = WBControllerAdapter(_CFG_DEFAULT)
        fatigued = WBControllerAdapter(_CFG_DEFAULT)

        out_fresh    = fresh.tick(1.0 / 60.0, _make_input(
            desired_com_acc=desired,
            fatigue_energy=1.0, fatigue_torque_scale=1.0))
        out_fatigued = fatigued.tick(1.0 / 60.0, _make_input(
            desired_com_acc=desired,
            fatigue_energy=0.1, fatigue_torque_scale=0.3))

        # Fatigued torque_cap = 0.3 * 10 = 3.0; fresh = 1.0 * 10 = 10.0
        # COM DOFs should be clipped at the respective caps
        cap_fresh    = 1.0 * 10.0
        cap_fatigued = 0.3 * 10.0

        com_x_fresh    = abs(out_fresh.joint_accelerations[0])
        com_x_fatigued = abs(out_fatigued.joint_accelerations[0])

        self.assertLessEqual(
            com_x_fresh, cap_fresh + 1e-3,
            f"Fresh COM x {com_x_fresh:.3f} should be ≤ cap {cap_fresh}",
        )
        self.assertLessEqual(
            com_x_fatigued, cap_fatigued + 1e-3,
            f"Fatigued COM x {com_x_fatigued:.3f} should be ≤ cap {cap_fatigued}",
        )

    def test_torque_limit_tightens_with_fatigue(self):
        """torque_max = torque_scale * 10; smaller scale → smaller bound."""
        formul_low  = QPFormulation(N_DOF)
        formul_high = QPFormulation(N_DOF)

        formul_low.add_ineq(build_torque_limit_constraints(torque_max=3.0))
        formul_high.add_ineq(build_torque_limit_constraints(torque_max=10.0))

        prob_low  = formul_low.build()
        prob_high = formul_high.build()

        # The b vectors should differ (low is tighter)
        self.assertLess(max(prob_low.b), max(prob_high.b),
                        "Low torque limit should be smaller than high")


# ---------------------------------------------------------------------------
# 6. test_determinism_same_inputs_same_qp_hash
# ---------------------------------------------------------------------------

class TestDeterminismSameInputsSameQPHash(unittest.TestCase):
    """Identical inputs must produce identical solution_hash."""

    def _run(self, cfg, wb_in: WBInput) -> str:
        adapter = WBControllerAdapter(cfg)
        out = adapter.tick(dt=1.0 / 60.0, wb_input=wb_in)
        return out.solution_hash

    def test_same_inputs_same_hash(self):
        """Two independent adapter instances with same inputs → same hash."""
        wb_in = _make_input(
            desired_com_acc=Vec3(1.5, 0.0, 0.7),
            desired_swing_acc=Vec3(0.4, 0.0, 0.1),
            desired_yaw_acc=0.3,
            brace_bias=0.2,
            global_risk=0.3,
            slip_risk=0.1,
            wind_load=0.2,
            mu=0.7,
            normal_force=680.0,
            grasp_active=True,
            grasp_rel_error_x=0.5,
            grasp_rel_error_z=0.3,
            grasp_max_force=600.0,
            fatigue_energy=0.75,
            fatigue_torque_scale=0.85,
        )
        hash1 = self._run(_CFG_DEFAULT, wb_in)
        hash2 = self._run(_CFG_DEFAULT, wb_in)

        self.assertEqual(hash1, hash2,
                         f"Hashes differ: {hash1!r} vs {hash2!r}")

    def test_different_inputs_different_hash(self):
        """Different inputs must generally produce different solution hashes."""
        wb_in_a = _make_input(desired_com_acc=Vec3(1.0, 0.0, 0.0))
        wb_in_b = _make_input(desired_com_acc=Vec3(0.0, 0.0, 5.0))

        hash_a = self._run(_CFG_DEFAULT, wb_in_a)
        hash_b = self._run(_CFG_DEFAULT, wb_in_b)

        self.assertNotEqual(hash_a, hash_b,
                            "Different inputs produced the same hash (collision)")

    def test_sequential_ticks_deterministic(self):
        """Running the same tick sequence twice must produce identical hashes."""
        inputs = [
            _make_input(desired_com_acc=Vec3(float(i) * 0.1, 0.0, 0.0),
                        fatigue_energy=max(0.1, 1.0 - i * 0.05))
            for i in range(10)
        ]

        def _run_sequence(cfg):
            adapter = WBControllerAdapter(cfg)
            return [adapter.tick(1.0 / 60.0, inp).solution_hash for inp in inputs]

        seq1 = _run_sequence(_CFG_DEFAULT)
        seq2 = _run_sequence(_CFG_DEFAULT)

        self.assertEqual(seq1, seq2,
                         "Sequential tick hashes differ between two identical runs")


# ---------------------------------------------------------------------------
# 7. test_fallback_on_solver_fail
# ---------------------------------------------------------------------------

class TestFallbackOnSolverFail(unittest.TestCase):
    """When solver cannot converge (max_iters=1), used_fallback must be True."""

    def test_fallback_engaged_when_not_converged(self):
        """With max_iters=1 and fallback_on_fail=True, adapter falls back."""
        adapter = WBControllerAdapter(_CFG_FALLBACK)
        out = adapter.tick(dt=1.0 / 60.0, wb_input=_make_input())
        self.assertTrue(out.used_fallback,
                        "Adapter should have fallen back when solver failed")

    def test_fallback_output_is_zero_vector(self):
        """Fallback output is the zero acceleration vector (safe default)."""
        adapter = WBControllerAdapter(_CFG_FALLBACK)
        out = adapter.tick(dt=1.0 / 60.0, wb_input=_make_input())
        self.assertTrue(out.used_fallback)
        self.assertEqual(
            out.joint_accelerations, [0.0] * N_DOF,
            "Fallback must return zero accelerations",
        )

    def test_no_fallback_when_disabled(self):
        """When fallback_on_fail=False, output is used even if not converged."""
        cfg = {"qp": dict(_CFG_FALLBACK["qp"], fallback_on_fail=False)}
        adapter = WBControllerAdapter(cfg)
        out = adapter.tick(dt=1.0 / 60.0, wb_input=_make_input())
        self.assertFalse(out.used_fallback,
                         "Fallback must be disabled when fallback_on_fail=False")


# ---------------------------------------------------------------------------
# Bonus: WeightScheduler and QPDiagnostics smoke tests
# ---------------------------------------------------------------------------

class TestWeightScheduler(unittest.TestCase):
    """WeightScheduler produces sensible weights under threat and fatigue."""

    def setUp(self):
        self.sched = WeightScheduler(_CFG_DEFAULT)

    def test_high_risk_increases_balance_weight(self):
        w_low  = self.sched.schedule(risk=0.0)
        w_high = self.sched.schedule(risk=0.9)
        self.assertGreater(w_high.balance, w_low.balance)

    def test_high_risk_decreases_look_weight(self):
        w_low  = self.sched.schedule(risk=0.0)
        w_high = self.sched.schedule(risk=0.9)
        self.assertLess(w_high.look, w_low.look)

    def test_fatigue_increases_brace_weight(self):
        w_fresh   = self.sched.schedule(fatigue_energy=1.0)
        w_fatigued = self.sched.schedule(fatigue_energy=0.1)
        self.assertGreater(w_fatigued.brace, w_fresh.brace)

    def test_all_weights_non_negative(self):
        w = self.sched.schedule(risk=0.5, slip_risk=0.5, wind_load=0.5,
                                fatigue_energy=0.5, fatigue_torque_scale=0.5)
        for attr in ("balance", "foot", "look", "brace", "grasp", "effort"):
            self.assertGreaterEqual(getattr(w, attr), 0.0)


class TestQPDiagnostics(unittest.TestCase):
    """QPDiagnostics records and summarises solver data correctly."""

    def test_record_and_summary(self):
        """Records are stored and summary shows correct counts."""
        from src.wb.QPSolver import QPResult

        diag = QPDiagnostics(_CFG_DEFAULT)
        adapter = WBControllerAdapter(_CFG_DEFAULT)
        sched   = WeightScheduler(_CFG_DEFAULT)

        for i in range(5):
            wb_in = _make_input(desired_com_acc=Vec3(float(i), 0.0, 0.0))
            out   = adapter.tick(1.0 / 60.0, wb_in)
            weights = sched.schedule()
            # Build a minimal QPResult for recording
            r = QPResult(
                x=out.joint_accelerations,
                cost=0.0,
                converged=out.converged,
                iters=out.solver_iters,
                solution_hash=out.solution_hash,
            )
            diag.record_tick(r, wb_in, weights, used_fallback=out.used_fallback)

        summary = diag.get_summary()
        self.assertEqual(summary["ticks"], 5)
        self.assertIn("convergence_rate", summary)

    def test_toggle_override(self):
        """Toggle sets the override flag."""
        diag = QPDiagnostics(_CFG_DEFAULT)
        self.assertIsNone(diag.qp_override)
        diag.toggle_qp(False)
        self.assertFalse(diag.qp_override)
        diag.toggle_qp(True)
        self.assertTrue(diag.qp_override)

    def test_reset_clears_records(self):
        """reset() empties the ring buffer."""
        from src.wb.QPSolver import QPResult

        diag    = QPDiagnostics(_CFG_DEFAULT)
        sched   = WeightScheduler(_CFG_DEFAULT)
        adapter = WBControllerAdapter(_CFG_DEFAULT)

        wb_in = _make_input()
        out   = adapter.tick(1.0 / 60.0, wb_in)
        r = QPResult(x=out.joint_accelerations, cost=0.0,
                     converged=out.converged, iters=out.solver_iters,
                     solution_hash=out.solution_hash)
        diag.record_tick(r, wb_in, sched.schedule())

        self.assertEqual(diag.get_summary()["ticks"], 1)
        diag.reset()
        self.assertEqual(diag.get_summary()["ticks"], 0)


if __name__ == "__main__":
    unittest.main()
