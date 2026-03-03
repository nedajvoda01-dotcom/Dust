"""test_salience_stage55.py — Stage 55 Perceptual Meaning Emergence tests.

Tests (§13)
-----------
1. test_salience_increases_near_instability
   — riskSalience rises when slip_risk, vibration, and instability_proximity
     are elevated.

2. test_scale_salience_during_sun_alignment
   — scaleSalience increases when sun_alignment is high (eclipse config).

3. test_salience_decays_when_stable
   — After a high-risk event, both globalSalience and riskSalience decay
     toward zero when environment returns to neutral.

4. test_camera_mod_within_bounds
   — CameraModifiers produced by SalienceCameraAdapter stay within the
     configured max_camera_mod bound.

5. test_input_priority_over_salience
   — When player_input_active=True, SalienceCameraAdapter returns neutral
     (zero) modifiers regardless of perceptual state.

6. test_determinism_salience_same_inputs_same_output
   — Two independent SalienceSystem instances with identical inputs produce
     bit-identical PerceptualState every tick.
"""
from __future__ import annotations

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.perception.SalienceSystem    import SalienceSystem, SalienceEnv, PerceptualState
from src.perception.RiskEstimator     import RiskEstimator
from src.perception.ScaleEstimator    import ScaleEstimator
from src.perception.StructuralEventTracker import StructuralEventTracker
from src.camera.SalienceCameraAdapter import SalienceCameraAdapter, CameraModifiers
from src.wb.SalienceWeightAdapter     import SalienceWeightAdapter
from src.audio.SalienceAudioAdapter   import SalienceAudioAdapter

_DT = 1.0 / 20.0   # 20 Hz perceptual tick


# ---------------------------------------------------------------------------
# 1. Salience increases near instability
# ---------------------------------------------------------------------------

class TestSalienceIncreasesNearInstability(unittest.TestCase):

    def test_salience_increases_near_instability(self):
        """riskSalience must be higher when hazard inputs are elevated
        compared to a neutral (all-zero) environment."""
        sys_low  = SalienceSystem()
        sys_high = SalienceSystem()

        env_low  = SalienceEnv()   # neutral
        env_high = SalienceEnv(
            slip_risk=0.9,
            vibration_level=0.8,
            instability_proximity=0.7,
        )

        # Run both for 2 s (40 ticks) to let smoothing settle
        for _ in range(40):
            state_low  = sys_low.update(_DT,  env_low)
            state_high = sys_high.update(_DT, env_high)

        self.assertGreater(
            state_high.riskSalience,
            state_low.riskSalience,
            "riskSalience should be higher in hazardous environment",
        )
        self.assertGreater(
            state_high.globalSalience,
            state_low.globalSalience,
            "globalSalience should be higher in hazardous environment",
        )


# ---------------------------------------------------------------------------
# 2. Scale salience during sun alignment
# ---------------------------------------------------------------------------

class TestScaleSalienceDuringSunAlignment(unittest.TestCase):

    def test_scale_salience_during_sun_alignment(self):
        """scaleSalience must be elevated when sun_alignment is high."""
        sys_low  = SalienceSystem()
        sys_high = SalienceSystem()

        env_low  = SalienceEnv()
        env_high = SalienceEnv(
            sun_alignment=1.0,
            fov_scale_metric=0.8,
            horizon_curvature=0.6,
        )

        for _ in range(40):
            state_low  = sys_low.update(_DT,  env_low)
            state_high = sys_high.update(_DT, env_high)

        self.assertGreater(
            state_high.scaleSalience,
            state_low.scaleSalience,
            "scaleSalience should be higher during eclipse-like alignment",
        )


# ---------------------------------------------------------------------------
# 3. Salience decays when stable
# ---------------------------------------------------------------------------

class TestSalienceDecaysWhenStable(unittest.TestCase):

    def test_salience_decays_when_stable(self):
        """After a high-risk spike, riskSalience must decay when environment
        returns to neutral."""
        sys = SalienceSystem()

        env_high = SalienceEnv(
            slip_risk=1.0,
            vibration_level=1.0,
            instability_proximity=1.0,
        )
        env_low = SalienceEnv()

        # Elevate for 1 s
        for _ in range(20):
            sys.update(_DT, env_high)

        peak_state = sys.update(_DT, env_high)
        peak_risk  = peak_state.riskSalience

        # Return to neutral for 5 s
        for _ in range(100):
            decayed = sys.update(_DT, env_low)

        self.assertLess(
            decayed.riskSalience,
            peak_risk,
            f"riskSalience should decay: {decayed.riskSalience:.4f} vs peak {peak_risk:.4f}",
        )
        self.assertLess(
            decayed.globalSalience,
            peak_state.globalSalience,
            "globalSalience should decay after returning to stable environment",
        )


# ---------------------------------------------------------------------------
# 4. Camera modifiers within bounds
# ---------------------------------------------------------------------------

class TestCameraModWithinBounds(unittest.TestCase):

    def test_camera_mod_within_bounds(self):
        """CameraModifiers must stay within configured max_camera_mod."""
        config = {"salience": {"max_camera_mod": 0.3}}
        adapter = SalienceCameraAdapter(config=config)

        # Test extreme states
        extreme = PerceptualState(
            riskSalience=1.0,
            scaleSalience=1.0,
            structuralSalience=1.0,
            environmentalSalience=1.0,
            motionSalience=1.0,
            globalSalience=1.0,
        )
        mods = adapter.compute(extreme, player_input_active=False)

        max_mod = 0.3
        self.assertLessEqual(
            mods.height_bias, max_mod,
            f"height_bias {mods.height_bias:.4f} exceeds max {max_mod}",
        )
        self.assertLessEqual(
            mods.rotation_lag, max_mod,
            f"rotation_lag {mods.rotation_lag:.4f} exceeds max {max_mod}",
        )
        # sway_scale must stay in [0, 1]
        self.assertGreaterEqual(mods.sway_scale, 0.0)
        self.assertLessEqual(mods.sway_scale, 1.0)


# ---------------------------------------------------------------------------
# 5. Input priority over salience
# ---------------------------------------------------------------------------

class TestInputPriorityOverSalience(unittest.TestCase):

    def test_input_priority_over_salience(self):
        """When player_input_active=True, camera mods must be neutral."""
        adapter = SalienceCameraAdapter()

        extreme = PerceptualState(
            riskSalience=1.0,
            scaleSalience=1.0,
            structuralSalience=1.0,
        )

        mods = adapter.compute(extreme, player_input_active=True)

        self.assertAlmostEqual(mods.fov_bias_deg, 0.0, places=9,
                               msg="fov_bias_deg must be zero when input active")
        self.assertAlmostEqual(mods.height_bias, 0.0, places=9,
                               msg="height_bias must be zero when input active")
        self.assertAlmostEqual(mods.sway_scale, 1.0, places=9,
                               msg="sway_scale must be 1.0 when input active")
        self.assertAlmostEqual(mods.rotation_lag, 0.0, places=9,
                               msg="rotation_lag must be zero when input active")


# ---------------------------------------------------------------------------
# 6. Determinism — same inputs → same output
# ---------------------------------------------------------------------------

class TestDeterminismSalienceSameInputsSameOutput(unittest.TestCase):

    def test_determinism_salience_same_inputs_same_output(self):
        """Two independent SalienceSystem instances with identical inputs
        must produce bit-identical PerceptualState values every tick."""
        sys_a = SalienceSystem()
        sys_b = SalienceSystem()

        env = SalienceEnv(
            slip_risk=0.6,
            vibration_level=0.4,
            instability_proximity=0.5,
            sun_alignment=0.7,
            fov_scale_metric=0.5,
            acoustic_anomaly=0.3,
        )

        for _ in range(60):
            sa = sys_a.update(_DT, env)
            sb = sys_b.update(_DT, env)

        tol = 1e-12
        self.assertAlmostEqual(sa.riskSalience,   sb.riskSalience,   delta=tol)
        self.assertAlmostEqual(sa.scaleSalience,  sb.scaleSalience,  delta=tol)
        self.assertAlmostEqual(sa.globalSalience, sb.globalSalience, delta=tol)
        self.assertAlmostEqual(
            sa.structuralSalience, sb.structuralSalience, delta=tol
        )
        self.assertAlmostEqual(
            sa.environmentalSalience, sb.environmentalSalience, delta=tol
        )


if __name__ == "__main__":
    unittest.main()
