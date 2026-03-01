"""test_character_environment_integration — Stage 15 tests.

Tests
-----
1. test_modifiers_change_with_wind
   — windSpeed↑ → windDragScale↑, upperBodyLean↑, speedScale↓

2. test_whiteout_caution
   — visibility↓ → turnResponsiveness↓, strideLength↓, braceRate↑

3. test_ice_slip_increase
   — ice↑ → effectiveFrictionScale↓, slipRate↑

4. test_geo_prebrace
   — Geo PRE intensity↑ → stance becomes braced/crouched, speedScale↓

5. test_seeded_variation_stable
   — same timeBucket + position → identical micro-variations
"""
from __future__ import annotations

import math
import sys
import os
import unittest
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.math.Vec3 import Vec3
from src.systems.CharacterPhysicalController import (
    CharacterPhysicalController,
    EnvironmentSampler,
    GroundHit,
    IGroundSampler,
)
from src.systems.CharacterEnvironmentIntegration import (
    AnimParamFrame,
    CharacterEnvironmentIntegration,
    CharacterRngStream,
    EnvContext,
    LocomotionModifiers,
)


# ---------------------------------------------------------------------------
# Shared helpers / stubs
# ---------------------------------------------------------------------------

PLANET_R = 1000.0


def _on_surface(height: float = 0.5) -> Vec3:
    return Vec3(0.0, PLANET_R + height, 0.0)


class _FlatGround(IGroundSampler):
    """Perfect-sphere ground sampler — inherits default implementation from IGroundSampler."""
    pass


class _MockClimate:
    """Controllable climate stub."""

    def __init__(
        self,
        wind_speed:  float = 0.0,
        dust:        float = 0.0,
        temp:        float = 290.0,
        wetness:     float = 0.0,
    ) -> None:
        self._wind_speed = wind_speed
        self._dust       = dust
        self._temp       = temp
        self._wetness    = wetness

    def sample_wind(self, pos: Vec3) -> Vec3:
        return Vec3(self._wind_speed, 0.0, 0.0)

    def sample_dust(self, pos: Vec3) -> float:
        return self._dust

    def sample_temperature(self, pos: Vec3) -> float:
        return self._temp

    def get_wetness(self, pos: Vec3) -> float:
        return self._wetness

    def get_visibility(self, pos: Vec3) -> float:
        import math
        return math.exp(-self._dust * 5.0)


class _MockGeoEvents:
    """Controllable geo-event stub."""

    def __init__(self, pre_intensity: float = 0.0, impact_intensity: float = 0.0) -> None:
        self._pre    = pre_intensity
        self._impact = impact_intensity

    def query_signals_near(self, pos: Vec3, radius: float):
        from src.systems.GeoEventSystem import GeoEventPhase

        @dataclass
        class _Sig:
            phase: object
            intensity: float
            position: Vec3
            radius: float = 500.0
            time_to_impact: float = 0.0

        results = []
        if self._pre > 0.0:
            results.append(_Sig(GeoEventPhase.PRE, self._pre, Vec3.zero()))
        if self._impact > 0.0:
            results.append(_Sig(GeoEventPhase.IMPACT, self._impact, Vec3.zero()))
        return results


def _make_integration(climate=None, geo_events=None, **kwargs) -> CharacterEnvironmentIntegration:
    return CharacterEnvironmentIntegration(
        config=None,
        global_seed=42,
        character_id=0,
        climate=climate,
        geo_field_sampler=None,
        geo_event_system=geo_events,
        planet_radius=PLANET_R,
        **kwargs,
    )


def _make_ctrl(climate=None, geo_events=None) -> CharacterPhysicalController:
    pos = _on_surface()
    gs  = _FlatGround(PLANET_R)
    env = EnvironmentSampler(climate=climate, geo_events=geo_events)
    return CharacterPhysicalController(
        pos, planet_radius=PLANET_R,
        ground_sampler=gs, env_sampler=env,
    )


# ---------------------------------------------------------------------------
# 1. test_modifiers_change_with_wind
# ---------------------------------------------------------------------------

class TestModifiersChangeWithWind(unittest.TestCase):
    """Wind speed increases → drag rises, lean rises, speed scale drops."""

    def _run_with_wind(self, wind_speed: float) -> LocomotionModifiers:
        climate = _MockClimate(wind_speed=wind_speed)
        integ   = _make_integration(climate=climate)
        ctrl    = _make_ctrl(climate=climate)
        # settle one tick
        ctrl.update(0.1)
        integ.update(ctrl, reflex=None, dt=0.1, game_time=1.0)
        return integ.loco_modifiers

    def test_wind_drag_scale_increases(self):
        mods_low  = self._run_with_wind(0.0)
        mods_high = self._run_with_wind(30.0)
        self.assertGreater(
            mods_high.wind_drag_scale, mods_low.wind_drag_scale,
            "wind_drag_scale should increase with wind speed"
        )

    def test_upper_body_lean_increases(self):
        mods_low  = self._run_with_wind(0.0)
        mods_high = self._run_with_wind(30.0)
        self.assertGreater(
            mods_high.upper_body_lean, mods_low.upper_body_lean,
            "upper_body_lean should increase with wind speed"
        )

    def test_speed_scale_decreases_in_headwind(self):
        mods_calm = self._run_with_wind(0.0)
        mods_gale = self._run_with_wind(30.0)
        self.assertLess(
            mods_gale.speed_scale, mods_calm.speed_scale,
            "speed_scale should decrease with strong wind"
        )


# ---------------------------------------------------------------------------
# 2. test_whiteout_caution
# ---------------------------------------------------------------------------

class TestWhiteoutCaution(unittest.TestCase):
    """Low visibility → reduced turn responsiveness, shorter stride, more bracing."""

    def _run_with_dust(self, dust: float) -> tuple:
        climate = _MockClimate(dust=dust)
        integ   = _make_integration(climate=climate)
        ctrl    = _make_ctrl(climate=climate)
        ctrl.update(0.1)
        integ.update(ctrl, reflex=None, dt=0.1, game_time=1.0)
        return integ.loco_modifiers, integ.anim_frame

    def test_turn_responsiveness_drops_in_whiteout(self):
        mods_clear, _ = self._run_with_dust(0.0)
        mods_dust,  _ = self._run_with_dust(0.95)
        self.assertLess(
            mods_dust.turn_responsiveness, mods_clear.turn_responsiveness,
            "turn_responsiveness should drop in dust/whiteout"
        )

    def test_stride_length_drops_in_whiteout(self):
        _, frame_clear = self._run_with_dust(0.0)
        _, frame_dust  = self._run_with_dust(0.95)
        self.assertLess(
            frame_dust.stride_length, frame_clear.stride_length,
            "stride_length should decrease in whiteout"
        )

    def test_brace_rate_rises_in_whiteout(self):
        mods_clear, _ = self._run_with_dust(0.0)
        mods_dust,  _ = self._run_with_dust(0.95)
        self.assertGreater(
            mods_dust.brace_rate, mods_clear.brace_rate,
            "brace_rate should increase in whiteout"
        )


# ---------------------------------------------------------------------------
# 3. test_ice_slip_increase
# ---------------------------------------------------------------------------

class TestIceSlipIncrease(unittest.TestCase):
    """High ice (cold temp) → lower effective friction, higher slip rate."""

    def _run_with_temp(self, temp: float) -> LocomotionModifiers:
        climate = _MockClimate(temp=temp, dust=0.0)
        integ   = _make_integration(climate=climate)
        ctrl    = _make_ctrl(climate=climate)
        ctrl.update(0.1)
        integ.update(ctrl, reflex=None, dt=0.1, game_time=1.0)
        return integ.loco_modifiers

    def test_friction_scale_lower_when_icy(self):
        mods_warm = self._run_with_temp(290.0)   # no ice
        mods_ice  = self._run_with_temp(240.0)   # strong ice
        self.assertLess(
            mods_ice.effective_friction_scale,
            mods_warm.effective_friction_scale,
            "effective_friction_scale should decrease with ice"
        )

    def test_slip_rate_higher_when_icy(self):
        mods_warm = self._run_with_temp(290.0)
        mods_ice  = self._run_with_temp(240.0)
        self.assertGreater(
            mods_ice.slip_rate, mods_warm.slip_rate,
            "slip_rate should increase with ice"
        )


# ---------------------------------------------------------------------------
# 4. test_geo_prebrace
# ---------------------------------------------------------------------------

class TestGeoPrebrace(unittest.TestCase):
    """Geo PRE signal → stance becomes braced/crouched, speed_scale drops."""

    def _run_with_pre(self, pre_intensity: float) -> LocomotionModifiers:
        geo   = _MockGeoEvents(pre_intensity=pre_intensity)
        integ = _make_integration(geo_events=geo)
        ctrl  = _make_ctrl(geo_events=geo)
        ctrl.update(0.1)
        integ.update(ctrl, reflex=None, dt=0.1, game_time=1.0)
        return integ.loco_modifiers

    def test_stance_becomes_braced_on_pre(self):
        mods = self._run_with_pre(0.8)
        self.assertIn(
            mods.stance, ("braced", "crouched"),
            f"Stance should be braced/crouched on strong PRE signal, got '{mods.stance}'"
        )

    def test_speed_scale_drops_on_pre(self):
        mods_calm = self._run_with_pre(0.0)
        mods_pre  = self._run_with_pre(0.8)
        self.assertLess(
            mods_pre.speed_scale, mods_calm.speed_scale,
            "speed_scale should drop when PRE signal is strong"
        )

    def test_brace_rate_rises_on_pre(self):
        mods_calm = self._run_with_pre(0.0)
        mods_pre  = self._run_with_pre(0.8)
        self.assertGreater(
            mods_pre.brace_rate, mods_calm.brace_rate,
            "brace_rate should increase with PRE signal"
        )


# ---------------------------------------------------------------------------
# 5. test_seeded_variation_stable
# ---------------------------------------------------------------------------

class TestSeededVariationStable(unittest.TestCase):
    """Same timeBucket + position → identical micro-variation values."""

    def test_same_bucket_same_values(self):
        rng1 = CharacterRngStream(global_seed=42, character_id=0, variation_window_sec=2.0)
        rng2 = CharacterRngStream(global_seed=42, character_id=0, variation_window_sec=2.0)

        # Same time bucket (both at t=1.5, bucket = floor(1.5/2) = 0)
        rng1.update(game_time=1.5, lat_cell=5, lon_cell=10)
        rng2.update(game_time=1.7, lat_cell=5, lon_cell=10)

        for i in range(8):
            self.assertAlmostEqual(
                rng1.value(i), rng2.value(i), places=10,
                msg=f"RNG value[{i}] should be identical in the same bucket"
            )

    def test_different_bucket_different_values(self):
        rng = CharacterRngStream(global_seed=42, character_id=0, variation_window_sec=2.0)

        rng.update(game_time=0.5, lat_cell=5, lon_cell=10)
        vals_bucket0 = [rng.value(i) for i in range(8)]

        rng.update(game_time=2.5, lat_cell=5, lon_cell=10)
        vals_bucket1 = [rng.value(i) for i in range(8)]

        # At least some values should differ between buckets
        self.assertFalse(
            vals_bucket0 == vals_bucket1,
            "Different time buckets should produce different RNG values"
        )

    def test_same_seed_reproducible_across_instances(self):
        """Two independent instances with identical seeds reproduce the same path."""
        rng_a = CharacterRngStream(global_seed=99, character_id=3, variation_window_sec=1.0)
        rng_b = CharacterRngStream(global_seed=99, character_id=3, variation_window_sec=1.0)

        for t in [0.3, 1.3, 2.7]:
            rng_a.update(t, lat_cell=2, lon_cell=7)
            rng_b.update(t, lat_cell=2, lon_cell=7)
            for i in range(8):
                self.assertAlmostEqual(
                    rng_a.value(i), rng_b.value(i), places=10,
                    msg=f"t={t} index={i}: values must match"
                )

    def test_integration_anim_frame_stable_in_same_bucket(self):
        """AnimParamFrame produced in same time bucket should have equal stride/cadence."""
        climate = _MockClimate(wind_speed=5.0, dust=0.2)

        def _run(game_time: float) -> AnimParamFrame:
            integ = _make_integration(climate=climate)
            ctrl  = _make_ctrl(climate=climate)
            ctrl.update(0.1)
            integ.update(ctrl, reflex=None, dt=0.1, game_time=game_time)
            return integ.anim_frame

        frame_a = _run(0.3)
        frame_b = _run(1.8)   # still bucket 0 (window=2.0)

        self.assertAlmostEqual(
            frame_a.stride_length, frame_b.stride_length, places=6,
            msg="stride_length should be stable within the same RNG bucket"
        )
        self.assertAlmostEqual(
            frame_a.cadence, frame_b.cadence, places=6,
            msg="cadence should be stable within the same RNG bucket"
        )


if __name__ == "__main__":
    unittest.main()
