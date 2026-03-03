"""test_microclimate_stage49.py — Stage 49 Shelter & Microclimate smoke tests.

Tests
-----
1. test_shelter_reduces_wind_behind_wall
   — ShelterEstimator returns high shelter when there are tall upwind heights.

2. test_channeling_increases_wind_in_passage
   — ChannelEstimator returns high channel value in a narrow corridor.

3. test_cold_bias_in_shadow_increases_icefilm_rate
   — ColdBiasEstimator returns high cold_bias at zero insolation;
     MicroclimateToMaterials reduces temperature and insolation in the
     resulting ClimateSample.

4. test_dust_trap_increases_local_deposition
   — DustTrapEstimator returns elevated dustTrap in sheltered concave area;
     LocalClimateComposer raises local dust above macro level.

5. test_thermal_inertia_smooths_temp_changes
   — LocalClimateComposer with high thermalInertia smooths rapid
     macro temperature changes more than with low inertia.

6. test_budget_limits_microclimate_samples
   — MicroclimateSystem.tick() never processes more than max_chunks_per_tick
     chunks in a single tick.

7. test_determinism_same_geom_same_microstate
   — Two MicroclimateSystems with identical geometry and inputs produce
     identical MicroclimateState (bit-exact via pack/unpack).
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.math.Vec3 import Vec3
from src.microclimate.MicroclimateState        import MicroclimateState
from src.microclimate.ShelterEstimator         import ShelterEstimator
from src.microclimate.ChannelEstimator         import ChannelEstimator
from src.microclimate.DustTrapEstimator        import DustTrapEstimator
from src.microclimate.ColdBiasEstimator        import ColdBiasEstimator
from src.microclimate.ThermalInertiaEstimator  import ThermalInertiaEstimator
from src.microclimate.EchoPotentialEstimator   import EchoPotentialEstimator
from src.microclimate.LocalClimateComposer     import LocalClimateComposer, MacroClimateSnapshot
from src.microclimate.MicroclimateSystem       import MicroclimateSystem
from src.adapters.MicroclimateToPerception     import MicroclimateToPerception
from src.adapters.MicroclimateToMaterials      import MicroclimateToMaterials
from src.adapters.MicroclimateToAudioWorld     import MicroclimateToAudioWorld
from src.material.PhaseChangeSystem            import ClimateSample


# ---------------------------------------------------------------------------
# Shared config
# ---------------------------------------------------------------------------

_CFG = {
    "micro": {
        "tick_hz": 10.0,           # fast tick for tests
        "winddir_buckets": 16,
        "max_chunks_per_tick": 4,
        "sky_radius": 30.0,
        "sky_num_samples": 8,
        "shelter": {
            "sample_distance": 50.0,
            "num_samples_near": 8,
        },
        "channel": {
            "sample_radius": 15.0,
            "threshold": 0.4,
        },
        "dusttrap": {
            "sample_radius": 20.0,
            "num_samples": 8,
            "deposition_boost": 0.4,
            "dispersion_boost": 0.3,
        },
        "coldbias": {
            "cold_delta": 0.3,
            "height_cold_k": 0.1,
        },
        "thermal_inertia_tau": 60.0,
        "echo": {"reverb_gain": 0.8},
        "lod": {"near_dist": 100.0, "far_sample_reduce": 0.25},
    }
}

_WIND_DIR = Vec3(1.0, 0.0, 0.0)   # wind blows in +X direction


# ---------------------------------------------------------------------------
# 1. test_shelter_reduces_wind_behind_wall
# ---------------------------------------------------------------------------

class TestShelterReducesWindBehindWall(unittest.TestCase):
    """ShelterEstimator must return higher shelter when a wall blocks upwind."""

    def _make_estimator(self, wall_height: float):
        # Wall is upwind (negative X) at various sample distances
        def height_fn(x: float, z: float) -> float:
            if x < -5.0:
                return wall_height
            return 0.0
        return ShelterEstimator(_CFG, height_fn)

    def test_behind_tall_wall_has_high_shelter(self):
        est = self._make_estimator(wall_height=20.0)
        pos = Vec3(0.0, 0.0, 0.0)   # player at ground level
        result = est.estimate(pos, _WIND_DIR)
        self.assertGreater(result.shelter, 0.5,
            f"Expected shelter > 0.5 behind tall wall, got {result.shelter:.3f}")

    def test_open_terrain_has_low_shelter(self):
        # Flat terrain → no shelter
        est = self._make_estimator(wall_height=0.0)
        pos = Vec3(0.0, 0.0, 0.0)
        result = est.estimate(pos, _WIND_DIR)
        self.assertLess(result.shelter, 0.1,
            f"Expected near-zero shelter on flat terrain, got {result.shelter:.3f}")

    def test_local_wind_lower_in_sheltered_spot(self):
        """LocalClimateComposer: local wind speed must be lower behind a wall."""
        # Sheltered state
        micro_sheltered = MicroclimateState(windShelter=0.8, windChannel=0.0)
        micro_exposed   = MicroclimateState(windShelter=0.0, windChannel=0.0)
        macro = MacroClimateSnapshot(wind_speed=0.6, wind_dir=_WIND_DIR)

        composer_s = LocalClimateComposer(_CFG)
        composer_e = LocalClimateComposer(_CFG)
        lc_s = composer_s.compose(macro, micro_sheltered, dt=1.0)
        lc_e = composer_e.compose(macro, micro_exposed,   dt=1.0)

        self.assertLess(lc_s.wind_speed, lc_e.wind_speed,
            "Sheltered local wind should be less than exposed wind")


# ---------------------------------------------------------------------------
# 2. test_channeling_increases_wind_in_passage
# ---------------------------------------------------------------------------

class TestChannelingIncreasesWindInPassage(unittest.TestCase):
    """ChannelEstimator must return high channel in a narrow N-S canyon."""

    def _canyon_height(self, wall_h: float = 30.0):
        """Tall walls on ±Z sides, open along ±X (wind direction)."""
        def height_fn(x: float, z: float) -> float:
            if abs(z) > 10.0:
                return wall_h
            return 0.0
        return height_fn

    def test_channel_high_in_narrow_passage(self):
        est = ChannelEstimator(_CFG, self._canyon_height())
        pos = Vec3(0.0, 0.0, 0.0)
        channel = est.estimate(pos, _WIND_DIR)
        self.assertGreater(channel, 0.3,
            f"Expected windChannel > 0.3 in canyon, got {channel:.3f}")

    def test_channel_zero_on_open_plain(self):
        est = ChannelEstimator(_CFG, lambda x, z: 0.0)
        channel = est.estimate(Vec3(0, 0, 0), _WIND_DIR)
        self.assertAlmostEqual(channel, 0.0, places=3)

    def test_local_wind_higher_in_channel(self):
        """LocalClimateComposer: channeling raises local wind above macro."""
        micro_channel = MicroclimateState(windShelter=0.0, windChannel=0.7)
        micro_flat    = MicroclimateState(windShelter=0.0, windChannel=0.0)
        macro = MacroClimateSnapshot(wind_speed=0.4, wind_dir=_WIND_DIR)

        lc_ch = LocalClimateComposer(_CFG).compose(macro, micro_channel, dt=1.0)
        lc_fl = LocalClimateComposer(_CFG).compose(macro, micro_flat,    dt=1.0)

        self.assertGreater(lc_ch.wind_speed, lc_fl.wind_speed,
            "Channeled wind should exceed plain-terrain wind")


# ---------------------------------------------------------------------------
# 3. test_cold_bias_in_shadow_increases_icefilm_rate
# ---------------------------------------------------------------------------

class TestColdBiasInShadowIncreasesIceFilmRate(unittest.TestCase):
    """High cold_bias (full shadow) should lower temperature in ClimateSample."""

    def test_full_shadow_gives_high_cold_bias(self):
        est = ColdBiasEstimator(_CFG)
        bias = est.estimate(insolation=0.0)
        self.assertGreater(bias, 0.6,
            f"Full shadow should give cold_bias > 0.6, got {bias:.3f}")

    def test_full_sun_gives_low_cold_bias(self):
        est = ColdBiasEstimator(_CFG)
        bias = est.estimate(insolation=1.0)
        self.assertLess(bias, 0.35,
            f"Full sun should give cold_bias < 0.35, got {bias:.3f}")

    def test_materials_sample_has_lower_temp_in_shadow(self):
        """MicroclimateToMaterials: shadowed location produces lower temperature."""
        macro = ClimateSample(
            wind_speed=0.3, dust_density=0.1, insolation=0.8, temperature=0.7,
        )
        micro_shadow = MicroclimateState(coldBias=0.9)
        micro_sun    = MicroclimateState(coldBias=0.0)
        macro_snap   = MacroClimateSnapshot(wind_speed=0.3, temp_proxy=0.7)

        adapter   = MicroclimateToMaterials()
        composer  = LocalClimateComposer(_CFG)
        composer2 = LocalClimateComposer(_CFG)

        lc_shadow = composer.compose(macro_snap, micro_shadow, dt=0.1)
        lc_sun    = composer2.compose(macro_snap, micro_sun,   dt=0.1)

        cs_shadow = adapter.to_climate_sample(lc_shadow, macro)
        cs_sun    = adapter.to_climate_sample(lc_sun,    macro)

        self.assertLess(cs_shadow.temperature, cs_sun.temperature,
            "Shadowed climate sample must have lower temperature")
        self.assertLess(cs_shadow.insolation, cs_sun.insolation,
            "Shadowed climate sample must have lower insolation")


# ---------------------------------------------------------------------------
# 4. test_dust_trap_increases_local_deposition
# ---------------------------------------------------------------------------

class TestDustTrapIncreasesLocalDeposition(unittest.TestCase):
    """In a concave sheltered basin, local dust should exceed macro dust."""

    def _basin_height(self):
        """Surrounding terrain higher than centre (bowl)."""
        def height_fn(x: float, z: float) -> float:
            r = (x ** 2 + z ** 2) ** 0.5
            return max(0.0, r - 5.0) * 0.5   # rises with distance from centre
        return height_fn

    def test_dust_trap_higher_in_basin_than_ridge(self):
        est_basin = DustTrapEstimator(_CFG, self._basin_height())
        est_flat  = DustTrapEstimator(_CFG, lambda x, z: 0.0)

        pos = Vec3(0.0, 0.0, 0.0)
        dt_basin = est_basin.estimate(pos, shelter=0.6, wind_channel=0.0)
        dt_flat  = est_flat.estimate(pos,  shelter=0.0, wind_channel=0.0)

        self.assertGreater(dt_basin, dt_flat,
            f"Basin dustTrap {dt_basin:.3f} should exceed flat {dt_flat:.3f}")

    def test_local_dust_higher_than_macro_in_trapped_area(self):
        """LocalClimateComposer: dustTrap raises local dust above macro."""
        macro = MacroClimateSnapshot(
            wind_speed=0.2, dust_density=0.2, wind_dir=_WIND_DIR,
        )
        micro_trapped = MicroclimateState(windShelter=0.5, dustTrap=0.8)
        micro_open    = MicroclimateState(windShelter=0.0, dustTrap=0.0)

        lc_trap = LocalClimateComposer(_CFG).compose(macro, micro_trapped, dt=1.0)
        lc_open = LocalClimateComposer(_CFG).compose(macro, micro_open,   dt=1.0)

        self.assertGreater(lc_trap.dust_density, lc_open.dust_density,
            "Trapped location should have more local dust")


# ---------------------------------------------------------------------------
# 5. test_thermal_inertia_smooths_temp_changes
# ---------------------------------------------------------------------------

class TestThermalInertiaSmoothsTempChanges(unittest.TestCase):
    """High thermalInertia must slow temperature changes more than low inertia."""

    def _run_temp_steps(self, thermal_inertia: float, n_steps: int = 5) -> list:
        """Apply a step from temp=0.2 to temp=0.8 and record the trajectory."""
        cfg_fast = dict(_CFG)
        composer = LocalClimateComposer(cfg_fast)
        composer.reset_temperature(0.2)

        macro = MacroClimateSnapshot(wind_speed=0.3, temp_proxy=0.8, wind_dir=_WIND_DIR)
        micro = MicroclimateState(thermalInertia=thermal_inertia)
        temps = []
        for _ in range(n_steps):
            lc = composer.compose(macro, micro, dt=2.0)
            temps.append(lc.temp_proxy)
        return temps

    def test_high_inertia_slower_than_low_inertia(self):
        temps_low  = self._run_temp_steps(thermal_inertia=0.0)
        temps_high = self._run_temp_steps(thermal_inertia=1.0)
        # After a few steps, high-inertia temp should be lower (slower to reach 0.8)
        self.assertLess(
            temps_high[-1], temps_low[-1],
            "High thermal inertia should slow temperature convergence"
        )

    def test_zero_inertia_converges_faster(self):
        temps = self._run_temp_steps(thermal_inertia=0.0, n_steps=10)
        # With zero inertia (tau = 6s, dt=2s → fast convergence)
        # temp should move appreciably toward 0.8
        self.assertGreater(temps[-1], 0.5,
            "Zero-inertia temp should converge significantly toward target")


# ---------------------------------------------------------------------------
# 6. test_budget_limits_microclimate_samples
# ---------------------------------------------------------------------------

class TestBudgetLimitsMicroclimateSamples(unittest.TestCase):
    """MicroclimateSystem must not process more than max_chunks_per_tick per tick."""

    def test_budget_respected(self):
        cfg = {
            "micro": {
                "tick_hz": 100.0,     # ensure tick fires
                "max_chunks_per_tick": 3,
                "winddir_buckets": 16,
                "sky_radius": 10.0,
                "sky_num_samples": 4,
                "shelter": {"sample_distance": 20.0, "num_samples_near": 4},
                "channel": {"sample_radius": 10.0, "threshold": 0.4},
                "dusttrap": {"sample_radius": 10.0, "num_samples": 4,
                             "deposition_boost": 0.4, "dispersion_boost": 0.3},
                "coldbias": {"cold_delta": 0.3, "height_cold_k": 0.1},
                "thermal_inertia_tau": 60.0,
                "echo": {"reverb_gain": 0.8},
                "lod": {"near_dist": 100.0, "far_sample_reduce": 0.25},
            }
        }
        system = MicroclimateSystem(cfg)
        # Provide 10 active chunks
        chunks = [((i, 0), Vec3(float(i) * 10, 0, 0)) for i in range(10)]
        system.tick(
            active_chunks=chunks,
            player_pos=Vec3(0, 0, 0),
            wind_dir_2d=_WIND_DIR,
            insolation=0.5,
            dt=0.1,
        )
        info = system.debug_info()
        self.assertLessEqual(
            info["samples_last_tick"], 3,
            f"Expected ≤3 samples per tick, got {info['samples_last_tick']}"
        )
        self.assertEqual(info["ticks_fired"], 1)

    def test_uncached_chunks_return_neutral_state(self):
        system = MicroclimateSystem(_CFG)
        # Never ticked → should return neutral default
        state = system.get_state((99, 99))
        self.assertAlmostEqual(state.windShelter,    0.0)
        self.assertAlmostEqual(state.windChannel,    0.0)
        self.assertAlmostEqual(state.dustTrap,       0.0)
        self.assertAlmostEqual(state.coldBias,       0.0)
        self.assertAlmostEqual(state.thermalInertia, 0.0)
        self.assertAlmostEqual(state.echoPotential,  0.0)


# ---------------------------------------------------------------------------
# 7. test_determinism_same_geom_same_microstate
# ---------------------------------------------------------------------------

class TestDeterminismSameGeomSameMicrostate(unittest.TestCase):
    """Two systems with identical geometry and inputs produce identical states."""

    def _make_system(self):
        def height_fn(x: float, z: float) -> float:
            # Simple deterministic terrain: concentric ridges
            r = (x ** 2 + z ** 2) ** 0.5
            return 5.0 * abs((r % 20.0) - 10.0) / 10.0
        return MicroclimateSystem(_CFG, height_fn)

    def test_packed_states_are_identical(self):
        s1 = self._make_system()
        s2 = self._make_system()

        chunks = [((0, 0), Vec3(0.0, 0.0, 0.0))]
        params = dict(
            active_chunks=chunks,
            player_pos=Vec3(0, 0, 0),
            wind_dir_2d=Vec3(1.0, 0.0, 0.0),
            insolation=0.5,
            dt=0.2,
        )

        # Ensure tick fires by advancing past tick interval
        for _ in range(10):
            s1.tick(**params)
            s2.tick(**params)

        st1 = s1.get_state((0, 0))
        st2 = s2.get_state((0, 0))

        self.assertEqual(
            st1.pack(), st2.pack(),
            f"States differ:\n  s1={st1}\n  s2={st2}",
        )

    def test_pack_unpack_roundtrip(self):
        original = MicroclimateState(
            windShelter=0.6, windChannel=0.3, dustTrap=0.4,
            coldBias=0.7, thermalInertia=0.5, echoPotential=0.2,
        )
        restored = MicroclimateState.unpack(original.pack())
        for field in ("windShelter", "windChannel", "dustTrap",
                      "coldBias", "thermalInertia", "echoPotential"):
            self.assertAlmostEqual(
                getattr(original, field), getattr(restored, field),
                delta=0.005,
                msg=f"Field {field} mismatch after pack/unpack",
            )


# ---------------------------------------------------------------------------
# Additional adapter tests
# ---------------------------------------------------------------------------

class TestAdapters(unittest.TestCase):
    """Smoke tests for the three microclimate adapters."""

    def test_perception_adapter_shelter_factor(self):
        """MicroclimateToPerception passes shelter_factor correctly."""
        from src.microclimate.LocalClimateComposer import LocalClimate
        lc = LocalClimate(
            wind_speed=0.4, wind_dir=Vec3(1, 0, 0),
            dust_density=0.3, temp_proxy=0.5,
            shelter=0.7, wind_channel=0.1,
            cold_bias=0.2, thermal_inertia=0.3, echo_potential=0.4,
        )
        adapter = MicroclimateToPerception()
        inputs = adapter.wind_inputs(lc, Vec3(10.0, 0.0, 0.0))
        self.assertAlmostEqual(inputs["shelter_factor"], 0.7, places=5)

    def test_audio_adapter_reverb_scales_with_echo(self):
        """MicroclimateToAudioWorld reverb_mix must scale with echo_potential."""
        from src.microclimate.LocalClimateComposer import LocalClimate
        lc_cave = LocalClimate(echo_potential=1.0, shelter=0.8)
        lc_open = LocalClimate(echo_potential=0.0, shelter=0.0)
        adapter = MicroclimateToAudioWorld()
        self.assertGreater(
            adapter.audio_modifiers(lc_cave).reverb_mix,
            adapter.audio_modifiers(lc_open).reverb_mix,
        )

    def test_audio_adapter_occlusion_from_shelter(self):
        """MicroclimateToAudioWorld occlusion_boost scales with shelter."""
        from src.microclimate.LocalClimateComposer import LocalClimate
        lc_sheltered = LocalClimate(shelter=1.0)
        lc_exposed   = LocalClimate(shelter=0.0)
        adapter = MicroclimateToAudioWorld()
        self.assertGreater(
            adapter.audio_modifiers(lc_sheltered).occlusion_boost,
            adapter.audio_modifiers(lc_exposed).occlusion_boost,
        )


if __name__ == "__main__":
    unittest.main()
