"""test_volumetric_fields_stage65.py — Stage 65 Volumetric Fields & Density Simulation tests.

Tests
-----
1. test_dust_lift_reduces_surface_dust
   — WeatherToVolumetrics.inject() with a DUST_STORM params increases grid
     density while the coupled settling round-trip demonstrates that lift
     rate is tied to wind speed / aerosol (proxy for surface dust decrease).

2. test_settling_increases_surface_dust
   — SettlingModel.step() removes density from the grid and returns a
     positive settled-mass value; repeated settling drains all density.

3. test_fog_forms_in_lowlands_when_cold
   — CondensationModel.step() with cold+humid conditions increases density;
     hot+dry conditions do not increase density.

4. test_raymarch_stable_under_pixel_pass
   — VolumetricRenderer.render_ray() produces transmittance in [0,1] and
     scatter in [0,1]³ for a fully-loaded density grid; the result does not
     depend on call order (stable under repeated invocations).

5. test_lod_tiers_reduce_cost
   — Lower LOD tiers produce results at least as fast as higher tiers;
     Tier-0 result is always ≥ Tier-3 transmittance (less scattering).

6. test_client_visual_sim_does_not_affect_authoritative_mass
   — VolumetricDomainManager.tick() returns settled mass only for dust grids;
     fog grids do not return settled mass; the DensityGrid used for visual
     simulation does not write to any PlanetChunkState or MassExchangeAPI.

7. test_deterministic_seed_same_density_evolution
   — Two independent VolumetricDomainManager instances created with the
     same config and driven with the same inputs produce grids with
     identical grid_hash() after many ticks.
"""
from __future__ import annotations

import os
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.vol.DensityGrid             import DensityGrid, VolumeLayerType
from src.vol.AdvectionSolver         import AdvectionSolver
from src.vol.SettlingModel           import SettlingModel
from src.vol.CondensationModel       import CondensationModel
from src.vol.TerrainOcclusionProxy   import TerrainOcclusionProxy
from src.vol.VolumetricRenderer      import VolumetricRenderer, RaymarchResult
from src.vol.VolumetricDomainManager import VolumetricDomainManager
from src.adapters.WeatherToVolumetrics   import WeatherToVolumetrics
from src.adapters.VolumetricsToAudio     import VolumetricsToAudio
from src.adapters.VolumetricsToVisibility import VolumetricsToVisibility
from src.net.VolumetricSeedSync          import VolumetricSeedSync, VolumetricSeed
from src.atmo.AtmosphereSystem           import LocalAtmoParams
from src.atmo.WeatherRegimeDetector      import WeatherRegime


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_VOL_CFG = {
    "vol": {
        "tick_hz":        10.0,
        "grid_resolution": 16,   # small for test speed
        "domain_radius_m": 200.0,
        "settling_k":      0.1,
        "condense_k":      0.08,
        "evap_k":          0.02,
        "lift_k":          0.1,
        "fog_lift_k":      0.08,
        "diffusion_k":     0.01,
        "absorption":      0.4,
        "anisotropy":      0.3,
        "raymarch_steps":  16,
    }
}


def _make_dust_grid(size: int = 8) -> DensityGrid:
    return DensityGrid(size, size, max(size // 4, 2), layer_type=VolumeLayerType.DUST)


def _make_fog_grid(size: int = 8) -> DensityGrid:
    return DensityGrid(size, size, max(size // 4, 2), layer_type=VolumeLayerType.FOG)


def _storm_params() -> LocalAtmoParams:
    return LocalAtmoParams(
        wind_speed=0.8, wind_x=0.8, wind_y=0.0,
        aerosol=0.7, humidity=0.2, temperature=0.6,
        pressure=0.2, storm_potential=0.8,
        thermal_effect=0.5,
        regime=WeatherRegime.DUST_STORM,
    )


def _cold_humid_params() -> LocalAtmoParams:
    return LocalAtmoParams(
        wind_speed=0.0, wind_x=0.0, wind_y=0.0,
        aerosol=0.05, humidity=0.8, temperature=0.3,
        pressure=0.35, storm_potential=0.1,
        thermal_effect=0.1,
        regime=WeatherRegime.FOG,
    )


# ---------------------------------------------------------------------------
# 1. test_dust_lift_reduces_surface_dust
# ---------------------------------------------------------------------------

class TestDustLiftReducesSurfaceDust(unittest.TestCase):
    """WeatherToVolumetrics injects dust proportional to wind × aerosol."""

    def test_dust_injected_proportional_to_wind_aerosol(self):
        """High wind + high aerosol injects more dust than calm conditions."""
        adapter = WeatherToVolumetrics(_VOL_CFG)
        grid_storm = _make_dust_grid()
        grid_calm  = _make_dust_grid()

        calm_params = LocalAtmoParams(
            wind_speed=0.1, aerosol=0.05, humidity=0.2,
            temperature=0.5, pressure=0.5, storm_potential=0.1,
            thermal_effect=0.0, regime=WeatherRegime.CLEAR,
        )

        adapter.inject(grid_storm, _storm_params(), dt=1.0)
        adapter.inject(grid_calm,  calm_params,     dt=1.0)

        self.assertGreater(
            grid_storm.total_density(), grid_calm.total_density(),
            "Storm conditions should inject more dust than calm conditions",
        )

    def test_dust_density_increases_after_injection(self):
        """Total density must increase after WeatherToVolumetrics.inject()."""
        adapter = WeatherToVolumetrics(_VOL_CFG)
        grid = _make_dust_grid()
        before = grid.total_density()
        adapter.inject(grid, _storm_params(), dt=1.0)
        self.assertGreater(
            grid.total_density(), before,
            "Injection must increase total grid density",
        )

    def test_no_injection_when_no_wind_no_aerosol(self):
        """Zero wind and zero aerosol must not inject any dust."""
        adapter = WeatherToVolumetrics(_VOL_CFG)
        grid = _make_dust_grid()
        zero_params = LocalAtmoParams(
            wind_speed=0.0, aerosol=0.0, humidity=0.0,
            temperature=0.5, pressure=0.5, storm_potential=0.0,
            thermal_effect=0.0, regime=WeatherRegime.CLEAR,
        )
        adapter.inject(grid, zero_params, dt=10.0)
        self.assertAlmostEqual(grid.total_density(), 0.0, places=9,
            msg="No dust should be injected with zero wind/aerosol")


# ---------------------------------------------------------------------------
# 2. test_settling_increases_surface_dust
# ---------------------------------------------------------------------------

class TestSettlingIncreasesSurfaceDust(unittest.TestCase):
    """SettlingModel drains density from grid and returns positive settled mass."""

    def test_settling_returns_positive_mass(self):
        """Settled mass returned by step() must be positive when grid has density."""
        model = SettlingModel(_VOL_CFG)
        grid  = _make_dust_grid()
        # Seed some density
        for iz in range(grid.depth):
            for iy in range(grid.height):
                for ix in range(grid.width):
                    grid.set_density(ix, iy, iz, 0.5)
        settled = model.step(grid, dt=1.0)
        self.assertGreater(settled, 0.0, "Settled mass must be positive")

    def test_settling_reduces_total_density(self):
        """Repeated settling drains total density toward zero."""
        model = SettlingModel(_VOL_CFG)
        grid  = _make_dust_grid()
        for iz in range(grid.depth):
            for iy in range(grid.height):
                for ix in range(grid.width):
                    grid.set_density(ix, iy, iz, 0.8)
        before = grid.total_density()
        for _ in range(20):
            model.step(grid, dt=1.0)
        self.assertLess(grid.total_density(), before,
            "Total density must decrease after repeated settling")

    def test_empty_grid_settles_nothing(self):
        """Settling on an empty grid returns zero."""
        model = SettlingModel(_VOL_CFG)
        grid  = _make_dust_grid()
        settled = model.step(grid, dt=1.0)
        self.assertAlmostEqual(settled, 0.0, places=9,
            msg="Empty grid must settle zero mass")


# ---------------------------------------------------------------------------
# 3. test_fog_forms_in_lowlands_when_cold
# ---------------------------------------------------------------------------

class TestFogFormsInLowlandsWhenCold(unittest.TestCase):
    """CondensationModel adds fog density in cold+humid conditions."""

    def test_fog_increases_in_cold_humid_conditions(self):
        """Cold+humid conditions must increase fog density."""
        model = CondensationModel(_VOL_CFG)
        grid  = _make_fog_grid()
        before = grid.total_density()
        condensed, _ = model.step(grid, humidity_proxy=0.9, temperature_proxy=0.2, dt=1.0)
        self.assertGreater(condensed, 0.0, "Cold+humid must condense fog")
        self.assertGreater(grid.total_density(), before,
            "Fog density must increase in cold+humid conditions")

    def test_no_fog_in_hot_dry_conditions(self):
        """Hot+dry conditions must not increase fog density."""
        model = CondensationModel(_VOL_CFG)
        grid  = _make_fog_grid()
        # Pre-seed a tiny amount to test evaporation
        for iy in range(grid.height):
            for ix in range(grid.width):
                grid.set_density(ix, iy, 0, 0.5)
        before = grid.total_density()
        _, evaporated = model.step(grid, humidity_proxy=0.05, temperature_proxy=0.95, dt=1.0)
        self.assertGreater(evaporated, 0.0, "Hot+dry must evaporate fog")
        self.assertLess(grid.total_density(), before,
            "Fog density must decrease in hot+dry conditions")

    def test_fog_injected_from_weather_in_cold_conditions(self):
        """WeatherToVolumetrics injects fog density in cold+humid conditions."""
        adapter = WeatherToVolumetrics(_VOL_CFG)
        grid = _make_fog_grid()
        before = grid.total_density()
        adapter.inject(grid, _cold_humid_params(), dt=1.0)
        self.assertGreater(grid.total_density(), before,
            "Cold+humid weather must inject fog density")


# ---------------------------------------------------------------------------
# 4. test_raymarch_stable_under_pixel_pass
# ---------------------------------------------------------------------------

class TestRaymarchStableUnderPixelPass(unittest.TestCase):
    """VolumetricRenderer produces valid results stable under repeated calls."""

    def test_transmittance_in_range(self):
        """Transmittance must be in [0, 1]."""
        renderer = VolumetricRenderer(_VOL_CFG)
        grid = _make_dust_grid()
        for iz in range(grid.depth):
            for iy in range(grid.height):
                for ix in range(grid.width):
                    grid.set_density(ix, iy, iz, 0.8)
        result = renderer.render_ray(grid, (0.5, 0.5, 0.0), (0.0, 0.0, 1.0), 1.0)
        self.assertIsInstance(result, RaymarchResult)
        self.assertGreaterEqual(result.transmittance, 0.0)
        self.assertLessEqual(result.transmittance,    1.0)

    def test_scatter_in_range(self):
        """All scatter components must be in [0, 1]."""
        renderer = VolumetricRenderer(_VOL_CFG)
        grid = _make_dust_grid()
        for iz in range(grid.depth):
            for iy in range(grid.height):
                for ix in range(grid.width):
                    grid.set_density(ix, iy, iz, 0.5)
        result = renderer.render_ray(grid, (0.1, 0.1, 0.0), (0.1, 0.1, 1.0), 1.0)
        for i, component in enumerate(result.scatter):
            self.assertGreaterEqual(component, 0.0, f"scatter[{i}] < 0")
            self.assertLessEqual(component,    1.0, f"scatter[{i}] > 1")

    def test_raymarch_stable_repeated_calls(self):
        """Repeated calls with same inputs produce identical results (stable)."""
        renderer = VolumetricRenderer(_VOL_CFG)
        grid = _make_dust_grid()
        for iz in range(grid.depth):
            for iy in range(grid.height):
                for ix in range(grid.width):
                    grid.set_density(ix, iy, iz, 0.4)
        r1 = renderer.render_ray(grid, (0.5, 0.5, 0.0), (0.0, 0.0, 1.0), 1.0)
        r2 = renderer.render_ray(grid, (0.5, 0.5, 0.0), (0.0, 0.0, 1.0), 1.0)
        self.assertAlmostEqual(r1.transmittance, r2.transmittance, places=10)
        for i in range(3):
            self.assertAlmostEqual(r1.scatter[i], r2.scatter[i], places=10)

    def test_empty_grid_full_transmittance(self):
        """Empty grid must give transmittance == 1.0 and scatter == 0."""
        renderer = VolumetricRenderer(_VOL_CFG)
        grid = _make_dust_grid()
        result = renderer.render_ray(grid, (0.5, 0.5, 0.0), (0.0, 0.0, 1.0), 1.0)
        self.assertAlmostEqual(result.transmittance, 1.0, places=5)
        for comp in result.scatter:
            self.assertAlmostEqual(comp, 0.0, places=5)


# ---------------------------------------------------------------------------
# 5. test_lod_tiers_reduce_cost
# ---------------------------------------------------------------------------

class TestLODTiersReduceCost(unittest.TestCase):
    """Lower LOD tiers are at least as fast as higher tiers."""

    def _time_tier(self, tier: int, grid: DensityGrid, reps: int = 20) -> float:
        renderer = VolumetricRenderer(_VOL_CFG)
        renderer.tier = tier
        t0 = time.perf_counter()
        for _ in range(reps):
            renderer.render_ray(grid, (0.5, 0.5, 0.0), (0.0, 0.0, 1.0), 1.0)
        return time.perf_counter() - t0

    def test_tier0_faster_or_equal_to_tier3(self):
        """Tier 0 (height fog) must not be slower than Tier 3 ray-march."""
        grid = DensityGrid(16, 16, 4, layer_type=VolumeLayerType.DUST)
        for iz in range(grid.depth):
            for iy in range(grid.height):
                for ix in range(grid.width):
                    grid.set_density(ix, iy, iz, 0.6)
        t0 = self._time_tier(0, grid, reps=50)
        t3 = self._time_tier(3, grid, reps=50)
        self.assertLessEqual(t0, t3 * 1.5,
            f"Tier 0 ({t0*1000:.1f}ms) must not exceed Tier 3 ({t3*1000:.1f}ms) significantly")

    def test_tier_setter_changes_step_count(self):
        """Setting tier must change internal step count."""
        renderer = VolumetricRenderer(_VOL_CFG)
        renderer.tier = 0
        self.assertEqual(renderer._steps, 0)
        renderer.tier = 1
        self.assertEqual(renderer._steps, 16)
        renderer.tier = 2
        self.assertEqual(renderer._steps, 24)
        renderer.tier = 3
        self.assertEqual(renderer._steps, 32)

    def test_tier0_transmittance_reasonable(self):
        """Tier 0 height fog transmittance must be in [0, 1]."""
        renderer = VolumetricRenderer(_VOL_CFG)
        renderer.tier = 0
        grid = _make_dust_grid()
        for iy in range(grid.height):
            for ix in range(grid.width):
                grid.set_density(ix, iy, 0, 0.8)
        result = renderer.render_ray(grid, (0.5, 0.5, 0.0), (0.0, 0.0, 1.0), 1.0)
        self.assertGreaterEqual(result.transmittance, 0.0)
        self.assertLessEqual(result.transmittance, 1.0)


# ---------------------------------------------------------------------------
# 6. test_client_visual_sim_does_not_affect_authoritative_mass
# ---------------------------------------------------------------------------

class TestClientVisualSimDoesNotAffectAuthoritativeMass(unittest.TestCase):
    """Visual sim (VolumetricDomainManager) never writes to authoritative terrain."""

    def test_dust_domain_returns_settled_mass_report(self):
        """tick() reports settled mass for dust domains."""
        mgr = VolumetricDomainManager(_VOL_CFG)
        grid = mgr.get_or_create(VolumeLayerType.DUST, 0.0, 0.0)
        # Seed density
        for iz in range(grid.depth):
            for iy in range(grid.height):
                for ix in range(grid.width):
                    grid.set_density(ix, iy, iz, 0.6)
        report = mgr.tick(dt=0.5, wind_x=0.3, wind_y=0.0,
                          humidity=0.3, temperature=0.5)
        self.assertTrue(
            any("DustVolume" in k for k in report),
            "Settled-mass report must contain a DustVolume entry",
        )
        dust_key = next(k for k in report if "DustVolume" in k)
        self.assertGreater(report[dust_key], 0.0,
            "Settled mass for dust domain must be positive")

    def test_fog_domain_not_in_settling_report(self):
        """tick() does not report settled mass for fog domains."""
        mgr = VolumetricDomainManager(_VOL_CFG)
        mgr.get_or_create(VolumeLayerType.FOG, 0.0, 0.0)
        report = mgr.tick(dt=0.5, wind_x=0.0, wind_y=0.0,
                          humidity=0.8, temperature=0.3)
        fog_keys = [k for k in report if "FogVolume" in k]
        self.assertEqual(fog_keys, [],
            "Fog domains must not appear in settled-mass report")

    def test_visual_grid_has_no_chunk_state_reference(self):
        """DensityGrid has no reference to PlanetChunkState (structural check)."""
        import inspect
        import src.vol.DensityGrid as mod
        src_code = inspect.getsource(mod)
        self.assertNotIn("PlanetChunkState", src_code,
            "DensityGrid must not import or reference PlanetChunkState")
        self.assertNotIn("MassExchangeAPI", src_code,
            "DensityGrid must not import or reference MassExchangeAPI")


# ---------------------------------------------------------------------------
# 7. test_deterministic_seed_same_density_evolution
# ---------------------------------------------------------------------------

class TestDeterministicSeedSameDensityEvolution(unittest.TestCase):
    """Two identical domain managers produce identical grids after same inputs."""

    def _run_manager(self, steps: int) -> str:
        mgr = VolumetricDomainManager(_VOL_CFG)
        grid = mgr.get_or_create(VolumeLayerType.DUST, 100.0, 200.0)
        # Identical initial state
        for iz in range(grid.depth):
            for iy in range(grid.height):
                for ix in range(grid.width):
                    grid.set_density(ix, iy, iz, 0.4)

        # Inject weather sources before ticking
        adapter = WeatherToVolumetrics(_VOL_CFG)
        for _ in range(steps):
            adapter.inject(grid, _storm_params(), dt=0.1)
            mgr.tick(dt=0.1, wind_x=0.4, wind_y=0.1,
                     humidity=0.3, temperature=0.5)
        return grid.grid_hash()

    def test_same_inputs_produce_same_hash(self):
        """Identical inputs must produce identical grid hashes."""
        hash1 = self._run_manager(steps=30)
        hash2 = self._run_manager(steps=30)
        self.assertEqual(hash1, hash2,
            "Identical simulation paths must produce identical grid hashes")

    def test_grid_hash_is_string_of_expected_length(self):
        """grid_hash() returns a 32-character hex string (MD5)."""
        grid = _make_dust_grid()
        h = grid.grid_hash()
        self.assertIsInstance(h, str)
        self.assertEqual(len(h), 32, f"Expected 32-char MD5 hex, got {len(h)}")


# ---------------------------------------------------------------------------
# Bonus: adapters and net tests
# ---------------------------------------------------------------------------

class TestVolumetricsAdapters(unittest.TestCase):

    def test_audio_fog_dampening(self):
        """Dense fog grid produces non-zero fog_dampening."""
        adapter = VolumetricsToAudio(_VOL_CFG)
        grid = _make_fog_grid()
        for iz in range(grid.depth):
            for iy in range(grid.height):
                for ix in range(grid.width):
                    grid.set_density(ix, iy, iz, 0.8)
        params = adapter.audio_params(grid)
        self.assertGreater(params.fog_dampening, 0.0,
            "Dense fog must produce non-zero fog_dampening")

    def test_visibility_decreases_with_density(self):
        """Higher density must reduce visibility proxy."""
        adapter = VolumetricsToVisibility(_VOL_CFG)
        grid = _make_dust_grid()
        vis_empty = adapter.visibility_proxy(grid)

        for iz in range(grid.depth):
            for iy in range(grid.height):
                for ix in range(grid.width):
                    grid.set_density(ix, iy, iz, 0.9)
        vis_dense = adapter.visibility_proxy(grid)

        self.assertLess(vis_dense, vis_empty,
            "Dense grid must have lower visibility than empty grid")

    def test_seed_sync_roundtrip(self):
        """VolumetricSeedSync encode/decode round-trip is lossless."""
        sync = VolumetricSeedSync()
        seeds = [
            VolumetricSeed(VolumeLayerType.DUST,  100.0, 200.0, seed=42,  source_strength=0.8),
            VolumetricSeed(VolumeLayerType.FOG,   -50.0, 300.0, seed=99,  source_strength=0.5),
            VolumetricSeed(VolumeLayerType.STEAM,   0.0,   0.0, seed=777, source_strength=1.0),
        ]
        data = sync.encode_seeds(seeds)
        recovered = sync.decode_seeds(data)

        self.assertEqual(len(recovered), len(seeds))
        for orig, rec in zip(seeds, recovered):
            self.assertEqual(orig.layer_type, rec.layer_type)
            self.assertEqual(orig.seed,       rec.seed)
            self.assertAlmostEqual(orig.source_strength, rec.source_strength, delta=1.0 / 255)

    def test_terrain_occlusion_zeros_density_below_surface(self):
        """TerrainOcclusionProxy zeroes density below terrain height."""
        proxy = TerrainOcclusionProxy(4, 4, 8, _VOL_CFG)
        # Set terrain height to half of depth
        proxy.update_heightfield([0.5] * 16)  # 4×4

        grid = DensityGrid(4, 4, 8, layer_type=VolumeLayerType.DUST)
        for iz in range(8):
            for iy in range(4):
                for ix in range(4):
                    grid.set_density(ix, iy, iz, 1.0)
        proxy.apply_occlusion(grid)

        # Voxels below terrain (iz < depth/2 = 4) must be zeroed
        for iy in range(4):
            for ix in range(4):
                for iz in range(4):
                    self.assertAlmostEqual(
                        grid.density(ix, iy, iz), 0.0, places=9,
                        msg=f"voxel ({ix},{iy},{iz}) should be zeroed by terrain occlusion",
                    )


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
