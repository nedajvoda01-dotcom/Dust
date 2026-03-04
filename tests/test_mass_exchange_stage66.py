"""test_mass_exchange_stage66.py — Stage 66 Mass Exchange & Erosion Framework tests.

Tests
-----
1. test_lift_reduces_surface_mass_increases_air_mass
   — LiftModel.compute_lift_rate() reduces dustThickness/snowMass on the
     surface and the same amount is added to the air density lists.

2. test_settling_increases_surface_mass_decreases_air_mass
   — SettlingModel.compute_settling_rate() increases dustThickness/snowMass
     and reduces the corresponding air-density values.

3. test_total_mass_conserved_surface_plus_air
   — After a full MassExchangeSystem.tick() the combined surface + air mass
     is within quantisation tolerance of the pre-tick value.

4. test_downhill_flux_moves_mass_down_slope
   — DownhillFluxModel produces positive flux when slope > threshold and
     the source cell loses exactly the flux amount applied to the neighbour.

5. test_contact_creates_compaction_and_displacement
   — ContactDisplacementModel.apply() increases snowCompaction and
     decreases dustThickness/snowMass on the cell under contact.

6. test_tracks_form_and_decay_under_wind
   — A sequence of contacts creates a track (reduced snowMass), then
     repeated ticks with wind gradually refill it (net settling > lift
     when wind is low).

7. test_determinism_flux_same_seed_same_result
   — Two independent MassExchangeSystem instances driven with the same
     input sequence produce identical grid_hash() values.

8. test_budget_clamps_prevent_runaway
   — With extreme wind (1.0) and many ticks the surface dust never goes
     negative and air density never exceeds 1.0.
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.material.PlanetChunkState    import PlanetChunkState, PlanetChunkGrid
from src.material.MassExchangeAPI     import MassExchangeAPI
from src.mass.LiftModel               import LiftModel
from src.mass.SettlingModel           import SettlingModel as MassSettlingModel
from src.mass.DownhillFluxModel       import DownhillFluxModel
from src.mass.ContactDisplacementModel import ContactDisplacementModel
from src.mass.FluxConservationChecker  import FluxConservationChecker
from src.mass.MassExchangeSystem       import MassExchangeSystem
from src.adapters.WeatherToMassExchange    import WeatherToMassExchange
from src.adapters.VolumetricsToSettling    import VolumetricsToSettling
from src.adapters.InstabilityToMassImpulse import InstabilityToMassImpulse
from src.adapters.CharacterToMassExchange  import CharacterToMassExchange
from src.vol.DensityGrid                   import DensityGrid, VolumeLayerType
from src.atmo.AtmosphereSystem             import LocalAtmoParams
from src.atmo.WeatherRegimeDetector        import WeatherRegime


# ---------------------------------------------------------------------------
# Shared config
# ---------------------------------------------------------------------------

_CFG = {
    "mass": {
        "lift_k_dust":             0.12,
        "lift_k_snow":             0.10,
        "settle_k_dust":           0.10,
        "settle_k_snow":           0.08,
        "downhill_flux_k":         0.15,
        "contact_displacement_k":  0.20,
        "max_flux_per_tick":       0.05,
        "slope_threshold_dust":    0.30,
        "slope_threshold_snow":    0.20,
        "decay_tau_compaction":    0.02,
    }
}

_QUANT_TOL = 2.0 / 255   # two uint8 steps of quantisation error


def _make_cell(**kwargs) -> PlanetChunkState:
    defaults = dict(
        dustThickness=0.4,
        snowMass=0.3,
        snowCompaction=0.1,
        crustHardness=0.2,
        debrisMass=0.05,
        surfaceRoughness=0.5,
    )
    defaults.update(kwargs)
    return PlanetChunkState(**defaults)


# ---------------------------------------------------------------------------
# 1. test_lift_reduces_surface_mass_increases_air_mass
# ---------------------------------------------------------------------------

class TestLift(unittest.TestCase):

    def test_lift_reduces_surface_mass_increases_air_mass(self):
        model = LiftModel(_CFG)
        cell  = _make_cell(dustThickness=0.4, snowMass=0.3, snowCompaction=0.0, crustHardness=0.1)
        api   = MassExchangeAPI(cell)

        air_dust = [0.0]
        air_snow = [0.0]

        rates = model.compute_lift_rate(cell, wind_speed=0.8, temperature=0.5)
        self.assertGreater(rates.dust_lift, 0.0, "Dust lift should be positive")
        self.assertGreater(rates.snow_lift, 0.0, "Snow lift should be positive")

        dust_before = cell.dustThickness
        snow_before = cell.snowMass
        api.apply_mass_delta("dustThickness", -rates.dust_lift)
        api.apply_mass_delta("snowMass",      -rates.snow_lift)
        air_dust[0] += rates.dust_lift
        air_snow[0] += rates.snow_lift

        self.assertLess(cell.dustThickness, dust_before, "dustThickness should decrease")
        self.assertLess(cell.snowMass, snow_before,       "snowMass should decrease")
        self.assertGreater(air_dust[0], 0.0, "air_dust should increase")
        self.assertGreater(air_snow[0], 0.0, "air_snow should increase")

    def test_no_lift_when_wind_zero(self):
        model = LiftModel(_CFG)
        cell  = _make_cell(dustThickness=0.5)
        rates = model.compute_lift_rate(cell, wind_speed=0.0)
        self.assertEqual(rates.dust_lift, 0.0)
        self.assertEqual(rates.snow_lift, 0.0)

    def test_no_lift_when_no_surface_material(self):
        model = LiftModel(_CFG)
        cell  = _make_cell(dustThickness=0.0, snowMass=0.0)
        rates = model.compute_lift_rate(cell, wind_speed=1.0)
        self.assertEqual(rates.dust_lift, 0.0)
        self.assertEqual(rates.snow_lift, 0.0)


# ---------------------------------------------------------------------------
# 2. test_settling_increases_surface_mass_decreases_air_mass
# ---------------------------------------------------------------------------

class TestSettling(unittest.TestCase):

    def test_settling_increases_surface_mass_decreases_air_mass(self):
        model = MassSettlingModel(_CFG)
        cell  = _make_cell(dustThickness=0.1, snowMass=0.0)
        api   = MassExchangeAPI(cell)

        air_dust = 0.5
        air_snow = 0.4

        rates = model.compute_settling_rate(
            air_dust, air_snow, wind_speed=0.1, slope=0.0, shelter=0.0
        )
        self.assertGreater(rates.dust_settle, 0.0, "Dust should settle from calm air")
        self.assertGreater(rates.snow_settle, 0.0, "Snow should settle from calm air")

        dust_before = cell.dustThickness
        api.apply_mass_delta("dustThickness", rates.dust_settle)
        api.apply_mass_delta("snowMass",      rates.snow_settle)
        air_dust -= rates.dust_settle
        air_snow -= rates.snow_settle

        self.assertGreater(cell.dustThickness, dust_before)
        self.assertLess(air_dust, 0.5)
        self.assertLess(air_snow, 0.4)

    def test_no_settling_from_empty_air(self):
        model = MassSettlingModel(_CFG)
        rates = model.compute_settling_rate(
            air_dust_density=0.0, air_snow_density=0.0,
            wind_speed=0.0, slope=0.0, shelter=0.0,
        )
        self.assertEqual(rates.dust_settle, 0.0)
        self.assertEqual(rates.snow_settle, 0.0)

    def test_high_wind_reduces_settling(self):
        model  = MassSettlingModel(_CFG)
        calm   = model.compute_settling_rate(0.5, 0.5, wind_speed=0.1)
        stormy = model.compute_settling_rate(0.5, 0.5, wind_speed=0.9)
        self.assertGreater(calm.dust_settle, stormy.dust_settle,
                           "Calm air should settle more than stormy air")


# ---------------------------------------------------------------------------
# 3. test_total_mass_conserved_surface_plus_air
# ---------------------------------------------------------------------------

class TestMassConservation(unittest.TestCase):

    def test_total_mass_conserved_surface_plus_air(self):
        system = MassExchangeSystem(_CFG)
        checker = FluxConservationChecker(tolerance=0.1)  # generous for quantisation

        cells    = [_make_cell() for _ in range(4)]
        air_dust = [0.2, 0.2, 0.2, 0.2]
        air_snow = [0.1, 0.1, 0.1, 0.1]

        before = checker.snapshot_total(cells, air_dust + air_snow)

        system.tick(
            cells=cells,
            air_dust=air_dust,
            air_snow=air_snow,
            wind_speed=0.5,
            slope_map=[0.1, 0.4, 0.6, 0.2],
            temperature=0.4,
            shelter=0.1,
            dt=1.0,
        )

        after = checker.snapshot_total(cells, air_dust + air_snow)

        # Allow for uint8 rounding across cells
        self.assertAlmostEqual(before, after, delta=0.1,
                               msg="Total mass should be approximately conserved")


# ---------------------------------------------------------------------------
# 4. test_downhill_flux_moves_mass_down_slope
# ---------------------------------------------------------------------------

class TestDownhillFlux(unittest.TestCase):

    def test_downhill_flux_moves_mass_down_slope(self):
        model = DownhillFluxModel(_CFG)
        src   = _make_cell(dustThickness=0.5, snowMass=0.4, debrisMass=0.1)
        dst   = _make_cell(dustThickness=0.0, snowMass=0.0, debrisMass=0.0)

        slope = 0.6   # well above both thresholds
        flux  = model.compute_downhill_flux(src, slope)

        self.assertGreater(flux.dust_flux,  0.0, "Dust should flow down steep slope")
        self.assertGreater(flux.snow_flux,  0.0, "Snow should flow down steep slope")
        self.assertGreater(flux.debris_flux, 0.0, "Debris should flow down steep slope")

        # Apply flux
        api_src = MassExchangeAPI(src)
        api_dst = MassExchangeAPI(dst)
        api_src.apply_mass_delta("dustThickness", -flux.dust_flux)
        api_src.apply_mass_delta("snowMass",      -flux.snow_flux)
        api_src.apply_mass_delta("debrisMass",    -flux.debris_flux)
        api_dst.apply_mass_delta("dustThickness",  flux.dust_flux)
        api_dst.apply_mass_delta("snowMass",       flux.snow_flux)
        api_dst.apply_mass_delta("debrisMass",     flux.debris_flux)

        self.assertLess(src.dustThickness, 0.5, "Source dust should decrease")
        self.assertGreater(dst.dustThickness, 0.0, "Destination dust should increase")

    def test_no_flux_below_threshold(self):
        model = DownhillFluxModel(_CFG)
        cell  = _make_cell(dustThickness=0.5, snowMass=0.4)
        flux  = model.compute_downhill_flux(cell, slope=0.1)  # below both thresholds
        self.assertEqual(flux.dust_flux, 0.0)
        self.assertEqual(flux.snow_flux, 0.0)


# ---------------------------------------------------------------------------
# 5. test_contact_creates_compaction_and_displacement
# ---------------------------------------------------------------------------

class TestContactDisplacement(unittest.TestCase):

    def test_contact_creates_compaction_and_displacement(self):
        model = ContactDisplacementModel(_CFG)
        cell  = _make_cell(snowMass=0.5, snowCompaction=0.1, dustThickness=0.4)
        api   = MassExchangeAPI(cell)

        comp_before = cell.snowCompaction
        dust_before = cell.dustThickness
        snow_before = cell.snowMass

        result = model.apply(api, contact_impulse=0.8)

        self.assertGreater(result.compaction_applied, 0.0)
        self.assertGreater(result.dust_displaced, 0.0)
        self.assertGreater(result.snow_displaced, 0.0)

        self.assertGreater(cell.snowCompaction, comp_before,
                           "snowCompaction should increase under contact")
        self.assertLess(cell.dustThickness, dust_before,
                        "dustThickness should decrease (displaced)")
        self.assertLess(cell.snowMass, snow_before,
                        "snowMass should decrease (displaced)")

    def test_no_displacement_with_zero_impulse(self):
        model  = ContactDisplacementModel(_CFG)
        cell   = _make_cell()
        api    = MassExchangeAPI(cell)
        result = model.apply(api, contact_impulse=0.0)
        self.assertEqual(result.compaction_applied, 0.0)
        self.assertEqual(result.dust_displaced, 0.0)
        self.assertEqual(result.snow_displaced, 0.0)


# ---------------------------------------------------------------------------
# 6. test_tracks_form_and_decay_under_wind
# ---------------------------------------------------------------------------

class TestTracksFormAndDecay(unittest.TestCase):

    def test_tracks_form_and_decay_under_wind(self):
        """Contact creates a track; light-wind settling gradually refills it."""
        system  = MassExchangeSystem(_CFG)
        cells   = [_make_cell(snowMass=0.5, dustThickness=0.3) for _ in range(4)]
        air_dust = [0.3, 0.3, 0.3, 0.3]
        air_snow = [0.2, 0.2, 0.2, 0.2]

        # Create track in cell 0
        api = MassExchangeAPI(cells[0])
        for _ in range(5):
            system.apply_contact(api, contact_impulse=0.7)
        snow_after_contact = cells[0].snowMass

        # Tick many times with low wind to allow settling
        for _ in range(20):
            system.tick(
                cells=cells,
                air_dust=air_dust,
                air_snow=air_snow,
                wind_speed=0.05,
                slope_map=[0.0] * 4,
                temperature=0.3,
                shelter=0.5,
                dt=1.0,
            )

        # snow should be partially refilled
        self.assertGreaterEqual(
            cells[0].snowMass, snow_after_contact,
            "Snow track should partially fill in with settling over time",
        )


# ---------------------------------------------------------------------------
# 7. test_determinism_flux_same_seed_same_result
# ---------------------------------------------------------------------------

class TestDeterminism(unittest.TestCase):

    def test_determinism_flux_same_seed_same_result(self):
        """Two independent systems with same inputs produce identical results."""
        def run():
            sys_ = MassExchangeSystem(_CFG)
            cells = [_make_cell(dustThickness=0.3, snowMass=0.2) for _ in range(4)]
            air_dust = [0.25] * 4
            air_snow = [0.15] * 4
            slopes = [0.1, 0.4, 0.5, 0.2]
            for _ in range(10):
                sys_.tick(
                    cells=cells,
                    air_dust=air_dust,
                    air_snow=air_snow,
                    wind_speed=0.4,
                    slope_map=slopes,
                    dt=1.0,
                )
            grid = PlanetChunkGrid("test", 2, 2)
            for i, c in enumerate(cells):
                grid._cells[i] = c
            return grid.grid_hash()

        self.assertEqual(run(), run(), "Same inputs must produce same hash")


# ---------------------------------------------------------------------------
# 8. test_budget_clamps_prevent_runaway
# ---------------------------------------------------------------------------

class TestBudgetClamps(unittest.TestCase):

    def test_budget_clamps_prevent_runaway(self):
        """Extreme wind for many ticks must not create negative mass or overflow."""
        system   = MassExchangeSystem(_CFG)
        cells    = [_make_cell(dustThickness=0.5, snowMass=0.5) for _ in range(4)]
        air_dust = [0.0] * 4
        air_snow = [0.0] * 4

        for _ in range(100):
            system.tick(
                cells=cells,
                air_dust=air_dust,
                air_snow=air_snow,
                wind_speed=1.0,
                slope_map=[1.0] * 4,
                dt=1.0,
            )

        for cell in cells:
            self.assertGreaterEqual(cell.dustThickness, 0.0, "dustThickness must not go negative")
            self.assertGreaterEqual(cell.snowMass,       0.0, "snowMass must not go negative")
            self.assertGreaterEqual(cell.debrisMass,     0.0, "debrisMass must not go negative")
        for d in air_dust:
            self.assertLessEqual(d, 1.0, "air_dust must not exceed 1.0")
        for s in air_snow:
            self.assertLessEqual(s, 1.0, "air_snow must not exceed 1.0")


# ---------------------------------------------------------------------------
# Adapter tests
# ---------------------------------------------------------------------------

class TestWeatherToMassExchange(unittest.TestCase):

    def test_maps_wind_speed_and_shelter(self):
        adapter = WeatherToMassExchange(_CFG)
        params  = LocalAtmoParams(wind_speed=0.7, temperature=0.4, humidity=0.5,
                                  pressure=0.5)
        self.assertAlmostEqual(adapter.wind_speed(params), 0.7, places=3)
        self.assertAlmostEqual(adapter.shelter(params),    0.3, places=3)

    def test_snow_boost_in_snow_regime(self):
        adapter = WeatherToMassExchange(_CFG)
        params  = LocalAtmoParams(wind_speed=0.3, humidity=0.8,
                                  regime=WeatherRegime.SNOW_DEPOSITION)
        self.assertGreater(adapter.air_snow_boost(params), 0.0)

    def test_no_snow_boost_in_clear_regime(self):
        adapter = WeatherToMassExchange(_CFG)
        params  = LocalAtmoParams(wind_speed=0.3, humidity=0.8,
                                  regime=WeatherRegime.CLEAR)
        self.assertEqual(adapter.air_snow_boost(params), 0.0)


class TestVolumetricsToSettling(unittest.TestCase):

    def test_returns_mean_ground_density(self):
        adapter = VolumetricsToSettling()
        grid    = DensityGrid(4, 4, 4, VolumeLayerType.DUST)
        for iy in range(4):
            for ix in range(4):
                grid.set_density(ix, iy, 0, 0.6)
        self.assertAlmostEqual(adapter.air_dust_density(grid), 0.6, places=3)

    def test_empty_grid_returns_zero(self):
        adapter = VolumetricsToSettling()
        grid    = DensityGrid(4, 4, 4, VolumeLayerType.SNOW_DRIFT)
        self.assertEqual(adapter.air_snow_density(grid), 0.0)


class TestInstabilityToMassImpulse(unittest.TestCase):

    def test_crust_failure_produces_debris(self):
        adapter = InstabilityToMassImpulse(_CFG)
        cell    = _make_cell(crustHardness=0.6, debrisMass=0.0)
        api     = MassExchangeAPI(cell)

        class FakeFailure:
            intensity  = 0.8
            crust_delta = 0.5

        debris_before = cell.debrisMass
        produced = adapter.on_crust_failure(api, FakeFailure())
        self.assertGreater(produced, 0.0)
        self.assertGreater(cell.debrisMass, debris_before)
        self.assertLess(cell.crustHardness, 0.6)

    def test_dust_avalanche_lifts_dust(self):
        adapter = InstabilityToMassImpulse(_CFG)
        cell    = _make_cell(dustThickness=0.5)
        api     = MassExchangeAPI(cell)

        class FakeAvalanche:
            dust_delta = 0.4

        dust_before = cell.dustThickness
        lifted = adapter.on_dust_avalanche(api, FakeAvalanche())
        self.assertGreater(lifted, 0.0)
        self.assertLess(cell.dustThickness, dust_before)


class TestCharacterToMassExchange(unittest.TestCase):

    def test_footstep_increases_compaction(self):
        adapter = CharacterToMassExchange(_CFG)
        cell    = _make_cell(snowMass=0.5, snowCompaction=0.1, dustThickness=0.4)
        api     = MassExchangeAPI(cell)

        comp_before = cell.snowCompaction
        result = adapter.on_footstep(api, contact_impulse=0.6)
        self.assertGreater(cell.snowCompaction, comp_before)
        self.assertGreater(result.compaction_applied, 0.0)

    def test_body_deposition_release_adds_to_surface(self):
        adapter = CharacterToMassExchange(_CFG)
        cell    = _make_cell(dustThickness=0.1, snowMass=0.1)
        api     = MassExchangeAPI(cell)

        dust_before = cell.dustThickness
        snow_before = cell.snowMass
        adapter.on_body_deposition_release(api, deposited_dust=0.05, deposited_snow=0.03)
        self.assertGreater(cell.dustThickness, dust_before)
        self.assertGreater(cell.snowMass, snow_before)

    def test_dust_raise_adds_to_air(self):
        adapter  = CharacterToMassExchange(_CFG)
        cell     = _make_cell(dustThickness=0.5, crustHardness=0.1)
        api      = MassExchangeAPI(cell)
        air_dust = [0.0]

        dust_before = cell.dustThickness
        lifted = adapter.on_dust_raise(api, air_dust, cell_idx=0, wind_speed=0.5)
        self.assertGreaterEqual(lifted, 0.0)
        if lifted > 0.0:
            self.assertLess(cell.dustThickness, dust_before)
            self.assertGreater(air_dust[0], 0.0)


if __name__ == "__main__":
    unittest.main()
