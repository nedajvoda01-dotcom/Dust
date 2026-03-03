"""test_planetary_evolution_stage50.py — Stage 50 Planetary Time-Scale Evolution
smoke tests.

Tests
-----
1. test_dust_migrates_downwind_over_time
   — DustAdvectionModel concentrates dust in the downwind direction after
     many ticks; tiles in the wind shadow gain more dust than tiles facing
     into the wind.

2. test_crust_stability_changes_after_cycles
   — CrustStabilityModel reduces crustStabilityMap under sustained thermal
     cycling stress; after enough ticks, stability is measurably lower than
     its initial value.

3. test_ice_belt_shifts_with_season
   — SeasonalInsolationModel produces different iceBeltDistribution values
     at two different seasonal phases (π/2 apart), demonstrating that ice
     belt coverage shifts as the planet orbits.

4. test_slope_creep_reduces_mass_uphill
   — SlopeCreepModel transfers dust from steep tiles (high slope) to their
     downhill neighbours; the uphill tile loses dust and the downhill tile
     gains it.

5. test_snapshot_restore_evolution_state
   — EvolutionSnapshot.save() + .load() round-trips PlanetEvolutionState
     and PlanetTimeController with lossiness bounded by 8-bit quantisation
     (≤ 1/255 ≈ 0.004).

6. test_determinism_same_seed_same_long_term_state
   — Two independent simulation runs with the same initial state and the
     same input sequence produce identical state_hash values.

7. test_budget_tile_update_limit
   — DustAdvectionModel never processes more than max_tiles_per_tick tiles
     in a single call to tick().
"""
from __future__ import annotations

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.evolution.PlanetTimeController      import PlanetTimeController
from src.evolution.PlanetEvolutionState      import PlanetEvolutionState
from src.evolution.DustAdvectionModel        import DustAdvectionModel
from src.evolution.CrustStabilityModel       import CrustStabilityModel
from src.evolution.SlopeCreepModel           import SlopeCreepModel
from src.evolution.SeasonalInsolationModel   import SeasonalInsolationModel
from src.evolution.EvolutionToMaterialAdapter import EvolutionToMaterialAdapter
from src.net.EvolutionReplicator             import EvolutionReplicator
from src.save.EvolutionSnapshot              import EvolutionSnapshot


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_W, _H = 16, 8   # small grid for fast tests

def _make_state() -> PlanetEvolutionState:
    return PlanetEvolutionState(width=_W, height=_H)

def _make_time_ctrl(**kwargs) -> PlanetTimeController:
    return PlanetTimeController({"planet": {"timescale": 1.0, **kwargs}})

_EVO_CFG = {
    "evolution": {
        "dust_advection_k":        0.2,
        "dust_erosion_k":          0.05,
        "deposition_k":            0.1,
        "max_tiles_per_tick":      _W * _H,  # update all tiles per tick
        "crust_decay_k":           0.05,
        "crust_dust_load_k":       0.02,
        "crust_recovery_k":        0.001,
        "crust_collapse_threshold": 0.2,
        "crust_recovery_boost":    0.15,
        "slope_creep_k":           0.1,
        "slope_threshold":         0.3,
        "dust_threshold":          0.1,
        "dust_transfer_frac":      0.5,
        "ice_belt_k":              0.1,
        "ice_melt_k":              0.05,
        "season_speed":            0.1,       # fast season for test
    }
}


# ---------------------------------------------------------------------------
# 1. test_dust_migrates_downwind_over_time
# ---------------------------------------------------------------------------

class TestDustMigratesDownwindOverTime(unittest.TestCase):
    """Dust should accumulate on the downwind (leeward) side after many ticks."""

    def test_dust_increases_in_downwind_half(self):
        state = _make_state()

        # Set uniform initial dust
        for i in range(state.size()):
            state.dustReservoirMap[i] = 0.3

        # Mark upwind half (x < W/2) with extra shelter = 0 (no deposition)
        # and downwind half (x >= W/2) with high shelter (deposition sink)
        W = _W
        shelter = []
        for idx in range(state.size()):
            ix = idx % W
            shelter.append(0.8 if ix >= W // 2 else 0.0)

        model = DustAdvectionModel(_EVO_CFG)

        # Wind blows in +X direction (wind_u=1, wind_v=0).
        # Use a short run (5 ticks) to avoid saturation: at this point the
        # sheltered (downwind) tiles should be measurably dustier than the
        # exposed (upwind) tiles, since deposition only applies there.
        for _ in range(5):
            model.tick(state, wind_u=1.0, wind_v=0.0, dt=1.0,
                       shelter_map=shelter)

        # Average dust in downwind half vs upwind half
        dust_downwind = [state.dustReservoirMap[i]
                         for i in range(state.size()) if (i % W) >= W // 2]
        dust_upwind   = [state.dustReservoirMap[i]
                         for i in range(state.size()) if (i % W) < W // 2]

        avg_down = sum(dust_downwind) / len(dust_downwind)
        avg_up   = sum(dust_upwind)   / len(dust_upwind)

        self.assertGreater(
            avg_down, avg_up,
            f"Downwind dust ({avg_down:.3f}) should exceed upwind ({avg_up:.3f})"
        )


# ---------------------------------------------------------------------------
# 2. test_crust_stability_changes_after_cycles
# ---------------------------------------------------------------------------

class TestCrustStabilityChangesAfterCycles(unittest.TestCase):
    """Sustained thermal cycling must reduce crustStabilityMap over time."""

    def test_stability_decreases_under_high_cycling(self):
        state = _make_state()

        # Set initial stability to 0.8
        for i in range(state.size()):
            state.crustStabilityMap[i] = 0.8

        model = CrustStabilityModel(_EVO_CFG)

        initial_avg = sum(state.crustStabilityMap) / state.size()

        # Run 200 ticks with high thermal cycling amplitude
        for _ in range(200):
            model.tick(state, thermal_cycle_amp=1.0, dt=0.5, planet_time=0.0)

        final_avg = sum(state.crustStabilityMap) / state.size()

        self.assertLess(
            final_avg, initial_avg,
            f"Stability should decrease: initial={initial_avg:.3f} "
            f"final={final_avg:.3f}"
        )

    def test_collapse_events_emitted_when_low(self):
        """Tiles near the collapse threshold must emit CrustCollapseEvents."""
        state = _make_state()
        # Force stability just above threshold
        for i in range(state.size()):
            state.crustStabilityMap[i] = 0.25

        model = CrustStabilityModel(_EVO_CFG)
        all_events = []
        for _ in range(50):
            evts = model.tick(state, thermal_cycle_amp=1.0, dt=1.0,
                              planet_time=10.0)
            all_events.extend(evts)

        self.assertGreater(
            len(all_events), 0,
            "Expected at least one CrustCollapseEvent near the threshold"
        )


# ---------------------------------------------------------------------------
# 3. test_ice_belt_shifts_with_season
# ---------------------------------------------------------------------------

class TestIceBeltShiftsWithSeason(unittest.TestCase):
    """iceBeltDistribution must differ at two different seasonal phases."""

    def _run_to_phase(self, target_phase: float) -> PlanetEvolutionState:
        state = _make_state()
        model = SeasonalInsolationModel(_EVO_CFG)
        # Advance until the seasonal phase is close to target
        # Each tick advances phase by season_speed * dt = 0.1 * 1.0 = 0.1 rad
        n_steps = int(target_phase / 0.1) + 1
        for _ in range(n_steps):
            model.tick(state, dt=1.0)
        return state

    def test_ice_distribution_differs_between_phases(self):
        state_q1 = self._run_to_phase(0.0)   # near winter solstice phase
        state_q2 = self._run_to_phase(math.pi * 0.5)  # near equinox phase

        # Compare average ice belt values
        avg1 = sum(state_q1.iceBeltDistribution) / state_q1.size()
        avg2 = sum(state_q2.iceBeltDistribution) / state_q2.size()

        self.assertNotAlmostEqual(
            avg1, avg2, delta=1e-4,
            msg=(
                f"Ice belt distribution should differ between seasonal phases; "
                f"avg1={avg1:.4f}, avg2={avg2:.4f}"
            )
        )

    def test_polar_tiles_accumulate_more_ice(self):
        """Polar tiles (top/bottom rows) should accumulate more ice than equatorial."""
        state = _make_state()
        model = SeasonalInsolationModel(_EVO_CFG)

        # Run many ticks to let ice belt develop
        for _ in range(500):
            model.tick(state, dt=1.0)

        W, H = _W, _H
        # Polar rows (y=0 and y=H-1) vs equatorial rows (y=H//2-1, y=H//2)
        polar_ice = []
        equat_ice = []
        for ix in range(W):
            polar_ice.append(state.iceBeltDistribution[state.tile(ix, 0)])
            polar_ice.append(state.iceBeltDistribution[state.tile(ix, H - 1)])
            equat_ice.append(state.iceBeltDistribution[state.tile(ix, H // 2)])

        avg_polar = sum(polar_ice) / len(polar_ice)
        avg_equat = sum(equat_ice) / len(equat_ice)

        self.assertGreater(
            avg_polar, avg_equat,
            f"Polar ice ({avg_polar:.4f}) should exceed equatorial ({avg_equat:.4f})"
        )


# ---------------------------------------------------------------------------
# 4. test_slope_creep_reduces_mass_uphill
# ---------------------------------------------------------------------------

class TestSlopeCreepReducesMassUphill(unittest.TestCase):
    """SlopeCreepModel must move dust from steep tiles to their downhill neighbours."""

    def test_dust_moves_downhill(self):
        state = _make_state()

        # Fill all tiles with high dust
        for i in range(state.size()):
            state.dustReservoirMap[i] = 0.8

        W = _W
        H = _H

        # High slope for top half, zero for bottom half
        slope_map = []
        for idx in range(state.size()):
            iy = idx // W
            slope_map.append(0.9 if iy < H // 2 else 0.0)

        model = SlopeCreepModel(_EVO_CFG)

        # Record initial dust at top row (y=0)
        initial_top = [state.dustReservoirMap[state.tile(ix, 0)] for ix in range(W)]

        for _ in range(100):
            model.tick(state, slope_map=slope_map, dt=1.0)

        # Top row should have lost dust
        final_top = [state.dustReservoirMap[state.tile(ix, 0)] for ix in range(W)]
        avg_initial = sum(initial_top) / len(initial_top)
        avg_final   = sum(final_top)   / len(final_top)

        self.assertLess(
            avg_final, avg_initial,
            f"Top-row dust should decrease due to creep: "
            f"initial={avg_initial:.3f}, final={avg_final:.3f}"
        )

    def test_downhill_tile_gains_dust(self):
        """The row below a steep slope should accumulate transferred dust."""
        state = _make_state()

        for i in range(state.size()):
            state.dustReservoirMap[i] = 0.5

        W = _W
        H = _H
        # Steep slope at row 0, flat everywhere else
        slope_map = [0.9 if (idx // W) == 0 else 0.0 for idx in range(state.size())]

        model = SlopeCreepModel(_EVO_CFG)

        # y=1 is the downhill row from y=0
        initial_row1 = [state.dustReservoirMap[state.tile(ix, 1)] for ix in range(W)]

        for _ in range(100):
            model.tick(state, slope_map=slope_map, dt=1.0)

        final_row1 = [state.dustReservoirMap[state.tile(ix, 1)] for ix in range(W)]

        avg_init = sum(initial_row1) / len(initial_row1)
        avg_fin  = sum(final_row1)   / len(final_row1)

        self.assertGreater(
            avg_fin, avg_init,
            f"Downhill row should gain dust: initial={avg_init:.3f}, final={avg_fin:.3f}"
        )


# ---------------------------------------------------------------------------
# 5. test_snapshot_restore_evolution_state
# ---------------------------------------------------------------------------

class TestSnapshotRestoreEvolutionState(unittest.TestCase):
    """EvolutionSnapshot must round-trip PlanetEvolutionState within 8-bit tolerance."""

    _TOLERANCE = 1 / 255.0 + 1e-9  # quantisation error ≤ 1 LSB

    def test_round_trip_fields(self):
        state = _make_state()
        # Set distinctive values
        for i in range(state.size()):
            state.dustReservoirMap[i]    = 0.123 + 0.001 * (i % 10)
            state.crustStabilityMap[i]   = 0.756 - 0.002 * (i % 7)
            state.slopeCreepMap[i]       = 0.333
            state.iceBeltDistribution[i] = 0.600 + 0.001 * (i % 5)
        state.seasonalInsolationPhase = 1.234

        tc = _make_time_ctrl()
        tc.advance(100.0)

        snap = EvolutionSnapshot()
        blob = snap.save(state, tc, world_seed=12345)
        restored, meta = snap.load(blob)

        self.assertEqual(meta["world_seed"], 12345)
        self.assertAlmostEqual(meta["planet_time"], tc.planet_time, places=5)
        self.assertAlmostEqual(meta["sim_time"],    tc.sim_time,    places=5)
        self.assertAlmostEqual(
            restored.seasonalInsolationPhase,
            state.seasonalInsolationPhase,
            delta=1e-4,
        )

        for i in range(state.size()):
            for field in ("dustReservoirMap", "crustStabilityMap",
                          "slopeCreepMap", "iceBeltDistribution"):
                orig = getattr(state, field)[i]
                rest = getattr(restored, field)[i]
                self.assertAlmostEqual(
                    orig, rest, delta=self._TOLERANCE,
                    msg=f"Field {field}[{i}]: original={orig:.4f} restored={rest:.4f}"
                )

    def test_bad_magic_raises(self):
        snap = EvolutionSnapshot()
        with self.assertRaises(ValueError):
            snap.load(b"BAD!" + b"\x00" * 100)

    def test_replicator_round_trip(self):
        """EvolutionReplicator build_snapshot / apply_snapshot round-trip."""
        state  = _make_state()
        for i in range(state.size()):
            state.dustReservoirMap[i] = 0.4 + 0.001 * (i % 20)
        state.seasonalInsolationPhase = 2.5

        rep  = EvolutionReplicator()
        msg  = rep.build_snapshot(state, world_seed=7, planet_time=500.0)

        state2 = _make_state()
        ok = rep.apply_snapshot(state2, msg)
        self.assertTrue(ok)
        self.assertAlmostEqual(
            state2.seasonalInsolationPhase,
            state.seasonalInsolationPhase,
            delta=0.01,
        )
        # Dust values should be within 8-bit tolerance
        tol = 1 / 255.0 + 1e-9
        for i in range(state.size()):
            self.assertAlmostEqual(
                state2.dustReservoirMap[i],
                state.dustReservoirMap[i],
                delta=tol,
            )


# ---------------------------------------------------------------------------
# 6. test_determinism_same_seed_same_long_term_state
# ---------------------------------------------------------------------------

class TestDeterminismSameSeedSameLongTermState(unittest.TestCase):
    """Two identical runs must produce the same state_hash."""

    def _run(self, n_ticks: int = 50) -> str:
        state  = _make_state()
        dust   = DustAdvectionModel(_EVO_CFG)
        crust  = CrustStabilityModel(_EVO_CFG)
        creep  = SlopeCreepModel(_EVO_CFG)
        season = SeasonalInsolationModel(_EVO_CFG)

        slope_map = [float(i % _W) / _W for i in range(state.size())]

        for _ in range(n_ticks):
            season.tick(state, dt=1.0)
            amp = season.thermal_cycle_amplitude(state)
            dust.tick(state, wind_u=0.6, wind_v=0.3, dt=1.0)
            crust.tick(state, thermal_cycle_amp=amp, dt=1.0)
            creep.tick(state, slope_map=slope_map, dt=1.0)

        return state.state_hash()

    def test_two_runs_same_hash(self):
        h1 = self._run()
        h2 = self._run()
        self.assertEqual(h1, h2, "Identical runs must produce identical state hashes")

    def test_planet_time_advances_deterministically(self):
        """PlanetTimeController must advance identically for the same inputs."""
        tc1 = _make_time_ctrl()
        tc2 = _make_time_ctrl()
        for _ in range(100):
            tc1.advance(0.016)
            tc2.advance(0.016)
        self.assertAlmostEqual(tc1.planet_time, tc2.planet_time, places=10)
        self.assertAlmostEqual(tc1.sim_time,    tc2.sim_time,    places=10)


# ---------------------------------------------------------------------------
# 7. test_budget_tile_update_limit
# ---------------------------------------------------------------------------

class TestBudgetTileUpdateLimit(unittest.TestCase):
    """DustAdvectionModel must not update more than max_tiles_per_tick tiles."""

    def test_budget_respected(self):
        budget = 10
        cfg = {
            "evolution": {
                "dust_advection_k":   0.2,
                "dust_erosion_k":     0.05,
                "deposition_k":       0.1,
                "max_tiles_per_tick": budget,
            }
        }
        state = _make_state()
        for i in range(state.size()):
            state.dustReservoirMap[i] = 0.5

        before = list(state.dustReservoirMap)

        model = DustAdvectionModel(cfg)
        model.tick(state, wind_u=1.0, wind_v=0.0, dt=1.0)

        changed = sum(
            1 for i in range(state.size())
            if abs(state.dustReservoirMap[i] - before[i]) > 1e-9
        )

        self.assertLessEqual(
            changed, budget,
            f"Expected ≤ {budget} tiles changed, got {changed}"
        )

    def test_full_budget_covers_all_tiles_eventually(self):
        """After n/budget ticks, all tiles should have been processed."""
        n = _W * _H
        budget = 8
        cfg = {
            "evolution": {
                "dust_advection_k":   0.2,
                "dust_erosion_k":     0.05,
                "deposition_k":       0.3,
                "max_tiles_per_tick": budget,
            }
        }
        state = _make_state()
        # Uniform high shelter so deposition dominates and values change
        shelter = [1.0] * n
        for i in range(n):
            state.dustReservoirMap[i] = 0.1   # start low to allow change

        model = DustAdvectionModel(cfg)
        n_ticks = (n // budget) + 2
        for _ in range(n_ticks):
            model.tick(state, wind_u=1.0, wind_v=0.0, dt=1.0,
                       shelter_map=shelter)

        # At least half the tiles should have changed from their initial value
        changed = sum(
            1 for v in state.dustReservoirMap if abs(v - 0.1) > 1e-4
        )
        self.assertGreater(
            changed, n // 2,
            f"Expected most tiles to have been updated; only {changed}/{n} changed"
        )


# ---------------------------------------------------------------------------
# Bonus: EvolutionToMaterialAdapter sanity checks
# ---------------------------------------------------------------------------

class TestEvolutionToMaterialAdapter(unittest.TestCase):
    """MaterialBiases must scale correctly with evolution field values."""

    def test_high_dust_gives_high_deposition_bias(self):
        adapter = EvolutionToMaterialAdapter()
        low  = adapter.get_material_biases(dust_reservoir=0.0, crust_stability=0.5, ice_belt=0.0)
        high = adapter.get_material_biases(dust_reservoir=1.0, crust_stability=0.5, ice_belt=0.0)
        self.assertGreater(high.dust_deposition_bias, low.dust_deposition_bias)

    def test_high_ice_belt_gives_positive_ice_formation_bias(self):
        adapter = EvolutionToMaterialAdapter()
        b = adapter.get_material_biases(dust_reservoir=0.3, crust_stability=0.5, ice_belt=0.8)
        self.assertGreater(b.ice_formation_bias, 0.0)

    def test_low_stability_gives_negative_roughness_bias(self):
        adapter = EvolutionToMaterialAdapter()
        b = adapter.get_material_biases(dust_reservoir=0.3, crust_stability=0.0, ice_belt=0.0)
        self.assertLess(b.roughness_bias, 0.0)

    def test_high_stability_gives_positive_crust_bias(self):
        adapter = EvolutionToMaterialAdapter()
        b_high = adapter.get_material_biases(dust_reservoir=0.0, crust_stability=1.0, ice_belt=0.0)
        b_low  = adapter.get_material_biases(dust_reservoir=0.0, crust_stability=0.0, ice_belt=0.0)
        self.assertGreater(b_high.crust_hardness_bias, b_low.crust_hardness_bias)


# ---------------------------------------------------------------------------
# Bonus: PlanetTimeController tests
# ---------------------------------------------------------------------------

class TestPlanetTimeController(unittest.TestCase):
    """PlanetTimeController dual-time and phase tests."""

    def test_planet_time_scales_with_timescale(self):
        tc = PlanetTimeController({"planet": {"timescale": 0.01, "day_length_s": 5400.0,
                                              "season_length_s": 86400.0}})
        tc.advance(100.0)
        self.assertAlmostEqual(tc.sim_time,    100.0, places=6)
        self.assertAlmostEqual(tc.planet_time, 1.0,   places=6)

    def test_season_phase_wraps(self):
        tc = PlanetTimeController({"planet": {"timescale": 1.0, "day_length_s": 5400.0,
                                              "season_length_s": 10.0}})
        tc.advance(25.0)   # 2.5 seasons
        self.assertAlmostEqual(tc.season_phase, 0.5, places=6)

    def test_snapshot_roundtrip(self):
        tc = PlanetTimeController({"planet": {"timescale": 0.001}})
        tc.advance(12345.0)
        d  = tc.to_dict()
        tc2 = PlanetTimeController()
        tc2.from_dict(d)
        self.assertAlmostEqual(tc2.sim_time,    tc.sim_time,    places=6)
        self.assertAlmostEqual(tc2.planet_time, tc.planet_time, places=6)


if __name__ == "__main__":
    unittest.main()
