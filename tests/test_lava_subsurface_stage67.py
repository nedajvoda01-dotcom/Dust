"""test_lava_subsurface_stage67.py — Stage 67 Lava & Subsurface Dynamics tests.

Tests (as specified in the problem statement §14)
-------------------------------------------------
1. test_vent_creates_magma_surface_state
   — VentSpawner fires a VentEvent; LavaSurfaceFlow.spawn_lava puts lava on the
     cell; CoolingModel reports MAGMA state.

2. test_magma_flows_down_slope
   — spawn_lava at cell 0; after tick with slope=1.0 some volume moves to cell 1.

3. test_magma_cools_to_crust
   — spawn lava; run many cooling ticks; eventually CoolingModel reports CRUST.

4. test_crack_reduces_crust_hardness
   — SubsurfaceToMaterials.apply_crack with non-zero energy reduces crustHardness.

5. test_subsurface_fields_deterministic
   — Two grids built with the same seed produce identical initial fields.

6. test_lava_interacts_with_snow_and_dust
   — SubsurfaceToMaterials.apply_lava reduces snowMass and dustThickness when
     the cell is in MAGMA state.

7. test_energy_budget_reduces_after_vent
   — SubsurfaceToEnergy.apply_vent reduces mechanical energy in EnergyLedger.

8. test_snapshot_restore_subsurface_state
   — snapshot() captures state; mutate; restore() returns to original values.
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.subsurface.SubsurfaceFieldGrid import SubsurfaceFieldGrid
from src.subsurface.MagmaPressureModel  import MagmaPressureModel
from src.subsurface.CrustWeaknessModel  import CrustWeaknessModel
from src.subsurface.VentDetector        import VentDetector
from src.subsurface.VentSpawner         import VentSpawner

from src.lava.LavaSurfaceFlow import LavaSurfaceFlow, LavaCell
from src.lava.CoolingModel    import CoolingModel, CellCrustState

from src.adapters.SubsurfaceToMaterials   import SubsurfaceToMaterials
from src.adapters.SubsurfaceToVolumetrics import SubsurfaceToVolumetrics
from src.adapters.SubsurfaceToAudio       import SubsurfaceToAudio
from src.adapters.SubsurfaceToEnergy      import SubsurfaceToEnergy

from src.material.PlanetChunkState import PlanetChunkState
from src.material.MassExchangeAPI  import MassExchangeAPI
from src.energy.EnergyLedger       import EnergyLedger
from src.vol.DensityGrid           import DensityGrid, VolumeLayerType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEED = 42


def _make_grid(seed: int = SEED, tile_count: int = 8) -> SubsurfaceFieldGrid:
    return SubsurfaceFieldGrid(world_seed=seed, tile_count=tile_count)


def _forced_grid(tile_count: int = 8, pressure: float = 0.9, weakness: float = 0.9, stress: float = 0.9) -> SubsurfaceFieldGrid:
    """Build a grid and force all tiles above vent threshold."""
    grid = SubsurfaceFieldGrid(world_seed=SEED, tile_count=tile_count)
    for i in range(tile_count):
        t = grid.tile(i)
        t.magmaPressureProxy   = pressure
        t.crustWeakness        = weakness
        t.subsurfaceStress     = stress
        t.ventPotential        = t.compute_vent_potential()
    return grid


# ---------------------------------------------------------------------------
# 1. test_vent_creates_magma_surface_state
# ---------------------------------------------------------------------------

class TestVentCreatesMagmaState(unittest.TestCase):
    """VentSpawner must fire events; spawned lava must be in MAGMA state."""

    def test_vent_event_spawns_lava_in_magma_state(self):
        grid     = _forced_grid()
        detector = VentDetector({"subsurface67": {"vent_threshold": 0.5}})
        spawner  = VentSpawner({"subsurface67": {"vent_cooldown_sec": 0.0}})
        events   = spawner.tick(grid, detector, game_time=0.0, dt=1.0)

        self.assertGreater(len(events), 0, "Expected at least one vent event")

        lava  = LavaSurfaceFlow(cell_count=8)
        model = CoolingModel()

        evt = events[0]
        lava.spawn_lava(evt.tile_idx % 8, evt.intensity)

        cell  = lava.cell(evt.tile_idx % 8)
        state = model.crust_state(cell)
        self.assertEqual(state, CellCrustState.MAGMA,
            "Freshly spawned lava cell must be in MAGMA state")

    def test_vent_event_has_positive_intensity(self):
        grid     = _forced_grid()
        detector = VentDetector({"subsurface67": {"vent_threshold": 0.5}})
        spawner  = VentSpawner({"subsurface67": {"vent_cooldown_sec": 0.0}})
        events   = spawner.tick(grid, detector, game_time=0.0, dt=1.0)
        for evt in events:
            self.assertGreater(evt.intensity, 0.0, "Vent intensity must be positive")


# ---------------------------------------------------------------------------
# 2. test_magma_flows_down_slope
# ---------------------------------------------------------------------------

class TestMagmaFlowsDownSlope(unittest.TestCase):
    """Lava must flow from cell 0 to adjacent cell 1 on a steep slope."""

    def test_volume_transfers_to_adjacent_cell(self):
        lava = LavaSurfaceFlow(cell_count=4)
        lava.spawn_lava(0, intensity=0.8)

        before_src = lava.cell(0).magma_volume
        before_dst = lava.cell(1).magma_volume

        slope_map = [1.0, 0.0, 0.0, 0.0]
        lava.tick(slope_map=slope_map, dt=1.0)

        after_src = lava.cell(0).magma_volume
        after_dst = lava.cell(1).magma_volume

        self.assertLess(after_src, before_src,
            "Source cell should lose volume after flowing downhill")
        self.assertGreater(after_dst, before_dst,
            "Destination cell should gain volume from flow")

    def test_no_flow_on_flat_terrain(self):
        lava = LavaSurfaceFlow(cell_count=4)
        lava.spawn_lava(0, intensity=0.5)

        before = lava.cell(0).magma_volume
        slope_map = [0.0, 0.0, 0.0, 0.0]
        lava.tick(slope_map=slope_map, dt=1.0)
        after = lava.cell(0).magma_volume

        self.assertAlmostEqual(before, after, places=5,
            msg="Flat terrain should not cause flow")

    def test_total_volume_conserved_during_flow(self):
        lava = LavaSurfaceFlow(cell_count=4)
        lava.spawn_lava(0, intensity=0.6)

        total_before = sum(lava.cell(i).magma_volume for i in range(4))
        slope_map = [0.8, 0.0, 0.0, 0.0]
        lava.tick(slope_map=slope_map, dt=1.0)
        total_after = sum(lava.cell(i).magma_volume for i in range(4))

        self.assertAlmostEqual(total_before, total_after, places=5,
            msg="Total lava volume must be conserved during flow")


# ---------------------------------------------------------------------------
# 3. test_magma_cools_to_crust
# ---------------------------------------------------------------------------

class TestMagmaCoolsToCrust(unittest.TestCase):
    """After many cooling ticks the cell must transition to CRUST."""

    def test_cools_to_crust_with_cold_air(self):
        lava    = LavaSurfaceFlow(cell_count=1)
        cooling = CoolingModel({"lava67": {"lava_cooling_rate": 0.1, "crust_temp_threshold": 0.15}})
        lava.spawn_lava(0, intensity=0.9)

        cell = lava.cell(0)
        # Tick until crust or timeout
        for _ in range(200):
            cooling.tick([cell], air_temp=0.0, wind_speed=0.5, dt=1.0)
            if cooling.crust_state(cell) == CellCrustState.CRUST:
                break

        self.assertEqual(cooling.crust_state(cell), CellCrustState.CRUST,
            "Magma must cool to CRUST state with sufficient ticks")

    def test_temp_decreases_monotonically(self):
        lava    = LavaSurfaceFlow(cell_count=1)
        cooling = CoolingModel()
        lava.spawn_lava(0, intensity=0.9)

        cell    = lava.cell(0)
        prev_t  = cell.temp_proxy
        for _ in range(10):
            cooling.tick([cell], air_temp=0.0, wind_speed=0.0, dt=1.0)
            self.assertLessEqual(cell.temp_proxy, prev_t,
                "Temperature proxy must not increase during cooling")
            prev_t = cell.temp_proxy


# ---------------------------------------------------------------------------
# 4. test_crack_reduces_crust_hardness
# ---------------------------------------------------------------------------

class TestCrackReducesCrustHardness(unittest.TestCase):
    """apply_crack must reduce crustHardness in the chunk."""

    def test_crack_reduces_crust_hardness(self):
        chunk   = PlanetChunkState(crustHardness=0.8)
        api     = MassExchangeAPI(chunk)
        adapter = SubsurfaceToMaterials()

        before = chunk.crustHardness
        adapter.apply_crack(api, crack_energy=0.8, dt=1.0)
        after  = chunk.crustHardness

        self.assertLess(after, before,
            "Crack must reduce crustHardness")

    def test_crack_adds_debris(self):
        chunk   = PlanetChunkState(crustHardness=0.8, debrisMass=0.0)
        api     = MassExchangeAPI(chunk)
        adapter = SubsurfaceToMaterials()

        adapter.apply_crack(api, crack_energy=0.8, dt=1.0)

        self.assertGreater(chunk.debrisMass, 0.0,
            "Crack must add debris fragments")


# ---------------------------------------------------------------------------
# 5. test_subsurface_fields_deterministic
# ---------------------------------------------------------------------------

class TestSubsurfaceFieldsDeterministic(unittest.TestCase):
    """Same seed must always produce identical initial fields."""

    def test_same_seed_same_fields(self):
        g1 = _make_grid(seed=SEED)
        g2 = _make_grid(seed=SEED)
        for i in range(g1.tile_count):
            t1, t2 = g1.tile(i), g2.tile(i)
            self.assertAlmostEqual(t1.magmaPressureProxy,   t2.magmaPressureProxy,   places=9)
            self.assertAlmostEqual(t1.thermalGradientProxy, t2.thermalGradientProxy, places=9)
            self.assertAlmostEqual(t1.crustWeakness,        t2.crustWeakness,        places=9)
            self.assertAlmostEqual(t1.subsurfaceStress,     t2.subsurfaceStress,     places=9)
            self.assertAlmostEqual(t1.ventPotential,        t2.ventPotential,        places=9)

    def test_different_seed_may_differ(self):
        g1 = _make_grid(seed=SEED)
        g2 = _make_grid(seed=SEED + 1)
        # Simply ensure neither raises
        self.assertIsNotNone(g1.tile(0))
        self.assertIsNotNone(g2.tile(0))

    def test_fields_in_range(self):
        grid = _make_grid()
        for i in range(grid.tile_count):
            t = grid.tile(i)
            for val in (t.magmaPressureProxy, t.thermalGradientProxy,
                        t.crustWeakness, t.subsurfaceStress, t.ventPotential):
                self.assertGreaterEqual(val, 0.0)
                self.assertLessEqual(val, 1.0)


# ---------------------------------------------------------------------------
# 6. test_lava_interacts_with_snow_and_dust
# ---------------------------------------------------------------------------

class TestLavaInteractsWithSnowAndDust(unittest.TestCase):
    """Active lava must reduce snowMass and dustThickness."""

    def test_lava_melts_snow(self):
        chunk   = PlanetChunkState(snowMass=0.8, dustThickness=0.5)
        api     = MassExchangeAPI(chunk)
        lava    = LavaSurfaceFlow(cell_count=1)
        cooling = CoolingModel()
        adapter = SubsurfaceToMaterials()

        lava.spawn_lava(0, intensity=1.0)
        cell = lava.cell(0)

        before_snow = chunk.snowMass
        adapter.apply_lava(api, cell, cooling, dt=1.0)
        self.assertLess(chunk.snowMass, before_snow,
            "Lava must melt snow (reduce snowMass)")

    def test_lava_burns_dust(self):
        chunk   = PlanetChunkState(snowMass=0.5, dustThickness=0.8)
        api     = MassExchangeAPI(chunk)
        lava    = LavaSurfaceFlow(cell_count=1)
        cooling = CoolingModel()
        adapter = SubsurfaceToMaterials()

        lava.spawn_lava(0, intensity=1.0)
        cell = lava.cell(0)

        before_dust = chunk.dustThickness
        adapter.apply_lava(api, cell, cooling, dt=1.0)
        self.assertLess(chunk.dustThickness, before_dust,
            "Lava must burn dust (reduce dustThickness)")

    def test_solidified_lava_does_not_melt_snow(self):
        chunk   = PlanetChunkState(snowMass=0.8)
        api     = MassExchangeAPI(chunk)
        lava    = LavaSurfaceFlow(cell_count=1)
        # Force cell to crust state via CoolingModel with zero threshold
        cooling = CoolingModel({"lava67": {"crust_temp_threshold": 1.0}})
        adapter = SubsurfaceToMaterials()

        lava.spawn_lava(0, intensity=0.5)
        cell = lava.cell(0)

        before_snow = chunk.snowMass
        adapter.apply_lava(api, cell, cooling, dt=1.0)
        # In CRUST state the cell should NOT melt snow
        self.assertGreaterEqual(chunk.snowMass, before_snow,
            "Solidified lava (CRUST) must not melt snow")


# ---------------------------------------------------------------------------
# 7. test_energy_budget_reduces_after_vent
# ---------------------------------------------------------------------------

class TestEnergyBudgetReducesAfterVent(unittest.TestCase):
    """apply_vent must reduce mechanical energy in the EnergyLedger."""

    def test_mechanical_energy_decreases(self):
        ledger  = EnergyLedger()
        ledger.add("mechanical", 0.8)

        adapter = SubsurfaceToEnergy()
        before  = ledger.get("mechanical")
        adapter.apply_vent(ledger, intensity=0.9)
        after   = ledger.get("mechanical")

        self.assertLess(after, before,
            "apply_vent must consume mechanical energy")

    def test_thermal_energy_increases(self):
        ledger  = EnergyLedger()
        ledger.add("mechanical", 1.0)

        adapter = SubsurfaceToEnergy()
        before_thermal = ledger.get("thermal")
        adapter.apply_vent(ledger, intensity=0.8)
        after_thermal  = ledger.get("thermal")

        self.assertGreater(after_thermal, before_thermal,
            "apply_vent must release heat into thermal reservoir")

    def test_crack_reduces_mechanical_energy(self):
        ledger  = EnergyLedger()
        ledger.add("mechanical", 0.7)

        adapter = SubsurfaceToEnergy()
        before  = ledger.get("mechanical")
        adapter.apply_crack(ledger, crack_energy=0.6)
        after   = ledger.get("mechanical")

        self.assertLess(after, before,
            "apply_crack must consume mechanical stress energy")

    def test_crack_raises_acoustic_energy(self):
        ledger  = EnergyLedger()
        adapter = SubsurfaceToEnergy()
        adapter.apply_crack(ledger, crack_energy=0.5)

        self.assertGreater(ledger.get("acoustic"), 0.0,
            "Crack must generate acoustic energy")


# ---------------------------------------------------------------------------
# 8. test_snapshot_restore_subsurface_state
# ---------------------------------------------------------------------------

class TestSnapshotRestoreSubsurfaceState(unittest.TestCase):
    """snapshot/restore must preserve subsurface field state."""

    def test_restore_recovers_original_values(self):
        grid = _make_grid(tile_count=4)
        snap = grid.snapshot()

        # Mutate
        for i in range(4):
            t = grid.tile(i)
            t.magmaPressureProxy   = 0.99
            t.crustWeakness        = 0.99
            t.subsurfaceStress     = 0.99

        grid.restore(snap)

        for i in range(4):
            t = grid.tile(i)
            orig = snap["tiles"][i]
            self.assertAlmostEqual(t.magmaPressureProxy,   orig["magmaPressureProxy"],   places=9)
            self.assertAlmostEqual(t.crustWeakness,        orig["crustWeakness"],        places=9)
            self.assertAlmostEqual(t.subsurfaceStress,     orig["subsurfaceStress"],     places=9)

    def test_snapshot_contains_all_fields(self):
        grid = _make_grid(tile_count=4)
        snap = grid.snapshot()
        for td in snap["tiles"]:
            for key in ("magmaPressureProxy", "thermalGradientProxy",
                        "crustWeakness", "subsurfaceStress", "ventPotential"):
                self.assertIn(key, td, f"Snapshot must contain '{key}'")

    def test_snapshot_roundtrip_preserves_tile_count(self):
        grid = _make_grid(tile_count=8)
        snap = grid.snapshot()
        self.assertEqual(len(snap["tiles"]), 8)

        grid2 = _make_grid(tile_count=8)
        grid2.restore(snap)
        self.assertEqual(grid2.tile_count, 8)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
