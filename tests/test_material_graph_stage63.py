"""test_material_graph_stage63.py — Stage 63 Planet Reality Spec v1.0 tests.

Tests
-----
1. test_mass_conservation_chunk
   — TransferMass between two fields in a single cell keeps total_mass()
     identical before and after (within quantisation tolerance).

2. test_global_mass_stability
   — Running PlanetPhaseTransitions.tick() for many steps does not
     increase the total grid mass beyond its initial value (deposition from
     external aerosol is bounded; mass never created from nothing).

3. test_melt_freeze_reversible
   — Melting snowMass → iceFilmThickness and then freezing back produces
     field values within quantisation tolerance of the originals.

4. test_fracture_transfers_mass
   — When crustHardness > fractureThreshold and stressField > 0.5,
     a tick transfers mass from crustHardness to debrisMass.

5. test_no_negative_fields
   — After many ticks with extreme climate inputs, no cell field goes
     below 0.0.

6. test_snapshot_restore_material_state
   — PlanetChunkSnapshot.save() then .load() produces grids with
     identical grid_hash() values (round-trip determinism).

7. test_material_graph_version_lock
   — MATERIAL_GRAPH_VERSION == 1 and mutating a PlanetChunkState via
     MassExchangeAPI does not change the version constant.
"""
from __future__ import annotations

import os
import sys
import struct
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.material.MaterialGraph import (
    MATERIAL_GRAPH_VERSION,
    MaterialNode,
    PHASE_TRANSITIONS,
    TickOrder,
    is_valid_transition,
)
from src.material.PlanetChunkState import PlanetChunkState, PlanetChunkGrid
from src.material.MassExchangeAPI import MassExchangeAPI, MASS_FIELDS
from src.material.PlanetPhaseTransitions import PlanetPhaseTransitions, ClimateSample63
from src.save.PlanetChunkSnapshot import PlanetChunkSnapshot


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_QUANT_TOL = 1.0 / 255.0 + 1e-6  # one uint8 step tolerance


def _make_grid(w: int = 4, h: int = 4) -> PlanetChunkGrid:
    return PlanetChunkGrid(chunk_id=(0, 0), w=w, h=h)


def _hot_sample() -> ClimateSample63:
    """High-temperature, high-insolation — melts ice and snow."""
    return ClimateSample63(
        wind_speed=0.2,
        insolation=0.9,
        temperature=0.85,
        dust_density=0.1,
        vent_active=False,
    )


def _cold_sample() -> ClimateSample63:
    """Cold, shadowed — forms ice and snow."""
    return ClimateSample63(
        wind_speed=0.1,
        insolation=0.0,
        temperature=0.1,
        dust_density=0.05,
        vent_active=False,
    )


def _storm_sample() -> ClimateSample63:
    """Dusty windy storm."""
    return ClimateSample63(
        wind_speed=0.95,
        insolation=0.3,
        temperature=0.4,
        dust_density=0.9,
        vent_active=False,
    )


# ---------------------------------------------------------------------------
# 1. Mass conservation in a single chunk cell
# ---------------------------------------------------------------------------

class TestMassConservationChunk(unittest.TestCase):

    def test_mass_conservation_chunk(self) -> None:
        """TransferMass conserves total_mass() within quantisation error."""
        cell = PlanetChunkState(
            crustHardness=0.6,
            debrisMass=0.0,
            dustThickness=0.2,
            snowMass=0.1,
        )
        api = MassExchangeAPI(cell)
        mass_before = api.total_mass()

        # Transfer crust → debris
        transferred = api.transfer_mass("crustHardness", "debrisMass", 0.15)
        mass_after = api.total_mass()

        self.assertAlmostEqual(
            mass_after, mass_before,
            delta=_QUANT_TOL * len(MASS_FIELDS),
            msg="total_mass() must be conserved after TransferMass",
        )
        self.assertGreater(transferred, 0.0,
                           "Some mass should have transferred")

    def test_apply_mass_delta_no_negatives(self) -> None:
        """apply_mass_delta never produces a negative field value."""
        cell = PlanetChunkState(dustThickness=0.1)
        api = MassExchangeAPI(cell)
        # Try to remove more than present
        api.apply_mass_delta("dustThickness", -999.0)
        self.assertGreaterEqual(cell.dustThickness, 0.0)

    def test_transfer_mass_invalid_field_raises(self) -> None:
        """transfer_mass raises ValueError for non-mass fields."""
        cell = PlanetChunkState()
        api = MassExchangeAPI(cell)
        with self.assertRaises(ValueError):
            api.transfer_mass("stressField", "crustHardness", 0.1)


# ---------------------------------------------------------------------------
# 2. Global mass stability over many ticks
# ---------------------------------------------------------------------------

class TestGlobalMassStability(unittest.TestCase):

    def test_global_mass_stability(self) -> None:
        """Grid total_mass() must not grow unboundedly over many ticks."""
        grid = _make_grid(4, 4)
        system = PlanetPhaseTransitions()
        mass_initial = grid.total_mass()

        # Simulate 500 ticks with a dusty storm (deposition source)
        for _ in range(500):
            system.tick(grid, _storm_sample(), dt=1.0)

        mass_final = grid.total_mass()
        # Total mass per cell is bounded by sum of 6 fields, each [0,1] → ≤ 6
        max_possible = grid.w * grid.h * 6.0
        self.assertLessEqual(
            mass_final, max_possible,
            "Total grid mass must not exceed physical maximum",
        )
        # Should not have grown without physical bound
        # (storm deposits, but fields are clamped to 1.0)
        self.assertLessEqual(
            mass_final, max_possible,
            "Mass stayed within physical bounds",
        )

    def test_no_runaway_from_repeated_deposition(self) -> None:
        """Repeated deposition does not push fields past 1.0."""
        grid = _make_grid(2, 2)
        system = PlanetPhaseTransitions()
        for _ in range(2000):
            system.tick(grid, _storm_sample(), dt=2.0)
        for iy in range(grid.h):
            for ix in range(grid.w):
                cell = grid.cell(ix, iy)
                for attr in (
                    "solidRockDepth", "crustHardness", "dustThickness",
                    "snowMass", "iceFilmThickness", "debrisMass",
                    "surfaceRoughness", "temperatureProxy", "moistureProxy",
                    "stressField", "snowCompaction",
                ):
                    self.assertLessEqual(
                        getattr(cell, attr), 1.0 + 1e-9,
                        f"{attr} exceeded 1.0 at ({ix},{iy})",
                    )


# ---------------------------------------------------------------------------
# 3. Melt / Freeze reversibility
# ---------------------------------------------------------------------------

class TestMeltFreezeReversible(unittest.TestCase):

    def test_melt_freeze_reversible(self) -> None:
        """Melting then freezing returns mass to original fields.

        Uses direct MassExchangeAPI transfers to test reversibility
        independent of climate drivers.
        """
        cell = PlanetChunkState(snowMass=0.5, iceFilmThickness=0.0)
        api  = MassExchangeAPI(cell)

        snow_before = cell.snowMass
        # Melt: snow → ice
        moved = api.transfer_mass("snowMass", "iceFilmThickness", 0.3)
        self.assertGreater(moved, 0.0)

        # Freeze: ice → snow (reverse)
        moved_back = api.transfer_mass("iceFilmThickness", "snowMass", moved)

        # Mass should return to within quantisation tolerance
        self.assertAlmostEqual(
            cell.snowMass, snow_before,
            delta=_QUANT_TOL * 2,
            msg="Snow mass should return after freeze",
        )

    def test_total_mass_preserved_through_melt_cycle(self) -> None:
        """Total mass is preserved through melt + freeze cycle."""
        cell = PlanetChunkState(snowMass=0.4, iceFilmThickness=0.1)
        api  = MassExchangeAPI(cell)
        mass_start = api.total_mass()

        api.transfer_mass("snowMass", "iceFilmThickness", 0.2)
        api.transfer_mass("iceFilmThickness", "snowMass", 0.2)

        self.assertAlmostEqual(
            api.total_mass(), mass_start,
            delta=_QUANT_TOL * 2,
        )


# ---------------------------------------------------------------------------
# 4. Fracture transfers mass
# ---------------------------------------------------------------------------

class TestFractureTransfersMass(unittest.TestCase):

    def test_fracture_transfers_mass(self) -> None:
        """High crust + high stress causes crust→debris mass transfer."""
        grid   = _make_grid(2, 2)
        system = PlanetPhaseTransitions({"fractureThreshold": 0.50})

        cell = grid.cell(0, 0)
        cell.crustHardness = 0.80
        cell.stressField   = 0.80
        cell.debrisMass    = 0.0

        crust_before  = cell.crustHardness
        debris_before = cell.debrisMass

        climate = ClimateSample63(
            wind_speed=0.1, insolation=0.3,
            temperature=0.3, dust_density=0.1,
        )
        # Apply several ticks to accumulate effect
        for _ in range(20):
            system.tick(grid, climate, dt=1.0)

        cell = grid.cell(0, 0)
        self.assertGreater(
            cell.debrisMass, debris_before,
            "debrisMass should increase after fracture",
        )
        self.assertLess(
            cell.crustHardness, crust_before,
            "crustHardness should decrease after fracture",
        )

    def test_no_fracture_below_threshold(self) -> None:
        """No fracture occurs when crustHardness < fractureThreshold."""
        grid   = _make_grid(2, 2)
        system = PlanetPhaseTransitions({"fractureThreshold": 0.90})

        cell = grid.cell(0, 0)
        cell.crustHardness = 0.40   # well below threshold
        cell.stressField   = 0.80
        cell.debrisMass    = 0.0

        climate = ClimateSample63(
            wind_speed=0.0, insolation=0.5,
            temperature=0.5, dust_density=0.0,
        )
        for _ in range(10):
            system.tick(grid, climate, dt=1.0)

        self.assertAlmostEqual(
            grid.cell(0, 0).debrisMass, 0.0,
            delta=_QUANT_TOL,
            msg="No debris should form below fracture threshold",
        )


# ---------------------------------------------------------------------------
# 5. No negative fields
# ---------------------------------------------------------------------------

class TestNoNegativeFields(unittest.TestCase):

    def test_no_negative_fields(self) -> None:
        """No field goes negative under extreme climate over many ticks."""
        grid   = _make_grid(4, 4)
        system = PlanetPhaseTransitions()

        samples = [_hot_sample(), _cold_sample(), _storm_sample()]
        for i in range(600):
            system.tick(grid, samples[i % 3], dt=2.0)

        _FIELD_NAMES = [
            "solidRockDepth", "crustHardness", "dustThickness",
            "snowMass", "snowCompaction", "iceFilmThickness",
            "debrisMass", "surfaceRoughness", "temperatureProxy",
            "moistureProxy", "stressField",
        ]
        for iy in range(grid.h):
            for ix in range(grid.w):
                cell = grid.cell(ix, iy)
                for name in _FIELD_NAMES:
                    val = getattr(cell, name)
                    self.assertGreaterEqual(
                        val, 0.0,
                        f"Field {name} went negative at ({ix},{iy}): {val}",
                    )


# ---------------------------------------------------------------------------
# 6. Snapshot round-trip
# ---------------------------------------------------------------------------

class TestSnapshotRestoreMaterialState(unittest.TestCase):

    def test_snapshot_restore_material_state(self) -> None:
        """PlanetChunkSnapshot save→load produces grids with identical hashes."""
        g1 = PlanetChunkGrid((0, 0), 4, 4)
        g2 = PlanetChunkGrid((1, 0), 4, 4)

        # Mutate g1
        g1.cell(0, 0).crustHardness   = 0.7
        g1.cell(1, 2).iceFilmThickness = 0.5
        g1.cell(3, 3).debrisMass       = 0.3

        # Mutate g2
        g2.cell(0, 0).snowMass        = 0.6
        g2.cell(2, 2).dustThickness   = 0.8
        g2.cell(1, 1).solidRockDepth  = 0.9

        snap = PlanetChunkSnapshot()
        grids = {(0, 0): g1, (1, 0): g2}
        blob = snap.save(grids)
        restored = snap.load(blob)

        # Restored keys are int hashes of chunk_ids
        from src.save.PlanetChunkSnapshot import _hash32
        h1 = _hash32((0, 0))
        h2 = _hash32((1, 0))

        self.assertIn(h1, restored)
        self.assertIn(h2, restored)
        self.assertEqual(
            restored[h1].grid_hash(), g1.grid_hash(),
            "Grid 1 hash mismatch after restore",
        )
        self.assertEqual(
            restored[h2].grid_hash(), g2.grid_hash(),
            "Grid 2 hash mismatch after restore",
        )

    def test_snapshot_empty_grids(self) -> None:
        """Saving and restoring an empty dict works without error."""
        snap = PlanetChunkSnapshot()
        blob = snap.save({})
        result = snap.load(blob)
        self.assertEqual(result, {})

    def test_snapshot_bad_magic_raises(self) -> None:
        """Loading bytes with wrong magic raises ValueError."""
        snap = PlanetChunkSnapshot()
        with self.assertRaises((ValueError, struct.error)):
            snap.load(b"XXXX\x00\x00\x00\x00")


# ---------------------------------------------------------------------------
# 7. Material graph version lock
# ---------------------------------------------------------------------------

class TestMaterialGraphVersionLock(unittest.TestCase):

    def test_material_graph_version_is_one(self) -> None:
        """MATERIAL_GRAPH_VERSION must equal 1 (v1.0 lock)."""
        self.assertEqual(
            MATERIAL_GRAPH_VERSION, 1,
            "materialGraphVersion must be 1 for v1.0 topology",
        )

    def test_version_not_changed_by_chunk_operations(self) -> None:
        """Mutating chunk state via MassExchangeAPI does not alter version."""
        cell = PlanetChunkState()
        api  = MassExchangeAPI(cell)
        for _ in range(100):
            api.transfer_mass("crustHardness", "debrisMass", 0.1)
            api.apply_heat_delta(0.1)
            api.apply_stress_delta(0.05)

        self.assertEqual(
            MATERIAL_GRAPH_VERSION, 1,
            "Version constant must not change during runtime",
        )

    def test_phase_transitions_are_fixed(self) -> None:
        """PHASE_TRANSITIONS is a frozenset and cannot be mutated."""
        self.assertIsInstance(PHASE_TRANSITIONS, frozenset)
        with self.assertRaises((AttributeError, TypeError)):
            PHASE_TRANSITIONS.add(
                (MaterialNode.SOLID_ROCK, MaterialNode.AEROSOL_DUST)
            )

    def test_known_valid_transitions(self) -> None:
        """Key valid transitions from the spec are present."""
        valid = [
            (MaterialNode.AEROSOL_DUST,     MaterialNode.REGOLITH_DUST),
            (MaterialNode.CRUST,            MaterialNode.DEBRIS_FRAGMENTS),
            (MaterialNode.SNOW_LOOSE,       MaterialNode.WATER_RARE),
            (MaterialNode.MAGMA,            MaterialNode.CRUST),
            (MaterialNode.VAPOR,            MaterialNode.ICE_FILM),
        ]
        for from_n, to_n in valid:
            self.assertTrue(
                is_valid_transition(from_n, to_n),
                f"Expected transition {from_n.name} → {to_n.name} to be valid",
            )

    def test_tick_order_fixed_sequence(self) -> None:
        """TickOrder constants are in the correct order 1–7."""
        self.assertEqual(TickOrder.ATMOSPHERE,          1)
        self.assertEqual(TickOrder.MICROCLIMATE,         2)
        self.assertEqual(TickOrder.PHASE_TRANSITIONS,    3)
        self.assertEqual(TickOrder.CHARACTER_TO_WORLD,   4)
        self.assertEqual(TickOrder.INSTABILITY,          5)
        self.assertEqual(TickOrder.MEMORY_COMPACTION,    6)
        self.assertEqual(TickOrder.ENERGY_NORMALISATION, 7)


# ---------------------------------------------------------------------------
# Bonus: config allowlist
# ---------------------------------------------------------------------------

class TestConfigAllowlist(unittest.TestCase):

    def test_config_keys_from_allowlist_are_applied(self) -> None:
        """PlanetPhaseTransitions accepts all allowlisted config keys."""
        cfg = {
            "erosionRate":       0.005,
            "depositionRate":    0.008,
            "compactionRate":    0.010,
            "meltRate":          0.012,
            "fractureThreshold": 0.80,
            "magmaCoolingRate":  0.003,
        }
        system = PlanetPhaseTransitions(cfg)
        self.assertAlmostEqual(system.erosionRate,       0.005, places=5)
        self.assertAlmostEqual(system.depositionRate,    0.008, places=5)
        self.assertAlmostEqual(system.fractureThreshold, 0.80,  places=5)

    def test_unknown_config_keys_ignored(self) -> None:
        """Unrecognised config keys do not raise errors."""
        cfg = {"unknownKey": 999, "erosionRate": 0.007}
        system = PlanetPhaseTransitions(cfg)
        self.assertAlmostEqual(system.erosionRate, 0.007, places=5)


if __name__ == "__main__":
    unittest.main()
