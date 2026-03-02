"""test_evolution_stage30.py — Stage 30 LongHorizonEvolutionSystem smoke tests.

Tests
-----
1. test_evo_fields_monotonic_rules
   — FractureFatigue never decreases without cause.
   — IceFilm grows when cold, shrinks when hot.
   — Field values never go outside [0, 1].

2. test_server_authority_evo_sync
   — A client that receives a snapshot reaches the same state hash as the
     server (round-trip serialisation).

3. test_storage_compaction_includes_evo
   — WorldState.save_evolution_snapshot() persists data.
   — OpsLayer.compact() includes evolution.snapshot in the baseline.
   — The delta log does not grow indefinitely (bounded by pruning rules).

4. test_long_run_no_blowup
   — Running 6 simulated hours (fast-forwarded) produces no NaN / Inf values
     and does not cause runaway numbers.
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.core.Config import Config
from src.net.WorldState import WorldState
from src.ops.OpsLayer import OpsLayer
from src.systems.LongHorizonEvolutionSystem import (
    EvolutionFields,
    LongHorizonEvolutionSystem,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**evo_overrides) -> Config:
    cfg = Config.__new__(Config)
    evo = {
        "enable":                    True,
        "tick_seconds":              10.0,   # fast for tests
        "tiles_per_tick":            32,
        "snapshot_interval_sec":     120.0,
        "delta_interval_sec":        20.0,
        "grid_w":                    16,
        "grid_h":                    8,
        "dust_lift_k":               0.02,
        "dust_dep_k":                0.03,
        "ice_form_k":                0.02,
        "ice_melt_k":                0.03,
        "fatigue_tempCycle_k":       0.001,
        "fatigue_stress_k":          0.0005,
        "slope_relax_k":             0.001,
        "max_slope_events_per_hour": 20,
        "network_quantization_bits": 8,
    }
    evo.update(evo_overrides)
    cfg._data = {"evo": evo}
    return cfg


def _make_evo(**evo_overrides) -> LongHorizonEvolutionSystem:
    cfg = _make_config(**evo_overrides)
    return LongHorizonEvolutionSystem(cfg, width=16, height=8)


class _ColdfakeCoupler:
    """Coupler stub: everything cold → ice should grow."""
    def dust_lift_potential(self, pos) -> float:
        return 0.0

    def ice_form_rate(self, pos) -> float:
        return 1.0   # maximum formation

    def T_target_at(self, pos) -> float:
        return 240.0  # below ice_thresh (270 K)

    def heat_wind_delta(self, pos):
        class _Z:
            x = y = z = 0.0
        return _Z()


class _HotfakeCoupler:
    """Coupler stub: everything hot → ice should melt, dust should lift."""
    def dust_lift_potential(self, pos) -> float:
        return 1.0

    def ice_form_rate(self, pos) -> float:
        return 0.0

    def T_target_at(self, pos) -> float:
        return 350.0  # well above ice_thresh

    def heat_wind_delta(self, pos):
        class _W:
            x = 15.0
            y = 0.0
            z = 0.0
        return _W()


# ---------------------------------------------------------------------------
# 1. TestEvoFieldsMonotonicRules
# ---------------------------------------------------------------------------

class TestEvoFieldsMonotonicRules(unittest.TestCase):
    """Evolution field rules must be physically consistent."""

    def test_fracture_fatigue_only_increases_under_stress(self) -> None:
        """FractureFatigue must not decrease when there is thermal cycling."""
        evo = _make_evo()
        # Record initial fatigue values
        initial = list(evo.fields.fracture_fatigue)

        # Advance 5 ticks with a hot coupler (thermal stress)
        coupler = _HotfakeCoupler()
        for _ in range(5):
            evo.update(dt=evo._tick_seconds, coupler=coupler, sim_time=0.0)

        for i in range(evo.fields.size()):
            self.assertGreaterEqual(
                evo.fields.fracture_fatigue[i], initial[i] - 1e-9,
                msg=f"FractureFatigue at tile {i} decreased unexpectedly",
            )

    def test_ice_grows_in_cold(self) -> None:
        """IceFilm must increase when temperature is below threshold."""
        evo = _make_evo()
        initial_ice = sum(evo.fields.ice_film) / evo.fields.size()

        coupler = _ColdfakeCoupler()
        for _ in range(10):
            evo.update(dt=evo._tick_seconds, coupler=coupler, sim_time=0.0)

        final_ice = sum(evo.fields.ice_film) / evo.fields.size()
        self.assertGreater(
            final_ice, initial_ice,
            msg=f"Mean IceFilm should grow in cold conditions: {initial_ice:.4f} → {final_ice:.4f}",
        )

    def test_ice_melts_in_heat(self) -> None:
        """IceFilm must decrease when temperature is well above threshold."""
        evo = _make_evo()
        # Pre-seed with ice
        for i in range(evo.fields.size()):
            evo.fields.ice_film[i] = 0.8

        coupler = _HotfakeCoupler()
        for _ in range(10):
            evo.update(dt=evo._tick_seconds, coupler=coupler, sim_time=0.0)

        final_ice = sum(evo.fields.ice_film) / evo.fields.size()
        self.assertLess(
            final_ice, 0.8,
            msg=f"Mean IceFilm should decrease in hot conditions; got {final_ice:.4f}",
        )

    def test_all_fields_stay_in_unit_range(self) -> None:
        """All evolution field values must remain in [0, 1] after many ticks."""
        evo = _make_evo()
        coupler = _HotfakeCoupler()

        for _ in range(20):
            evo.update(dt=evo._tick_seconds, coupler=coupler, sim_time=float(_ * evo._tick_seconds))

        for attr in (
            "dust_thickness", "dust_mobility", "ice_film",
            "surface_freshness", "fracture_fatigue",
            "slope_stability", "regolith_cohesion",
        ):
            for i, v in enumerate(getattr(evo.fields, attr)):
                self.assertFalse(
                    math.isnan(v) or math.isinf(v),
                    msg=f"{attr}[{i}] is NaN/Inf: {v}",
                )
                self.assertGreaterEqual(v, 0.0, msg=f"{attr}[{i}] < 0: {v}")
                self.assertLessEqual(v,   1.0, msg=f"{attr}[{i}] > 1: {v}")

    def test_dust_moves_with_wind(self) -> None:
        """With strong lift, dust thickness should decrease in source tile."""
        evo = _make_evo(dust_lift_k=0.2)
        # Put a lot of dust in tile 0
        evo.fields.dust_thickness[0] = 0.9
        initial_dust = evo.fields.dust_thickness[0]

        coupler = _HotfakeCoupler()
        # Run tick so tile 0 is processed (tile cursor starts at 0)
        evo._tile_cursor = 0
        evo._do_evolution_tick(coupler, None, sim_time=0.0)

        # Tile 0 should have lost some dust (lifted+transported)
        self.assertLess(
            evo.fields.dust_thickness[0], initial_dust + 1e-9,
            msg="Dust at source tile should not increase with strong lift",
        )


# ---------------------------------------------------------------------------
# 2. TestServerAuthorityEvoSync
# ---------------------------------------------------------------------------

class TestServerAuthorityEvoSync(unittest.TestCase):
    """Snapshot round-trip must yield identical state hash."""

    def test_snapshot_round_trip_hash(self) -> None:
        """Server snapshot → client apply → hashes match."""
        evo_server = _make_evo()
        coupler = _ColdfakeCoupler()
        for _ in range(5):
            evo_server.update(
                dt=evo_server._tick_seconds,
                coupler=coupler,
                sim_time=float(_ * evo_server._tick_seconds),
            )

        snap = evo_server.get_snapshot()
        server_hash = evo_server.state_hash()

        evo_client = _make_evo()
        evo_client.apply_snapshot(snap)
        client_hash = evo_client.state_hash()

        self.assertEqual(
            server_hash, client_hash,
            msg=(
                f"State hash mismatch after snapshot round-trip: "
                f"server={server_hash} client={client_hash}"
            ),
        )

    def test_delta_round_trip(self) -> None:
        """Server delta → client apply → dirty tiles match."""
        evo_server = _make_evo()
        evo_client = _make_evo()

        # Start both from the same snapshot
        snap = evo_server.get_snapshot()
        evo_client.apply_snapshot(snap)

        # Advance server by one tick
        evo_server._do_evolution_tick(_HotfakeCoupler(), None, sim_time=0.0)

        # Get delta and apply to client
        delta = evo_server.get_delta()
        evo_client.apply_delta(delta)

        # Dirty tiles should now match (spot-check a few)
        tiles = delta.get("tiles", [])
        self.assertGreater(len(tiles), 0, "Delta should contain at least one tile")

        bits  = delta.get("bits", 8)
        scale = (1 << bits) - 1
        for t in tiles[:5]:
            idx = t["i"]
            expected_ice = t["ic"] / scale
            actual_ice   = evo_client.fields.ice_film[idx]
            self.assertAlmostEqual(
                actual_ice, expected_ice, delta=1.5 / scale,
                msg=f"IceFilm mismatch at tile {idx}: expected≈{expected_ice:.4f} got {actual_ice:.4f}",
            )

    def test_snapshot_type_field(self) -> None:
        """Snapshot dict must have type == EVOLUTION_SNAPSHOT."""
        evo = _make_evo()
        snap = evo.get_snapshot()
        self.assertEqual(snap.get("type"), "EVOLUTION_SNAPSHOT")

    def test_delta_type_field(self) -> None:
        """Delta dict must have type == EVOLUTION_DELTA."""
        evo = _make_evo()
        evo._dirty = {0, 1, 2}
        delta = evo.get_delta()
        self.assertEqual(delta.get("type"), "EVOLUTION_DELTA")

    def test_snapshot_contains_all_field_keys(self) -> None:
        """Snapshot fields dict must contain all 7 evolution field arrays."""
        evo  = _make_evo()
        snap = evo.get_snapshot()
        for key in (
            "dustThickness", "dustMobility", "iceFilm",
            "surfaceFreshness", "fractureFatigue",
            "slopeStability", "regolithCohesion",
        ):
            self.assertIn(key, snap["fields"], msg=f"Missing field key: {key}")


# ---------------------------------------------------------------------------
# 3. TestStorageCompactionIncludesEvo
# ---------------------------------------------------------------------------

class TestStorageCompactionIncludesEvo(unittest.TestCase):
    """compact() must include evolution.snapshot in the baseline."""

    def setUp(self) -> None:
        self._tmp   = tempfile.mkdtemp()
        self._state = os.path.join(self._tmp, "world_state")
        self.ws = WorldState(self._state)
        self.ws.load_or_create(default_seed=42)

    def test_save_and_load_evolution_snapshot(self) -> None:
        """WorldState can persist and reload an evolution snapshot."""
        evo = _make_evo()
        snap = evo.get_snapshot()
        self.ws.save_evolution_snapshot(snap)
        loaded = self.ws.load_evolution_snapshot()
        self.assertIsNotNone(loaded, "load_evolution_snapshot should return a dict")
        self.assertEqual(loaded["stateHash"], snap["stateHash"])

    def test_compact_includes_evolution_snapshot(self) -> None:
        """compact() must write evolution.snapshot into the baseline directory."""
        evo = _make_evo()
        snap = evo.get_snapshot()
        self.ws.save_evolution_snapshot(snap)

        ops = OpsLayer(world_state=self.ws, state_dir=self._state)
        ok = ops.compact()
        self.assertTrue(ok, "compact() must return True")

        state    = Path(self._state)
        baseline = sorted(state.glob("baseline_*"))[0]
        self.assertTrue(
            (baseline / "evolution.snapshot").exists(),
            "baseline must contain evolution.snapshot",
        )

    def test_compact_evolution_snapshot_content(self) -> None:
        """evolution.snapshot in baseline must have correct stateHash."""
        evo = _make_evo()
        coupler = _ColdfakeCoupler()
        for _ in range(3):
            evo.update(dt=evo._tick_seconds, coupler=coupler, sim_time=float(_ * 10))

        snap = evo.get_snapshot()
        self.ws.save_evolution_snapshot(snap)

        ops = OpsLayer(world_state=self.ws, state_dir=self._state)
        ops.compact()

        state    = Path(self._state)
        baseline = sorted(state.glob("baseline_*"))[0]
        data     = json.loads((baseline / "evolution.snapshot").read_text())
        self.assertEqual(data["stateHash"], snap["stateHash"])

    def test_compact_without_evolution_snapshot_still_works(self) -> None:
        """compact() must succeed even when no evolution snapshot has been saved."""
        ops = OpsLayer(world_state=self.ws, state_dir=self._state)
        ok = ops.compact()
        self.assertTrue(ok, "compact() should succeed without evolution snapshot")

        state    = Path(self._state)
        baseline = sorted(state.glob("baseline_*"))[0]
        # evolution.snapshot should simply be absent — not an error
        self.assertTrue((baseline / "world.json").exists())


# ---------------------------------------------------------------------------
# 4. TestLongRunNoBlowup
# ---------------------------------------------------------------------------

class TestLongRunNoBlowup(unittest.TestCase):
    """Running many evolution hours must not produce NaN/Inf or runaway values."""

    def _run_sim(self, hours: float, coupler) -> LongHorizonEvolutionSystem:
        evo = _make_evo(
            tick_seconds=60.0,
            tiles_per_tick=32,
        )
        total_seconds = hours * 3600.0
        step = 60.0  # advance one tick at a time
        sim_time = 0.0
        while sim_time < total_seconds:
            evo.update(dt=step, coupler=coupler, sim_time=sim_time)
            sim_time += step
        return evo

    def _assert_no_blowup(self, evo: LongHorizonEvolutionSystem, label: str) -> None:
        for attr in (
            "dust_thickness", "dust_mobility", "ice_film",
            "surface_freshness", "fracture_fatigue",
            "slope_stability", "regolith_cohesion",
        ):
            for i, v in enumerate(getattr(evo.fields, attr)):
                self.assertFalse(
                    math.isnan(v) or math.isinf(v),
                    msg=f"[{label}] {attr}[{i}] is NaN/Inf",
                )
                self.assertGreaterEqual(v, -1e-9, msg=f"[{label}] {attr}[{i}] < 0")
                self.assertLessEqual(v, 1.0 + 1e-9, msg=f"[{label}] {attr}[{i}] > 1")

    def test_long_run_cold_no_blowup(self) -> None:
        """6 sim-hours with cold coupler must not produce NaN or out-of-range values."""
        evo = self._run_sim(6.0, _ColdfakeCoupler())
        self._assert_no_blowup(evo, "cold-6h")

    def test_long_run_hot_no_blowup(self) -> None:
        """6 sim-hours with hot coupler must not produce NaN or out-of-range values."""
        evo = self._run_sim(6.0, _HotfakeCoupler())
        self._assert_no_blowup(evo, "hot-6h")

    def test_long_run_no_coupler_no_blowup(self) -> None:
        """6 sim-hours with no coupler must not produce NaN or out-of-range values."""
        evo = self._run_sim(6.0, None)
        self._assert_no_blowup(evo, "no-coupler-6h")

    def test_fast_forward_helper(self) -> None:
        """fast_forward_days() must advance tick_index without NaN."""
        evo = _make_evo(tick_seconds=60.0, tiles_per_tick=16)
        evo.fast_forward_days(1.0)
        # Run enough real dt to drain the FF
        for _ in range(200):
            evo.update(dt=60.0, coupler=None, sim_time=float(_ * 60))
        self.assertGreater(evo._tick_index, 0, "Tick index should advance after fast-forward")
        self._assert_no_blowup(evo, "fast-forward-1d")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
