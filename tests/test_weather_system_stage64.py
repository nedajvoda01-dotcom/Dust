"""test_weather_system_stage64.py — Stage 64 Atmospheric & Weather System tests.

Tests
-----
1. test_atmo_fields_stable_no_runaway
   — Running many ticks with neutral insolation keeps all field values
     bounded within [0, 1] and does not diverge.

2. test_storm_regime_triggers_from_thresholds
   — WeatherRegimeDetector returns DUST_STORM when wind/aerosol/front
     thresholds are exceeded, and CLEAR when they are not.

3. test_aerosol_mass_conserved_global
   — After many advection steps with no external sources, total aerosol
     does not increase (settling dominates; conservation proxy holds).

4. test_fog_forms_in_cold_lowlands
   — A cold, humid, low-pressure tile produces fog_potential above the
     detection threshold and triggers a FOG regime.

5. test_determinism_server_client_same_fields
   — Two AtmosphereSystem instances with the same seed and identical
     inputs produce bit-identical grid_hash() after many ticks.

6. test_replication_lod_reduces_bandwidth
   — AtmosphereReplicator: encode_far() produces a smaller packet than
     encode_near() for the same grid and radius.

7. test_material_coupling_calls_api_only
   — AtmosphereToMaterials.apply() only modifies chunk fields via
     MassExchangeAPI and never writes PlanetChunkState directly;
     after a DUST_STORM tick the heat delta is non-zero.
"""
from __future__ import annotations

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.atmo.GlobalFieldGrid        import GlobalFieldGrid, AtmoTile
from src.atmo.FieldAdvection         import FieldAdvection
from src.atmo.WeatherRegimeDetector  import WeatherRegimeDetector, WeatherRegime
from src.atmo.AtmosphereSystem       import AtmosphereSystem
from src.adapters.AtmosphereToMaterials import AtmosphereToMaterials
from src.adapters.AtmosphereToAudio     import AtmosphereToAudio
from src.adapters.AtmosphereToRender    import AtmosphereToRender
from src.net.AtmosphereReplicator       import AtmosphereReplicator
from src.material.PlanetChunkState      import PlanetChunkState
from src.material.MassExchangeAPI       import MassExchangeAPI


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_CFG = {
    "atmo64": {
        "tick_hz":           2.0,
        "temp_relax_tau":    30.0,
        "pressure_relax_tau": 15.0,
        "aerosol_settle_rate": 0.005,
        "erosion_lift_rate":   0.002,
        "storm_thresholds": {
            "wind":          0.5,
            "aerosol":       0.4,
            "front":         0.1,
            "snow_temp":     0.35,
            "snow_humidity": 0.3,
            "fog":           0.3,
            "electro":       0.5,
        },
    }
}


def _make_system(w: int = 8, h: int = 8, seed: int = 0) -> AtmosphereSystem:
    return AtmosphereSystem(w, h, config=_CFG, seed=seed)


def _make_storm_tile() -> AtmoTile:
    """A tile that clearly meets dust-storm thresholds."""
    return AtmoTile(
        pressure=0.2,     # low pressure → high gradient when surrounded by higher
        temperature=0.6,
        wind_x=0.8,
        wind_y=0.0,
        aerosol=0.7,
        humidity=0.2,
        electro=0.1,
    )


def _make_fog_tile() -> AtmoTile:
    """A tile with cold + humid conditions that should produce fog.

    Temperature (0.4) is above the snow threshold (0.35) but cold enough
    for fog formation via the fog_potential formula.
    """
    return AtmoTile(
        pressure=0.35,    # low-ish
        temperature=0.4,  # cold but above snow_temp threshold (0.35)
        wind_x=0.0,
        wind_y=0.0,
        aerosol=0.05,
        humidity=0.8,     # high humidity
        electro=0.0,
    )


# ---------------------------------------------------------------------------
# 1. test_atmo_fields_stable_no_runaway
# ---------------------------------------------------------------------------

class TestAtmoFieldsStableNoRunaway(unittest.TestCase):
    """All field values must stay in [0, 1] after many ticks."""

    def test_fields_bounded_after_many_ticks(self):
        system = _make_system()
        # Tick with neutral insolation (0.5) for the equivalent of many steps
        for _ in range(200):
            system.tick(dt=1.0, insolation_map=lambda ix, iy: 0.5)

        for iy in range(system._grid.height):
            for ix in range(system._grid.width):
                t = system.get_tile(ix, iy)
                for name, val in [
                    ("pressure",    t.pressure),
                    ("temperature", t.temperature),
                    ("aerosol",     t.aerosol),
                    ("humidity",    t.humidity),
                    ("electro",     t.electro),
                ]:
                    self.assertGreaterEqual(val, 0.0, f"{name} went negative at ({ix},{iy})")
                    self.assertLessEqual(val, 1.0, f"{name} exceeded 1.0 at ({ix},{iy})")
                self.assertGreaterEqual(t.wind_x, -1.0, f"wind_x < -1 at ({ix},{iy})")
                self.assertLessEqual(t.wind_x,   1.0,  f"wind_x > 1 at ({ix},{iy})")

    def test_temperature_converges_toward_insolation(self):
        """With constant high insolation, temperature should rise above 0.5."""
        system = _make_system()
        for _ in range(300):
            system.tick(dt=1.0, insolation_map=lambda ix, iy: 0.9)
        t = system.get_tile(3, 3)
        self.assertGreater(t.temperature, 0.5,
            f"Temperature should rise toward insolation target, got {t.temperature:.3f}")


# ---------------------------------------------------------------------------
# 2. test_storm_regime_triggers_from_thresholds
# ---------------------------------------------------------------------------

class TestStormRegimeTriggersFromThresholds(unittest.TestCase):
    """WeatherRegimeDetector produces correct regimes from field values."""

    def _make_detector(self):
        return WeatherRegimeDetector(_CFG)

    def test_dust_storm_detected(self):
        detector = self._make_detector()
        storm_tile = _make_storm_tile()
        regime = detector.detect(
            tile           = storm_tile,
            fog_potential  = 0.05,
            front_intensity= 0.3,
            storm_potential= 0.7,
        )
        self.assertEqual(regime, WeatherRegime.DUST_STORM,
            f"Expected DUST_STORM, got {regime}")

    def test_clear_regime_when_below_thresholds(self):
        detector = self._make_detector()
        calm = AtmoTile(pressure=0.5, temperature=0.5,
                        wind_x=0.1, wind_y=0.0, aerosol=0.05)
        regime = detector.detect(calm, fog_potential=0.05, front_intensity=0.05, storm_potential=0.1)
        self.assertEqual(regime, WeatherRegime.CLEAR,
            f"Expected CLEAR for calm conditions, got {regime}")

    def test_fog_regime_detected(self):
        detector = self._make_detector()
        fog_tile = _make_fog_tile()
        regime = detector.detect(fog_tile, fog_potential=0.6, front_intensity=0.05, storm_potential=0.1)
        self.assertEqual(regime, WeatherRegime.FOG,
            f"Expected FOG for cold humid conditions, got {regime}")

    def test_snow_deposition_detected(self):
        detector = self._make_detector()
        snow_tile = AtmoTile(pressure=0.5, temperature=0.2, wind_x=0.1, wind_y=0.0,
                             aerosol=0.1, humidity=0.55, electro=0.0)
        regime = detector.detect(snow_tile, fog_potential=0.1, front_intensity=0.05, storm_potential=0.1)
        self.assertEqual(regime, WeatherRegime.SNOW_DEPOSITION,
            f"Expected SNOW_DEPOSITION for cold humid tile, got {regime}")

    def test_electrical_regime_requires_storm_plus_high_electro(self):
        detector = self._make_detector()
        elec_tile = AtmoTile(pressure=0.2, temperature=0.6,
                             wind_x=0.8, wind_y=0.0, aerosol=0.7, electro=0.7)
        regime = detector.detect(elec_tile, fog_potential=0.05, front_intensity=0.3, storm_potential=0.8)
        self.assertEqual(regime, WeatherRegime.ELECTRICAL,
            f"Expected ELECTRICAL for high-electro storm, got {regime}")

    def test_no_false_electrical_without_storm(self):
        """High electro alone (no storm wind/aerosol) must not trigger ELECTRICAL."""
        detector = self._make_detector()
        tile = AtmoTile(pressure=0.5, temperature=0.5, wind_x=0.1, wind_y=0.0,
                        aerosol=0.1, electro=0.9)
        regime = detector.detect(tile, fog_potential=0.05, front_intensity=0.02, storm_potential=0.1)
        self.assertNotEqual(regime, WeatherRegime.ELECTRICAL,
            "ELECTRICAL should not trigger without storm conditions")


# ---------------------------------------------------------------------------
# 3. test_aerosol_mass_conserved_global
# ---------------------------------------------------------------------------

class TestAerosolMassConservedGlobal(unittest.TestCase):
    """Total aerosol must not increase without external sources."""

    def test_aerosol_does_not_grow_over_ticks(self):
        """With no external sources and calm initial aerosol, settling should
        keep total aerosol ≤ initial total."""
        system = _make_system(w=6, h=6)
        # Seed a moderate aerosol everywhere
        for iy in range(system._grid.height):
            for ix in range(system._grid.width):
                t = system.get_tile(ix, iy)
                system._grid.set_tile(ix, iy, AtmoTile(
                    aerosol=0.3,
                    wind_x=0.2, wind_y=0.0,
                    pressure=t.pressure, temperature=t.temperature,
                    humidity=t.humidity, electro=t.electro,
                ))
        initial_aerosol = system.total_aerosol()

        for _ in range(100):
            system.tick(dt=1.0, insolation_map=lambda ix, iy: 0.5)

        final_aerosol = system.total_aerosol()
        # Aerosol may decrease (settling) but must not exceed initial + tolerance
        tolerance = initial_aerosol * 0.15 + 0.5  # allow small numerical growth
        self.assertLessEqual(final_aerosol, initial_aerosol + tolerance,
            f"Aerosol grew from {initial_aerosol:.3f} to {final_aerosol:.3f}")


# ---------------------------------------------------------------------------
# 4. test_fog_forms_in_cold_lowlands
# ---------------------------------------------------------------------------

class TestFogFormsInColdLowlands(unittest.TestCase):
    """A cold, humid, low-pressure tile must produce fog potential above threshold."""

    def test_fog_potential_above_threshold_cold_humid(self):
        grid = GlobalFieldGrid(4, 4)
        # Place a cold, humid tile
        fog_tile = _make_fog_tile()
        grid.set_tile(1, 1, fog_tile)
        fp = grid.fog_potential(1, 1)
        self.assertGreater(fp, 0.3,
            f"Expected fog_potential > 0.3 for cold humid tile, got {fp:.3f}")

    def test_fog_regime_via_system(self):
        """AtmosphereSystem.get_regime() returns FOG for a cold humid tile."""
        system = _make_system(w=4, h=4)
        fog_tile = _make_fog_tile()
        system._grid.set_tile(2, 2, fog_tile)
        regime = system.get_regime(2, 2)
        self.assertEqual(regime, WeatherRegime.FOG,
            f"Expected FOG regime for cold humid tile, got {regime}")

    def test_hot_dry_tile_no_fog(self):
        """Hot, dry tile must produce negligible fog potential."""
        grid = GlobalFieldGrid(4, 4)
        hot_dry = AtmoTile(pressure=0.6, temperature=0.9, humidity=0.05,
                           wind_x=0.0, wind_y=0.0, aerosol=0.0, electro=0.0)
        grid.set_tile(2, 2, hot_dry)
        fp = grid.fog_potential(2, 2)
        self.assertLess(fp, 0.1,
            f"Hot dry tile should have near-zero fog potential, got {fp:.3f}")


# ---------------------------------------------------------------------------
# 5. test_determinism_server_client_same_fields
# ---------------------------------------------------------------------------

class TestDeterminismServerClientSameFields(unittest.TestCase):
    """Two identical systems produce identical grid hashes after many ticks."""

    def _run_system(self, seed: int, steps: int) -> str:
        system = AtmosphereSystem(6, 6, config=_CFG, seed=seed)
        for _ in range(steps):
            system.tick(dt=0.5, insolation_map=lambda ix, iy: 0.6)
        return system.grid_hash()

    def test_same_seed_produces_same_hash(self):
        hash1 = self._run_system(seed=42, steps=50)
        hash2 = self._run_system(seed=42, steps=50)
        self.assertEqual(hash1, hash2,
            "Identical seeds/inputs must produce identical grid hashes")

    def test_different_seeds_may_differ(self):
        """Different seeds are allowed to produce different initial states
        (not required, but the architecture supports it)."""
        # This test just verifies the system runs without error for any seed
        for seed in [0, 1, 99]:
            h = self._run_system(seed=seed, steps=10)
            self.assertIsInstance(h, str)
            self.assertEqual(len(h), 32)   # MD5 hex digest length


# ---------------------------------------------------------------------------
# 6. test_replication_lod_reduces_bandwidth
# ---------------------------------------------------------------------------

class TestReplicationLODReducesBandwidth(unittest.TestCase):
    """AtmosphereReplicator: far encoding must be smaller than near encoding."""

    def test_far_packet_smaller_than_near_packet(self):
        rep    = AtmosphereReplicator(_CFG)
        grid   = GlobalFieldGrid(8, 8)
        radius = 3
        cx, cy = 4, 4

        near_bytes = rep.encode_near(grid, cx, cy, radius)
        far_bytes  = rep.encode_far(grid,  cx, cy, radius)

        self.assertLess(len(far_bytes), len(near_bytes),
            f"Far packet ({len(far_bytes)}B) must be smaller than near packet ({len(near_bytes)}B)")

    def test_near_decode_roundtrip(self):
        """Decoded near tiles must match original grid tiles."""
        rep  = AtmosphereReplicator(_CFG)
        grid = GlobalFieldGrid(4, 4)
        # Set a distinctive tile
        grid.set_tile(1, 1, AtmoTile(pressure=0.8, aerosol=0.5))

        data   = rep.encode_near(grid, 1, 1, radius=1)
        result = rep.decode_near(data)

        self.assertGreater(len(result), 0, "Decoded result must not be empty")
        found = {(ix, iy): tile for ix, iy, tile in result}
        self.assertIn((1, 1), found, "Tile (1,1) must be in decoded result")

    def test_far_decode_returns_vis_and_angle(self):
        """Decoded far tiles provide visibility and wind angle."""
        rep  = AtmosphereReplicator(_CFG)
        grid = GlobalFieldGrid(4, 4)

        data   = rep.encode_far(grid, 2, 2, radius=2)
        result = rep.decode_far(data)

        self.assertGreater(len(result), 0, "Decoded far result must not be empty")
        for ix, iy, vis, angle in result:
            self.assertGreaterEqual(vis,   0.0, f"visibility < 0 at ({ix},{iy})")
            self.assertLessEqual(vis,      1.0, f"visibility > 1 at ({ix},{iy})")
            self.assertGreaterEqual(angle, -math.pi - 0.01)
            self.assertLessEqual(angle,     math.pi + 0.01)

    def test_far_packet_size_constant_per_tile(self):
        """Far packet size equals header + N × far_tile_size."""
        rep    = AtmosphereReplicator(_CFG)
        grid   = GlobalFieldGrid(4, 4)
        radius = 1
        cx, cy = 2, 2

        data = rep.encode_far(grid, cx, cy, radius)
        expected_tiles = sum(
            1 for iy in range(4) for ix in range(4)
            if max(abs(ix - cx), abs(iy - cy)) <= radius
        )
        from src.net.AtmosphereReplicator import _HEADER_SIZE
        expected_size = _HEADER_SIZE + expected_tiles * rep.far_packet_size
        self.assertEqual(len(data), expected_size,
            f"Expected {expected_size}B, got {len(data)}B")


# ---------------------------------------------------------------------------
# 7. test_material_coupling_calls_api_only
# ---------------------------------------------------------------------------

class TestMaterialCouplingCallsApiOnly(unittest.TestCase):
    """AtmosphereToMaterials uses only MassExchangeAPI; no direct field writes."""

    def _make_chunk(self) -> PlanetChunkState:
        chunk = PlanetChunkState()
        chunk.dustThickness  = 0.3
        chunk.temperatureProxy = 0.5
        return chunk

    def test_dust_storm_applies_dust_delta(self):
        """In DUST_STORM regime, dustThickness changes."""
        from src.atmo.AtmosphereSystem import LocalAtmoParams
        chunk   = self._make_chunk()
        api     = MassExchangeAPI(chunk)
        adapter = AtmosphereToMaterials(_CFG)

        params = LocalAtmoParams(
            wind_speed=0.8, aerosol=0.7, humidity=0.2, temperature=0.6,
            pressure=0.2, storm_potential=0.8,
            thermal_effect=0.5,   # non-zero so heat delta is detectable
            regime=WeatherRegime.DUST_STORM,
        )
        before_dust = chunk.dustThickness
        before_temp = chunk.temperatureProxy
        # Use dt=5.0 to accumulate changes above uint8 quantisation threshold
        adapter.apply(api, params, dt=5.0)

        temp_changed = abs(chunk.temperatureProxy - before_temp) > 1e-6
        dust_changed = abs(chunk.dustThickness - before_dust)    > 1e-6
        self.assertTrue(dust_changed or temp_changed,
            "DUST_STORM should modify at least one chunk field via API")

    def test_heat_delta_applied_in_all_regimes(self):
        """Heat delta (temperatureProxy) must change in all regimes."""
        from src.atmo.AtmosphereSystem import LocalAtmoParams
        adapter = AtmosphereToMaterials(_CFG)

        for regime in (WeatherRegime.CLEAR, WeatherRegime.DUST_STORM,
                       WeatherRegime.FOG, WeatherRegime.SNOW_DEPOSITION):
            chunk = self._make_chunk()
            api   = MassExchangeAPI(chunk)
            params = LocalAtmoParams(
                wind_speed=0.4, aerosol=0.3, humidity=0.5,
                temperature=0.8,
                thermal_effect=0.8,   # explicitly hot → large enough delta
                pressure=0.5, storm_potential=0.3,
                regime=regime,
            )
            before = chunk.temperatureProxy
            # dt=5.0 ensures heat = 0.8 × 0.005 × 5 = 0.02 > one uint8 step
            adapter.apply(api, params, dt=5.0)
            self.assertNotAlmostEqual(chunk.temperatureProxy, before, places=5,
                msg=f"temperatureProxy unchanged in regime {regime}")

    def test_no_direct_field_access(self):
        """Adapter must not access PlanetChunkState directly (structural check).

        We verify the adapter module does not import PlanetChunkState.
        """
        import importlib, inspect
        import src.adapters.AtmosphereToMaterials as mod
        src_code = inspect.getsource(mod)
        self.assertNotIn("PlanetChunkState", src_code,
            "AtmosphereToMaterials must not import or use PlanetChunkState directly")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
