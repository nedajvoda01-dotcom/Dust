"""test_observer_stage53.py — Stage 53 Observer Influence Without Agency tests.

Tests
-----
1. test_repeated_steps_increase_local_stress
   — Multiple ContactStressInjector.inject calls raise
     stressAccumulationField above zero on the target tile.

2. test_thermal_footprint_reduces_icefilm_locally
   — ThermalFootprintInjector.inject reduces ice_film on the target cell
     and slightly raises snow_compaction.

3. test_impulse_can_trigger_near_threshold_instability
   — ImpulseToShearInjector.inject on a tile with shearStressField just
     below the instability threshold raises it above the threshold.

4. test_player_cannot_trigger_instability_in_stable_zone
   — Repeated ImpulseToShearInjector.inject calls on a tile with
     shearStressField = 0.0 cannot raise it above max_influence_per_tile.

5. test_influence_decays_over_time
   — After accumulating influence via ContactStressInjector, calling
     InfluenceLimiter.tick with a large dt reduces the accumulated value
     toward zero.

6. test_network_authoritative_player_influence
   — PlayerInfluenceAdapter only mutates state when enabled; when
     disabled (enable=False) all inject calls are no-ops.

7. test_determinism_replay_observer_influence
   — Two independent replays of the same contact/impulse sequence
     produce identical WorldMemoryState.state_hash values.
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.observer.InfluenceLimiter         import InfluenceLimiter
from src.observer.ContactStressInjector    import ContactStressInjector
from src.observer.ThermalFootprintInjector import ThermalFootprintInjector
from src.observer.ImpulseToShearInjector   import ImpulseToShearInjector
from src.observer.PlayerInfluenceAdapter   import PlayerInfluenceAdapter

from src.memory.WorldMemoryState           import WorldMemoryState
from src.material.SurfaceMaterialState     import SurfaceMaterialState
from src.instability.InstabilityState      import InstabilityState
from src.microclimate.MicroclimateState    import MicroclimateState


# ---------------------------------------------------------------------------
# Shared config — aggressive gain coefficients so tests can observe effects
# ---------------------------------------------------------------------------

_W, _H = 8, 4

_CFG = {
    "observer": {
        "enable":                 True,
        "k_player_stress":        0.1,   # fast gain for tests
        "k_thermal_melt":         0.05,  # fast melt for tests
        "k_impulse_shear":        0.08,  # fast shear for tests
        "wind_wake_factor":       0.1,
        "max_influence_per_tile": 0.5,   # generous cap for most tests
        "influence_radius":       2.0,
        "decay_tau":              10.0,  # fast decay for decay test
    }
}

_CFG_STRICT = {
    "observer": {
        **_CFG["observer"],
        "max_influence_per_tile": 0.05,  # tight cap for stability test
    }
}

_CFG_DISABLED = {
    "observer": {
        **_CFG["observer"],
        "enable": False,
    }
}


def _make_memory() -> WorldMemoryState:
    return WorldMemoryState(width=_W, height=_H)


def _make_material(ice: float = 0.8, snow: float = 0.1) -> SurfaceMaterialState:
    return SurfaceMaterialState(ice_film=ice, snow_compaction=snow)


def _make_instability() -> InstabilityState:
    return InstabilityState(width=_W, height=_H)


# ---------------------------------------------------------------------------
# 1. test_repeated_steps_increase_local_stress
# ---------------------------------------------------------------------------

class TestRepeatedStepsIncreaseLocalStress(unittest.TestCase):
    """Repeated footstep contacts must raise stressAccumulationField."""

    def test_repeated_steps_increase_local_stress(self):
        memory  = _make_memory()
        inj     = ContactStressInjector(_CFG)
        tile    = memory.tile(2, 1)

        before = memory.stressAccumulationField[tile]
        for _ in range(10):
            inj.inject(memory, tile, contact_force=0.5, dt=1.0)

        self.assertGreater(
            memory.stressAccumulationField[tile], before,
            "Repeated steps must raise stressAccumulationField above zero",
        )

    def test_zero_force_no_change(self):
        memory = _make_memory()
        inj    = ContactStressInjector(_CFG)
        tile   = memory.tile(0, 0)
        inj.inject(memory, tile, contact_force=0.0, dt=1.0)
        self.assertEqual(memory.stressAccumulationField[tile], 0.0)

    def test_compaction_also_increases(self):
        memory = _make_memory()
        inj    = ContactStressInjector(_CFG)
        tile   = memory.tile(3, 2)
        for _ in range(5):
            inj.inject(memory, tile, contact_force=0.8, dt=1.0)
        self.assertGreater(
            memory.compactionHistoryField[tile], 0.0,
            "Repeated steps must also increase compactionHistoryField",
        )


# ---------------------------------------------------------------------------
# 2. test_thermal_footprint_reduces_icefilm_locally
# ---------------------------------------------------------------------------

class TestThermalFootprintReducesIcefilm(unittest.TestCase):
    """Body heat must reduce ice_film and slightly raise snow_compaction."""

    def test_thermal_footprint_reduces_icefilm_locally(self):
        mat  = _make_material(ice=0.8, snow=0.1)
        inj  = ThermalFootprintInjector(_CFG)
        tile = 0  # arbitrary index for limiter key

        ice_before  = mat.ice_film
        snow_before = mat.snow_compaction

        for _ in range(20):
            inj.inject(mat, tile, body_heat=1.0, dt=1.0)

        self.assertLess(mat.ice_film, ice_before,
                        "Body heat must reduce ice_film")
        self.assertGreater(mat.snow_compaction, snow_before,
                           "Body heat must raise snow_compaction slightly")

    def test_zero_heat_no_change(self):
        mat = _make_material(ice=0.5)
        inj = ThermalFootprintInjector(_CFG)
        # Capture the stored (already quantised) value; we compare against it
        # directly, so any change – however tiny – should fail the test.
        before = mat.ice_film
        inj.inject(mat, 0, body_heat=0.0, dt=1.0)
        self.assertAlmostEqual(mat.ice_film, before, places=9)

    def test_ice_film_does_not_go_negative(self):
        mat = _make_material(ice=0.01)
        inj = ThermalFootprintInjector(_CFG)
        for _ in range(100):
            inj.inject(mat, 0, body_heat=1.0, dt=1.0)
        self.assertGreaterEqual(mat.ice_film, 0.0)


# ---------------------------------------------------------------------------
# 3. test_impulse_can_trigger_near_threshold_instability
# ---------------------------------------------------------------------------

class TestImpulseCanTriggerNearThreshold(unittest.TestCase):
    """An impulse on a near-threshold tile must push shear above threshold."""

    def test_impulse_can_trigger_near_threshold_instability(self):
        instab = _make_instability()
        inj    = ImpulseToShearInjector(_CFG)
        tile   = instab.tile(1, 1)

        # Place tile just below a typical threshold (0.6)
        near_threshold = 0.58
        instab.shearStressField[tile] = near_threshold

        inj.inject(instab, tile, impulse_magnitude=1.0)

        self.assertGreater(
            instab.shearStressField[tile], near_threshold,
            "Impulse must raise shear stress above the near-threshold value",
        )

    def test_zero_impulse_no_change(self):
        instab = _make_instability()
        inj    = ImpulseToShearInjector(_CFG)
        tile   = instab.tile(0, 0)
        instab.shearStressField[tile] = 0.3
        inj.inject(instab, tile, impulse_magnitude=0.0)
        self.assertAlmostEqual(instab.shearStressField[tile], 0.3, places=5)


# ---------------------------------------------------------------------------
# 4. test_player_cannot_trigger_instability_in_stable_zone
# ---------------------------------------------------------------------------

class TestPlayerCannotTriggerInstabilityInStableZone(unittest.TestCase):
    """In a stable zone the accumulated shear must not exceed the cap."""

    def test_player_cannot_trigger_instability_in_stable_zone(self):
        instab = _make_instability()
        inj    = ImpulseToShearInjector(_CFG_STRICT)
        tile   = instab.tile(4, 2)

        # Stable zone: shear starts at zero
        for _ in range(200):
            inj.inject(instab, tile, impulse_magnitude=1.0)

        cap = _CFG_STRICT["observer"]["max_influence_per_tile"]
        self.assertLessEqual(
            instab.shearStressField[tile], cap + 1e-6,
            "Player cannot raise shear above max_influence_per_tile in a stable zone",
        )


# ---------------------------------------------------------------------------
# 5. test_influence_decays_over_time
# ---------------------------------------------------------------------------

class TestInfluenceDecaysOverTime(unittest.TestCase):
    """Accumulated influence must decay toward zero when limiter is ticked."""

    def test_influence_decays_over_time(self):
        limiter = InfluenceLimiter(_CFG)
        tile    = 7

        # Build up some influence
        limiter.record(tile, 0.04)
        initial = limiter.accumulated(tile)
        self.assertGreater(initial, 0.0)

        # Advance time well beyond decay_tau (10 s in test config)
        for _ in range(50):
            limiter.tick(dt=2.0)

        self.assertLess(
            limiter.accumulated(tile), initial,
            "Accumulated influence must decay over time",
        )

    def test_zero_dt_no_decay(self):
        limiter = InfluenceLimiter(_CFG)
        tile    = 3
        limiter.record(tile, 0.03)
        before = limiter.accumulated(tile)
        limiter.tick(dt=0.0)
        self.assertAlmostEqual(limiter.accumulated(tile), before, places=9)

    def test_accumulated_starts_zero(self):
        limiter = InfluenceLimiter(_CFG)
        self.assertEqual(limiter.accumulated(99), 0.0)


# ---------------------------------------------------------------------------
# 6. test_network_authoritative_player_influence
# ---------------------------------------------------------------------------

class TestNetworkAuthoritativePlayerInfluence(unittest.TestCase):
    """When enable=False the adapter must be a complete no-op."""

    def test_disabled_adapter_is_noop(self):
        adapter = PlayerInfluenceAdapter(_CFG_DISABLED)

        memory  = _make_memory()
        mat     = _make_material(ice=0.7)
        instab  = _make_instability()
        mc      = MicroclimateState(windShelter=0.2)

        # Capture the stored (already quantised) value before injection so the
        # no-op check compares against it directly rather than the raw float 0.7.
        mat_ice_before = mat.ice_film

        tile = 5
        adapter.apply_contact(memory,  tile, contact_force=1.0, dt=10.0)
        adapter.apply_thermal(mat,     tile, body_heat=1.0,     dt=10.0)
        adapter.apply_impulse(instab,  tile, impulse_magnitude=1.0)
        adapter.apply_wind_wake(mc)
        adapter.tick(dt=10.0)

        self.assertEqual(memory.stressAccumulationField[tile], 0.0,
                         "Disabled adapter must not modify stressAccumulationField")
        self.assertAlmostEqual(mat.ice_film, mat_ice_before, places=9,
                               msg="Disabled adapter must not modify ice_film")
        self.assertEqual(instab.shearStressField[tile], 0.0,
                         "Disabled adapter must not modify shearStressField")
        self.assertAlmostEqual(mc.windShelter, 0.2, places=5,
                               msg="Disabled adapter must not modify windShelter")

    def test_enabled_adapter_modifies_state(self):
        adapter = PlayerInfluenceAdapter(_CFG)
        memory  = _make_memory()
        tile    = 2
        adapter.apply_contact(memory, tile, contact_force=0.8, dt=1.0)
        self.assertGreater(memory.stressAccumulationField[tile], 0.0,
                           "Enabled adapter must modify stressAccumulationField")

    def test_wind_wake_increases_shelter(self):
        adapter = PlayerInfluenceAdapter(_CFG)
        mc      = MicroclimateState(windShelter=0.3)
        adapter.apply_wind_wake(mc)
        self.assertGreater(mc.windShelter, 0.3,
                           "Wind wake must increase windShelter")


# ---------------------------------------------------------------------------
# 7. test_determinism_replay_observer_influence
# ---------------------------------------------------------------------------

class TestDeterminismReplayObserverInfluence(unittest.TestCase):
    """Two identical replay sequences must produce the same state hash."""

    def _run(self) -> str:
        memory  = _make_memory()
        adapter = PlayerInfluenceAdapter(_CFG)

        contacts = [
            (memory.tile(i % _W, i % _H), (i % 5 + 1) * 0.1, 0.5)
            for i in range(30)
        ]

        for tile, force, dt in contacts:
            adapter.apply_contact(memory, tile, contact_force=force, dt=dt)
            adapter.tick(dt=dt)

        return memory.state_hash()

    def test_determinism_replay_observer_influence(self):
        hash_a = self._run()
        hash_b = self._run()
        self.assertEqual(hash_a, hash_b,
                         "Determinism violated: identical replays produced different hashes")


if __name__ == "__main__":
    unittest.main()
