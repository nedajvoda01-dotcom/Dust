"""test_energy_stage54.py — Stage 54 Global Energy Budget & Entropy Balance tests.

Tests
-----
1. test_total_dust_mass_conserved
   — After EnergyToEvolutionAdapter.dust_conservation_tick() the total
     dust mass in the field list does not exceed max_dust_mass.

2. test_mechanical_energy_reduced_after_instability
   — After EnergyToInstabilityAdapter.on_crust_failure() the mechanical
     reservoir is lower than before.

3. test_entropy_stays_within_bounds_over_long_run
   — After many repeated energy_balance_tick() calls with high-entropy
     input, planet_entropy stays within [0, 1].

4. test_no_runaway_instability
   — Even with repeated on_crust_failure() calls, mechanical energy
     never grows without bound (capped at max_mech_stress).

5. test_long_term_simulation_stable
   — After 1000 inject_solar / wind_tick / energy_balance_tick cycles
     all reservoir values remain in [0, 1].

6. test_snapshot_restore_energy_state
   — EnergySnapshot.save() + .load() + load_state_dict() round-trips
     reservoirs and planet_entropy with float32 precision.

7. test_determinism_energy_balance
   — Two independent GlobalEnergySystem runs with the same input sequence
     produce identical state_dict() outputs.
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.energy.GlobalEnergySystem         import GlobalEnergySystem
from src.energy.EnergyLedger               import EnergyLedger
from src.energy.EnergyNormalizer            import EnergyNormalizer
from src.energy.EnergyToInstabilityAdapter  import EnergyToInstabilityAdapter
from src.energy.EnergyToEvolutionAdapter    import EnergyToEvolutionAdapter
from src.energy.EnergySnapshot              import EnergySnapshot


# ---------------------------------------------------------------------------
# Shared config
# ---------------------------------------------------------------------------

_CFG = {
    "energy": {
        "enable":               True,
        "tick_hz":              1.0,   # run balance every second for tests
        "max_mech_stress":      0.9,
        "max_dust_mass":        1.0,
        "max_ice_mass":         1.0,
        "entropy_upper_bound":  0.8,
        "entropy_lower_bound":  0.2,
        "auto_normalize_k":     0.05,
        "transfer_efficiency":  0.85,
        "mech_per_crust_event": 0.08,
        "mech_per_dust_event":  0.04,
        "atmo_per_dust_event":  0.03,
        "thermal_per_frac_event": 0.06,
        "mech_from_frac_event": 0.04,
        "min_mech_trigger":     0.02,
        "dust_target_mean":     0.5,
        "ice_thermal_threshold": 0.3,
        "ice_melt_return_k":    0.6,
        "wind_erosion_base":    0.4,
    }
}


def _make_system() -> GlobalEnergySystem:
    return GlobalEnergySystem(_CFG)


# ---------------------------------------------------------------------------
# Dummy instability event
# ---------------------------------------------------------------------------

class _FakeCrustEvent:
    def __init__(self, intensity: float = 0.7):
        self.intensity = intensity


# ---------------------------------------------------------------------------
# 1. test_total_dust_mass_conserved
# ---------------------------------------------------------------------------

class TestDustMassConserved(unittest.TestCase):

    def test_total_dust_mass_conserved(self):
        adapter = EnergyToEvolutionAdapter(_CFG)
        ledger  = EnergyLedger(_CFG)

        # Start with total dust well above max_dust_mass
        dust_fields = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]   # total = 3.3 > 1.0
        total_before = sum(dust_fields)
        self.assertGreater(total_before, 1.0)

        adapter.dust_conservation_tick(ledger, dust_fields, dt=1.0)

        total_after = sum(dust_fields)
        self.assertLessEqual(
            total_after, 1.0 + 1e-9,
            "Dust total must not exceed max_dust_mass after conservation tick",
        )

    def test_dust_unchanged_when_within_budget(self):
        adapter = EnergyToEvolutionAdapter(_CFG)
        ledger  = EnergyLedger(_CFG)

        dust_fields = [0.1, 0.1, 0.1]   # total = 0.3 < 1.0
        original = list(dust_fields)
        adapter.dust_conservation_tick(ledger, dust_fields, dt=1.0)
        self.assertEqual(dust_fields, original,
                         "Dust fields should be unchanged when within budget")


# ---------------------------------------------------------------------------
# 2. test_mechanical_energy_reduced_after_instability
# ---------------------------------------------------------------------------

class TestMechanicalEnergyReducedAfterInstability(unittest.TestCase):

    def test_mechanical_energy_reduced_after_instability(self):
        adapter = EnergyToInstabilityAdapter(_CFG)
        ledger  = EnergyLedger(_CFG)

        # Prime the mechanical reservoir
        ledger.set("mechanical", 0.5)
        mech_before = ledger.get("mechanical")

        event = _FakeCrustEvent(intensity=0.7)
        adapter.on_crust_failure(ledger, event)

        mech_after = ledger.get("mechanical")
        self.assertLess(
            mech_after, mech_before,
            "Mechanical energy must decrease after a crust failure event",
        )

    def test_zero_intensity_no_change(self):
        adapter = EnergyToInstabilityAdapter(_CFG)
        ledger  = EnergyLedger(_CFG)
        ledger.set("mechanical", 0.5)

        adapter.on_crust_failure(ledger, _FakeCrustEvent(intensity=0.0))
        self.assertAlmostEqual(ledger.get("mechanical"), 0.5, places=5)


# ---------------------------------------------------------------------------
# 3. test_entropy_stays_within_bounds_over_long_run
# ---------------------------------------------------------------------------

class TestEntropyBounds(unittest.TestCase):

    def test_entropy_stays_within_bounds_over_long_run(self):
        system = _make_system()

        for _ in range(500):
            system.record_instability_event(intensity=0.9)
            system.inject_solar(0.8)
            system.wind_tick(0.9, dt=1.0)
            # Force balance tick every iteration (tick_hz=1.0, dt=1.0)
            system.energy_balance_tick(dt=1.0)

        entropy = system.planet_entropy
        self.assertGreaterEqual(entropy, 0.0, "Entropy must not go below 0")
        self.assertLessEqual(entropy, 1.0, "Entropy must not exceed 1")

    def test_entropy_decreases_with_smoothing(self):
        system = _make_system()
        # Drive entropy high
        for _ in range(20):
            system.record_instability_event(intensity=1.0)
        high_entropy = system.planet_entropy

        # Then smooth it down
        for _ in range(200):
            system.record_erosion_smoothing(1.0)
            system.energy_balance_tick(dt=1.0)

        self.assertLess(system.planet_entropy, high_entropy,
                        "Erosion smoothing should lower entropy")


# ---------------------------------------------------------------------------
# 4. test_no_runaway_instability
# ---------------------------------------------------------------------------

class TestNoRunawayInstability(unittest.TestCase):

    def test_no_runaway_instability(self):
        system  = _make_system()
        adapter = EnergyToInstabilityAdapter(_CFG)

        # Repeatedly fire crust events AND run balance ticks
        for _ in range(200):
            system.ledger.add("mechanical", 0.05)
            adapter.on_crust_failure(system.ledger, _FakeCrustEvent(0.8))
            system.record_instability_event(0.8)
            system.energy_balance_tick(dt=1.0)

        mech = system.ledger.get("mechanical")
        self.assertLessEqual(
            mech, _CFG["energy"]["max_mech_stress"] + 1e-9,
            f"Mechanical energy must not exceed max_mech_stress; got {mech}",
        )

    def test_cascade_gate_prevents_cascade_when_empty(self):
        adapter = EnergyToInstabilityAdapter(_CFG)
        ledger  = EnergyLedger(_CFG)
        # mechanical is 0 — cascade should be gated
        self.assertFalse(adapter.can_trigger_cascade(ledger),
                         "Cascade should be blocked when mechanical energy is 0")

        ledger.set("mechanical", 0.5)
        self.assertTrue(adapter.can_trigger_cascade(ledger),
                        "Cascade should be allowed when mechanical energy is high")


# ---------------------------------------------------------------------------
# 5. test_long_term_simulation_stable
# ---------------------------------------------------------------------------

class TestLongTermSimulationStable(unittest.TestCase):

    def test_long_term_simulation_stable(self):
        system = _make_system()

        for step in range(1000):
            system.inject_solar(0.6)
            system.wind_tick(0.5, dt=0.1)
            system.record_instability_event(0.1)
            system.record_erosion_smoothing(0.05)
            system.energy_balance_tick(dt=0.1)

        reservoirs = system.ledger.reservoirs()
        for name, val in reservoirs.items():
            self.assertGreaterEqual(val, 0.0,
                                    f"Reservoir '{name}' went below 0")
            self.assertLessEqual(val, 1.0,
                                 f"Reservoir '{name}' exceeded 1")

        self.assertGreaterEqual(system.planet_entropy, 0.0)
        self.assertLessEqual(system.planet_entropy, 1.0)


# ---------------------------------------------------------------------------
# 6. test_snapshot_restore_energy_state
# ---------------------------------------------------------------------------

class TestSnapshotRestore(unittest.TestCase):

    def test_snapshot_restore_energy_state(self):
        system = _make_system()

        # Set non-trivial state
        system.ledger.set("mechanical", 0.42)
        system.ledger.set("thermal",    0.65)
        system.ledger.set("acoustic",   0.17)
        system._planet_entropy = 0.73

        snap = EnergySnapshot()
        blob = snap.save(system, sim_time=99.5)

        state_dict, meta = snap.load(blob)

        self.assertAlmostEqual(meta["sim_time"], 99.5, places=4)
        self.assertEqual(state_dict["type"], "GLOBAL_ENERGY_STATE_54")

        # Restore into a fresh system
        system2 = _make_system()
        system2.load_state_dict(state_dict)

        # float32 precision: max error ≈ 6e-8 for values in [0,1]
        tolerance = 1e-5
        self.assertAlmostEqual(
            system2.ledger.get("mechanical"),
            system.ledger.get("mechanical"),
            delta=tolerance,
        )
        self.assertAlmostEqual(
            system2.ledger.get("thermal"),
            system.ledger.get("thermal"),
            delta=tolerance,
        )
        self.assertAlmostEqual(
            system2.planet_entropy,
            system._planet_entropy,
            delta=tolerance,
        )

    def test_bad_magic_raises(self):
        snap = EnergySnapshot()
        with self.assertRaises(ValueError):
            snap.load(b"XXXX" + b"\x00" * 60)


# ---------------------------------------------------------------------------
# 7. test_determinism_energy_balance
# ---------------------------------------------------------------------------

class TestDeterminismEnergyBalance(unittest.TestCase):

    def _run_simulation(self) -> dict:
        system = _make_system()
        adapter = EnergyToInstabilityAdapter(_CFG)
        evo_adapter = EnergyToEvolutionAdapter(_CFG)
        dust = [0.2, 0.3, 0.4, 0.5, 0.6]

        for step in range(50):
            solar = (step % 7) / 7.0
            wind  = (step % 5) / 5.0
            system.inject_solar(solar)
            system.wind_tick(wind, dt=0.5)
            if step % 3 == 0:
                adapter.on_crust_failure(system.ledger, _FakeCrustEvent(0.6))
                system.record_instability_event(0.6)
            evo_adapter.dust_conservation_tick(system.ledger, dust, dt=0.5)
            system.energy_balance_tick(dt=0.5)

        return system.state_dict()

    def test_determinism_energy_balance(self):
        result_a = self._run_simulation()
        result_b = self._run_simulation()

        self.assertAlmostEqual(
            result_a["planet_entropy"],
            result_b["planet_entropy"],
            places=10,
            msg="Determinism violated: planet_entropy differs",
        )
        for name in result_a["reservoirs"]:
            self.assertAlmostEqual(
                result_a["reservoirs"][name],
                result_b["reservoirs"][name],
                places=10,
                msg=f"Determinism violated: reservoir '{name}' differs",
            )


if __name__ == "__main__":
    unittest.main()
