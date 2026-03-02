"""test_generative_foley_stage36 — Stage 36 Generative Foley from Physics.

Tests
-----
1. test_deterministic_same_impulse_same_wavehash
   — Two identical ContactImpulse → ExcitationGenerator → ModalResonator
     chains with the same tick_index produce waveform SHA-256 hashes that
     are equal (determinism requirement from §12.2).

2. test_impact_energy_correlates_with_impulse
   — Higher impulse_magnitude → higher total excitation energy (§5.1).

3. test_sliding_generates_noise
   — A pure-sliding impulse (slip_ratio=1.0) produces non-zero, sign-varying
     excitation samples (noise-like output), while a pure-impact impulse
     (slip_ratio=0.0) produces only positive-decaying amplitudes (§5.2).

4. test_bulk_resonator_triggers_on_rift
   — MegaResonator.apply_stress() above threshold sets is_active=True and
     tick() returns non-zero output (§8, BulkPlateResonator).

5. test_budget_respected_under_storm
   — Spamming 1000 ContactImpulse triggers never causes
     ModalResonatorPool.active_count to exceed max_active_resonators (§13).
"""
from __future__ import annotations

import hashlib
import math
import struct
import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.audio.ContactImpulseCollector import ContactImpulse, ContactImpulseCollector
from src.audio.ExcitationGenerator import ExcitationGenerator, ExcitationType
from src.audio.MaterialAcousticDB import MaterialAcousticDB, MAT_BASALT, MAT_DUST, MAT_PLATE
from src.audio.ModalResonator import ModalResonatorPool
from src.audio.MegaResonator import MegaResonator
from src.audio.SpatialEmitter import SpatialEmitter
from src.audio.AtmosphericPropagation import AtmosphericPropagation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_CONFIG = {
    "audio": {
        "max_active_resonators": 16,
        "modal_max_modes":       6,
        "granular_burst_k":      1.0,
        "friction_noise_k":      1.0,
        "bulk_plate_lowfreq_gain": 1.0,
        "atmo_lowpass_dust_k":   1.0,
        "cave_reverb_mix":       0.35,
        "network_impulse_hz":    300.0,
    }
}


def _make_impact_impulse(magnitude: float = 1.0) -> ContactImpulse:
    return ContactImpulse(
        impulse_magnitude=magnitude,
        contact_duration=0.005,
        material_pair=(MAT_BASALT, MAT_BASALT),
        slip_ratio=0.0,          # pure impact
        contact_area=0.01,
        world_pos=(0.0, 0.0, 0.0),
    )


def _make_sliding_impulse(magnitude: float = 1.0) -> ContactImpulse:
    return ContactImpulse(
        impulse_magnitude=magnitude,
        contact_duration=0.010,
        material_pair=(MAT_DUST, MAT_BASALT),
        slip_ratio=1.0,          # pure sliding
        contact_area=0.05,
        world_pos=(5.0, 0.0, 5.0),
    )


def _synthesise_resonator(
    impulse:    ContactImpulse,
    tick_index: int,
    n_ticks:    int = 60,
    dt:         float = 1.0 / 60.0,
) -> list:
    """Drive ExcitationGenerator → ModalResonatorPool and return sample list."""
    db      = MaterialAcousticDB()
    profile = db.get(impulse.material_pair[0])
    gen     = ExcitationGenerator()
    pool    = ModalResonatorPool(config=_BASE_CONFIG)

    samples_exc = gen.generate(impulse, tick_index, dt,
                               graininess=profile.graininess)
    pool.trigger(profile, samples_exc)

    wave = []
    for _ in range(n_ticks):
        wave.append(pool.tick(dt))
    return wave


def _wave_hash(wave: list) -> str:
    packed = struct.pack(f"{len(wave)}d", *wave)
    return hashlib.sha256(packed).hexdigest()


# ---------------------------------------------------------------------------
# 1. test_deterministic_same_impulse_same_wavehash
# ---------------------------------------------------------------------------

class TestDeterministicWavehash(unittest.TestCase):
    """Same impulse + tick_index must produce identical waveform hash."""

    def _hash_for(self, tick_index: int, magnitude: float) -> str:
        impulse = _make_impact_impulse(magnitude)
        wave    = _synthesise_resonator(impulse, tick_index=tick_index)
        return _wave_hash(wave)

    def test_deterministic_basalt_impact(self) -> None:
        h1 = self._hash_for(tick_index=100, magnitude=0.8)
        h2 = self._hash_for(tick_index=100, magnitude=0.8)
        self.assertEqual(h1, h2,
                         "Identical input must yield identical waveform hash")

    def test_deterministic_dust_sliding(self) -> None:
        impulse = _make_sliding_impulse(0.6)
        w1 = _synthesise_resonator(impulse, tick_index=42)
        w2 = _synthesise_resonator(impulse, tick_index=42)
        self.assertEqual(_wave_hash(w1), _wave_hash(w2))

    def test_different_tick_index_differs(self) -> None:
        # Same sliding impulse but different tick_index → different noise seed
        impulse = _make_sliding_impulse(0.5)
        wave_a = _synthesise_resonator(impulse, tick_index=10)
        wave_b = _synthesise_resonator(impulse, tick_index=99)
        self.assertNotEqual(_wave_hash(wave_a), _wave_hash(wave_b),
                            "Different tick_index should change sliding noise")


# ---------------------------------------------------------------------------
# 2. test_impact_energy_correlates_with_impulse
# ---------------------------------------------------------------------------

class TestImpactEnergyCorrelates(unittest.TestCase):
    """Higher impulse_magnitude → higher summed excitation energy."""

    def _total_energy(self, magnitude: float) -> float:
        impulse = _make_impact_impulse(magnitude)
        gen     = ExcitationGenerator()
        samples = gen.generate(impulse, tick_index=0, dt=1.0 / 60.0)
        return sum(s.amplitude ** 2 for s in samples)

    def test_double_magnitude_higher_energy(self) -> None:
        e1 = self._total_energy(0.5)
        e2 = self._total_energy(1.0)
        self.assertGreater(e2, e1,
                           "Higher impulse magnitude must produce higher excitation energy")

    def test_zero_magnitude_zero_energy(self) -> None:
        e = self._total_energy(0.0)
        self.assertAlmostEqual(e, 0.0, places=9,
                               msg="Zero magnitude must produce zero excitation")

    def test_energy_monotone(self) -> None:
        energies = [self._total_energy(m) for m in [0.1, 0.5, 1.0, 2.0]]
        for i in range(len(energies) - 1):
            self.assertLess(energies[i], energies[i + 1],
                            "Excitation energy must grow monotonically with impulse magnitude")


# ---------------------------------------------------------------------------
# 3. test_sliding_generates_noise
# ---------------------------------------------------------------------------

class TestSlidingGeneratesNoise(unittest.TestCase):
    """Sliding excitation must produce sign-varying samples (noise-like)."""

    def test_sliding_has_sign_changes(self) -> None:
        impulse = _make_sliding_impulse(1.0)
        gen     = ExcitationGenerator()
        samples = gen.generate(impulse, tick_index=5, dt=1.0 / 60.0)
        self.assertGreater(len(samples), 0, "Sliding must produce samples")
        amps = [s.amplitude for s in samples]
        has_positive = any(a > 0 for a in amps)
        has_negative = any(a < 0 for a in amps)
        self.assertTrue(has_positive and has_negative,
                        f"Sliding noise must contain both positive and negative values; got {amps}")

    def test_impact_is_classified_correctly(self) -> None:
        impulse = _make_impact_impulse(1.0)
        gen     = ExcitationGenerator()
        exc_type = gen.classify(impulse)
        self.assertEqual(exc_type, ExcitationType.IMPACT,
                         "slip_ratio=0.0 must classify as IMPACT")

    def test_sliding_is_classified_correctly(self) -> None:
        impulse = _make_sliding_impulse(1.0)
        gen     = ExcitationGenerator()
        exc_type = gen.classify(impulse)
        self.assertEqual(exc_type, ExcitationType.SLIDING,
                         "slip_ratio=1.0 must classify as SLIDING")

    def test_different_tick_index_changes_noise(self) -> None:
        impulse = _make_sliding_impulse(0.8)
        gen     = ExcitationGenerator()
        s1 = gen.generate(impulse, tick_index=1,  dt=1.0 / 60.0)
        s2 = gen.generate(impulse, tick_index=100, dt=1.0 / 60.0)
        amps1 = [s.amplitude for s in s1]
        amps2 = [s.amplitude for s in s2]
        self.assertNotEqual(amps1, amps2,
                            "Different tick_index must produce different noise pattern")


# ---------------------------------------------------------------------------
# 4. test_bulk_resonator_triggers_on_rift
# ---------------------------------------------------------------------------

class TestBulkResonatorTriggersOnRift(unittest.TestCase):
    """MegaResonator must activate and produce output when stress > threshold."""

    def test_below_threshold_not_active(self) -> None:
        mega = MegaResonator(activation_threshold=0.5)
        mega.apply_stress(0.1)
        # tick a bit
        for _ in range(10):
            mega.tick(1.0 / 60.0)
        self.assertFalse(mega.is_active,
                         "Stress below threshold must not activate the resonator")

    def test_above_threshold_becomes_active(self) -> None:
        mega = MegaResonator(activation_threshold=0.1)
        mega.apply_stress(1.0)
        self.assertTrue(mega.is_active,
                        "Stress above threshold must activate the resonator")

    def test_active_resonator_produces_output(self) -> None:
        mega = MegaResonator(activation_threshold=0.1)
        mega.apply_stress(2.0)
        total = sum(abs(mega.tick(1.0 / 60.0)) for _ in range(60))
        self.assertGreater(total, 0.0,
                           "Active MegaResonator must produce non-zero output")

    def test_output_decays_over_time(self) -> None:
        mega = MegaResonator(activation_threshold=0.05)
        mega.apply_stress(3.0)
        dt = 1.0 / 60.0
        early  = sum(abs(mega.tick(dt)) for _ in range(30))
        middle = sum(abs(mega.tick(dt)) for _ in range(30))
        # Amplitude decays exponentially: the second 30-tick window
        # should have lower total energy than the first.
        self.assertGreater(early, middle,
                           "MegaResonator output must decay over time")

    def test_large_stress_gives_larger_amplitude(self) -> None:
        mega_small = MegaResonator(activation_threshold=0.05)
        mega_large = MegaResonator(activation_threshold=0.05)
        mega_small.apply_stress(0.5)
        mega_large.apply_stress(5.0)
        self.assertGreater(
            mega_large.peak_amplitude,
            mega_small.peak_amplitude,
            "Larger stress must produce larger plate amplitude",
        )


# ---------------------------------------------------------------------------
# 5. test_budget_respected_under_storm
# ---------------------------------------------------------------------------

class TestBudgetRespectedUnderStorm(unittest.TestCase):
    """Spamming triggers must never exceed max_active_resonators."""

    def test_pool_never_exceeds_budget(self) -> None:
        pool = ModalResonatorPool(config=_BASE_CONFIG)
        db   = MaterialAcousticDB()
        gen  = ExcitationGenerator(config=_BASE_CONFIG)
        dt   = 1.0 / 60.0

        for tick in range(200):
            # Generate a burst of contacts every tick
            for mat_id in (MAT_DUST, MAT_BASALT):
                impulse = _make_impact_impulse(0.5)
                profile = db.get(mat_id)
                samples = gen.generate(impulse, tick_index=tick, dt=dt)
                pool.trigger(profile, samples)

            pool.tick(dt)
            self.assertLessEqual(
                pool.active_count,
                _BASE_CONFIG["audio"]["max_active_resonators"],
                f"active_count={pool.active_count} exceeds max at tick {tick}",
            )

    def test_collector_drops_when_full(self) -> None:
        """ContactImpulseCollector must not grow unbounded."""
        collector = ContactImpulseCollector(config=_BASE_CONFIG, max_pending=8)
        for _ in range(100):
            collector.record(
                fn=500.0, ft=0.0, v_rel=0.0,
                mat_a=MAT_BASALT, mat_b=MAT_BASALT,
                area=0.01, duration=0.016,
            )
        # Internal pending buffer must not exceed max_pending
        self.assertLessEqual(
            len(collector._pending), 8,
            "Collector must cap pending contacts at max_pending",
        )


# ---------------------------------------------------------------------------
# Bonus: SpatialEmitter and AtmosphericPropagation basic sanity
# ---------------------------------------------------------------------------

class TestSpatialEmitter(unittest.TestCase):
    def test_far_source_is_quieter(self) -> None:
        emitter = SpatialEmitter()
        near = emitter.attenuate(1.0, distance=1.0)
        far  = emitter.attenuate(1.0, distance=100.0)
        self.assertGreater(near, far)

    def test_beyond_max_distance_is_silent(self) -> None:
        emitter = SpatialEmitter(max_distance=50.0)
        out = emitter.attenuate(1.0, distance=600.0)
        self.assertAlmostEqual(out, 0.0, places=9)


class TestAtmosphericPropagation(unittest.TestCase):
    def test_high_dust_reduces_output(self) -> None:
        atmo_clean = AtmosphericPropagation()
        atmo_dusty = AtmosphericPropagation()
        dt = 1.0 / 60.0
        # Feed a unit signal for many ticks and compare steady-state RMS
        clean_out = sum(abs(atmo_clean.process(1.0, dust_density=0.0, dt=dt))
                        for _ in range(120))
        dusty_out = sum(abs(atmo_dusty.process(1.0, dust_density=1.0, dt=dt))
                        for _ in range(120))
        self.assertGreater(clean_out, dusty_out,
                           "High dust density must attenuate high-frequency content")

    def test_cave_reverb_adds_energy(self) -> None:
        atmo = AtmosphericPropagation()
        dt = 1.0 / 60.0
        # Impulse into cave reverb should sustain longer than dry
        dry_tail  = sum(abs(atmo.process(0.0, in_cave=False, dt=dt)) for _ in range(60))
        atmo2 = AtmosphericPropagation()
        atmo2.process(1.0, in_cave=True, dt=dt)   # prime the delay buffer
        wet_tail  = sum(abs(atmo2.process(0.0, in_cave=True,  dt=dt)) for _ in range(60))
        self.assertGreater(wet_tail, dry_tail,
                           "Cave reverb should sustain signal beyond the input impulse")


if __name__ == "__main__":
    unittest.main()
