"""test_acoustic_world_stage46 — Stage 46 Long-range Acoustic World Model.

Tests
-----
1. test_distance_attenuation_monotonic
   — PropagationModel gain decreases strictly as distance increases.

2. test_dust_reduces_high_freq_and_snr
   — High dust_density lowers lp_cutoff_norm and snr compared to clean air.

3. test_occlusion_blocks_audible_more_than_infra
   — A fully occluded source has audible gain reduced much more than infra gain.

4. test_valley_gain_increases_range_in_concave_areas
   — ValleyConcavityProxy returns higher valley_gain for a concave position
     than for a flat/convex position.

5. test_infra_field_persists_and_decays
   — InfraField retains injected energy for several ticks before decaying.

6. test_budget_limits_respected
   — EmitterAggregator never exceeds max_emitters active records.
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.math.Vec3 import Vec3
from src.audio.audio_world.PropagationModel       import PropagationModel
from src.audio.audio_world.OcclusionCache         import OcclusionCache
from src.audio.audio_world.ValleyConcavityProxy   import ValleyConcavityProxy
from src.audio.audio_world.InfraField             import InfraField
from src.audio.audio_world.EmitterAggregator      import (
    EmitterAggregator, AcousticEmitterRecord, EmitterType,
)
from src.audio.audio_world.AtmosphereAcousticAdapter import AtmosphereAcousticAdapter
from src.perception.AudioWorldAdapter             import AudioWorldAdapter
from src.net.EmitterReplicator                    import EmitterReplicator


# ---------------------------------------------------------------------------
# Config fixture
# ---------------------------------------------------------------------------

_CFG = {
    "audio_world": {
        "max_emitters":           32,
        "max_raycasts_per_sec":   50,
        "occlusion_cache_ttl_sec": 0.5,
        "valley_gain_min":        0.8,
        "valley_gain_max":        1.3,
        "dust_lowpass_k":         2.0,
        "wind_snr_k":             1.5,
        "infra_decay_tau":        4.0,
        "lod": {"near_dist": 50.0, "mid_dist": 200.0, "far_update_hz": 3.0},
    }
}


# ---------------------------------------------------------------------------
# 1. test_distance_attenuation_monotonic
# ---------------------------------------------------------------------------

class TestDistanceAttenuationMonotonic(unittest.TestCase):
    """PropagationModel gain must decrease (or stay equal) as distance grows."""

    def test_audible_gain_monotonic(self):
        model = PropagationModel(_CFG)
        distances = [1, 10, 50, 100, 300, 600, 1000]
        gains = [
            model.propagate(d, band_energy_audible=1.0).gain_audible
            for d in distances
        ]
        for i in range(len(gains) - 1):
            self.assertGreaterEqual(
                gains[i], gains[i + 1],
                f"gain at d={distances[i]} ({gains[i]:.4f}) must be >= "
                f"gain at d={distances[i+1]} ({gains[i+1]:.4f})",
            )

    def test_infra_gain_monotonic(self):
        model = PropagationModel(_CFG)
        distances = [1, 50, 200, 500, 1000]
        gains = [
            model.propagate(d, band_energy_infra=1.0).gain_infra
            for d in distances
        ]
        for i in range(len(gains) - 1):
            self.assertGreaterEqual(
                gains[i], gains[i + 1],
                f"infra gain not monotonic at d={distances[i]}",
            )


# ---------------------------------------------------------------------------
# 2. test_dust_reduces_high_freq_and_snr
# ---------------------------------------------------------------------------

class TestDustReducesHighFreqAndSNR(unittest.TestCase):
    """High dust must reduce lp_cutoff_norm and snr."""

    def test_dust_lowers_lp_cutoff(self):
        model = PropagationModel(_CFG)
        clean = model.propagate(50.0, dust_density=0.0)
        dusty = model.propagate(50.0, dust_density=1.0)
        self.assertGreater(
            clean.lp_cutoff_norm, dusty.lp_cutoff_norm,
            "High dust must lower lp_cutoff_norm (more muffling)",
        )

    def test_dust_lowers_snr(self):
        model = PropagationModel(_CFG)
        clean = model.propagate(50.0, dust_density=0.0, wind_speed=0.0)
        dusty = model.propagate(50.0, dust_density=1.0, wind_speed=0.0)
        self.assertGreater(
            clean.snr, dusty.snr,
            "High dust must lower SNR",
        )

    def test_wind_lowers_snr(self):
        model = PropagationModel(_CFG)
        calm  = model.propagate(50.0, dust_density=0.0, wind_speed=0.0)
        windy = model.propagate(50.0, dust_density=0.0, wind_speed=1.0)
        self.assertGreater(
            calm.snr, windy.snr,
            "High wind must lower SNR",
        )

    def test_atmosphere_adapter_dust_range(self):
        adapter = AtmosphereAcousticAdapter(_CFG)
        calm  = adapter.compute(dust_density=0.0, wind_speed=0.0)
        storm = adapter.compute(dust_density=1.0, wind_speed=1.0)
        self.assertGreater(calm.range_factor, storm.range_factor,
                           "Storm conditions must reduce effective range_factor")
        self.assertLess(storm.range_factor, 1.0,
                        "Storm range_factor must be < 1.0")


# ---------------------------------------------------------------------------
# 3. test_occlusion_blocks_audible_more_than_infra
# ---------------------------------------------------------------------------

class TestOcclusionBlocksAudibleMoreThanInfra(unittest.TestCase):
    """Occlusion must hit audible band harder than infra band."""

    def test_audible_vs_infra_occlusion_ratio(self):
        model = PropagationModel(_CFG)
        d = 100.0

        # Unoccluded reference
        ref = model.propagate(
            d, band_energy_audible=1.0, band_energy_infra=1.0, occlusion=0.0
        )
        # Fully occluded
        occ = model.propagate(
            d, band_energy_audible=1.0, band_energy_infra=1.0, occlusion=1.0
        )

        # Audible should drop more in relative terms
        if ref.gain_audible > 1e-9:
            ratio_audible = occ.gain_audible / ref.gain_audible
        else:
            ratio_audible = 0.0
        if ref.gain_infra > 1e-9:
            ratio_infra = occ.gain_infra / ref.gain_infra
        else:
            ratio_infra = 0.0

        self.assertLess(
            ratio_audible, ratio_infra,
            "Occlusion must reduce audible more than infra "
            f"(ratio_audible={ratio_audible:.3f}, ratio_infra={ratio_infra:.3f})",
        )


# ---------------------------------------------------------------------------
# 4. test_valley_gain_increases_range_in_concave_areas
# ---------------------------------------------------------------------------

class TestValleyGainInConcaveAreas(unittest.TestCase):
    """Concave terrain must produce higher valley_gain than flat/convex terrain."""

    def test_concave_higher_than_flat(self):
        # Flat terrain: centre height == surrounding height
        flat_fn = lambda x, z: 0.0
        # Concave terrain: centre is lowest, surroundings are higher
        concave_fn = lambda x, z: (x ** 2 + z ** 2) ** 0.5 * 0.1

        proxy_flat    = ValleyConcavityProxy(_CFG, height_fn=flat_fn)
        proxy_concave = ValleyConcavityProxy(_CFG, height_fn=concave_fn)

        pos = Vec3(0.0, 0.0, 0.0)
        gain_flat    = proxy_flat.compute(pos)
        gain_concave = proxy_concave.compute(pos)

        self.assertGreater(
            gain_concave, gain_flat,
            f"Concave terrain gain ({gain_concave:.3f}) must exceed "
            f"flat terrain gain ({gain_flat:.3f})",
        )

    def test_convex_lower_than_flat(self):
        # Convex (dome): centre is highest
        convex_fn = lambda x, z: -((x ** 2 + z ** 2) ** 0.5) * 0.1
        flat_fn   = lambda x, z: 0.0

        proxy_convex = ValleyConcavityProxy(_CFG, height_fn=convex_fn)
        proxy_flat   = ValleyConcavityProxy(_CFG, height_fn=flat_fn)

        pos = Vec3(0.0, 0.0, 0.0)
        self.assertLessEqual(
            proxy_convex.compute(pos),
            proxy_flat.compute(pos),
            "Convex terrain gain must not exceed flat terrain gain",
        )

    def test_gain_within_bounds(self):
        proxy = ValleyConcavityProxy(_CFG)
        gain = proxy.compute(Vec3(10.0, 0.0, 10.0))
        self.assertGreaterEqual(gain, _CFG["audio_world"]["valley_gain_min"])
        self.assertLessEqual(gain, _CFG["audio_world"]["valley_gain_max"])


# ---------------------------------------------------------------------------
# 5. test_infra_field_persists_and_decays
# ---------------------------------------------------------------------------

class TestInfraFieldPersistsAndDecays(unittest.TestCase):
    """InfraField must retain energy for several ticks then decay to near zero."""

    def test_energy_persists_initially(self):
        field = InfraField(_CFG)
        pos   = Vec3(0.0, 0.0, 0.0)
        field.inject(pos, energy=1.0)

        # Immediately after injection, energy must be non-negligible
        field.tick(dt=0.1)
        level = field.sample(pos)
        self.assertGreater(level, 0.1,
                           "InfraField must retain energy shortly after injection")

    def test_energy_decays_over_time(self):
        field = InfraField(_CFG)
        pos   = Vec3(0.0, 0.0, 0.0)
        field.inject(pos, energy=2.0)

        # Tick for 2 seconds
        for _ in range(20):
            field.tick(dt=0.1)
        level_early = field.sample(pos)

        # Tick another 10 seconds
        for _ in range(100):
            field.tick(dt=0.1)
        level_late = field.sample(pos)

        self.assertGreater(level_early, level_late,
                           "InfraField energy must decay over time")

    def test_energy_eventually_reaches_near_zero(self):
        field = InfraField(_CFG)
        pos   = Vec3(0.0, 0.0, 0.0)
        field.inject(pos, energy=1.0)

        # Tick for 60 seconds (well beyond any reasonable tau)
        for _ in range(600):
            field.tick(dt=0.1)

        self.assertAlmostEqual(field.sample(pos), 0.0, places=3,
                               msg="InfraField must decay to near zero over long time")

    def test_urgency_equals_sample(self):
        field = InfraField(_CFG)
        pos   = Vec3(50.0, 0.0, 50.0)
        field.inject(pos, energy=0.5)
        field.tick(dt=0.05)
        self.assertAlmostEqual(field.urgency(pos), field.sample(pos))


# ---------------------------------------------------------------------------
# 6. test_budget_limits_respected
# ---------------------------------------------------------------------------

class TestBudgetLimitsRespected(unittest.TestCase):
    """EmitterAggregator must never exceed max_emitters active records."""

    def test_active_count_never_exceeds_budget(self):
        agg = EmitterAggregator(_CFG)
        max_e = _CFG["audio_world"]["max_emitters"]

        for tick in range(50):
            # Add several emitters per tick
            for i in range(10):
                rec = AcousticEmitterRecord(
                    pos=Vec3(float(tick * 10 + i), 0.0, 0.0),
                    band_energy_audible=0.5,
                    band_energy_infra=0.2,
                    emitter_type=EmitterType.IMPACT_MEDIUM,
                    created_tick=tick,
                    ttl=120,
                )
                agg.add(rec)
            agg.tick(tick)
            self.assertLessEqual(
                len(agg.active_emitters), max_e,
                f"active_emitters={len(agg.active_emitters)} > max={max_e} at tick {tick}",
            )

    def test_expired_emitters_removed(self):
        agg = EmitterAggregator(_CFG)
        rec = AcousticEmitterRecord(
            pos=Vec3(0.0, 0.0, 0.0),
            band_energy_audible=1.0,
            created_tick=0,
            ttl=5,
        )
        agg.add(rec)
        # Should be alive for ticks 0-4
        agg.tick(2)
        self.assertEqual(len(agg.active_emitters), 1)
        # Expired after tick 5
        agg.tick(10)
        self.assertEqual(len(agg.active_emitters), 0,
                         "Expired emitters must be removed")


# ---------------------------------------------------------------------------
# Bonus: OcclusionCache + EmitterReplicator basic sanity
# ---------------------------------------------------------------------------

class TestOcclusionCache(unittest.TestCase):
    def test_blocked_returns_one(self):
        cache = OcclusionCache(_CFG, raycast_fn=lambda a, b: True)
        occ = cache.get_occlusion(1, Vec3.zero(), Vec3(10, 0, 0), energy=0.5)
        self.assertAlmostEqual(occ, 1.0)

    def test_clear_returns_zero(self):
        cache = OcclusionCache(_CFG, raycast_fn=lambda a, b: False)
        occ = cache.get_occlusion(2, Vec3.zero(), Vec3(10, 0, 0), energy=0.5)
        self.assertAlmostEqual(occ, 0.0)

    def test_cache_reuses_within_ttl(self):
        calls = [0]
        def counting_raycast(a, b):
            calls[0] += 1
            return False

        cache = OcclusionCache(_CFG, raycast_fn=counting_raycast)
        cache.get_occlusion(3, Vec3.zero(), Vec3(5, 0, 0), energy=0.5)
        cache.tick(0.1)  # age < TTL (0.5 s)
        cache.get_occlusion(3, Vec3.zero(), Vec3(5, 0, 0), energy=0.5)
        self.assertEqual(calls[0], 1, "Second call within TTL must reuse cached value")


class TestEmitterReplicator(unittest.TestCase):
    def test_roundtrip(self):
        rep  = EmitterReplicator(_CFG)
        orig = AcousticEmitterRecord(
            id=42,
            pos=Vec3(100.0, 5.0, -200.0),
            band_energy_audible=0.8,
            band_energy_infra=0.3,
            directivity=0.6,
            emitter_type=EmitterType.STRUCTURAL,
            ttl=90,
        )
        records = rep.serialise([orig], client_pos=Vec3(0, 0, 0))
        decoded = rep.deserialise(records)
        self.assertEqual(len(decoded), 1)
        d = decoded[0]
        self.assertEqual(d.id, orig.id)
        self.assertEqual(d.emitter_type, orig.emitter_type)
        # Positions are quantised to 4 m grid — within 2 m
        self.assertAlmostEqual(d.pos.x, orig.pos.x, delta=4.0)

    def test_interest_radius_filter(self):
        rep = EmitterReplicator(_CFG)
        far_emitter = AcousticEmitterRecord(
            id=1, pos=Vec3(1000.0, 0.0, 0.0), band_energy_audible=1.0
        )
        records = rep.serialise([far_emitter], client_pos=Vec3(0, 0, 0))
        # Default interest radius (800 m) — 1000 m emitter should be filtered out
        # unless we override the config. In _CFG we don't set it, so default=800.
        self.assertEqual(len(records), 0,
                         "Emitter beyond interest radius must be filtered")


class TestAudioWorldAdapter(unittest.TestCase):
    def test_adapter_produces_sources(self):
        adapter = AudioWorldAdapter(_CFG)
        emitters = [
            AcousticEmitterRecord(
                id=1,
                pos=Vec3(30.0, 0.0, 0.0),
                band_energy_audible=0.9,
                band_energy_infra=0.1,
                emitter_type=EmitterType.IMPACT_MEDIUM,
            )
        ]
        result = adapter.build_audio_sources(
            listener_pos=Vec3(0, 0, 0),
            emitters=emitters,
        )
        self.assertGreater(len(result.audio_sources), 0,
                           "Adapter must produce at least one AudioSource for nearby emitter")
        self.assertGreaterEqual(result.audio_snr, 0.0)
        self.assertLessEqual(result.audio_snr,    1.0)

    def test_adapter_infra_urgency_from_field(self):
        adapter = AudioWorldAdapter(_CFG)
        infra   = InfraField(_CFG)
        pos     = Vec3(0.0, 0.0, 0.0)
        infra.inject(pos, energy=2.0)
        infra.tick(0.05)

        result = adapter.build_audio_sources(
            listener_pos=pos,
            emitters=[],
            infra_field=infra,
        )
        self.assertGreater(result.infra_urgency, 0.0,
                           "Adapter must propagate infra urgency from InfraField")


if __name__ == "__main__":
    unittest.main()
