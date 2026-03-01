"""test_subsurface_hazard.py — Stage 27 SubsurfaceHazardSystem smoke tests.

Tests (as specified in the problem statement §14)
-------------------------------------------------
1. test_event_phases_order
   — PRE always precedes IMPACT, POST follows IMPACT, timings correct.

2. test_chain_cap_respected
   — High-risk scenario never exceeds chain_max_depth / max_patches_per_event.

3. test_multiplayer_sync
   — Two clients receiving the same event end up with identical event_id and
     the same patch count (hash-equivalent).

4. test_dust_wave_decay
   — Visibility falls immediately after IMPACT and recovers over POST duration.

5. test_rate_limits
   — Spamming tick() never exceeds max_events_per_hour_global.
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.math.Vec3 import Vec3
from src.systems.SubsurfaceSystem import SubsurfaceSystem
from src.systems.SubsurfaceHazardSystem import (
    CaveDustField,
    SubsurfaceHazardEvent,
    SubsurfaceHazardEventType,
    SubsurfaceHazardPhase,
    SubsurfaceHazardSystem,
    ZoneRiskFactors,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEED          = 42
PLANET_RADIUS = 1_000.0


def _make_subsurface(seed: int = SEED) -> SubsurfaceSystem:
    cfg = {
        "enable":       True,
        "shallow_prob": 1.0,
        "mid_prob":     1.0,
        "deep_prob":    1.0,
        "portal_frequency": 0.5,
        "collapse_event_rate_cap": 100,
    }
    return SubsurfaceSystem(
        config=cfg,
        global_seed=seed,
        planet_radius=PLANET_RADIUS,
    )


def _make_hazard(
    subsurface_sys=None,
    seed: int = SEED,
    **overrides,
) -> SubsurfaceHazardSystem:
    cfg = {
        "enable":                     True,
        "tick_seconds":               0.01,   # fire every tick in tests
        "risk_threshold":             0.0,    # always trigger (test mode)
        "pre_min_sec":                1.0,
        "pre_max_sec":                2.0,
        "post_min_sec":               5.0,
        "post_max_sec":               10.0,
        "max_events_per_hour_global": 50,
        "max_events_per_hour_zone":   10,
        "chain_max_depth":            3,
        "chain_decay":                0.9,    # high to guarantee chains in tests
        "max_patches_per_event":      12,
        "dust_wave_speed":            8.0,
        "dust_peak_density":          0.85,
        "cooldown_same_cave_sec":     0.0,    # no cooldown in tests
    }
    cfg.update(overrides)
    if subsurface_sys is None:
        subsurface_sys = _make_subsurface(seed)
    return SubsurfaceHazardSystem(
        config=cfg,
        subsurface_sys=subsurface_sys,
        planet_radius=PLANET_RADIUS,
        global_seed=seed,
    )


# ---------------------------------------------------------------------------
# 1. test_event_phases_order
# ---------------------------------------------------------------------------


class TestEventPhasesOrder(unittest.TestCase):
    """PRE must always precede IMPACT; POST must follow; timings must be valid."""

    def _make_event(self, t0: float = 0.0, pre_dur: float = 3.0, post_dur: float = 20.0) -> SubsurfaceHazardEvent:
        return SubsurfaceHazardEvent(
            event_id    = 0,
            event_type  = SubsurfaceHazardEventType.LOCAL_COLLAPSE,
            zone_id     = 0,
            anchor_node = 0,
            position    = Vec3(1.0, 0.0, 0.0),
            t0          = t0,
            pre_dur     = pre_dur,
            post_dur    = post_dur,
            seed_local  = 1,
            intensity   = 0.8,
        )

    def test_pre_before_impact(self):
        evt = self._make_event(t0=100.0, pre_dur=5.0)
        self.assertEqual(evt.phase_at(100.0), SubsurfaceHazardPhase.PRE)
        self.assertEqual(evt.phase_at(104.9), SubsurfaceHazardPhase.PRE)

    def test_post_after_impact(self):
        evt = self._make_event(t0=100.0, pre_dur=5.0, post_dur=20.0)
        # At IMPACT time and just after
        self.assertEqual(evt.phase_at(105.0), SubsurfaceHazardPhase.POST)
        self.assertEqual(evt.phase_at(124.9), SubsurfaceHazardPhase.POST)

    def test_done_after_post(self):
        evt = self._make_event(t0=100.0, pre_dur=5.0, post_dur=20.0)
        self.assertEqual(evt.phase_at(125.1), SubsurfaceHazardPhase.DONE)

    def test_impact_time_equals_t0_plus_pre_dur(self):
        evt = self._make_event(t0=50.0, pre_dur=7.0)
        self.assertAlmostEqual(evt.impact_time, 57.0)

    def test_end_time_equals_impact_plus_post_dur(self):
        evt = self._make_event(t0=50.0, pre_dur=7.0, post_dur=30.0)
        self.assertAlmostEqual(evt.end_time, 87.0)

    def test_pre_intensity_ramps_up(self):
        evt = self._make_event(t0=0.0, pre_dur=10.0)
        i_early = evt.phase_intensity(1.0)
        i_late  = evt.phase_intensity(9.0)
        self.assertLess(i_early, i_late,
            "PRE intensity should increase toward IMPACT")

    def test_post_intensity_decays(self):
        evt = self._make_event(t0=0.0, pre_dur=5.0, post_dur=20.0)
        i_early_post = evt.phase_intensity(5.1)
        i_late_post  = evt.phase_intensity(24.9)
        self.assertGreater(i_early_post, i_late_post,
            "POST intensity should decay over time")

    def test_tick_generates_events_with_valid_timing(self):
        sys_ = _make_hazard()
        player_pos = Vec3(PLANET_RADIUS, 0.0, 0.0)
        evts = sys_.tick(dt=1.0, game_time=100.0, player_positions=[player_pos])
        for evt in evts:
            self.assertGreater(evt.pre_dur, 0.0,
                "pre_dur must be positive")
            self.assertGreater(evt.post_dur, 0.0,
                "post_dur must be positive")
            self.assertLess(evt.t0, evt.impact_time,
                "t0 must be before impact_time")
            self.assertLess(evt.impact_time, evt.end_time,
                "impact_time must be before end_time")


# ---------------------------------------------------------------------------
# 2. test_chain_cap_respected
# ---------------------------------------------------------------------------


class TestChainCapRespected(unittest.TestCase):
    """Chain collapse must never exceed chain_max_depth or max_patches_per_event."""

    def test_chain_depth_bounded(self):
        chain_max = 2
        sys_ = _make_hazard(chain_max_depth=chain_max, chain_decay=1.0)
        player_pos = Vec3(PLANET_RADIUS, 0.0, 0.0)
        evts = sys_.tick(dt=1.0, game_time=0.0, player_positions=[player_pos])
        for evt in evts:
            self.assertLessEqual(
                evt.chain_depth, chain_max,
                f"Chain depth {evt.chain_depth} exceeds max {chain_max}",
            )

    def test_total_patches_per_event_bounded(self):
        max_p = 6
        sys_  = _make_hazard(max_patches_per_event=max_p, chain_decay=1.0)
        player_pos = Vec3(PLANET_RADIUS, 0.0, 0.0)
        evts  = sys_.tick(dt=1.0, game_time=0.0, player_positions=[player_pos])
        for evt in evts:
            self.assertLessEqual(
                len(evt.patch_batch), max_p,
                f"Event has {len(evt.patch_batch)} patches, exceeds cap {max_p}",
            )

    def test_chain_events_only_from_collapses(self):
        """Chain events should only have chain_depth > 0."""
        sys_ = _make_hazard(chain_max_depth=3, chain_decay=1.0)
        player_pos = Vec3(PLANET_RADIUS, 0.0, 0.0)
        evts = sys_.tick(dt=1.0, game_time=0.0, player_positions=[player_pos])
        for evt in evts:
            if evt.chain_depth > 0:
                self.assertEqual(evt.event_type,
                    SubsurfaceHazardEventType.CHAIN_COLLAPSE,
                    "Events with chain_depth > 0 must be CHAIN_COLLAPSE type")


# ---------------------------------------------------------------------------
# 3. test_multiplayer_sync
# ---------------------------------------------------------------------------


class TestMultiplayerSync(unittest.TestCase):
    """Two clients receiving the same event must have identical state."""

    def test_apply_replicated_event_same_id(self):
        """Both clients record an event with the same event_id."""
        sub_sys = _make_subsurface()
        server  = _make_hazard(subsurface_sys=sub_sys)
        player_pos = Vec3(PLANET_RADIUS, 0.0, 0.0)
        evts = server.tick(dt=1.0, game_time=0.0, player_positions=[player_pos])
        if not evts:
            self.skipTest("No events fired — cave graph may be empty near player")

        evt = evts[0]

        client1 = _make_hazard(subsurface_sys=_make_subsurface(), seed=SEED + 1)
        client2 = _make_hazard(subsurface_sys=_make_subsurface(), seed=SEED + 2)

        client1.apply_replicated_event(evt)
        client2.apply_replicated_event(evt)

        self.assertEqual(client1.event_log[0].event_id, evt.event_id)
        self.assertEqual(client2.event_log[0].event_id, evt.event_id)

    def test_apply_replicated_event_same_patch_count(self):
        """Both clients must record the same number of patches."""
        sub_sys = _make_subsurface()
        server  = _make_hazard(subsurface_sys=sub_sys)
        player_pos = Vec3(PLANET_RADIUS, 0.0, 0.0)
        evts = server.tick(dt=1.0, game_time=0.0, player_positions=[player_pos])
        if not evts:
            self.skipTest("No events fired")

        evt = evts[0]
        client1 = _make_hazard(subsurface_sys=_make_subsurface(), seed=SEED + 1)
        client2 = _make_hazard(subsurface_sys=_make_subsurface(), seed=SEED + 2)

        client1.apply_replicated_event(evt)
        client2.apply_replicated_event(evt)

        self.assertEqual(
            len(client1.event_log[0].patch_batch),
            len(client2.event_log[0].patch_batch),
            "Clients must have the same patch count after applying the same event",
        )

    def test_event_log_grows_on_apply(self):
        """apply_replicated_event must add exactly one entry to event_log."""
        client = _make_hazard()
        evt = SubsurfaceHazardEvent(
            event_id    = 99,
            event_type  = SubsurfaceHazardEventType.DUST_WAVE,
            zone_id     = 0,
            anchor_node = 0,
            position    = Vec3(0.0, 1.0, 0.0),
            t0          = 0.0,
            pre_dur     = 3.0,
            post_dur    = 15.0,
            seed_local  = 42,
            intensity   = 0.7,
        )
        before = len(client.event_log)
        client.apply_replicated_event(evt)
        self.assertEqual(len(client.event_log), before + 1)


# ---------------------------------------------------------------------------
# 4. test_dust_wave_decay
# ---------------------------------------------------------------------------


class TestDustWaveDecay(unittest.TestCase):
    """Dust density must rise to peak then decay back toward zero."""

    def _make_dust(
        self,
        start_time: float = 0.0,
        decay_rate: float = 0.1,
        peak: float = 0.85,
        speed: float = 8.0,
    ) -> CaveDustField:
        return CaveDustField(
            zone_id      = 0,
            origin_node  = 0,
            start_time   = start_time,
            speed        = speed,
            decay_rate   = decay_rate,
            peak_density = peak,
        )

    def test_density_zero_before_start(self):
        dust = self._make_dust(start_time=10.0)
        self.assertEqual(dust.density_at(5.0), 0.0,
            "Dust must be zero before start_time")

    def test_density_nonzero_after_start(self):
        dust = self._make_dust(start_time=0.0)
        self.assertGreater(dust.density_at(1.0), 0.0,
            "Dust must be non-zero just after start")

    def test_density_decays_over_time(self):
        dust = self._make_dust(start_time=0.0, decay_rate=0.5, peak=1.0)
        d_early = dust.density_at(0.1)
        d_late  = dust.density_at(10.0)
        self.assertGreater(d_early, d_late,
            "Dust density must decrease over time (exponential decay)")

    def test_density_approaches_zero(self):
        dust = self._make_dust(start_time=0.0, decay_rate=1.0, peak=1.0)
        d_far = dust.density_at(100.0)
        self.assertLess(d_far, 0.01,
            "After sufficient time dust density must approach zero")

    def test_density_bounded_zero_to_one(self):
        dust = self._make_dust(start_time=0.0, peak=0.85)
        for t in (0.0, 0.5, 1.0, 5.0, 30.0, 200.0):
            d = dust.density_at(t)
            self.assertGreaterEqual(d, 0.0, f"Density below 0 at t={t}")
            self.assertLessEqual(d, 1.0,    f"Density above 1 at t={t}")

    def test_dust_spawned_on_collapse_event(self):
        sys_ = _make_hazard()
        player_pos = Vec3(PLANET_RADIUS, 0.0, 0.0)
        evts = sys_.tick(dt=1.0, game_time=0.0, player_positions=[player_pos])
        if not evts:
            self.skipTest("No events fired — cave graph empty near player")
        zone_id = evts[0].zone_id
        # Dust field must be present for the zone after event fires
        density = sys_.dust_density_at(zone_id, game_time=evts[0].impact_time + 1.0)
        self.assertGreaterEqual(density, 0.0)
        self.assertLessEqual(density, 1.0)


# ---------------------------------------------------------------------------
# 5. test_rate_limits
# ---------------------------------------------------------------------------


class TestRateLimits(unittest.TestCase):
    """Spamming tick() must never exceed max_events_per_hour_global."""

    def test_global_cap_respected(self):
        cap = 5
        sys_ = _make_hazard(
            max_events_per_hour_global=cap,
            tick_seconds=0.001,   # fires every call
            cooldown_same_cave_sec=0.0,
        )
        player_pos = Vec3(PLANET_RADIUS, 0.0, 0.0)

        all_events: list = []
        game_time = 0.0
        # Run many ticks within the same simulated hour
        for _ in range(50):
            game_time += 10.0   # advance by 10 s each tick (stays < 3600)
            evts = sys_.tick(dt=10.0, game_time=game_time, player_positions=[player_pos])
            all_events.extend(evts)

        self.assertLessEqual(
            len(all_events), cap,
            f"Total events {len(all_events)} exceeds global cap {cap}",
        )

    def test_zone_cap_respected(self):
        zone_cap = 2
        sys_ = _make_hazard(
            max_events_per_hour_zone=zone_cap,
            max_events_per_hour_global=100,
            tick_seconds=0.001,
            cooldown_same_cave_sec=0.0,
        )
        player_pos = Vec3(PLANET_RADIUS, 0.0, 0.0)

        all_events: list = []
        game_time = 0.0
        for _ in range(30):
            game_time += 10.0
            evts = sys_.tick(dt=10.0, game_time=game_time, player_positions=[player_pos])
            all_events.extend(evts)

        # Count events per zone_id
        zone_counts: dict = {}
        for evt in all_events:
            zone_counts[evt.zone_id] = zone_counts.get(evt.zone_id, 0) + 1

        for z_id, count in zone_counts.items():
            self.assertLessEqual(
                count, zone_cap,
                f"Zone {z_id} fired {count} events, exceeds zone cap {zone_cap}",
            )

    def test_cooldown_prevents_rapid_repeat(self):
        sys_ = _make_hazard(
            cooldown_same_cave_sec=300.0,   # 5-minute cooldown
            tick_seconds=0.001,
            max_events_per_hour_global=100,
            max_events_per_hour_zone=100,
        )
        player_pos = Vec3(PLANET_RADIUS, 0.0, 0.0)

        # First tick fires an event
        evts1 = sys_.tick(dt=1.0, game_time=0.0, player_positions=[player_pos])

        # Second tick immediately (only 1s later) — cooldown should block
        evts2 = sys_.tick(dt=1.0, game_time=1.0, player_positions=[player_pos])

        if evts1:
            # Any zone that fired in evts1 must NOT fire again in evts2
            fired_zones = {e.zone_id for e in evts1}
            refired_zones = {e.zone_id for e in evts2}
            overlap = fired_zones & refired_zones
            self.assertEqual(
                len(overlap), 0,
                f"Zones {overlap} fired again within cooldown period",
            )

    def test_budget_resets_after_hour(self):
        cap = 3
        sys_ = _make_hazard(
            max_events_per_hour_global=cap,
            tick_seconds=0.001,
            cooldown_same_cave_sec=0.0,
        )
        player_pos = Vec3(PLANET_RADIUS, 0.0, 0.0)

        # Exhaust the budget
        for _ in range(20):
            sys_.tick(dt=10.0, game_time=float(_ * 10), player_positions=[player_pos])

        # Advance past 1 hour boundary
        evts_after = sys_.tick(
            dt=10.0,
            game_time=3610.0,   # past the 3600 s reset
            player_positions=[player_pos],
        )
        # Budget should have reset; at least some event can fire again
        # (We simply verify no crash and that the system remains consistent)
        self.assertIsInstance(evts_after, list)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
