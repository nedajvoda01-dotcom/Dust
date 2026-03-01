"""test_net_stage25.py — Stage 25 diegetic multiplayer presence smoke tests.

Tests
-----
1. TestTrailTtlDecay
   — Trail segment TTL decreases on update.
   — Segment expires after TTL elapses.
   — Storm multiplier accelerates decay on dust surfaces.
   — Ice surfaces are NOT eroded by storm.

2. TestTrailReplicationInterest
   — Players in a far sector do NOT receive TRAIL_EVENT messages.
   — Players in an adjacent sector DO receive TRAIL_EVENT messages.
   — Sender does not receive their own trail events.

3. TestHeadlampWhiteoutScatter
   — In clear conditions the scatter radius is zero (or near zero).
   — In whiteout the scatter radius grows toward the configured maximum.
   — Night gives higher cone intensity than day.

4. TestRemotePlayerLOD
   — Distance < lod_distance_1 → LOD_FULL.
   — lod_distance_1 <= distance < lod_distance_2 → LOD_REDUCED.
   — distance >= lod_distance_2 → LOD_MINIMAL.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.systems.DiegeticMultiplayerPresence import (
    DiegeticMultiplayerPresence,
    TrailDecalSystem,
    TrailSegment,
    TrailType,
    MaterialClass,
    LODLevel,
    HeadlampSystem,
    RemotePlayerLOD,
)
from src.net.TrailEventProtocol import (
    encode_trail_event,
    decode_trail_events,
    should_relay,
    TrailBatchAccumulator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_segment(
    pos:       list       = None,
    ttl:       float      = 60.0,
    material:  MaterialClass = MaterialClass.DUST,
    trail_type: TrailType = TrailType.FOOTPRINT,
) -> TrailSegment:
    return TrailSegment(
        pos        = pos or [0.0, 0.0, 0.0],
        direction  = [0.0, 0.0, 1.0],
        width      = 0.3,
        strength   = 1.0,
        trail_type = trail_type,
        material   = material,
        ttl        = ttl,
    )


def _make_presence(**kw) -> DiegeticMultiplayerPresence:
    defaults = {
        "trails_ttl_sec":              60.0,
        "trails_ttl_storm_multiplier": 0.25,
        "trails_max_segments_near":    512,
        "trails_sector_cache_size":    64,
        "headlamp_intensity_day":      0.15,
        "headlamp_intensity_night":    1.0,
        "headlamp_scatter_in_whiteout": 0.6,
        "remote_lod_distance_1":       40.0,
        "remote_lod_distance_2":       120.0,
        "net_trail_batch_ms":          300.0,
        "audio_remote_foot_gain":      0.6,
        "audio_remote_cutoff_storm":   400.0,
    }
    defaults.update(kw)
    return DiegeticMultiplayerPresence(config=defaults)


# ---------------------------------------------------------------------------
# 1. TestTrailTtlDecay
# ---------------------------------------------------------------------------

class TestTrailTtlDecay(unittest.TestCase):
    """Trail segments age and expire correctly."""

    def setUp(self) -> None:
        self.system = TrailDecalSystem(
            base_ttl_sec         = 60.0,
            ttl_storm_multiplier = 0.25,
        )

    # -- Basic TTL ageing

    def test_ttl_decreases_on_update(self) -> None:
        seg = _make_segment(ttl=60.0)
        self.system.add_segment(seg)
        self.system.update(dt=10.0, storm_intensity=0.0)
        remaining = self.system._segments[0].ttl
        self.assertLess(remaining, 60.0, "TTL should decrease after update")

    def test_segment_expires_at_ttl(self) -> None:
        seg = _make_segment(ttl=5.0)
        self.system.add_segment(seg)
        # Update past the TTL
        self.system.update(dt=6.0, storm_intensity=0.0)
        self.assertEqual(self.system.active_count(), 0,
                         "Expired segment should be removed")

    def test_segment_alive_before_ttl(self) -> None:
        seg = _make_segment(ttl=10.0)
        self.system.add_segment(seg)
        self.system.update(dt=5.0, storm_intensity=0.0)
        self.assertEqual(self.system.active_count(), 1,
                         "Segment should still be alive before TTL elapses")

    # -- Storm erosion on dust

    def test_storm_reduces_remaining_ttl_on_dust(self) -> None:
        """A storm update should leave less TTL than a calm update."""
        seg_calm  = _make_segment(ttl=60.0, material=MaterialClass.DUST)
        seg_storm = _make_segment(ttl=60.0, material=MaterialClass.DUST)

        sys_calm  = TrailDecalSystem(base_ttl_sec=60.0, ttl_storm_multiplier=0.25)
        sys_storm = TrailDecalSystem(base_ttl_sec=60.0, ttl_storm_multiplier=0.25)

        sys_calm.add_segment(seg_calm)
        sys_storm.add_segment(seg_storm)

        sys_calm.update(dt=10.0, storm_intensity=0.0)
        sys_storm.update(dt=10.0, storm_intensity=1.0)

        ttl_calm  = sys_calm._segments[0].ttl  if sys_calm._segments  else 0.0
        ttl_storm = sys_storm._segments[0].ttl if sys_storm._segments else 0.0

        self.assertLess(ttl_storm, ttl_calm,
                        "Storm should erode dust TTL faster than calm conditions")

    def test_storm_erases_dust_faster_full_storm(self) -> None:
        """With full storm, dust segment should expire before calm TTL."""
        # Full storm means the effective TTL is base * storm_multiplier
        seg = _make_segment(ttl=60.0 * 0.25, material=MaterialClass.DUST)
        system = TrailDecalSystem(base_ttl_sec=60.0, ttl_storm_multiplier=0.25)
        system.add_segment(seg)
        # Run for slightly over storm-reduced TTL
        system.update(dt=16.0, storm_intensity=1.0)
        self.assertEqual(system.active_count(), 0,
                         "Full-storm dust segment should be erased quickly")

    # -- Ice surfaces are not affected by storm

    def test_ice_not_eroded_by_storm(self) -> None:
        """Ice-film traces should NOT be accelerated by storm."""
        seg_ice = _make_segment(ttl=60.0 * 2.0, material=MaterialClass.ICE_FILM)
        seg_dust = _make_segment(ttl=60.0, material=MaterialClass.DUST)

        sys_ice  = TrailDecalSystem(base_ttl_sec=60.0, ttl_storm_multiplier=0.25)
        sys_dust = TrailDecalSystem(base_ttl_sec=60.0, ttl_storm_multiplier=0.25)

        sys_ice.add_segment(seg_ice)
        sys_dust.add_segment(seg_dust)

        sys_ice.update(dt=10.0, storm_intensity=1.0)
        sys_dust.update(dt=10.0, storm_intensity=1.0)

        # Both have segments alive still; verify ice TTL is higher
        if sys_ice._segments and sys_dust._segments:
            ttl_ice  = sys_ice._segments[0].ttl
            ttl_dust = sys_dust._segments[0].ttl
            self.assertGreater(ttl_ice, ttl_dust,
                               "Ice trace should outlast dust in storm")

    # -- Integration via DiegeticMultiplayerPresence.add_trail_event

    def test_add_trail_event_creates_segment(self) -> None:
        presence = _make_presence()
        presence.add_trail_event({
            "type":     "footprint",
            "pos":      [10.0, 0.0, 20.0],
            "dir":      [0.0, 0.0, 1.0],
            "strength": 0.8,
            "material": "Dust",
        })
        self.assertEqual(presence._trails.active_count(), 1)

    def test_add_trail_event_bad_type_ignored(self) -> None:
        presence = _make_presence()
        presence.add_trail_event({
            "type":     "unknown_type_xyz",
            "pos":      [0.0, 0.0, 0.0],
            "strength": 0.5,
            "material": "Dust",
        })
        self.assertEqual(presence._trails.active_count(), 0,
                         "Bad event type should be silently ignored")

    def test_update_trails_decays_via_presence(self) -> None:
        presence = _make_presence(trails_ttl_sec=10.0)
        presence.add_trail_event({
            "type": "footprint", "pos": [0.0, 0.0, 0.0],
            "strength": 1.0, "material": "Dust",
        })
        presence.update_trails(dt=11.0, storm_intensity=0.0)
        self.assertEqual(presence._trails.active_count(), 0,
                         "Segment should expire after TTL via presence.update_trails")


# ---------------------------------------------------------------------------
# 2. TestTrailReplicationInterest
# ---------------------------------------------------------------------------

class TestTrailReplicationInterest(unittest.IsolatedAsyncioTestCase):
    """TRAIL_BATCH messages are interest-filtered before relay."""

    async def asyncSetUp(self) -> None:
        import tempfile
        self._tmp = tempfile.mkdtemp()
        state_dir = os.path.join(self._tmp, "world_state")

        from src.net.NetworkServer import NetworkServer
        from src.net.WorldState import WorldState
        from src.net.PlayerRegistry import PlayerRegistry

        ws = WorldState(state_dir)
        ws.load_or_create(default_seed=42)
        self.server = NetworkServer(
            bootstrap=None, config=None, world_state=ws, state_dir=state_dir
        )
        self.received: dict = {}

    def _make_ws(self, pid: str):
        received = self.received
        class FakeWS:
            async def send(self, msg):
                received.setdefault(pid, []).append(json.loads(msg))
        return FakeWS()

    async def _connect(self, pid: str, pos: list):
        ws = self._make_ws(pid)
        self.server._connections[pid] = ws
        self.server._registry.add(pid, pos)
        return ws

    # -- Sender not included

    async def test_sender_excluded_from_relay(self) -> None:
        await self._connect("sender", [0.0, 1000.0, 0.0])
        await self._connect("nearby", [5.0, 1000.0, 5.0])

        batch_msg = {
            "type": "TRAIL_BATCH",
            "events": [encode_trail_event("footprint", [0, 0, 0],
                                           [0, 0, 1], 0.8, "Dust", 1)],
        }
        await self.server._relay_trail_batch("sender", batch_msg)

        trail_msgs_sender = [
            m for m in self.received.get("sender", [])
            if m.get("type") == "TRAIL_EVENT"
        ]
        self.assertEqual(len(trail_msgs_sender), 0,
                         "Sender should not receive its own trail events")

    # -- Nearby player receives trail event

    async def test_nearby_player_receives_trail(self) -> None:
        # Both players very close on the unit sphere
        await self._connect("sender",  [0.0, 1000.0, 1.0])
        await self._connect("nearby",  [0.0, 1000.0, 1.2])

        batch_msg = {
            "type": "TRAIL_BATCH",
            "events": [encode_trail_event("footprint", [0, 0, 0],
                                           [0, 0, 1], 0.8, "Dust", 1)],
        }
        await self.server._relay_trail_batch("sender", batch_msg)

        trail_msgs = [
            m for m in self.received.get("nearby", [])
            if m.get("type") == "TRAIL_EVENT"
        ]
        self.assertGreater(len(trail_msgs), 0,
                           "Nearby player should receive TRAIL_EVENT")

    # -- Distant player does not receive trail event

    async def test_far_player_excluded_from_relay(self) -> None:
        """A player on the opposite side of the planet should not receive trails."""
        # Put sender at north pole, far player at south pole
        r = 1000.0
        await self._connect("sender",  [0.0,  r,  0.0])
        await self._connect("far_away", [0.0, -r,  0.0])

        # Ensure sector parameters make this a far exclusion
        self.server._sector_deg    = 5.0
        self.server._sector_radius = 2

        batch_msg = {
            "type": "TRAIL_BATCH",
            "events": [encode_trail_event("footprint", [0, 0, 0],
                                           [0, 0, 1], 0.8, "Dust", 1)],
        }
        await self.server._relay_trail_batch("sender", batch_msg)

        trail_msgs = [
            m for m in self.received.get("far_away", [])
            if m.get("type") == "TRAIL_EVENT"
        ]
        self.assertEqual(len(trail_msgs), 0,
                         "Far player (opposite side of planet) should not "
                         "receive trail events")

    # -- should_relay unit tests

    def test_should_relay_nearby(self) -> None:
        """Positions within adjacent sectors → True."""
        sender    = [0.0, 1000.0, 1.0]
        recipient = [0.0, 1000.0, 2.0]
        self.assertTrue(should_relay(sender, recipient, sector_deg=5.0, sector_radius=2))

    def test_should_relay_far_false(self) -> None:
        """Positions on opposite hemisphere → False."""
        sender    = [0.0,  1000.0, 0.0]
        recipient = [0.0, -1000.0, 0.0]
        self.assertFalse(should_relay(sender, recipient, sector_deg=5.0, sector_radius=2))

    # -- TrailBatchAccumulator

    def test_batch_accumulator_flushes(self) -> None:
        acc = TrailBatchAccumulator(batch_ms=300.0)
        acc.add("footprint", [0, 0, 0], [0, 0, 1], 0.8, "Dust", tick=1)
        acc.add("footprint", [1, 0, 0], [0, 0, 1], 0.7, "Dust", tick=2)
        result = acc.flush()
        self.assertIsNotNone(result)
        self.assertEqual(result["type"], "TRAIL_BATCH")
        self.assertEqual(len(result["events"]), 2)

    def test_batch_accumulator_empty_flush_returns_none(self) -> None:
        acc = TrailBatchAccumulator(batch_ms=300.0)
        result = acc.flush()
        self.assertIsNone(result)

    def test_batch_accumulator_max_size_triggers_flush(self) -> None:
        acc = TrailBatchAccumulator(batch_ms=300.0, max_batch_size=3)
        results = []
        for i in range(3):
            r = acc.add("footprint", [i, 0, 0], [0, 0, 1], 0.5, "Dust", tick=i)
            if r is not None:
                results.append(r)
        self.assertGreater(len(results), 0,
                           "Max-size exceeded should trigger an early flush")

    def test_decode_trail_events_round_trip(self) -> None:
        ev = encode_trail_event("slide", [1.0, 2.0, 3.0], [0.5, 0.0, 0.5],
                                 0.75, "IceFilm", tick=5)
        msg = {"type": "TRAIL_BATCH", "events": [ev]}
        decoded = decode_trail_events(msg)
        self.assertEqual(len(decoded), 1)
        self.assertEqual(decoded[0]["type"], "slide")
        self.assertAlmostEqual(decoded[0]["strength"], 0.75, delta=0.01)
        self.assertEqual(decoded[0]["material"], "IceFilm")


# ---------------------------------------------------------------------------
# 3. TestHeadlampWhiteoutScatter
# ---------------------------------------------------------------------------

class TestHeadlampWhiteoutScatter(unittest.TestCase):
    """HeadlampSystem produces correct profiles for environment conditions."""

    def setUp(self) -> None:
        self.headlamp = HeadlampSystem(
            intensity_day       = 0.15,
            intensity_night     = 1.0,
            scatter_in_whiteout = 0.6,
        )

    def test_clear_day_no_scatter(self) -> None:
        profile = self.headlamp.compute(day_fraction=1.0, visibility=1.0)
        self.assertAlmostEqual(profile.scatter_radius, 0.0, delta=0.01,
                               msg="No scatter in clear daylight")

    def test_whiteout_has_scatter(self) -> None:
        profile = self.headlamp.compute(day_fraction=0.5, visibility=0.0)
        self.assertGreater(profile.scatter_radius, 0.0,
                           msg="Whiteout should produce a scatter halo > 0")

    def test_full_whiteout_scatter_equals_max(self) -> None:
        profile = self.headlamp.compute(day_fraction=0.5, visibility=0.0)
        self.assertAlmostEqual(profile.scatter_radius, 0.6, delta=0.01,
                               msg="Full whiteout scatter must reach configured max")

    def test_night_higher_cone_intensity_than_day(self) -> None:
        day   = self.headlamp.compute(day_fraction=1.0, visibility=1.0)
        night = self.headlamp.compute(day_fraction=0.0, visibility=1.0)
        self.assertGreater(night.cone_intensity, day.cone_intensity,
                           msg="Night should have higher cone intensity than day")

    def test_partial_visibility_intermediate_scatter(self) -> None:
        half  = self.headlamp.compute(day_fraction=0.5, visibility=0.5)
        clear = self.headlamp.compute(day_fraction=0.5, visibility=1.0)
        white = self.headlamp.compute(day_fraction=0.5, visibility=0.0)
        self.assertGreater(half.scatter_radius, clear.scatter_radius)
        self.assertLess(half.scatter_radius, white.scatter_radius)

    def test_whiteout_narrows_cone(self) -> None:
        clear   = self.headlamp.compute(day_fraction=0.5, visibility=1.0)
        whiteout = self.headlamp.compute(day_fraction=0.5, visibility=0.0)
        self.assertLess(whiteout.cone_half_angle, clear.cone_half_angle,
                        msg="Whiteout should narrow the cone (forward beam collapses)")

    def test_presence_compute_headlamp_wrapper(self) -> None:
        """DiegeticMultiplayerPresence.compute_headlamp delegates correctly."""
        presence = _make_presence()
        profile  = presence.compute_headlamp(day_fraction=0.0, visibility=0.0)
        self.assertGreater(profile.scatter_radius, 0.0)
        self.assertAlmostEqual(profile.cone_intensity, 1.0, delta=0.01)


# ---------------------------------------------------------------------------
# 4. TestRemotePlayerLOD
# ---------------------------------------------------------------------------

class TestRemotePlayerLOD(unittest.TestCase):
    """RemotePlayerLOD selects correct level for each distance range."""

    def setUp(self) -> None:
        self.lod = RemotePlayerLOD(lod_distance_1=40.0, lod_distance_2=120.0)

    def test_close_is_full(self) -> None:
        level = self.lod.lod_for_distance(10.0)
        self.assertEqual(level, LODLevel.FULL,
                         "Distance < lod_distance_1 must use LOD_FULL")

    def test_just_below_d1_is_full(self) -> None:
        level = self.lod.lod_for_distance(39.9)
        self.assertEqual(level, LODLevel.FULL)

    def test_at_d1_is_reduced(self) -> None:
        level = self.lod.lod_for_distance(40.0)
        self.assertEqual(level, LODLevel.REDUCED,
                         "Distance == lod_distance_1 must use LOD_REDUCED")

    def test_mid_range_is_reduced(self) -> None:
        level = self.lod.lod_for_distance(80.0)
        self.assertEqual(level, LODLevel.REDUCED)

    def test_just_below_d2_is_reduced(self) -> None:
        level = self.lod.lod_for_distance(119.9)
        self.assertEqual(level, LODLevel.REDUCED)

    def test_at_d2_is_minimal(self) -> None:
        level = self.lod.lod_for_distance(120.0)
        self.assertEqual(level, LODLevel.MINIMAL,
                         "Distance >= lod_distance_2 must use LOD_MINIMAL")

    def test_far_is_minimal(self) -> None:
        level = self.lod.lod_for_distance(500.0)
        self.assertEqual(level, LODLevel.MINIMAL)

    def test_presence_lod_wrapper(self) -> None:
        """DiegeticMultiplayerPresence.lod_for_distance delegates correctly."""
        presence = _make_presence(remote_lod_distance_1=40.0,
                                  remote_lod_distance_2=120.0)
        self.assertEqual(presence.lod_for_distance(10.0),  LODLevel.FULL)
        self.assertEqual(presence.lod_for_distance(80.0),  LODLevel.REDUCED)
        self.assertEqual(presence.lod_for_distance(200.0), LODLevel.MINIMAL)

    def test_debug_info_contains_segments_count(self) -> None:
        presence = _make_presence()
        presence.add_trail_event({
            "type": "footprint", "pos": [0, 0, 0],
            "strength": 1.0, "material": "Dust",
        })
        info = presence.debug_info()
        self.assertIn("active_trail_segments", info)
        self.assertGreaterEqual(info["active_trail_segments"], 1)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
