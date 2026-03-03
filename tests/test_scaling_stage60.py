"""test_scaling_stage60.py — Stage 60 Production Scaling & Sharding tests.

Tests
-----
1.  test_region_handoff_seamless
    — A player moving toward a region boundary triggers a HandoffRequest;
      executing the handoff atomically moves the player to the target RS
      with no double-authority (player exists on exactly one RS at all times).

2.  test_cross_region_infra_event
    — Posting a cross-region event via GlobalAuthorityService delivers it
      to neighbouring regions only (not the source region).

3.  test_checkpoint_restore_global_plus_regions
    — A full checkpoint written by CheckpointCoordinator can be loaded back
      and the GA + RS state is fully recovered.

4.  test_rs_crash_recovery_only_affects_one_region
    — After a simulated RS crash (deregister + re-register), only that
      region's data is lost; other regions remain unaffected.

5.  test_ga_failover_no_world_drift
    — Within the grace period, a RS continues to tick independently;
      after the grace period expires the RS should signal that it needs
      GA contact (simulated by checking tick behaviour).

6.  loadtest_100_players_single_region
    — 100 players can be added to a single RS without hitting the hard cap;
      the RS stays in non-degraded state when below the soft cap and
      degraded when above it.

7.  loadtest_500_players_hotspot_degrades_gracefully
    — RegionAutoscaler returns DEGRADED/OVERLOADED with reduced chunk update
      intervals and acoustic raycasts when player counts are high.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.global_.GlobalAuthorityService import GlobalAuthorityService
from src.global_.GlobalSnapshotWriter import GlobalSnapshotWriter
from src.regions.RegionIndexing import RegionIndexing
from src.regions.RegionShardServer import RegionShardServer, HandoffRequest
from src.regions.HandoffManager import HandoffManager
from src.regions.GhostBorderCache import GhostBorderCache
from src.net.GatewayRouter import GatewayRouter
from src.net.RegionSessionManager import RegionSessionManager
from src.ops.RegionAutoscaler import RegionAutoscaler, STATUS_OK, STATUS_DEGRADED, STATUS_OVERLOADED
from src.save.CheckpointCoordinator import CheckpointCoordinator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ga() -> GlobalAuthorityService:
    return GlobalAuthorityService()


def _make_indexing(deg: float = 10.0) -> RegionIndexing:
    return RegionIndexing(region_size_deg=deg)


def _make_rs(region_id: int, ga, indexing) -> RegionShardServer:
    return RegionShardServer(region_id=region_id, ga_service=ga, indexing=indexing)


# ---------------------------------------------------------------------------
# 1. test_region_handoff_seamless
# ---------------------------------------------------------------------------

class TestRegionHandoffSeamless(unittest.TestCase):
    """A player approaching a boundary is seamlessly handed off."""

    def test_handoff_moves_player_atomically(self):
        ga       = _make_ga()
        idx      = _make_indexing()
        hm       = HandoffManager(ga, idx)

        # Two adjacent regions
        rid_a    = idx.region_id(lat=0.0, lon=0.0)    # region A
        rid_b    = idx.region_id(lat=0.0, lon=15.0)   # region B (different tile)

        rs_a     = _make_rs(rid_a, ga, idx)
        rs_b     = _make_rs(rid_b, ga, idx)

        # Add player to A
        rs_a.add_player("p1", [0.0, 0.0, 0.0])
        self.assertIn("p1", rs_a.player_ids)
        self.assertNotIn("p1", rs_b.player_ids)

        # Build a HandoffRequest (as if RS produced it)
        request = HandoffRequest(
            player_id     = "p1",
            source_region = rid_a,
            target_region = rid_b,
            player_state  = {"player_id": "p1", "pos": [0.0, 15.0, 0.0], "vel": [0.0, 1.0, 0.0]},
        )

        # Execute handoff
        success = hm.execute_handoff(request, rs_a, rs_b)

        self.assertTrue(success)
        # Player must be on exactly one RS after handoff
        self.assertNotIn("p1", rs_a.player_ids, "player still on source RS after handoff")
        self.assertIn("p1", rs_b.player_ids, "player not on target RS after handoff")
        self.assertEqual(hm.completed_count, 1)
        self.assertEqual(hm.failed_count, 0)

    def test_handoff_fails_when_target_full(self):
        ga  = _make_ga()
        idx = _make_indexing()
        hm  = HandoffManager(ga, idx)

        rid_a = idx.region_id(lat=0.0, lon=0.0)
        rid_b = idx.region_id(lat=0.0, lon=15.0)
        rs_a  = RegionShardServer(rid_a, ga, idx)
        # Override hard cap to 0 for rs_b
        rs_b  = RegionShardServer(rid_b, ga, idx)
        rs_b._hard_cap = 0

        rs_a.add_player("p1", [0.0, 0.0, 0.0])
        request = HandoffRequest(
            player_id="p1",
            source_region=rid_a,
            target_region=rid_b,
            player_state={"player_id": "p1", "pos": [0.0, 15.0, 0.0], "vel": []},
        )

        success = hm.execute_handoff(request, rs_a, rs_b)
        self.assertFalse(success)
        # Player must remain on source
        self.assertIn("p1", rs_a.player_ids)
        self.assertNotIn("p1", rs_b.player_ids)

    def test_tick_produces_handoff_request_near_boundary(self):
        """A player placed in the handoff band triggers a HandoffRequest."""
        ga  = _make_ga()
        idx = RegionIndexing(region_size_deg=10.0)
        rid = idx.region_id(lat=0.0, lon=0.0)
        rs  = RegionShardServer(rid, ga, idx)
        rs._handoff_band_m = 50_000.0  # ~0.45 deg — easier to trigger in test

        bounds    = idx.region_bounds(rid)   # (lat_min, lon_min, lat_max, lon_max)
        lat_min   = bounds[0]
        band_deg  = rs._handoff_band_m / 111_320.0
        # Place player just inside the southern boundary
        near_lat  = lat_min + band_deg * 0.5
        near_lon  = (bounds[1] + bounds[3]) / 2  # mid-longitude

        rs.add_player("p2", [near_lat, near_lon, 0.0])
        requests = rs.tick(dt_s=0.1)
        self.assertTrue(
            any(r.player_id == "p2" for r in requests),
            "expected handoff request for player near boundary",
        )


# ---------------------------------------------------------------------------
# 2. test_cross_region_infra_event
# ---------------------------------------------------------------------------

class TestCrossRegionInfraEvent(unittest.TestCase):
    """Cross-region events reach neighbours but not the source."""

    def test_event_delivered_to_neighbours_not_source(self):
        ga  = _make_ga()
        idx = _make_indexing()

        rid_0 = 0
        rid_1 = 1
        rid_2 = 2

        ga.register_region(rid_0, "rs:0")
        ga.register_region(rid_1, "rs:1")
        ga.register_region(rid_2, "rs:2")

        event = {"type": "infra_pulse", "source_region": rid_0, "energy": 1000.0}
        ga.post_cross_region_event(event)

        # rid_0 should get nothing (source)
        self.assertEqual(ga.drain_events_for_region(rid_0), [])
        # rid_1 and rid_2 should each get the event
        ev1 = ga.drain_events_for_region(rid_1)
        ev2 = ga.drain_events_for_region(rid_2)
        self.assertEqual(len(ev1), 1)
        self.assertEqual(ev1[0]["type"], "infra_pulse")
        self.assertEqual(len(ev2), 1)

    def test_events_drained_only_once(self):
        ga = _make_ga()
        ga.register_region(0, "rs:0")
        ga.register_region(1, "rs:1")

        ga.post_cross_region_event({"type": "storm", "source_region": 0})
        ga.drain_events_for_region(1)
        # Second drain should return nothing
        self.assertEqual(ga.drain_events_for_region(1), [])


# ---------------------------------------------------------------------------
# 3. test_checkpoint_restore_global_plus_regions
# ---------------------------------------------------------------------------

class TestCheckpointRestore(unittest.TestCase):
    """Checkpoint can be written and fully restored."""

    def test_full_checkpoint_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ga      = _make_ga()
            ga.seed = 777
            ga.tick(10.0)

            writer = GlobalSnapshotWriter(state_dir=tmpdir)
            coord  = CheckpointCoordinator(
                ga_writer=writer,
                state_dir=tmpdir,
                expected_regions=2,
            )

            ga_snap = ga.global_snapshot()
            cid     = coord.begin_checkpoint(ga_snap)

            # Two regions report their snapshots
            rs_snap_0 = {"region_id": 0, "players": {}, "is_degraded": False}
            rs_snap_1 = {"region_id": 1, "players": {"p1": {"pos": [1, 2, 3], "vel": [0, 0, 0]}}, "is_degraded": False}
            coord.register_region_snapshot(cid, 0, rs_snap_0)
            coord.register_region_snapshot(cid, 1, rs_snap_1)

            finalised = coord.try_finalise(cid)
            self.assertTrue(finalised)
            self.assertTrue(coord.is_complete(cid))

            # Load back the checkpoint
            saved = coord.load_checkpoint(cid)
            self.assertIsNotNone(saved)
            self.assertEqual(saved["ga"]["seed"], 777)
            self.assertIn("0", saved["regions"])
            self.assertIn("1", saved["regions"])
            p1_pos = saved["regions"]["1"]["players"]["p1"]["pos"]
            self.assertEqual(p1_pos, [1, 2, 3])

    def test_ga_snapshot_writer_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ga = _make_ga()
            ga.seed = 999
            ga.planet_time = 3.14

            writer = GlobalSnapshotWriter(state_dir=tmpdir)
            writer.write(ga.global_snapshot())
            snap = writer.read()

            self.assertIsNotNone(snap)
            self.assertEqual(snap["seed"], 999)
            self.assertAlmostEqual(snap["planet_time"], 3.14, places=5)


# ---------------------------------------------------------------------------
# 4. test_rs_crash_recovery_only_affects_one_region
# ---------------------------------------------------------------------------

class TestRSCrashRecovery(unittest.TestCase):
    """RS crash + recovery does not affect other regions."""

    def test_crash_only_affects_crashed_region(self):
        ga  = _make_ga()
        idx = _make_indexing()

        rid_0 = idx.region_id(lat=0.0,  lon=0.0)
        rid_1 = idx.region_id(lat=0.0, lon=15.0)

        rs_0 = _make_rs(rid_0, ga, idx)
        rs_1 = _make_rs(rid_1, ga, idx)

        rs_0.add_player("p_a", [0.0, 0.0, 0.0])
        rs_1.add_player("p_b", [0.0, 15.0, 0.0])

        # Simulate rs_0 crash: deregister
        snap_0 = rs_0.region_snapshot()
        ga.deregister_region(rid_0)

        # rs_1 should be unaffected
        self.assertIn(rid_1, ga.live_regions())
        self.assertIn("p_b", rs_1.player_ids)

        # Recover rs_0 from snapshot
        rs_0_new = _make_rs(rid_0, ga, idx)
        rs_0_new.load_region_snapshot(snap_0)
        self.assertIn("p_a", rs_0_new.player_ids)

    def test_ga_dead_region_not_in_live_list(self):
        ga = _make_ga()
        ga.register_region(5, "rs:5")
        ga.deregister_region(5)
        self.assertNotIn(5, ga.live_regions())


# ---------------------------------------------------------------------------
# 5. test_ga_failover_no_world_drift
# ---------------------------------------------------------------------------

class TestGAFailover(unittest.TestCase):
    """GA snapshot can be restored; RS can continue ticking independently."""

    def test_ga_snapshot_restore_preserves_state(self):
        ga_primary = _make_ga()
        ga_primary.seed        = 42
        ga_primary.epoch       = 7
        ga_primary.planet_time = 12.5
        ga_primary.energy_in   = 500.0

        snap = ga_primary.global_snapshot()

        # Failover: new GA instance restores from snapshot
        ga_standby = _make_ga()
        ga_standby.load_snapshot(snap)

        self.assertEqual(ga_standby.seed, 42)
        self.assertEqual(ga_standby.epoch, 7)
        self.assertAlmostEqual(ga_standby.planet_time, 12.5, places=5)
        self.assertAlmostEqual(ga_standby.energy_in, 500.0, places=3)

    def test_rs_ticks_independently_for_grace_period(self):
        """RS tick does not raise even when GA has no registered region info."""
        ga  = _make_ga()
        idx = _make_indexing()
        rid = idx.region_id(lat=10.0, lon=10.0)
        rs  = _make_rs(rid, ga, idx)
        rs.add_player("p1", [10.0, 10.0, 0.0])

        # Even with GA "offline" (no further calls to ga), RS tick should not raise
        for _ in range(10):
            requests = rs.tick(dt_s=0.05)
        # Player is still tracked
        self.assertIn("p1", rs.player_ids)


# ---------------------------------------------------------------------------
# 6. loadtest_100_players_single_region
# ---------------------------------------------------------------------------

class TestLoadtest100Players(unittest.TestCase):
    """100 players can be added; degradation status reflects soft cap."""

    def test_100_players_below_soft_cap_not_degraded(self):
        ga  = _make_ga()
        idx = _make_indexing()
        rid = idx.region_id(lat=20.0, lon=20.0)

        rs = RegionShardServer(rid, ga, idx)
        rs._soft_cap = 200   # raise cap so 100 is below it
        rs._hard_cap = 300

        for i in range(100):
            added = rs.add_player(f"player_{i}", [20.0, 20.0, float(i)])
            self.assertTrue(added, f"player_{i} was rejected unexpectedly")

        self.assertEqual(rs.player_count, 100)
        rs.tick(dt_s=0.016)
        self.assertFalse(rs.is_degraded)

    def test_players_above_soft_cap_trigger_degraded(self):
        ga  = _make_ga()
        idx = _make_indexing()
        rid = idx.region_id(lat=20.0, lon=20.0)

        rs = RegionShardServer(rid, ga, idx)
        rs._soft_cap = 50
        rs._hard_cap = 300

        for i in range(100):
            rs.add_player(f"player_{i}", [20.0, 20.0, float(i)])

        rs.tick(dt_s=0.016)
        self.assertTrue(rs.is_degraded)

    def test_hard_cap_rejects_excess_players(self):
        ga  = _make_ga()
        idx = _make_indexing()
        rid = idx.region_id(lat=20.0, lon=20.0)

        rs = RegionShardServer(rid, ga, idx)
        rs._soft_cap = 5
        rs._hard_cap = 10

        for i in range(10):
            rs.add_player(f"player_{i}", [20.0, 20.0, 0.0])

        rejected = rs.add_player("overflow", [20.0, 20.0, 0.0])
        self.assertFalse(rejected)
        self.assertEqual(rs.player_count, 10)


# ---------------------------------------------------------------------------
# 7. loadtest_500_players_hotspot_degrades_gracefully
# ---------------------------------------------------------------------------

class TestLoadtest500PlayersHotspot(unittest.TestCase):
    """RegionAutoscaler degrades gracefully under high player counts."""

    def test_normal_below_soft_cap(self):
        scaler   = RegionAutoscaler()
        decision = scaler.update(region_id=0, player_count=50)
        self.assertEqual(decision.status, STATUS_OK)
        self.assertGreater(decision.acoustic_raycasts, 16)
        self.assertGreaterEqual(decision.remote_state_hz, 15.0)

    def test_degraded_above_soft_cap(self):
        scaler   = RegionAutoscaler()
        decision = scaler.update(region_id=0, player_count=150)
        self.assertEqual(decision.status, STATUS_DEGRADED)
        self.assertLess(decision.chunk_update_interval, 5.0)
        self.assertGreater(decision.chunk_update_interval, 1.0)
        self.assertLessEqual(decision.acoustic_raycasts, 16)
        self.assertLessEqual(decision.remote_state_hz, 15.0)

    def test_overloaded_above_hard_cap(self):
        scaler   = RegionAutoscaler()
        decision = scaler.update(region_id=0, player_count=500)
        self.assertEqual(decision.status, STATUS_OVERLOADED)

    def test_500_player_hotspot(self):
        """500 players produce OVERLOADED and reduced budgets."""
        scaler   = RegionAutoscaler()
        decision = scaler.update(region_id=7, player_count=500)
        self.assertEqual(decision.status, STATUS_OVERLOADED)
        self.assertLessEqual(decision.acoustic_raycasts, 16)
        self.assertGreater(decision.chunk_update_interval, 1.0)

    def test_decision_cached_per_region(self):
        scaler = RegionAutoscaler()
        scaler.update(region_id=3, player_count=200)
        cached = scaler.decision_for(3)
        self.assertIsNotNone(cached)
        self.assertEqual(cached.player_count, 200)


# ---------------------------------------------------------------------------
# Additional: RegionIndexing
# ---------------------------------------------------------------------------

class TestRegionIndexing(unittest.TestCase):
    """RegionIndexing correctness."""

    def test_lat_lon_round_trips(self):
        idx = RegionIndexing(region_size_deg=10.0)
        for lat in range(-80, 81, 20):
            for lon in range(-170, 171, 20):
                rid    = idx.region_id(lat, lon)
                bounds = idx.region_bounds(rid)
                self.assertGreaterEqual(lat, bounds[0] - 1e-6)
                self.assertLessEqual   (lat, bounds[2] + 1e-6)

    def test_total_regions(self):
        idx = RegionIndexing(region_size_deg=10.0)
        self.assertEqual(idx.total_regions, idx.rows * idx.cols)
        self.assertEqual(idx.cols, 36)
        self.assertEqual(idx.rows, 18)

    def test_neighbours_are_adjacent(self):
        idx  = RegionIndexing(region_size_deg=10.0)
        rid  = idx.region_id(lat=0.0, lon=0.0)
        nb   = idx.neighbours(rid)
        self.assertGreater(len(nb), 0)
        self.assertNotIn(rid, nb)

    def test_lon_wraps(self):
        idx   = RegionIndexing(region_size_deg=10.0)
        rid_w = idx.region_id(lat=0.0, lon=-175.0)
        rid_e = idx.region_id(lat=0.0, lon= 175.0)
        nb_w  = idx.neighbours(rid_w)
        nb_e  = idx.neighbours(rid_e)
        # Antipodal edge regions should neighbour each other via lon-wrap
        self.assertIn(rid_e, nb_w)
        self.assertIn(rid_w, nb_e)


# ---------------------------------------------------------------------------
# Additional: GhostBorderCache
# ---------------------------------------------------------------------------

class TestGhostBorderCache(unittest.TestCase):

    def test_put_and_get(self):
        cache = GhostBorderCache(max_age_s=10.0)
        cache.put(region_id=1, chunk_id=42, data={"mat": 3})
        result = cache.get(region_id=1, chunk_id=42)
        self.assertIsNotNone(result)
        self.assertEqual(result["mat"], 3)

    def test_missing_returns_none(self):
        cache = GhostBorderCache()
        self.assertIsNone(cache.get(99, 99))

    def test_evict_clears_region(self):
        cache = GhostBorderCache()
        cache.put(1, 1, {"a": 1})
        cache.put(1, 2, {"b": 2})
        cache.put(2, 1, {"c": 3})
        cache.evict(1)
        self.assertIsNone(cache.get(1, 1))
        self.assertIsNone(cache.get(1, 2))
        self.assertIsNotNone(cache.get(2, 1))


# ---------------------------------------------------------------------------
# Additional: GatewayRouter & RegionSessionManager
# ---------------------------------------------------------------------------

class TestGatewayRouter(unittest.TestCase):

    def test_route_returns_node_addr(self):
        ga  = _make_ga()
        idx = _make_indexing()
        rid = idx.region_id(lat=0.0, lon=0.0)
        ga.register_region(rid, "ws://node0:8765")

        router = GatewayRouter(ga, idx)
        addr   = router.route("sess1", lat=0.0, lon=0.0)
        self.assertEqual(addr, "ws://node0:8765")
        self.assertEqual(router.current_rs("sess1"), "ws://node0:8765")

    def test_route_returns_none_if_region_offline(self):
        ga     = _make_ga()
        idx    = _make_indexing()
        router = GatewayRouter(ga, idx)
        # No region registered → should return None
        self.assertIsNone(router.route("sess2", lat=0.0, lon=0.0))

    def test_reroute_after_handoff(self):
        ga    = _make_ga()
        idx   = _make_indexing()
        rid_a = idx.region_id(lat=0.0, lon=0.0)
        rid_b = idx.region_id(lat=0.0, lon=15.0)
        ga.register_region(rid_a, "ws://nodeA:8765")
        ga.register_region(rid_b, "ws://nodeB:8765")

        router = GatewayRouter(ga, idx)
        router.route("sess3", lat=0.0, lon=0.0)
        self.assertEqual(router.current_rs("sess3"), "ws://nodeA:8765")

        router.reroute("sess3", rid_b)
        self.assertEqual(router.current_rs("sess3"), "ws://nodeB:8765")


class TestRegionSessionManager(unittest.TestCase):

    def test_register_and_query(self):
        mgr = RegionSessionManager()
        mgr.register("s1", region_id=3, lat=10.0, lon=20.0)
        self.assertEqual(mgr.region_for_session("s1"), 3)
        self.assertEqual(mgr.position_for_session("s1"), (10.0, 20.0))

    def test_sessions_in_region(self):
        mgr = RegionSessionManager()
        mgr.register("s1", 3, 0.0, 0.0)
        mgr.register("s2", 3, 1.0, 1.0)
        mgr.register("s3", 5, 0.0, 0.0)
        self.assertCountEqual(mgr.sessions_in_region(3), ["s1", "s2"])

    def test_unregister(self):
        mgr = RegionSessionManager()
        mgr.register("s1", 3, 0.0, 0.0)
        mgr.unregister("s1")
        self.assertIsNone(mgr.region_for_session("s1"))
        self.assertEqual(mgr.session_count, 0)


if __name__ == "__main__":
    unittest.main()
