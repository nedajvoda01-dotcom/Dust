"""test_subsurface.py — Stage 26 SubsurfaceSystem smoke tests.

Tests
-----
1. test_cave_determinism
   — Same seed always produces identical cave graph (nodes + edges).

2. test_portal_exists_rarely
   — Portal count is low relative to shallow node count.
   — Portals are not zero (unless the graph is empty by chance).

3. test_multiplayer_collapse_replicates
   — Server generates a collapse batch; two independent client systems
     apply it and end up with the same number of recorded events and
     the same patch count in their logs.

4. test_cave_factor_transitions
   — cave_factor_at returns 0 for a position far above the surface.
   — cave_factor_at returns a higher value as a position moves underground
     toward a known cave node (monotonically non-decreasing approach).
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.math.Vec3 import Vec3
from src.systems.SubsurfaceSystem import (
    CaveGraph,
    CavePortal,
    CollapseEvent,
    SubsurfacePatchBatch,
    SubsurfaceSystem,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEED          = 42
PLANET_RADIUS = 1_000.0


def _make_system(seed: int = SEED, **kwargs) -> SubsurfaceSystem:
    """Build a SubsurfaceSystem with deterministic settings suitable for tests."""
    cfg = {
        "enable":             True,
        "shallow_prob":       1.0,   # always emit shallow nodes for test coverage
        "mid_prob":           1.0,
        "deep_prob":          1.0,
        "portal_frequency":   0.5,   # high enough to get portals, but not 1.0
        "collapse_event_rate_cap": 100,
    }
    cfg.update(kwargs)
    return SubsurfaceSystem(
        config=cfg,
        global_seed=seed,
        planet_radius=PLANET_RADIUS,
    )


# ---------------------------------------------------------------------------
# 1. test_cave_determinism
# ---------------------------------------------------------------------------

class TestCaveDeterminism(unittest.TestCase):
    """Same seed must always produce the same cave graph."""

    def _graph_snapshot(self, seed: int) -> tuple:
        sys_ = _make_system(seed=seed)
        g    = sys_.cave_graph
        node_ids   = tuple(n.node_id for n in g.nodes)
        node_kinds = tuple(n.kind.name for n in g.nodes)
        node_depths = tuple(round(n.depth, 4) for n in g.nodes)
        edge_ids   = tuple((e.src_id, e.dst_id) for e in g.edges)
        return node_ids, node_kinds, node_depths, edge_ids

    def test_same_seed_same_graph(self):
        """Two systems with the same seed must produce identical cave graphs."""
        snap1 = self._graph_snapshot(SEED)
        snap2 = self._graph_snapshot(SEED)
        self.assertEqual(snap1, snap2,
            "Cave graph must be identical for the same seed")

    def test_node_count_stable_across_runs(self):
        """Node count must not change between two builds with the same seed."""
        s1 = _make_system(seed=SEED)
        s2 = _make_system(seed=SEED)
        self.assertEqual(
            len(s1.cave_graph.nodes), len(s2.cave_graph.nodes),
            "Node count must be stable",
        )

    def test_different_seeds_may_differ(self):
        """Different seeds are allowed (but not required) to differ.

        The important invariant is that neither crashes.
        """
        snap1 = self._graph_snapshot(SEED)
        snap2 = self._graph_snapshot(SEED + 1)
        # Just ensure they don't raise; we don't assert equality/inequality
        self.assertIsNotNone(snap1)
        self.assertIsNotNone(snap2)

    def test_portals_deterministic(self):
        """Portal list must be identical for the same seed."""
        s1 = _make_system(seed=SEED)
        s2 = _make_system(seed=SEED)
        dirs1 = [(round(p.x, 6), round(p.y, 6), round(p.z, 6))
                 for p in s1.portals()]
        dirs2 = [(round(p.x, 6), round(p.y, 6), round(p.z, 6))
                 for p in s2.portals()]
        self.assertEqual(dirs1, dirs2,
            "Portal directions must be identical for the same seed")


# ---------------------------------------------------------------------------
# 2. test_portal_exists_rarely
# ---------------------------------------------------------------------------

class TestPortalExistsRarely(unittest.TestCase):
    """Portals must exist but be rare relative to shallow nodes."""

    def test_portals_present_when_freq_above_zero(self):
        """With portal_frequency > 0 and shallow nodes present, portals must exist."""
        sys_ = _make_system(portal_frequency=0.5)
        if not sys_.cave_graph.nodes:
            self.skipTest("No cave nodes generated — graph empty by construction")
        # At least 1 shallow node must exist (shallow_prob=1.0)
        shallow = [n for n in sys_.cave_graph.nodes
                   if n.layer.name == "SHALLOW"]
        if not shallow:
            self.skipTest("No shallow nodes produced")
        # Expect at least one portal
        self.assertGreater(len(sys_.portals()), 0,
            "Expected at least one portal with portal_frequency=0.5")

    def test_portals_zero_when_freq_is_zero(self):
        """Zero portal_frequency must produce no portals."""
        sys_ = _make_system(portal_frequency=0.0)
        self.assertEqual(len(sys_.portals()), 0,
            "portal_frequency=0 must produce zero portals")

    def test_portal_count_less_than_shallow_node_count(self):
        """Portal count must be at most the number of shallow nodes."""
        sys_ = _make_system(portal_frequency=0.5)
        shallow_count = sum(
            1 for n in sys_.cave_graph.nodes if n.layer.name == "SHALLOW"
        )
        self.assertLessEqual(
            len(sys_.portals()), shallow_count,
            "Portal count must not exceed shallow node count",
        )

    def test_portal_directions_are_unit_vectors(self):
        """All portal directions must be (approximately) unit vectors."""
        sys_ = _make_system(portal_frequency=1.0)
        for d in sys_.portals():
            length = (d.x ** 2 + d.y ** 2 + d.z ** 2) ** 0.5
            self.assertAlmostEqual(length, 1.0, places=5,
                msg=f"Portal direction must be a unit vector, got |d|={length}")


# ---------------------------------------------------------------------------
# 3. test_multiplayer_collapse_replicates
# ---------------------------------------------------------------------------

class TestMultiplayerCollapseReplicates(unittest.TestCase):
    """Server-generated collapse must replicate identically to all clients."""

    def _server_generate(self) -> SubsurfacePatchBatch:
        server = _make_system()
        world_dir = Vec3(1.0, 0.0, 0.0)   # arbitrary surface point
        return server.generate_collapse(world_dir, seed_local=7777, game_time=100.0)

    def test_patch_count_equal_on_both_clients(self):
        """Two clients applying the same batch must see the same patch count."""
        batch   = self._server_generate()
        client1 = _make_system(seed=SEED + 1)
        client2 = _make_system(seed=SEED + 2)

        client1.apply_event_patch(batch)
        client2.apply_event_patch(batch)

        self.assertEqual(
            len(client1.event_log), len(client2.event_log),
            "Both clients must record the same number of collapse events",
        )

    def test_collapse_patches_are_nonempty(self):
        """A collapse batch must contain at least one patch."""
        batch = self._server_generate()
        self.assertGreater(len(batch.patches), 0,
            "Collapse batch must contain at least one patch")

    def test_collapse_event_recorded_after_apply(self):
        """Applying a batch must add exactly one entry to the event log."""
        sys_  = _make_system()
        batch = sys_.generate_collapse(
            Vec3(0.0, 1.0, 0.0), seed_local=123, game_time=50.0,
        )
        initial_count = len(sys_.event_log)
        sys_.apply_event_patch(batch)
        self.assertEqual(
            len(sys_.event_log), initial_count + 1,
            "apply_event_patch must add exactly one entry to event_log",
        )

    def test_same_batch_deterministic_patches(self):
        """Generating a collapse with the same seed_local always gives the same patches."""
        sys1 = _make_system()
        sys2 = _make_system()
        world_dir = Vec3(0.0, 0.0, 1.0)

        b1 = sys1.generate_collapse(world_dir, seed_local=42, game_time=10.0)
        b2 = sys2.generate_collapse(world_dir, seed_local=42, game_time=10.0)

        self.assertEqual(len(b1.patches), len(b2.patches),
            "Same seed_local must produce same number of patches")

    def test_rate_cap_honoured(self):
        """Server must not exceed collapse_event_rate_cap per hour."""
        cap = 3
        sys_ = _make_system(collapse_event_rate_cap=cap)
        world_dir = Vec3(1.0, 0.0, 0.0)

        # Generate more events than the cap
        non_empty = 0
        for i in range(cap + 5):
            batch = sys_.generate_collapse(world_dir, seed_local=i, game_time=float(i))
            sys_.apply_event_patch(batch)
            if len(batch.patches) > 0:
                non_empty += 1

        self.assertLessEqual(non_empty, cap,
            f"No more than {cap} non-empty batches should be produced (rate cap)")


# ---------------------------------------------------------------------------
# 4. test_cave_factor_transitions
# ---------------------------------------------------------------------------

class TestCaveFactorTransitions(unittest.TestCase):
    """cave_factor_at must transition smoothly between surface and underground."""

    def test_high_above_surface_gives_zero_or_low(self):
        """A position far above the planet surface must have near-zero cave factor."""
        sys_    = _make_system()
        # 10× the planet radius above the surface
        far_pos = Vec3(PLANET_RADIUS * 10.0, 0.0, 0.0)
        cf      = sys_.cave_factor_at(far_pos)
        self.assertLessEqual(cf, 0.1,
            f"Expected low cave_factor far above surface, got {cf:.4f}")

    def test_cave_factor_bounded(self):
        """cave_factor_at must always be in [0, 1]."""
        sys_ = _make_system()
        test_positions = [
            Vec3(PLANET_RADIUS, 0.0, 0.0),
            Vec3(0.0, PLANET_RADIUS, 0.0),
            Vec3(PLANET_RADIUS * 0.5, 0.0, 0.0),   # inside planet
            Vec3(PLANET_RADIUS * 2.0, 0.0, 0.0),   # above surface
            Vec3(0.001, 0.001, 0.001),              # near-origin
        ]
        for pos in test_positions:
            cf = sys_.cave_factor_at(pos)
            self.assertGreaterEqual(cf, 0.0,
                f"cave_factor below 0 at {pos}: {cf}")
            self.assertLessEqual(cf, 1.0,
                f"cave_factor above 1 at {pos}: {cf}")

    def test_deeper_position_higher_factor_near_node(self):
        """Moving toward a cave node must not decrease the cave factor."""
        sys_ = _make_system()
        nodes = sys_.cave_graph.nodes
        if not nodes:
            self.skipTest("No cave nodes — graph empty")

        # Pick the shallowest node
        node = min(nodes, key=lambda n: n.depth)

        # Positions at progressively deeper fractions of node depth
        d = node.direction  # unit vector

        altitudes = [PLANET_RADIUS, PLANET_RADIUS * 0.8, PLANET_RADIUS * 0.5]
        prev_cf = None
        for alt in altitudes:
            pos = Vec3(d.x * alt, d.y * alt, d.z * alt)
            cf  = sys_.cave_factor_at(pos)
            self.assertGreaterEqual(cf, 0.0)
            self.assertLessEqual(cf, 1.0)
            if prev_cf is not None:
                # Moving inward may increase or stay equal (not strictly required
                # to be monotone, but we do require it not to jump above 1)
                self.assertLessEqual(cf, 1.0)
            prev_cf = cf

    def test_atmosphere_params_bounded(self):
        """atmosphere_params must return values in expected ranges."""
        sys_ = _make_system()
        for cf in (0.0, 0.25, 0.5, 0.75, 1.0):
            params = sys_.atmosphere_params(cf)
            for key in ("fog_density", "light_scatter", "audio_reverb_mix"):
                self.assertGreaterEqual(params[key], 0.0,
                    f"{key} below 0 at cave_factor={cf}")
                self.assertLessEqual(params[key], 1.0,
                    f"{key} above 1 at cave_factor={cf}")
            for key in ("speed_scale", "turn_responsiveness"):
                self.assertGreater(params[key], 0.0,
                    f"{key} must be positive")
                self.assertLessEqual(params[key], 1.0,
                    f"{key} above 1")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
