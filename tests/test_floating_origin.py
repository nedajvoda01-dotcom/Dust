"""test_floating_origin — validates FloatingOrigin precision management."""
import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.math.Vec3 import Vec3
from src.math.FloatingOrigin import FloatingOrigin


class _SceneObj:
    """Minimal stand-in for a scene object with a local_position."""
    def __init__(self, local_pos: Vec3) -> None:
        self.local_position = local_pos


class TestFloatingOrigin(unittest.TestCase):
    def test_no_rebase_below_threshold(self):
        fo = FloatingOrigin(rebase_threshold=2000.0, planet_radius=1000.0)
        fo.local_position = Vec3(100.0, 1000.0, 0.0)
        rebased = fo.try_rebase([])
        self.assertFalse(rebased)

    def test_rebase_triggered_above_threshold(self):
        fo = FloatingOrigin(rebase_threshold=500.0, planet_radius=1000.0)
        fo.local_position = Vec3(0.0, 600.0, 0.0)
        rebased = fo.try_rebase([])
        self.assertTrue(rebased)

    def test_local_pos_small_after_rebase(self):
        fo = FloatingOrigin(rebase_threshold=500.0, planet_radius=1000.0)
        large_pos = Vec3(0.0, 600.0, 0.0)
        fo.local_position = large_pos
        fo.try_rebase([])
        self.assertLess(fo.local_position.length(), 1e-6)

    def test_world_position_preserved_after_rebase(self):
        fo = FloatingOrigin(rebase_threshold=500.0, planet_radius=1000.0)
        large_pos = Vec3(300.0, 400.0, 0.0)
        fo.local_position = large_pos
        world_before = fo.world_position()
        fo.try_rebase([])
        world_after = fo.world_position()
        diff = (world_after - world_before).length()
        self.assertLess(diff, 1e-5, f"World position changed after rebase: {diff}")

    def test_relative_distance_preserved_after_rebase(self):
        """Objects' relative distance must not change after rebase."""
        fo = FloatingOrigin(rebase_threshold=500.0, planet_radius=1000.0)
        fo.local_position = Vec3(0.0, 600.0, 0.0)

        a = _SceneObj(Vec3(10.0, 600.0, 0.0))
        b = _SceneObj(Vec3(10.0, 605.0, 3.0))
        dist_before = (a.local_position - b.local_position).length()

        fo.try_rebase([a, b])
        dist_after = (a.local_position - b.local_position).length()
        self.assertAlmostEqual(dist_before, dist_after, places=5)

    def test_multiple_rebases_accumulate_offset(self):
        fo = FloatingOrigin(rebase_threshold=100.0, planet_radius=1000.0)
        total_offset = Vec3(0.0, 0.0, 0.0)
        for i in range(5):
            move = Vec3(0.0, float(150 * (i + 1)), 0.0)
            fo.local_position = move
            fo.try_rebase([])
            total_offset = fo.origin_offset
        # Origin offset should be non-zero
        self.assertGreater(total_offset.length(), 0.0)

    def test_to_local_round_trip(self):
        fo = FloatingOrigin(rebase_threshold=1000.0)
        world = Vec3(500.0, 200.0, 300.0)
        local = fo.to_local(world)
        world2 = local + fo.origin_offset
        diff = (world2 - world).length()
        self.assertLess(diff, 1e-9)


if __name__ == "__main__":
    unittest.main()
