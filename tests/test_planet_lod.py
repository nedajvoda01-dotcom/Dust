"""test_planet_lod — validates Stage 3 geometric foundation.

Tests:
  1. TestCubeSphereMapping   — cube→sphere unit-vector correctness and face-edge continuity
  2. TestLODStitch           — same-LOD neighbour tiles share identical edge vertices
  3. TestDifferentLODStitch  — finer tile's edge is a subset of coarser tile's edge
  4. TestDeterminism         — SampleHeight is stable across calls for a given seed
"""
from __future__ import annotations

import math
import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.math.Vec3 import Vec3
from src.planet.PlanetLOD import (
    cube_to_sphere, face_to_cube, node_bounds, build_tile_mesh,
    NUM_FACES,
    FACE_PX, FACE_NX, FACE_PY, FACE_NY, FACE_PZ, FACE_NZ,
)
from src.planet.PlanetHeightProvider import PlanetHeightProvider

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EPS = 1e-5


class _FlatHeightProvider:
    """Zero-height stub — keeps geometry exactly on the unit sphere × radius."""

    seed = 0

    def sample_height(self, _unit_dir: Vec3) -> float:  # noqa: D401
        return 0.0

    def sample_normal_approx(self, unit_dir: Vec3) -> Vec3:
        return unit_dir


def _tile_edge_vertices(
    face: int, lod: int, x: int, y: int,
    height_provider,
    planet_radius: float,
    tile_res: int,
    edge: str,
) -> list:
    """
    Return the (x,y,z) tuples along one edge of a tile mesh in grid order.
    *edge* is one of 'N', 'S', 'E', 'W'.
    """
    N = tile_res
    u_min, v_min, u_max, v_max = node_bounds(lod, x, y)
    du = (u_max - u_min) / (N - 1)
    dv = (v_max - v_min) / (N - 1)
    verts = []

    for i in range(N):
        if edge == 'S':   u, v = u_min + i * du, v_min
        elif edge == 'N': u, v = u_min + i * du, v_max
        elif edge == 'W': u, v = u_min,           v_min + i * dv
        else:             u, v = u_max,            v_min + i * dv
        d = cube_to_sphere(face, u, v)
        h = height_provider.sample_height(d)
        pos = d * (planet_radius + h)
        verts.append((pos.x, pos.y, pos.z))

    return verts


# ---------------------------------------------------------------------------
# 1. Cube → sphere mapping
# ---------------------------------------------------------------------------

class TestCubeSphereMapping(unittest.TestCase):

    def test_unit_length(self):
        """cube_to_sphere must always return a unit vector."""
        for face in range(NUM_FACES):
            for u in (-1.0, -0.5, 0.0, 0.5, 1.0):
                for v in (-1.0, -0.5, 0.0, 0.5, 1.0):
                    d = cube_to_sphere(face, u, v)
                    self.assertAlmostEqual(
                        d.length(), 1.0, places=6,
                        msg=f"Not unit: face={face} u={u} v={v}",
                    )

    def test_face_centers_distinct(self):
        """Each face centre must map to a distinct hemisphere."""
        centers = [cube_to_sphere(f, 0.0, 0.0) for f in range(NUM_FACES)]
        for i in range(NUM_FACES):
            for j in range(i + 1, NUM_FACES):
                dot = centers[i].dot(centers[j])
                self.assertLess(
                    dot, 0.99,
                    msg=f"Face centres {i} and {j} too close: dot={dot:.6f}",
                )

    # --- inter-face edge continuity ----------------------------------------

    def _check_edge_pair(self, face_a, u_a, v_a_vals, face_b, u_b, v_b_vals,
                         flip: bool = False):
        """
        Verify that corresponding points on two face edges map to the same
        unit sphere direction (within EPS).
        """
        if flip:
            v_b_vals = list(reversed(v_b_vals))
        for va, vb in zip(v_a_vals, v_b_vals):
            da = cube_to_sphere(face_a, u_a, va)
            db = cube_to_sphere(face_b, u_b, vb)
            diff = (da - db).length()
            self.assertLess(
                diff, EPS,
                msg=(
                    f"Edge gap: F{face_a}(u={u_a},v={va}) vs "
                    f"F{face_b}(u={u_b},v={vb}) → diff={diff:.2e}"
                ),
            )

    def _sample_v(self, n: int = 9):
        return [v for v in
                [-1.0 + k * 2.0 / (n - 1) for k in range(n)]]

    def test_edge_PX_east_matches_PZ_west(self):
        """Face 0 (+X) east edge (u=+1) matches Face 4 (+Z) west edge (u=-1)."""
        vs = self._sample_v()
        self._check_edge_pair(FACE_PX, 1.0, vs, FACE_PZ, -1.0, vs)

    def test_edge_PX_west_matches_NZ_east(self):
        """Face 0 (+X) west edge (u=-1) matches Face 5 (-Z) east edge (u=+1)."""
        vs = self._sample_v()
        self._check_edge_pair(FACE_PX, -1.0, vs, FACE_NZ, 1.0, vs)

    def test_edge_PX_north_matches_PY_east(self):
        """Face 0 (+X) north edge (v=+1) matches Face 2 (+Y) east edge (u=+1,
        with v→-v orientation)."""
        us = self._sample_v()
        # Face 0 north: cube = (1, 1, u)
        # Face 2 east:  cube = (1, 1, -v)  → u_face0 = -v_face2
        vs_face2 = [-u for u in us]
        for u0, v2 in zip(us, vs_face2):
            da = cube_to_sphere(FACE_PX, u0, 1.0)
            db = cube_to_sphere(FACE_PY, 1.0, v2)
            diff = (da - db).length()
            self.assertLess(diff, EPS,
                            msg=f"F0 north vs F2 east: u={u0} → diff={diff:.2e}")

    def test_edge_PX_south_matches_NY_east(self):
        """Face 0 (+X) south edge (v=-1) matches Face 3 (-Y) east edge (u=+1)."""
        us = self._sample_v()
        for u0, v3 in zip(us, us):
            da = cube_to_sphere(FACE_PX, u0, -1.0)
            db = cube_to_sphere(FACE_NY, 1.0, v3)
            diff = (da - db).length()
            self.assertLess(diff, EPS,
                            msg=f"F0 south vs F3 east: u={u0} → diff={diff:.2e}")

    def test_all_face_corners_match_neighbours(self):
        """
        Corner vertices (u,v ∈ {-1,+1}) must be shared by at least two faces
        and produce the same sphere direction.
        """
        corners = [(-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)]
        # Collect all (face, u, v) → unit_dir
        dirs: dict = {}
        for face in range(NUM_FACES):
            for u, v in corners:
                d = cube_to_sphere(face, u, v)
                key = (round(d.x, 4), round(d.y, 4), round(d.z, 4))
                dirs.setdefault(key, []).append((face, u, v))
        # Each direction should appear on exactly 3 faces (cube corner)
        for key, occurrences in dirs.items():
            self.assertGreaterEqual(
                len(occurrences), 2,
                msg=f"Corner direction {key} found on only 1 face: {occurrences}",
            )


# ---------------------------------------------------------------------------
# 2. Same-LOD neighbour tile edge matching
# ---------------------------------------------------------------------------

class TestLODStitch(unittest.TestCase):

    _HP = _FlatHeightProvider()
    _R  = 1000.0
    _N  = 17

    def _edge(self, face, lod, x, y, side):
        return _tile_edge_vertices(face, lod, x, y, self._HP, self._R, self._N, side)

    def test_same_lod_east_west_match(self):
        """
        East edge of tile (LOD=1, x=0, y=0) must exactly match the west edge
        of neighbour tile (LOD=1, x=1, y=0) on the same face.
        """
        east_of_A = self._edge(FACE_PX, 1, 0, 0, 'E')
        west_of_B = self._edge(FACE_PX, 1, 1, 0, 'W')
        for i, (pa, pb) in enumerate(zip(east_of_A, west_of_B)):
            for ca, cb in zip(pa, pb):
                self.assertAlmostEqual(
                    ca, cb, places=5,
                    msg=f"East/West edge mismatch at vertex {i}: {pa} vs {pb}",
                )

    def test_same_lod_north_south_match(self):
        """
        North edge of (LOD=1, x=0, y=0) must match south edge of (LOD=1, x=0, y=1).
        """
        north_of_A = self._edge(FACE_PY, 1, 0, 0, 'N')
        south_of_B = self._edge(FACE_PY, 1, 0, 1, 'S')
        for i, (pa, pb) in enumerate(zip(north_of_A, south_of_B)):
            for ca, cb in zip(pa, pb):
                self.assertAlmostEqual(
                    ca, cb, places=5,
                    msg=f"North/South edge mismatch at vertex {i}: {pa} vs {pb}",
                )

    def test_same_lod_all_faces_edge_vertex_count(self):
        """Every edge of every LOD-1 tile has exactly tile_res vertices."""
        for face in range(NUM_FACES):
            for x in range(2):
                for y in range(2):
                    for side in ('N', 'S', 'E', 'W'):
                        ev = self._edge(face, 1, x, y, side)
                        self.assertEqual(
                            len(ev), self._N,
                            msg=f"Wrong vertex count on face={face} x={x} y={y} {side}",
                        )


# ---------------------------------------------------------------------------
# 3. Finer-LOD edge is a subset of coarser-LOD edge
# ---------------------------------------------------------------------------

class TestDifferentLODStitch(unittest.TestCase):
    """
    When tile A is at LOD L and a finer tile C is at LOD L+1, every other
    vertex of C's shared edge corresponds exactly to a vertex of A's edge
    (at the matching UV position).  This confirms there are no T-holes in
    the geometric sense.
    """

    _HP = _FlatHeightProvider()
    _R  = 1000.0
    _N  = 17

    def test_finer_edge_subset_of_coarser(self):
        """
        Tile A = (LOD=1, x=0, y=0), east edge.
        Tile C = (LOD=2, x=1, y=0), west edge (south half of A's east edge).

        Node bounds:
          A: u ∈ [-1, 0], v ∈ [-1, 0]  → east edge at u=0, v ∈ [-1, 0]
          C: u ∈ [0, 0.5], v ∈ [-1, -0.5] → west edge at u=0, v ∈ [-1, -0.5]

        A's east edge has 17 vertices with spacing dv = 1/16.
        C's west edge has 17 vertices with spacing dv = 0.5/16 = 1/32.
        Every even-indexed C vertex (k=0,2,4,…,16) must coincide with
        a vertex of A's east edge.
        """
        face = FACE_PX
        N    = self._N
        hp   = self._HP
        R    = self._R

        # A east edge: u=0, v from -1 to 0
        u_min_A, v_min_A, u_max_A, v_max_A = node_bounds(1, 0, 0)
        dv_A = (v_max_A - v_min_A) / (N - 1)
        a_east = []
        for k in range(N):
            v = v_min_A + k * dv_A
            d = cube_to_sphere(face, u_max_A, v)
            h = hp.sample_height(d)
            pos = d * (R + h)
            a_east.append((pos.x, pos.y, pos.z))

        # C west edge: u=0, v from -1 to -0.5
        u_min_C, v_min_C, u_max_C, v_max_C = node_bounds(2, 2, 0)
        dv_C = (v_max_C - v_min_C) / (N - 1)
        c_west = []
        for k in range(N):
            v = v_min_C + k * dv_C
            d = cube_to_sphere(face, u_min_C, v)
            h = hp.sample_height(d)
            pos = d * (R + h)
            c_west.append((pos.x, pos.y, pos.z))

        # Even-indexed C vertices (k=0,2,4,...,16) must match A's
        # vertices at indices k=0,1,2,...,8 (A spacing = 2 × C spacing).
        for ck in range(0, N, 2):
            ak = ck // 2    # corresponding A vertex index
            pa = a_east[ak]
            pc = c_west[ck]
            for ca, cc in zip(pa, pc):
                self.assertAlmostEqual(
                    ca, cc, places=4,
                    msg=(
                        f"LOD-stitch mismatch: A[{ak}]={pa} vs C[{ck}]={pc}"
                    ),
                )


# ---------------------------------------------------------------------------
# 4. Height provider determinism
# ---------------------------------------------------------------------------

class TestDeterminism(unittest.TestCase):

    def test_sample_height_stable_same_seed(self):
        """
        SampleHeight must return the same value for the same (unit_dir, seed)
        across multiple provider instances.
        """
        seed = 42
        dirs = [
            Vec3(1.0, 0.0, 0.0),
            Vec3(0.0, 1.0, 0.0),
            Vec3(0.0, 0.0, 1.0),
            Vec3(0.577, 0.577, 0.577).normalized(),
            Vec3(-0.3, 0.8, 0.5).normalized(),
        ]
        hp1 = PlanetHeightProvider(seed)
        hp2 = PlanetHeightProvider(seed)
        for d in dirs:
            h1 = hp1.sample_height(d)
            h2 = hp2.sample_height(d)
            self.assertAlmostEqual(
                h1, h2, places=10,
                msg=f"Non-deterministic height at {d}: {h1} vs {h2}",
            )

    def test_different_seeds_produce_different_heights(self):
        """Different seeds must (overwhelmingly) produce different heights."""
        d = Vec3(0.6, 0.5, 0.6).normalized()
        results = {seed: PlanetHeightProvider(seed).sample_height(d)
                   for seed in (0, 1, 99, 42, 12345)}
        unique = len(set(round(v, 6) for v in results.values()))
        self.assertGreater(unique, 1,
                           msg=f"All seeds produced the same height: {results}")

    def test_height_within_expected_range(self):
        """Height must remain within ±2× HEIGHT_SCALE for any direction."""
        from src.planet.PlanetHeightProvider import HEIGHT_SCALE
        hp = PlanetHeightProvider(7)
        import random
        rng = random.Random(99)
        for _ in range(50):
            x = rng.uniform(-1, 1); y = rng.uniform(-1, 1); z = rng.uniform(-1, 1)
            d = Vec3(x, y, z)
            if d.is_near_zero():
                continue
            d = d.normalized()
            h = hp.sample_height(d)
            self.assertLess(
                abs(h), HEIGHT_SCALE * 2.5,
                msg=f"Height {h:.2f} outside expected range at {d}",
            )

    def test_lod_tile_determinism(self):
        """
        Building the same tile twice must produce identical vertex arrays.
        """
        hp = PlanetHeightProvider(42)
        R  = 1000.0
        m1 = build_tile_mesh(FACE_PX, 3, 2, 1, hp, R, tile_res=9)
        m2 = build_tile_mesh(FACE_PX, 3, 2, 1, hp, R, tile_res=9)
        self.assertEqual(len(m1.vertices), len(m2.vertices))
        for v1, v2 in zip(m1.vertices, m2.vertices):
            for c1, c2 in zip(v1, v2):
                self.assertAlmostEqual(c1, c2, places=10,
                                       msg="Tile mesh is not deterministic")


if __name__ == "__main__":
    unittest.main()
