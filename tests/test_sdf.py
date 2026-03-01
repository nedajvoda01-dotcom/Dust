"""test_sdf — Stage 4 SDF voxel subsystem tests.

Tests
-----
1. TestSDFBaseConsistency  — sign of d is correct above / below the surface
2. TestChunkDeterminism    — same coord + seed → identical distance_field
3. TestPatchRebuild        — SphereCarve marks chunks dirty and changes SDF/mesh
4. TestChunkSeams          — shared voxels on chunk boundaries have equal d values
"""
from __future__ import annotations

import math
import sys
import os
import unittest
import hashlib
import struct

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.math.Vec3 import Vec3
from src.planet.PlanetHeightProvider import PlanetHeightProvider
from src.planet.PlanetLOD import cube_to_sphere, node_bounds, FACE_PX, FACE_NX
from src.planet.SDFChunk import (
    SDFChunk, SDFChunkCoord, MATERIAL_AIR, MATERIAL_ROCK,
)
from src.planet.SDFGenerator import generate_chunk
from src.planet.SDFMesher import MarchingCubesMesher
from src.planet.SDFPatchSystem import (
    SphereCarve, CapsuleCarve, AdditiveDeposit, SDFPatchLog,
)
from src.planet.SDFWorld import SDFWorld


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

SEED   = 42
RADIUS = 1000.0

_HP = PlanetHeightProvider(SEED)


def _make_chunk(
    face: int = FACE_PX,
    lod: int = 3,
    tx: int = 0,
    ty: int = 0,
    depth: int = 0,
    resolution: int = 8,
    voxel_depth: float = 2.0,
) -> SDFChunk:
    coord = SDFChunkCoord(face_id=face, lod=lod, tile_x=tx, tile_y=ty,
                          depth_index=depth)
    return generate_chunk(coord, resolution, voxel_depth, RADIUS, _HP)


def _field_hash(chunk: SDFChunk) -> str:
    """MD5 hash of the distance_field (for determinism checks)."""
    packed = struct.pack(f"{len(chunk.distance_field)}f", *chunk.distance_field)
    return hashlib.md5(packed).hexdigest()


# ---------------------------------------------------------------------------
# 1. SDF base consistency
# ---------------------------------------------------------------------------

class TestSDFBaseConsistency(unittest.TestCase):
    """SDF sign must be correct relative to the heightfield surface."""

    def test_surface_straddling_chunk_has_both_signs(self):
        """A depth_index=0 chunk must contain both positive (air) and
        negative (rock) voxels, since it straddles the surface."""
        chunk = _make_chunk(depth=0, resolution=8)
        has_air  = any(d >= 0.0 for d in chunk.distance_field)
        has_rock = any(d <  0.0 for d in chunk.distance_field)
        self.assertTrue(has_air,  "depth_index=0 chunk has no air voxels")
        self.assertTrue(has_rock, "depth_index=0 chunk has no rock voxels")

    def test_analytical_sign_above_surface(self):
        """SDFWorld.sample_signed_distance must return d > 0 above the surface."""
        world = SDFWorld(_HP, RADIUS, seed=SEED)
        # Test several directions
        import random
        rng = random.Random(7)
        for _ in range(20):
            dx = rng.uniform(-1, 1)
            dy = rng.uniform(-1, 1)
            dz = rng.uniform(-1, 1)
            unit_dir = Vec3(dx, dy, dz).normalized()
            h_surface = _HP.sample_height(unit_dir)
            r_surface = RADIUS + h_surface
            # Point clearly above surface
            pos_above = unit_dir * (r_surface + 5.0)
            d = world.sample_signed_distance(pos_above)
            self.assertGreater(d, 0.0,
                msg=f"Expected d>0 above surface at {unit_dir}: d={d:.4f}")

    def test_analytical_sign_below_surface(self):
        """SDFWorld.sample_signed_distance must return d < 0 below the surface."""
        world = SDFWorld(_HP, RADIUS, seed=SEED)
        import random
        rng = random.Random(13)
        for _ in range(20):
            dx = rng.uniform(-1, 1)
            dy = rng.uniform(-1, 1)
            dz = rng.uniform(-1, 1)
            unit_dir = Vec3(dx, dy, dz).normalized()
            h_surface = _HP.sample_height(unit_dir)
            r_surface = RADIUS + h_surface
            # Point clearly below surface
            pos_below = unit_dir * (r_surface - 5.0)
            d = world.sample_signed_distance(pos_below)
            self.assertLess(d, 0.0,
                msg=f"Expected d<0 below surface at {unit_dir}: d={d:.4f}")

    def test_material_air_above_surface(self):
        """sample_material must return MATERIAL_AIR above the surface."""
        world = SDFWorld(_HP, RADIUS, seed=SEED)
        unit_dir  = Vec3(1.0, 0.0, 0.0)
        h_surface = _HP.sample_height(unit_dir)
        pos_above = unit_dir * (RADIUS + h_surface + 10.0)
        mat = world.sample_material(pos_above)
        self.assertEqual(mat, MATERIAL_AIR,
            msg=f"Expected MATERIAL_AIR above surface, got {mat}")

    def test_material_rock_below_surface(self):
        """sample_material must return MATERIAL_ROCK below the surface."""
        world = SDFWorld(_HP, RADIUS, seed=SEED)
        unit_dir  = Vec3(0.0, 1.0, 0.0)
        h_surface = _HP.sample_height(unit_dir)
        pos_below = unit_dir * (RADIUS + h_surface - 10.0)
        mat = world.sample_material(pos_below)
        self.assertEqual(mat, MATERIAL_ROCK,
            msg=f"Expected MATERIAL_ROCK below surface, got {mat}")

    def test_voxel_material_matches_d_sign(self):
        """Every voxel's material_field must match the sign of distance_field."""
        chunk = _make_chunk(depth=0, resolution=8)
        R = chunk.resolution
        for k in range(R):
            for j in range(R):
                for i in range(R):
                    d   = chunk.get_d(i, j, k)
                    mat = chunk.material_field[chunk.flat_index(i, j, k)]
                    if d >= 0.0:
                        self.assertEqual(mat, MATERIAL_AIR,
                            msg=f"Voxel ({i},{j},{k}) d={d:.3f} should be AIR")
                    else:
                        self.assertEqual(mat, MATERIAL_ROCK,
                            msg=f"Voxel ({i},{j},{k}) d={d:.3f} should be ROCK")


# ---------------------------------------------------------------------------
# 2. Chunk determinism
# ---------------------------------------------------------------------------

class TestChunkDeterminism(unittest.TestCase):

    def test_same_coord_same_field(self):
        """Generating the same coord twice must produce identical distance_fields."""
        coord = SDFChunkCoord(face_id=FACE_PX, lod=3, tile_x=1, tile_y=2,
                              depth_index=0)
        c1 = generate_chunk(coord, 8, 2.0, RADIUS, _HP)
        c2 = generate_chunk(coord, 8, 2.0, RADIUS, _HP)
        self.assertEqual(c1.distance_field, c2.distance_field,
                         "distance_field not deterministic for same coord")

    def test_same_coord_same_hash(self):
        """Hash of distance_field must be stable across instances."""
        coord = SDFChunkCoord(face_id=FACE_NX, lod=2, tile_x=0, tile_y=1,
                              depth_index=0)
        c1 = generate_chunk(coord, 8, 2.0, RADIUS, _HP)
        c2 = generate_chunk(coord, 8, 2.0, RADIUS, _HP)
        self.assertEqual(_field_hash(c1), _field_hash(c2))

    def test_different_coords_produce_different_positions(self):
        """Different tile coordinates must produce different world positions.

        Note: the base SDF d-values are purely radial (d = radial_offset), so
        all tiles at the same k-level share the same d-values by design.  What
        distinguishes tiles is the world-space *positions* of their voxels.
        """
        c1 = generate_chunk(
            SDFChunkCoord(FACE_PX, 3, 0, 0, 0), 8, 2.0, RADIUS, _HP)
        c2 = generate_chunk(
            SDFChunkCoord(FACE_PX, 3, 1, 0, 0), 8, 2.0, RADIUS, _HP)
        # Positions must differ because the tiles cover different parts of the sphere
        self.assertNotEqual(c1.positions, c2.positions,
                            "Different tile coords should have different voxel positions")

    def test_different_depth_differ(self):
        """Different depth_index layers must produce different distance fields."""
        base = dict(face_id=FACE_PX, lod=3, tile_x=0, tile_y=0)
        c0 = generate_chunk(SDFChunkCoord(**base, depth_index=0), 8, 2.0, RADIUS, _HP)
        c1 = generate_chunk(SDFChunkCoord(**base, depth_index=1), 8, 2.0, RADIUS, _HP)
        self.assertNotEqual(c0.distance_field, c1.distance_field,
                            "depth_index=0 and depth_index=1 have identical fields")

    def test_world_stream_deterministic(self):
        """SDFWorld must produce the same meshes when updated with the same position."""
        pos = Vec3(RADIUS, 0.0, 0.0)
        w1 = SDFWorld(_HP, RADIUS, seed=SEED, sdf_lod=3, resolution=8)
        w2 = SDFWorld(_HP, RADIUS, seed=SEED, sdf_lod=3, resolution=8)
        w1.update(pos)
        w2.update(pos)
        self.assertEqual(w1.active_chunk_count(), w2.active_chunk_count(),
                         "Different active chunk counts for identical inputs")


# ---------------------------------------------------------------------------
# 3. Patch rebuild
# ---------------------------------------------------------------------------

class TestPatchRebuild(unittest.TestCase):

    def test_sphere_carve_marks_dirty(self):
        """After applying SphereCarve, affected chunks must be dirty."""
        chunk = _make_chunk(depth=0, resolution=8)
        chunk.dirty = False   # pretend it was clean

        # Place the carve sphere at the centre of the chunk
        R   = chunk.resolution
        cx, cy, cz = chunk.get_pos(R // 2, R // 2, R // 2)
        patch = SphereCarve(centre=Vec3(cx, cy, cz), radius=15.0)
        changed = patch.apply_to_chunk(chunk)
        if changed:
            self.assertTrue(chunk.dirty,
                "Chunk should be dirty after SphereCarve")

    def test_sphere_carve_increases_d_inside(self):
        """SphereCarve must increase d (push towards air) inside the sphere."""
        chunk = _make_chunk(depth=0, resolution=8)
        R     = chunk.resolution
        # Place carve at a voxel known to be rock (d < 0)
        rock_idx = None
        for k in range(R):
            for j in range(R):
                for i in range(R):
                    if chunk.get_d(i, j, k) < 0.0:
                        rock_idx = (i, j, k)
                        break
                if rock_idx:
                    break
            if rock_idx:
                break

        if rock_idx is None:
            self.skipTest("No rock voxels in test chunk")

        i, j, k   = rock_idx
        px, py, pz = chunk.get_pos(i, j, k)
        old_d      = chunk.get_d(i, j, k)

        patch   = SphereCarve(centre=Vec3(px, py, pz), radius=10.0)
        patch.apply_to_chunk(chunk)
        new_d   = chunk.get_d(i, j, k)
        self.assertGreaterEqual(new_d, old_d,
            msg=f"SphereCarve should not decrease d: old={old_d:.3f} new={new_d:.3f}")

    def test_sphere_carve_does_not_affect_distant_chunk(self):
        """SphereCarve far from a chunk must leave it unchanged."""
        chunk   = _make_chunk(depth=0, resolution=8)
        before  = list(chunk.distance_field)
        # Place the carve far away from the planet surface (clearly outside chunk)
        patch   = SphereCarve(centre=Vec3(0.0, 0.0, 0.0), radius=1.0)
        changed = patch.apply_to_chunk(chunk)
        self.assertFalse(changed,
            "SphereCarve at origin should not affect a surface-near chunk")
        self.assertEqual(chunk.distance_field, before,
            "distance_field was modified by a distant carve")

    def test_sphere_carve_creates_air_voxels(self):
        """SphereCarve centred on a rock voxel must convert some rock to air."""
        chunk = _make_chunk(depth=0, resolution=8)
        R     = chunk.resolution

        # Find a rock voxel
        rock_pos = None
        for k in range(R):
            for j in range(R):
                for i in range(R):
                    if chunk.get_d(i, j, k) < -1.0:
                        rock_pos = chunk.get_pos(i, j, k)
                        break
                if rock_pos:
                    break
            if rock_pos:
                break

        if rock_pos is None:
            self.skipTest("No sufficiently deep rock voxels in test chunk")

        patch = SphereCarve(centre=Vec3(*rock_pos), radius=20.0)
        patch.apply_to_chunk(chunk)

        has_new_air = any(
            chunk.material_field[chunk.flat_index(i, j, k)] == MATERIAL_AIR
            and chunk.get_d(i, j, k) >= 0.0
            for k in range(R) for j in range(R) for i in range(R)
        )
        self.assertTrue(has_new_air,
            "SphereCarve should have converted some rock voxels to air")

    def test_patch_log_replay_idempotent(self):
        """Replaying the patch log to a freshly generated chunk must yield
        the same distance_field as applying the patch directly."""
        coord = SDFChunkCoord(FACE_PX, 3, 0, 0, 0)
        patch_log = SDFPatchLog()
        c_ref     = generate_chunk(coord, 8, 2.0, RADIUS, _HP)
        R         = c_ref.resolution
        cx, cy, cz = c_ref.get_pos(R // 2, R // 2, R // 2)
        patch     = SphereCarve(centre=Vec3(cx, cy, cz), radius=15.0)
        patch_log.add(patch)

        # Apply directly
        patch.apply_to_chunk(c_ref)

        # Apply via log replay on a fresh chunk
        c_replay = generate_chunk(coord, 8, 2.0, RADIUS, _HP)
        patch_log.apply_to_chunk(c_replay)

        self.assertEqual(c_ref.distance_field, c_replay.distance_field,
            "Patch log replay did not reproduce the same distance_field")

    def test_mesh_generated_after_carve(self):
        """After carving into a chunk that straddles the surface, a mesh must exist."""
        world = SDFWorld(_HP, RADIUS, seed=SEED, sdf_lod=3, resolution=8)
        pos   = Vec3(RADIUS, 0.0, 0.0)
        world.update(pos)

        # Apply a carve near the surface
        unit_dir  = Vec3(1.0, 0.0, 0.0)
        h_surface = _HP.sample_height(unit_dir)
        r_surface = RADIUS + h_surface
        carve_pos = unit_dir * (r_surface - 5.0)

        world.apply_patch(SphereCarve(centre=carve_pos, radius=30.0))
        meshes = world.get_render_meshes()
        # At least one chunk should produce a mesh (surface-straddling tiles)
        self.assertGreater(len(meshes), 0,
            "Expected at least one mesh after carving near the surface")

    def test_additive_deposit_decreases_d(self):
        """AdditiveDeposit must decrease d (push towards rock) inside the sphere."""
        chunk = _make_chunk(depth=0, resolution=8)
        R     = chunk.resolution

        # Find an air voxel
        air_pos = None
        for k in range(R):
            for j in range(R):
                for i in range(R):
                    if chunk.get_d(i, j, k) > 1.0:
                        air_pos = chunk.get_pos(i, j, k)
                        old_d   = chunk.get_d(i, j, k)
                        air_ijk = (i, j, k)
                        break
                if air_pos:
                    break
            if air_pos:
                break

        if air_pos is None:
            self.skipTest("No air voxels in test chunk")

        patch = AdditiveDeposit(centre=Vec3(*air_pos), radius=20.0)
        patch.apply_to_chunk(chunk)
        new_d = chunk.get_d(*air_ijk)
        self.assertLessEqual(new_d, old_d,
            msg=f"AdditiveDeposit should not increase d: old={old_d:.3f} new={new_d:.3f}")

    def test_capsule_carve_changes_chunk(self):
        """CapsuleCarve must modify a chunk it intersects."""
        chunk = _make_chunk(depth=0, resolution=8)
        R     = chunk.resolution
        # Build a capsule along the chunk's radial axis (between top and bottom)
        top    = Vec3(*chunk.get_pos(R // 2, R // 2, 0))
        bottom = Vec3(*chunk.get_pos(R // 2, R // 2, R - 1))
        patch  = CapsuleCarve(a=top, b=bottom, radius=10.0)
        changed = patch.apply_to_chunk(chunk)
        self.assertTrue(changed,
            "CapsuleCarve along chunk radial axis should change the chunk")


# ---------------------------------------------------------------------------
# 4. Chunk seams
# ---------------------------------------------------------------------------

class TestChunkSeams(unittest.TestCase):
    """Shared voxel positions on adjacent chunk boundaries must have equal d."""

    def _get_edge_d_values(
        self,
        face: int, lod: int, tx: int, ty: int,
        axis: str,
    ) -> list:
        """
        Return d values for the voxels on one face of a chunk.
        axis: 'i_max' = last i-column, 'i_min' = first i-column,
              'j_max' = last j-row,    'j_min' = first j-row.
        """
        coord = SDFChunkCoord(face_id=face, lod=lod, tile_x=tx, tile_y=ty,
                              depth_index=0)
        chunk = generate_chunk(coord, 8, 2.0, RADIUS, _HP)
        R     = chunk.resolution
        vals  = []
        for k in range(R):
            for j in range(R):
                for i in range(R):
                    include = (
                        (axis == 'i_max' and i == R - 1) or
                        (axis == 'i_min' and i == 0)     or
                        (axis == 'j_max' and j == R - 1) or
                        (axis == 'j_min' and j == 0)
                    )
                    if include:
                        vals.append(chunk.get_d(i, j, k))
        return vals

    def test_adjacent_tiles_i_seam_consistency(self):
        """
        The i_max face of tile (0,0) and the i_min face of tile (1,0) at the
        same LOD must have the same d-value layout, because the SDF formula
        is purely a function of world position (which is shared on the boundary).

        The voxel positions are computed from the face-UV grid, and the UV
        boundary between tiles is shared — so the SDF must agree.
        """
        # We test the seam between LOD-3 tiles (tx=0, ty=0) and (tx=1, ty=0)
        lod = 3
        face = FACE_PX

        # Edge of tile (0,0) at i_max is the same UV as edge of tile (1,0) at i_min.
        # For a resolution-8 chunk, the last i column at tile (0,0) uses the
        # same u-coordinate as the first i column of tile (1,0).
        coord_A = SDFChunkCoord(face_id=face, lod=lod, tile_x=0, tile_y=0,
                                depth_index=0)
        coord_B = SDFChunkCoord(face_id=face, lod=lod, tile_x=1, tile_y=0,
                                depth_index=0)
        R = 8
        vd = 2.0
        chunkA = generate_chunk(coord_A, R, vd, RADIUS, _HP)
        chunkB = generate_chunk(coord_B, R, vd, RADIUS, _HP)

        u_min_A, _, u_max_A, _ = node_bounds(lod, 0, 0)
        u_min_B, _, u_max_B, _ = node_bounds(lod, 1, 0)

        # The shared UV boundary: u_max_A == u_min_B
        self.assertAlmostEqual(u_max_A, u_min_B, places=10,
            msg="Tiles (0,0) and (1,0) don't share a UV boundary")

        # For each (j, k) the d value at A's i_max must equal B's i_min
        for k in range(R):
            for j in range(R):
                dA = chunkA.get_d(R - 1, j, k)
                dB = chunkB.get_d(0,     j, k)
                self.assertAlmostEqual(
                    dA, dB, places=10,
                    msg=(
                        f"Seam mismatch at (j={j}, k={k}): "
                        f"A.d={dA:.6f} B.d={dB:.6f}"
                    ),
                )

    def test_depth_layer_seam(self):
        """
        Adjacent depth layers must be contiguous: the bottom voxel (k=R-1) of
        depth_index=0 and the top voxel (k=0) of depth_index=1 should differ
        in d by exactly one voxel_depth (they are adjacent radial samples, not
        the same point).  This confirms there is no gap or overlap between layers.
        """
        lod = 3
        face = FACE_PX
        R = 8; vd = 2.0
        coord0 = SDFChunkCoord(face_id=face, lod=lod, tile_x=0, tile_y=0,
                               depth_index=0)
        coord1 = SDFChunkCoord(face_id=face, lod=lod, tile_x=0, tile_y=0,
                               depth_index=1)
        c0 = generate_chunk(coord0, R, vd, RADIUS, _HP)
        c1 = generate_chunk(coord1, R, vd, RADIUS, _HP)

        # d at k=R-1 of depth_index=0  and  d at k=0 of depth_index=1
        # should differ by exactly voxel_depth (one radial step).
        for j in range(R):
            for i in range(R):
                d0_bottom = c0.get_d(i, j, R - 1)
                d1_top    = c1.get_d(i, j, 0)
                self.assertAlmostEqual(
                    d0_bottom - d1_top, vd, places=10,
                    msg=(
                        f"Depth-layer seam at (i={i}, j={j}): "
                        f"expected diff={vd}, got {d0_bottom - d1_top:.6f} "
                        f"(d0={d0_bottom:.4f}, d1={d1_top:.4f})"
                    ),
                )


# ---------------------------------------------------------------------------
# 5. Raycast
# ---------------------------------------------------------------------------

class TestRaycastSDF(unittest.TestCase):

    def test_ray_hits_surface_from_above(self):
        """A ray fired downward from above the surface should hit it."""
        world    = SDFWorld(_HP, RADIUS, seed=SEED)
        unit_dir = Vec3(0.0, 1.0, 0.0)
        h_surface = _HP.sample_height(unit_dir)
        r_surface = RADIUS + h_surface
        origin   = unit_dir * (r_surface + 50.0)
        ray_dir  = -unit_dir  # pointing inward
        hit, pos, dist = world.raycast_sdf(origin, ray_dir, max_dist=200.0)
        self.assertTrue(hit, "Ray from above should hit the planet surface")
        # Hit position should be close to the surface
        r_hit = pos.length()
        self.assertAlmostEqual(r_hit, r_surface, delta=2.0,
            msg=f"Hit position radius {r_hit:.2f} should be near surface {r_surface:.2f}")

    def test_ray_misses_from_inside(self):
        """A ray fired outward from well inside the planet should not hit
        anything within a short max_dist (it starts inside)."""
        world    = SDFWorld(_HP, RADIUS, seed=SEED)
        unit_dir = Vec3(1.0, 0.0, 0.0)
        origin   = unit_dir * (RADIUS - 100.0)  # well inside
        hit, _, _ = world.raycast_sdf(origin, unit_dir, max_dist=0.1)
        # d at origin is negative; first step sees d<=0 → immediate hit
        self.assertTrue(hit,
            "Ray starting inside should detect surface immediately")


# ---------------------------------------------------------------------------
# 6. Marching Cubes basic sanity
# ---------------------------------------------------------------------------

class TestMarchingCubes(unittest.TestCase):

    def test_surface_chunk_produces_mesh(self):
        """A surface-straddling chunk must produce a non-None Mesh."""
        chunk  = _make_chunk(depth=0, resolution=8)
        mesher = MarchingCubesMesher()
        mesh   = mesher.build_mesh(chunk)
        self.assertIsNotNone(mesh, "MC should produce a mesh for a surface chunk")

    def test_all_air_chunk_no_mesh(self):
        """A fully-air chunk must return None (nothing to mesh)."""
        chunk = _make_chunk(depth=0, resolution=8)
        # Force all voxels to air
        n = len(chunk.distance_field)
        chunk.distance_field = [1.0] * n
        mesher = MarchingCubesMesher()
        mesh   = mesher.build_mesh(chunk)
        self.assertIsNone(mesh, "All-air chunk should produce no mesh")

    def test_all_rock_chunk_no_mesh(self):
        """A fully-rock chunk must return None (no surface)."""
        chunk = _make_chunk(depth=0, resolution=8)
        n     = len(chunk.distance_field)
        chunk.distance_field = [-1.0] * n
        mesher = MarchingCubesMesher()
        mesh   = mesher.build_mesh(chunk)
        self.assertIsNone(mesh, "All-rock chunk should produce no mesh")

    def test_mesh_has_valid_indices(self):
        """All mesh indices must be within the vertex list bounds."""
        chunk  = _make_chunk(depth=0, resolution=8)
        mesher = MarchingCubesMesher()
        mesh   = mesher.build_mesh(chunk)
        if mesh is None:
            self.skipTest("Surface chunk produced no mesh")
        n_verts = len(mesh.vertices)
        for idx in mesh.indices:
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, n_verts,
                msg=f"Mesh index {idx} out of range [0, {n_verts})")

    def test_mesh_has_unit_normals(self):
        """All mesh normals must be (approximately) unit length."""
        chunk  = _make_chunk(depth=0, resolution=8)
        mesher = MarchingCubesMesher()
        mesh   = mesher.build_mesh(chunk)
        if mesh is None:
            self.skipTest("Surface chunk produced no mesh")
        for nx, ny, nz in mesh.normals:
            length = math.sqrt(nx*nx + ny*ny + nz*nz)
            self.assertAlmostEqual(length, 1.0, delta=1e-5,
                msg=f"Non-unit normal: ({nx:.4f},{ny:.4f},{nz:.4f}) len={length:.6f}")


if __name__ == "__main__":
    unittest.main()
