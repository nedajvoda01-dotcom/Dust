"""SDFWorld — streaming SDF chunk cache, sampling API, and patch dispatch.

Public API (for use by future physics / geo-event systems):
    sample_signed_distance(world_pos)   →  float
    sample_material(world_pos)          →  int  (MATERIAL_AIR / MATERIAL_ROCK)
    raycast_sdf(origin, dir, max_dist)  →  (hit: bool, hit_pos: Vec3, dist: float)
    get_ground_height_at(unit_dir)      →  float
    get_friction_at(world_pos)          →  float  (stub)
    get_stability_at(world_pos)         →  float  (stub)
    apply_patch(patch)                  →  None
    get_render_meshes()                 →  list[Mesh]

Streaming
---------
``update(player_world_pos)`` refreshes the active chunk set each frame.
Active chunks are determined by the LOD tiles that are closest to the player
(using the same LOD quadtree concept as the surface tile streamer, but limited
to the SDF bubble radius).  Only depth_index=0 (surface-straddling) chunks
are loaded by default; deeper layers can be added in future stages.

Memory limit
------------
``max_cached_chunks`` evicts the least-recently-used chunks when exceeded.
"""
from __future__ import annotations

import math
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

from src.math.Vec3 import Vec3
from src.planet.PlanetHeightProvider import PlanetHeightProvider
from src.planet.PlanetLOD import (
    NUM_FACES, node_bounds, cube_to_sphere, node_center_dir,
)
from src.planet.SDFChunk import (
    SDFChunk, SDFChunkCoord, MATERIAL_AIR, MATERIAL_ROCK,
)
from src.planet.SDFGenerator import generate_chunk
from src.planet.SDFMesher import MarchingCubesMesher
from src.planet.SDFPatchSystem import SDFPatch, SDFPatchLog
from src.render.MeshBuilder import Mesh


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_LOD         = 4     # LOD level for SDF tiles near the player
_DEFAULT_RESOLUTION  = 16    # voxels per side
_DEFAULT_VOXEL_DEPTH = 2.0   # metres per radial voxel
_DEFAULT_DEPTH_RANGE = (0, 0) # depth_index range (0 = surface straddling only)
_DEFAULT_MAX_CHUNKS  = 128


# ---------------------------------------------------------------------------
# SDFWorld
# ---------------------------------------------------------------------------

class SDFWorld:
    """
    Manages the SDF voxel world around the player.

    Parameters
    ----------
    height_provider  : PlanetHeightProvider (or compatible duck-type)
    planet_radius    : planet base radius in simulation units
    seed             : world seed (for future per-chunk noise; currently unused
                       because the base SDF is fully determined by the heightfield)
    sdf_lod          : LOD level of surface tiles used as SDF chunk anchors
    resolution       : voxels per side per chunk
    voxel_depth      : radial metres per voxel
    depth_layers     : (min_depth_index, max_depth_index) inclusive range
    max_cached_chunks: eviction limit
    """

    def __init__(
        self,
        height_provider,
        planet_radius:     float,
        seed:              int   = 42,
        sdf_lod:           int   = _DEFAULT_LOD,
        resolution:        int   = _DEFAULT_RESOLUTION,
        voxel_depth:       float = _DEFAULT_VOXEL_DEPTH,
        depth_layers:      tuple = _DEFAULT_DEPTH_RANGE,
        max_cached_chunks: int   = _DEFAULT_MAX_CHUNKS,
    ) -> None:
        self._hp            = height_provider
        self._radius        = planet_radius
        self._seed          = seed
        self._lod           = sdf_lod
        self._resolution    = resolution
        self._voxel_depth   = voxel_depth
        self._depth_min     = depth_layers[0]
        self._depth_max     = depth_layers[1]
        self._max_chunks    = max_cached_chunks

        self._mesher     = MarchingCubesMesher()
        self._patch_log  = SDFPatchLog()

        # LRU cache: coord → SDFChunk
        self._chunks: OrderedDict[SDFChunkCoord, SDFChunk] = OrderedDict()

        # Angular radius of the SDF bubble (2 tiles at the chosen LOD)
        tile_half_angle  = math.pi / (1 << sdf_lod)
        self._bubble_rad = tile_half_angle * 3.0   # roughly 3-tile radius

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def update(self, player_world_pos: Vec3) -> None:
        """
        Refresh active chunks for the given player position.

        Called once per frame.  Loads chunks near the player and evicts
        distant ones when the cache limit is exceeded.
        """
        player_dir = player_world_pos.normalized()
        needed     = self._needed_coords(player_dir)

        # Generate any missing chunks
        for coord in needed:
            if coord not in self._chunks:
                self._load_chunk(coord)
            else:
                # Touch for LRU ordering
                self._chunks.move_to_end(coord)

        # Evict excess chunks (those not in *needed* and beyond cap)
        if len(self._chunks) > self._max_chunks:
            needed_set = set(needed)
            to_evict   = [
                k for k in list(self._chunks.keys())
                if k not in needed_set
            ]
            for k in to_evict:
                if len(self._chunks) <= self._max_chunks:
                    break
                del self._chunks[k]

    # ------------------------------------------------------------------
    # Sampling API
    # ------------------------------------------------------------------

    def sample_signed_distance(self, world_pos: Vec3) -> float:
        """
        Return the signed distance to the planet surface at *world_pos*.

        Uses the analytical formula:   d = r_point - r_surface(dir)
        where r_surface = planet_radius + height_provider.sample_height(dir).

        This always returns a value (no chunk is required) because the formula
        is the same one used to populate the chunks.
        """
        dist_from_center = world_pos.length()
        if dist_from_center < 1e-12:
            return -self._radius
        unit_dir  = world_pos / dist_from_center
        h_surface = self._hp.sample_height(unit_dir)
        r_surface = self._radius + h_surface
        return dist_from_center - r_surface

    def sample_material(self, world_pos: Vec3) -> int:
        """
        Return MATERIAL_AIR or MATERIAL_ROCK at *world_pos*.

        Tries to find a loaded chunk first; falls back to the analytical SDF.
        """
        chunk = self._find_chunk_for(world_pos)
        if chunk is not None:
            R   = chunk.resolution
            # Find nearest voxel (brute-force; fine for small resolution)
            best_dist_sq = math.inf
            best_mat     = MATERIAL_AIR
            for k in range(R):
                for j in range(R):
                    for i in range(R):
                        px, py, pz = chunk.get_pos(i, j, k)
                        dx = px - world_pos.x
                        dy = py - world_pos.y
                        dz = pz - world_pos.z
                        d2 = dx*dx + dy*dy + dz*dz
                        if d2 < best_dist_sq:
                            best_dist_sq = d2
                            best_mat = chunk.material_field[
                                chunk.flat_index(i, j, k)
                            ]
            return best_mat

        # Fall back to analytical SDF
        d = self.sample_signed_distance(world_pos)
        return MATERIAL_AIR if d >= 0.0 else MATERIAL_ROCK

    def raycast_sdf(
        self,
        origin:   Vec3,
        direction: Vec3,
        max_dist: float,
        step_scale: float = 0.5,
    ) -> Tuple[bool, Vec3, float]:
        """
        Sphere-trace along *direction* from *origin* up to *max_dist*.

        Returns (hit, hit_position, distance_travelled).
        *step_scale* < 1 adds safety margin against over-stepping.
        """
        d_norm = direction.normalized()
        if d_norm.is_near_zero():
            return (False, origin, 0.0)

        t         = 0.0
        min_step  = 0.01
        pos       = origin

        while t < max_dist:
            sdf = self.sample_signed_distance(pos)
            if sdf <= 0.0:
                return (True, pos, t)
            step = max(abs(sdf) * step_scale, min_step)
            t   += step
            pos  = origin + d_norm * t

        return (False, origin + d_norm * max_dist, max_dist)

    # ------------------------------------------------------------------
    # Ground / friction / stability stubs (API for future physics)
    # ------------------------------------------------------------------

    def get_ground_height_at(self, unit_dir: Vec3) -> float:
        """Height offset above planet radius at *unit_dir* (from heightfield)."""
        return self._hp.sample_height(unit_dir.normalized())

    def get_friction_at(self, world_pos: Vec3) -> float:
        """Surface friction at *world_pos*.  Stub — returns constant."""
        return 0.6

    def get_stability_at(self, world_pos: Vec3) -> float:
        """Geological stability at *world_pos*.  Stub — returns constant."""
        return 1.0

    # ------------------------------------------------------------------
    # Patch API
    # ------------------------------------------------------------------

    def apply_patch(self, patch: SDFPatch) -> None:
        """
        Add *patch* to the log and apply it to all currently loaded chunks.
        Affected chunks are marked dirty and their meshes rebuilt.
        """
        self._patch_log.add(patch)
        for coord, chunk in list(self._chunks.items()):
            if patch.apply_to_chunk(chunk):
                self._rebuild_mesh(chunk)

    def patch_log(self) -> SDFPatchLog:
        """Access to the full patch log (for serialisation / replay)."""
        return self._patch_log

    # ------------------------------------------------------------------
    # Render output
    # ------------------------------------------------------------------

    def get_render_meshes(self) -> List[Mesh]:
        """Return all ready meshes for the currently active chunks."""
        return [
            c.mesh_handle
            for c in self._chunks.values()
            if c.mesh_handle is not None and not c.dirty
        ]

    def active_chunk_count(self) -> int:
        return len(self._chunks)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _needed_coords(self, player_dir: Vec3) -> List[SDFChunkCoord]:
        """
        Collect all chunk coords that should be active for *player_dir*.

        Strategy: iterate over all tiles at ``self._lod`` for all 6 faces;
        include those whose centre direction is within the bubble radius.
        """
        coords: List[SDFChunkCoord] = []
        num_tiles = 1 << self._lod
        cos_thresh = math.cos(self._bubble_rad)

        for face in range(NUM_FACES):
            for tx in range(num_tiles):
                for ty in range(num_tiles):
                    cdir     = node_center_dir(face, self._lod, tx, ty)
                    cos_dist = cdir.dot(player_dir)
                    if cos_dist >= cos_thresh:
                        for depth in range(self._depth_min, self._depth_max + 1):
                            coords.append(SDFChunkCoord(
                                face_id     = face,
                                lod         = self._lod,
                                tile_x      = tx,
                                tile_y      = ty,
                                depth_index = depth,
                            ))
        return coords

    def _load_chunk(self, coord: SDFChunkCoord) -> SDFChunk:
        """Generate, patch, mesh, and cache a new chunk."""
        chunk = generate_chunk(
            coord          = coord,
            resolution     = self._resolution,
            voxel_depth    = self._voxel_depth,
            planet_radius  = self._radius,
            height_provider= self._hp,
        )
        # Replay all existing patches
        self._patch_log.apply_to_chunk(chunk)
        # Build mesh
        self._rebuild_mesh(chunk)
        self._chunks[coord] = chunk
        return chunk

    def _rebuild_mesh(self, chunk: SDFChunk) -> None:
        """Run the mesher on *chunk* and clear the dirty flag."""
        mesh            = self._mesher.build_mesh(chunk)
        chunk.mesh_handle = mesh
        chunk.dirty       = False

    def _find_chunk_for(self, world_pos: Vec3) -> Optional[SDFChunk]:
        """Return the loaded chunk containing *world_pos*, or None."""
        if not self._chunks:
            return None
        # Find chunk whose voxels are closest to world_pos on average
        # (simplified: use chunk centre heuristic)
        best_chunk = None
        best_dist  = math.inf
        for chunk in self._chunks.values():
            R  = chunk.resolution
            cx, cy, cz = chunk.get_pos(R // 2, R // 2, R // 2)
            dx = cx - world_pos.x; dy = cy - world_pos.y; dz = cz - world_pos.z
            d2 = dx*dx + dy*dy + dz*dz
            if d2 < best_dist:
                best_dist  = d2
                best_chunk = chunk
        return best_chunk
