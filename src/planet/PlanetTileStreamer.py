"""PlanetTileStreamer — LOD quadtree update, job queue, tile streaming.

Responsibilities:
  * Maintain one QuadtreeNode tree per cube face (6 total).
  * Each frame: walk trees, decide which nodes to split / merge based on
    angular distance to the player.
  * Schedule BuildTileMeshJob tasks for newly-visible leaf nodes.
  * Process a bounded number of jobs per frame (job-queue scaffold for
    future async/threaded parallelism).
  * Return the set of READY leaf meshes for rendering.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from src.math.Vec3 import Vec3
from src.planet.PlanetLOD import (
    QuadtreeNode, NodeState, NUM_FACES,
    build_tile_mesh,
)
from src.render.MeshBuilder import Mesh


# ---------------------------------------------------------------------------
# Job type (scaffold for future parallelism)
# ---------------------------------------------------------------------------

@dataclass
class BuildTileMeshJob:
    """Pending tile-mesh build request."""
    face: int
    lod:  int
    x:    int
    y:    int
    node: QuadtreeNode


# ---------------------------------------------------------------------------
# PlanetTileStreamer
# ---------------------------------------------------------------------------

class PlanetTileStreamer:
    """
    Manages LOD quadtrees for all six planet faces and streams tile meshes.

    Call ``update(player_unit_dir)`` every frame to refresh the LOD tree
    and process pending build jobs.
    """

    def __init__(
        self,
        height_provider,
        planet_radius: float,
        min_lod: int = 0,
        max_lod: int = 7,
        tile_res: int = 17,
        max_active_tiles: int = 256,
        debug: bool = False,
    ) -> None:
        self._hp              = height_provider
        self._radius          = planet_radius
        self._min_lod         = min_lod
        self._max_lod         = max_lod
        self._tile_res        = tile_res
        self._max_active      = max_active_tiles
        self._debug           = debug

        # Split threshold per LOD level: split a node when the player's
        # arc-distance to the node centre is less than this value.
        # Halves with each LOD level so detail increases near the player.
        self._split_thresholds: list[float] = [
            math.pi * 0.75 / (1 << lod)
            for lod in range(max_lod + 1)
        ]

        # Job queue (FIFO; can be replaced with a thread pool later)
        self._jobs: list[BuildTileMeshJob] = []

        # Flat list of current leaf nodes (rebuilt each update)
        self._active_leaves: list[QuadtreeNode] = []

        # Root node for each of the 6 cube faces
        self._roots: list[QuadtreeNode] = [
            QuadtreeNode(face_id=f, lod_level=0, x=0, y=0)
            for f in range(NUM_FACES)
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, player_unit_dir: Vec3, max_jobs_per_frame: int = 4) -> None:
        """
        Refresh the LOD quadtrees and process a bounded number of mesh
        build jobs.

        *player_unit_dir* — normalised direction from the planet centre to
        the player (i.e. the player's position on the unit sphere).
        """
        # Step 1: update split / merge decisions for every face
        for root in self._roots:
            self._update_node(root, player_unit_dir)

        # Step 2: collect all leaf nodes
        self._active_leaves.clear()
        for root in self._roots:
            self._collect_leaves(root, self._active_leaves)

        # Step 3: process job queue (bounded per frame)
        processed = 0
        while self._jobs and processed < max_jobs_per_frame:
            job = self._jobs.pop(0)
            # Only build if the node is still expecting a mesh
            if job.node.state == NodeState.LOADING:
                job.node.mesh_handle = build_tile_mesh(
                    job.face, job.lod, job.x, job.y,
                    self._hp, self._radius, self._tile_res,
                )
                job.node.state = NodeState.READY
            processed += 1

        if self._debug:
            ready = sum(1 for n in self._active_leaves
                        if n.state == NodeState.READY)
            print(
                f"[TileStreamer] active={len(self._active_leaves)} "
                f"ready={ready} queue={len(self._jobs)}"
            )

    def get_render_tiles(self) -> list[tuple]:
        """
        Return ``[(Mesh, lod_level), …]`` for every READY leaf node.
        Only nodes with a built mesh_handle are included.
        """
        return [
            (n.mesh_handle, n.lod_level)
            for n in self._active_leaves
            if n.state == NodeState.READY and n.mesh_handle is not None
        ]

    def active_tile_count(self) -> int:
        """Total number of active leaf nodes (including those still loading)."""
        return len(self._active_leaves)

    def job_queue_length(self) -> int:
        return len(self._jobs)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _update_node(self, node: QuadtreeNode, player_dir: Vec3) -> None:
        """Recursively split or merge nodes based on player proximity."""
        cdir    = node.center_dir()
        cos_a   = max(-1.0, min(1.0, cdir.dot(player_dir)))
        arc_dist = math.acos(cos_a)

        should_split = (
            node.lod_level < self._max_lod
            and arc_dist < self._split_thresholds[node.lod_level]
        )

        if should_split:
            if node.is_leaf:
                node.split()
                # The parent no longer renders; free its mesh
                node.mesh_handle = None
                node.state = NodeState.UNLOADED
            for child in node.children:
                self._update_node(child, player_dir)
        else:
            # Should be a leaf — collapse children if any
            if not node.is_leaf:
                node.merge()
            # Schedule mesh build if not already done
            if node.state == NodeState.UNLOADED:
                node.state = NodeState.LOADING
                self._jobs.append(BuildTileMeshJob(
                    face=node.face_id,
                    lod=node.lod_level,
                    x=node.x,
                    y=node.y,
                    node=node,
                ))

    def _collect_leaves(self, node: QuadtreeNode, out: list) -> None:
        """Depth-first collection of all leaf nodes into *out*."""
        if node.is_leaf:
            out.append(node)
        else:
            for child in node.children:
                self._collect_leaves(child, out)
