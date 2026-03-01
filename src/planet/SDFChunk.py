"""SDFChunk — core data types for the SDF voxel subsystem.

Coordinate system (Path A: surface-tile-anchored):
  Each SDFChunkCoord is tied to a cube-sphere surface tile and a radial
  depth layer.  This ensures deterministic chunk identity as the player
  moves and aligns naturally with the existing LOD quadtree.

  face_id     : 0–5 (matches PlanetLOD face constants)
  lod         : quadtree LOD level of the anchor tile
  tile_x / y  : tile indices at that LOD level
  depth_index : 0 = chunk straddles the surface (half above, half below);
                positive integers go deeper underground.

SDF convention (OpenVDB-style, used throughout this subsystem):
  d > 0  →  outside / air
  d < 0  →  inside / rock
  d = 0  →  isosurface

Voxel indexing within a chunk (resolution R):
  flat_index = i  +  j * R  +  k * R²
  i  : horizontal, along face-U   (0 .. R-1)
  j  : horizontal, along face-V   (0 .. R-1)
  k  : radial, 0 = outermost/top  (0 .. R-1)

Radial position of voxel k for a chunk at depth_index D:
  half_depth    = (R // 2) * voxel_depth
  radial_offset = half_depth - D * R * voxel_depth - k * voxel_depth
  (positive → above surface, negative → below)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from src.render.MeshBuilder import Mesh


# ---------------------------------------------------------------------------
# Material IDs (minimal; no textures)
# ---------------------------------------------------------------------------

MATERIAL_AIR:  int = 0
MATERIAL_ROCK: int = 1


# ---------------------------------------------------------------------------
# SDFChunkCoord
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SDFChunkCoord:
    """Unique, hashable identifier for one SDF chunk."""
    face_id:     int
    lod:         int
    tile_x:      int
    tile_y:      int
    depth_index: int   # 0 = surface-straddling, positive = deeper underground


# ---------------------------------------------------------------------------
# SDFChunk
# ---------------------------------------------------------------------------

@dataclass
class SDFChunk:
    """One 3-D block of SDF data anchored to a surface tile.

    ``positions``      — world-space (x,y,z) for every voxel corner.
    ``distance_field`` — signed distance value for every voxel.
    ``material_field`` — MATERIAL_AIR or MATERIAL_ROCK per voxel.
    ``mesh_handle``    — most recently built Mesh (or None if dirty/empty).
    ``dirty``          — True when the SDF has been modified since last mesh build.
    """
    coord:          SDFChunkCoord
    resolution:     int                      # voxels per side (e.g. 16)
    voxel_size:     float                    # approximate horizontal metres per voxel
    voxel_depth:    float                    # radial metres per voxel
    positions:      List[tuple]              # len = resolution³, each (x,y,z)
    distance_field: List[float]              # len = resolution³
    material_field: List[int]                # len = resolution³
    mesh_handle:    Optional[Mesh] = field(default=None, compare=False)
    dirty:          bool           = True

    # ------------------------------------------------------------------
    # Voxel accessors
    # ------------------------------------------------------------------

    def flat_index(self, i: int, j: int, k: int) -> int:
        return i + j * self.resolution + k * self.resolution * self.resolution

    def get_d(self, i: int, j: int, k: int) -> float:
        return self.distance_field[self.flat_index(i, j, k)]

    def set_d(self, i: int, j: int, k: int, value: float) -> None:
        self.distance_field[self.flat_index(i, j, k)] = value

    def get_pos(self, i: int, j: int, k: int) -> tuple:
        return self.positions[self.flat_index(i, j, k)]

    # ------------------------------------------------------------------
    # Optimisation helpers
    # ------------------------------------------------------------------

    def is_empty(self) -> bool:
        """True when all voxels are air (d ≥ 0) — no surface to mesh."""
        return all(d >= 0.0 for d in self.distance_field)

    def is_full(self) -> bool:
        """True when all voxels are rock (d ≤ 0) — no surface to mesh."""
        return all(d <= 0.0 for d in self.distance_field)
