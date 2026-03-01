"""PlanetLOD — cube→sphere parameterisation, quadtree nodes, tile mesh generation.

Face layout (u, v ∈ [-1, 1]):
    FACE_PX (0):  cube = ( 1,  v,  u)
    FACE_NX (1):  cube = (-1,  v, -u)
    FACE_PY (2):  cube = ( u,  1, -v)
    FACE_NY (3):  cube = ( u, -1,  v)
    FACE_PZ (4):  cube = (-u,  v,  1)
    FACE_NZ (5):  cube = ( u,  v, -1)

All six adjacent face-edge pairs produce matching unit-sphere directions,
so the sphere has no geometric seams.

Tile meshes include perimeter skirts (slightly inset toward the planet
centre) to prevent visual gaps at LOD boundaries.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from src.math.Vec3 import Vec3
from src.render.MeshBuilder import Mesh


# ---------------------------------------------------------------------------
# Face IDs
# ---------------------------------------------------------------------------

FACE_PX: int = 0
FACE_NX: int = 1
FACE_PY: int = 2
FACE_NY: int = 3
FACE_PZ: int = 4
FACE_NZ: int = 5
NUM_FACES: int = 6


# ---------------------------------------------------------------------------
# Cube → sphere mapping
# ---------------------------------------------------------------------------

def face_to_cube(face: int, u: float, v: float) -> Vec3:
    """Map (face, u, v) ∈ [-1,1]² to a point on the surface of the unit cube."""
    if face == FACE_PX:
        return Vec3( 1.0,  v,  u)
    if face == FACE_NX:
        return Vec3(-1.0,  v, -u)
    if face == FACE_PY:
        return Vec3( u,  1.0, -v)
    if face == FACE_NY:
        return Vec3( u, -1.0,  v)
    if face == FACE_PZ:
        return Vec3(-u,  v,  1.0)
    # FACE_NZ
    return Vec3( u,  v, -1.0)


def cube_to_sphere(face: int, u: float, v: float) -> Vec3:
    """Map (face, u, v) to a unit-sphere direction (normalize the cube point)."""
    return face_to_cube(face, u, v).normalized()


# ---------------------------------------------------------------------------
# Node bounds / centre helpers
# ---------------------------------------------------------------------------

def node_bounds(lod: int, x: int, y: int) -> tuple:
    """
    Return (u_min, v_min, u_max, v_max) face-space coordinates for a
    quadtree node at *lod* level and grid position *(x, y)*.
    """
    scale = 2.0 / (1 << lod)
    u_min = -1.0 + x * scale
    v_min = -1.0 + y * scale
    return (u_min, v_min, u_min + scale, v_min + scale)


def node_center_dir(face: int, lod: int, x: int, y: int) -> Vec3:
    """Unit sphere direction of the quadtree node's centre."""
    u_min, v_min, u_max, v_max = node_bounds(lod, x, y)
    return cube_to_sphere(face, (u_min + u_max) * 0.5, (v_min + v_max) * 0.5)


# ---------------------------------------------------------------------------
# QuadtreeNode
# ---------------------------------------------------------------------------

class NodeState(Enum):
    UNLOADED = auto()
    LOADING  = auto()
    READY    = auto()


@dataclass
class QuadtreeNode:
    """A single node in the per-face quadtree LOD structure."""

    face_id:     int
    lod_level:   int
    x:           int              # column index at this LOD level (0 .. 2^lod - 1)
    y:           int              # row    index
    bounds:      tuple            = field(init=False)
    children:    list             = field(default_factory=list)
    mesh_handle: Optional[Mesh]   = field(default=None, compare=False)
    state:       NodeState        = NodeState.UNLOADED

    def __post_init__(self) -> None:
        self.bounds = node_bounds(self.lod_level, self.x, self.y)

    # ------------------------------------------------------------------
    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def center_dir(self) -> Vec3:
        u_min, v_min, u_max, v_max = self.bounds
        return cube_to_sphere(
            self.face_id,
            (u_min + u_max) * 0.5,
            (v_min + v_max) * 0.5,
        )

    # ------------------------------------------------------------------
    def split(self) -> None:
        """Subdivide into four children (NW, NE, SW, SE in grid order)."""
        assert self.is_leaf, "Node already split"
        nl  = self.lod_level + 1
        cx  = self.x * 2
        cy  = self.y * 2
        self.children = [
            QuadtreeNode(self.face_id, nl, cx,     cy    ),
            QuadtreeNode(self.face_id, nl, cx + 1, cy    ),
            QuadtreeNode(self.face_id, nl, cx,     cy + 1),
            QuadtreeNode(self.face_id, nl, cx + 1, cy + 1),
        ]

    def merge(self) -> None:
        """Collapse children back to a leaf (frees mesh handle too)."""
        self.children.clear()
        self.mesh_handle = None
        self.state = NodeState.UNLOADED


# ---------------------------------------------------------------------------
# Procedural vertex colour (height-based, no textures)
# ---------------------------------------------------------------------------

_C_LOW    = (0.35, 0.28, 0.22)   # dark basalt / dried mud
_C_MID    = (0.66, 0.50, 0.38)   # dusty red soil
_C_HIGH   = (0.76, 0.62, 0.48)   # pale ochre terrace
_C_PEAK   = (0.88, 0.84, 0.80)   # bright highland / dust-covered ridge

HEIGHT_SCALE_FOR_COLOR: float = 40.0   # matches PlanetHeightProvider.HEIGHT_SCALE


def _lerp_color(a: tuple, b: tuple, t: float) -> tuple:
    t = max(0.0, min(1.0, t))
    return (
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    )


def _height_color(h: float) -> tuple:
    """Map height offset to an RGB colour triple (no textures)."""
    t = (h / HEIGHT_SCALE_FOR_COLOR + 1.0) * 0.5   # 0 .. 1

    if t < 0.25:
        return _lerp_color(_C_LOW, _C_MID, t / 0.25)
    if t < 0.55:
        return _lerp_color(_C_MID, _C_HIGH, (t - 0.25) / 0.30)
    return _lerp_color(_C_HIGH, _C_PEAK, (t - 0.55) / 0.45)


# ---------------------------------------------------------------------------
# Tile mesh generation
# ---------------------------------------------------------------------------

_SKIRT_SCALE: float = 0.991    # pull skirt vertices ~0.9 % toward planet centre


def build_tile_mesh(
    face: int,
    lod: int,
    x: int,
    y: int,
    height_provider,
    planet_radius: float,
    tile_res: int = 17,
) -> Mesh:
    """
    Build an NxN mesh for the given quadtree tile, with perimeter skirts.

    *tile_res* must be 2^k + 1 (e.g. 9, 17, 33) so that adjacent tiles
    at the same LOD always share identical edge vertex positions.
    """
    N = tile_res
    u_min, v_min, u_max, v_max = node_bounds(lod, x, y)
    du = (u_max - u_min) / (N - 1)
    dv = (v_max - v_min) / (N - 1)

    vertices: list = []
    normals:  list = []
    colors:   list = []

    # ------------------------------------------------------------------
    # Main N × N grid
    # ------------------------------------------------------------------
    for j in range(N):
        for i in range(N):
            u = u_min + i * du
            v = v_min + j * dv
            d = cube_to_sphere(face, u, v)
            h = height_provider.sample_height(d)
            r = planet_radius + h
            pos = d * r
            vertices.append((pos.x, pos.y, pos.z))
            normals.append((d.x, d.y, d.z))    # sphere normal (approx)
            colors.append(_height_color(h))

    # ------------------------------------------------------------------
    # Skirt vertices — perimeter of the tile pulled toward planet centre
    # ------------------------------------------------------------------
    # Layout of skirt vertex buffer (relative to skirt_start):
    #   [0 ..  N-1]       : bottom row  j = 0
    #   [N .. 2N-1]       : top row     j = N-1
    #   [2N .. 3N-5]      : left col    i = 0,  j = 1 .. N-2
    #   [3N-4 .. 4N-9]    : right col   i = N-1, j = 1 .. N-2

    skirt_start = len(vertices)

    def _add_skirt(si: int, sj: int) -> None:
        u = u_min + si * du
        v = v_min + sj * dv
        d = cube_to_sphere(face, u, v)
        h = height_provider.sample_height(d)
        r = (planet_radius + h) * _SKIRT_SCALE
        pos = d * r
        vertices.append((pos.x, pos.y, pos.z))
        normals.append((d.x, d.y, d.z))
        colors.append((0.28, 0.22, 0.18))   # hidden skirt colour

    for i in range(N):        _add_skirt(i, 0)       # bottom
    for i in range(N):        _add_skirt(i, N - 1)   # top
    for j in range(1, N - 1): _add_skirt(0, j)       # left  (skip corners)
    for j in range(1, N - 1): _add_skirt(N - 1, j)   # right (skip corners)

    # ------------------------------------------------------------------
    # Index generation
    # ------------------------------------------------------------------
    indices = _build_indices(N, skirt_start)

    return Mesh(vertices=vertices, normals=normals, indices=indices, colors=colors)


def _build_indices(N: int, skirt_start: int) -> list:
    """Build triangle indices for the main grid and its four skirts."""
    indices: list = []

    # Main grid
    for j in range(N - 1):
        for i in range(N - 1):
            a = j * N + i
            b = a + 1
            c = (j + 1) * N + i
            d = c + 1
            indices += [a, c, b, b, c, d]

    # Helper: skirt vertex index
    def sk(idx: int) -> int:
        return skirt_start + idx

    # Bottom skirt row (j = 0)
    for i in range(N - 1):
        a = i;         b = i + 1
        sa = sk(i);    sb = sk(i + 1)
        indices += [a, sa, b, b, sa, sb]

    # Top skirt row (j = N-1)
    for i in range(N - 1):
        a = (N - 1) * N + i;   b = a + 1
        sa = sk(N + i);         sb = sk(N + i + 1)
        indices += [a, b, sa, b, sb, sa]

    # Left skirt column (i = 0)
    # Corner mapping: j=0 → sk(0), j=1..N-2 → sk(2N + j-1), j=N-1 → sk(N)
    def _left_sk(j: int) -> int:
        if j == 0:     return sk(0)
        if j == N - 1: return sk(N)
        return sk(2 * N + j - 1)

    for j in range(N - 1):
        a = j * N;           b = (j + 1) * N
        sa = _left_sk(j);    sb = _left_sk(j + 1)
        indices += [a, b, sa, b, sb, sa]

    # Right skirt column (i = N-1)
    # Corner mapping: j=0 → sk(N-1), j=1..N-2 → sk(3N-4 + j-1), j=N-1 → sk(2N-1)
    def _right_sk(j: int) -> int:
        if j == 0:     return sk(N - 1)
        if j == N - 1: return sk(2 * N - 1)
        return sk(3 * N - 4 + j - 1)

    for j in range(N - 1):
        a = j * N + (N - 1);    b = (j + 1) * N + (N - 1)
        sa = _right_sk(j);       sb = _right_sk(j + 1)
        indices += [a, sa, b, b, sa, sb]

    return indices
