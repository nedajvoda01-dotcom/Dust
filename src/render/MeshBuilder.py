"""MeshBuilder — procedural mesh generation (no file assets)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from src.math.Vec3 import Vec3


@dataclass
class Mesh:
    vertices: list              # list of (x,y,z) tuples
    normals: list               # list of (x,y,z) tuples
    indices: list               # list of int triangles (triplets)
    colors: list = field(default_factory=list)  # optional per-vertex (r,g,b)


class MeshBuilder:
    @staticmethod
    def uv_sphere(radius: float = 1.0, stacks: int = 16, slices: int = 24) -> Mesh:
        """
        Build a UV sphere mesh.
        stacks: latitudinal divisions
        slices: longitudinal divisions
        """
        vertices = []
        normals = []
        indices = []

        for i in range(stacks + 1):
            phi = math.pi * i / stacks          # 0 .. pi
            sin_phi = math.sin(phi)
            cos_phi = math.cos(phi)
            for j in range(slices + 1):
                theta = 2.0 * math.pi * j / slices  # 0 .. 2pi
                x = sin_phi * math.cos(theta)
                y = cos_phi
                z = sin_phi * math.sin(theta)
                vertices.append((x * radius, y * radius, z * radius))
                normals.append((x, y, z))

        for i in range(stacks):
            for j in range(slices):
                a = i * (slices + 1) + j
                b = a + slices + 1
                indices.extend([a, b, a + 1, b, b + 1, a + 1])

        return Mesh(vertices=vertices, normals=normals, indices=indices)

    @staticmethod
    def debug_line_sphere(radius: float = 1.0, segments: int = 64) -> list:
        """
        Return a list of (start, end) Vec3 pairs forming three great circles
        for debug visualisation of the sphere.
        """
        lines = []
        for axis in range(3):
            prev = None
            for i in range(segments + 1):
                angle = 2.0 * math.pi * i / segments
                if axis == 0:
                    p = Vec3(0.0, math.cos(angle) * radius, math.sin(angle) * radius)
                elif axis == 1:
                    p = Vec3(math.cos(angle) * radius, 0.0, math.sin(angle) * radius)
                else:
                    p = Vec3(math.cos(angle) * radius, math.sin(angle) * radius, 0.0)
                if prev is not None:
                    lines.append((prev, p))
                prev = p
        return lines
