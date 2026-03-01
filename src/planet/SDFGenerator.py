"""SDFGenerator — deterministic SDF chunk generation from the planet heightfield.

For every voxel (i, j, k) in the chunk the world-space position is:
    dir       = cube_to_sphere(face_id, u, v)          # horizontal direction
    h_surface = height_provider.sample_height(dir)
    r_surface = planet_radius + h_surface
    r_point   = r_surface + radial_offset
    world_pos = dir * r_point

And the SDF value is:
    d = r_point - r_surface = radial_offset

This guarantees d=0 coincides exactly with the heightfield surface.
Positive d → air (above surface), negative d → rock (below surface).

Material channels (hardness, fracture, porosity) are populated from the
optional GeoFieldSampler when provided, enabling SDF-based systems to
know about geological weakness zones and rift porosity.
"""
from __future__ import annotations

import math

from src.planet.PlanetLOD import node_bounds, cube_to_sphere
from src.planet.SDFChunk import (
    SDFChunk, SDFChunkCoord,
    MATERIAL_AIR, MATERIAL_ROCK,
)


def generate_chunk(
    coord: SDFChunkCoord,
    resolution: int,
    voxel_depth: float,
    planet_radius: float,
    height_provider,
    geo_sampler=None,
) -> SDFChunk:
    """
    Build and return a fully populated SDFChunk for *coord*.

    Parameters
    ----------
    coord            : identifies which tile and depth layer to generate
    resolution       : voxels per side (same in i, j, k)
    voxel_depth      : metres per radial voxel step
    planet_radius    : base planet radius in simulation units
    height_provider  : object with ``sample_height(unit_dir) -> float``
    geo_sampler      : optional GeoFieldSampler; when supplied the
                       hardness/fracture/porosity channels are populated
                       from the tectonic field
    """
    R = resolution
    half_depth = (R // 2) * voxel_depth
    # Top of this chunk (k=0) in metres relative to the local surface height.
    top_offset = half_depth - coord.depth_index * R * voxel_depth

    u_min, v_min, u_max, v_max = node_bounds(coord.lod, coord.tile_x, coord.tile_y)
    du = (u_max - u_min) / max(R - 1, 1)
    dv = (v_max - v_min) / max(R - 1, 1)

    n = R * R * R
    positions      = [(0.0, 0.0, 0.0)] * n
    distance_field = [0.0] * n
    material_field = [MATERIAL_AIR] * n
    hardness_field = [1.0] * n
    fracture_field = [0.0] * n
    porosity_field = [0.0] * n

    for k in range(R):
        radial_offset = top_offset - k * voxel_depth
        for j in range(R):
            v = v_min + j * dv
            for i in range(R):
                u = u_min + i * du
                direction = cube_to_sphere(coord.face_id, u, v)
                h_surface = height_provider.sample_height(direction)
                r_surface = planet_radius + h_surface
                r_point   = r_surface + radial_offset
                # Clamp to a small positive radius to avoid singularities
                r_point   = max(r_point, 1e-3)
                wx = direction.x * r_point
                wy = direction.y * r_point
                wz = direction.z * r_point

                idx = i + j * R + k * R * R
                positions[idx]      = (wx, wy, wz)
                distance_field[idx] = radial_offset
                material_field[idx] = (
                    MATERIAL_AIR if radial_offset >= 0.0 else MATERIAL_ROCK
                )

                if geo_sampler is not None:
                    sample = geo_sampler.sample(direction)
                    hardness_field[idx] = sample.hardness
                    fracture_field[idx] = sample.fracture
                    # Porosity is elevated under rifts (divergent boundaries)
                    from src.planet.TectonicPlatesSystem import BoundaryType
                    if sample.boundary_type == BoundaryType.DIVERGENT:
                        porosity_field[idx] = min(1.0, sample.boundary_strength * 0.8)
                    else:
                        porosity_field[idx] = 0.0

    # Approximate horizontal voxel size from tile angular width
    tile_center_u = (u_min + u_max) * 0.5
    tile_center_v = (v_min + v_max) * 0.5
    center_dir    = cube_to_sphere(coord.face_id, tile_center_u, tile_center_v)
    h_center      = height_provider.sample_height(center_dir)
    r_center      = planet_radius + h_center
    uv_per_voxel = du if R > 1 else 1.0   # face UV units per voxel
    voxel_size    = r_center * uv_per_voxel * math.sqrt(2.0) / math.sqrt(3.0)
    voxel_size    = max(voxel_size, voxel_depth)

    return SDFChunk(
        coord          = coord,
        resolution     = R,
        voxel_size     = voxel_size,
        voxel_depth    = voxel_depth,
        positions      = positions,
        distance_field = distance_field,
        material_field = material_field,
        hardness_field = hardness_field,
        fracture_field = fracture_field,
        porosity_field = porosity_field,
    )
