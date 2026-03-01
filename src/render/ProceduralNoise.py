"""ProceduralNoise — deterministic 3D value-noise for procedural materials.

No bitmap textures or samplers are used anywhere in this module.

Public API
----------
noise3(p: Vec3) -> float
    Single-octave 3D value noise in [0, 1].

fbm3(p: Vec3, octaves: int = 2, lacunarity: float = 2.0,
     gain: float = 0.5) -> float
    Fractional Brownian Motion (2–3 octaves; higher octaves are forbidden
    by the Stage-10 spec to preserve world scale).  Returns value in
    approximately [-1, 1].

gradient3(p: Vec3, eps: float = 0.01) -> Vec3
    Numeric gradient of noise3 via central finite differences.
    Used for micro-normal perturbation.
"""
from __future__ import annotations

import math

from src.math.Vec3 import Vec3

# ---------------------------------------------------------------------------
# Internal hash helpers
# ---------------------------------------------------------------------------

def _hash3(ix: int, iy: int, iz: int) -> float:
    """Deterministic pseudo-random float in [0, 1) from three integers."""
    n = ix * 1531 + iy * 3571 + iz * 7919
    n = (n ^ (n >> 13)) * 1664525 + 1013904223
    n = (n ^ (n >> 16)) & 0x7FFFFFFF
    return n / 2147483648.0


def _smooth(t: float) -> float:
    """Smootherstep: 6t^5 - 15t^4 + 10t^3 (Ken Perlin's quintic)."""
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _lerp(a: float, b: float, t: float) -> float:
    return a + t * (b - a)


# ---------------------------------------------------------------------------
# Single-octave 3D value noise
# ---------------------------------------------------------------------------

def noise3(p: Vec3) -> float:
    """3D value noise returning a value in [0, 1].

    Uses trilinear interpolation with smootherstep blending so the
    output is C1-continuous.
    """
    ix = math.floor(p.x)
    iy = math.floor(p.y)
    iz = math.floor(p.z)
    fx = p.x - ix
    fy = p.y - iy
    fz = p.z - iz

    ux = _smooth(fx)
    uy = _smooth(fy)
    uz = _smooth(fz)

    # Eight lattice-corner hashes
    v000 = _hash3(ix,     iy,     iz    )
    v100 = _hash3(ix + 1, iy,     iz    )
    v010 = _hash3(ix,     iy + 1, iz    )
    v110 = _hash3(ix + 1, iy + 1, iz    )
    v001 = _hash3(ix,     iy,     iz + 1)
    v101 = _hash3(ix + 1, iy,     iz + 1)
    v011 = _hash3(ix,     iy + 1, iz + 1)
    v111 = _hash3(ix + 1, iy + 1, iz + 1)

    # Trilinear interpolation
    x0 = _lerp(_lerp(v000, v100, ux), _lerp(v010, v110, ux), uy)
    x1 = _lerp(_lerp(v001, v101, ux), _lerp(v011, v111, ux), uy)
    return _lerp(x0, x1, uz)


# ---------------------------------------------------------------------------
# Fractional Brownian Motion (max 3 octaves — spec requirement)
# ---------------------------------------------------------------------------

_MAX_OCTAVES = 3


def fbm3(
    p: Vec3,
    octaves: int = 2,
    lacunarity: float = 2.0,
    gain: float = 0.5,
) -> float:
    """Sum of *octaves* noise layers.

    Returns a value roughly in [0, 1]; clamped to that range.
    Octave count is capped at ``_MAX_OCTAVES`` (3) to prevent small
    high-frequency patterns from destroying world scale.
    """
    octaves = max(1, min(octaves, _MAX_OCTAVES))
    amplitude = 1.0
    frequency = 1.0
    total = 0.0
    norm = 0.0
    for _ in range(octaves):
        total += noise3(Vec3(p.x * frequency, p.y * frequency, p.z * frequency)) * amplitude
        norm += amplitude
        amplitude *= gain
        frequency *= lacunarity
    result = total / norm if norm > 0.0 else 0.5
    return max(0.0, min(1.0, result))


# ---------------------------------------------------------------------------
# Noise gradient (finite differences, world-space)
# ---------------------------------------------------------------------------

def gradient3(p: Vec3, eps: float = 0.01) -> Vec3:
    """Central-difference gradient of *noise3* at position *p*.

    The returned vector is **not** normalised so the caller can scale it
    by a micro-strength factor before blending with the surface normal.
    """
    dx = Vec3(eps, 0.0, 0.0)
    dy = Vec3(0.0, eps, 0.0)
    dz = Vec3(0.0, 0.0, eps)
    gx = noise3(p + dx) - noise3(p - dx)
    gy = noise3(p + dy) - noise3(p - dy)
    gz = noise3(p + dz) - noise3(p - dz)
    return Vec3(gx, gy, gz)
