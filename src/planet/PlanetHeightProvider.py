"""PlanetHeightProvider — deterministic procedural height for the planet surface.

Produces Martian-style relief:
  * large-scale fBm base terrain
  * craters (small/medium/large with age-based erosion)
  * fault-line canyons
  * soft terracing

No external dependencies; all noise is hash-based value noise.
"""
from __future__ import annotations

import math
import random

from src.math.Vec3 import Vec3


# ---------------------------------------------------------------------------
# Low-level 3-D value noise (pure integer arithmetic, no imports needed)
# ---------------------------------------------------------------------------

def _hash3i(a: int, b: int, c: int) -> int:
    """Fast integer hash, result in [0, 0x7FFF_FFFF]."""
    h = (a * 1_664_525 + b * 22_695_477 + c * 1_013_904_223) & 0xFFFF_FFFF
    h ^= h >> 16
    h = (h * 1_664_525 + 1_013_904_223) & 0xFFFF_FFFF
    h ^= h >> 10
    return h & 0x7FFF_FFFF


def _h3f(a: int, b: int, c: int) -> float:
    """Float in [0, 1) from integer triple."""
    return _hash3i(a & 0xFF, b & 0xFF, c & 0xFF) / 2_147_483_647.0


def _s5(t: float) -> float:
    """Ken Perlin quintic smoothstep: 6t⁵-15t⁴+10t³."""
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _value_noise3(x: float, y: float, z: float) -> float:
    """3-D value noise ∈ [-1, 1] (trilinear interpolation of hashed lattice)."""
    ix = int(math.floor(x)); fx = x - ix
    iy = int(math.floor(y)); fy = y - iy
    iz = int(math.floor(z)); fz = z - iz
    sx = _s5(fx); sy = _s5(fy); sz = _s5(fz)

    def h(a: int, b: int, c: int) -> float:
        return _h3f(a, b, c) * 2.0 - 1.0

    v000 = h(ix,     iy,     iz    )
    v100 = h(ix + 1, iy,     iz    )
    v010 = h(ix,     iy + 1, iz    )
    v110 = h(ix + 1, iy + 1, iz    )
    v001 = h(ix,     iy,     iz + 1)
    v101 = h(ix + 1, iy,     iz + 1)
    v011 = h(ix,     iy + 1, iz + 1)
    v111 = h(ix + 1, iy + 1, iz + 1)

    v00 = v000 + sx * (v100 - v000)
    v10 = v010 + sx * (v110 - v010)
    v01 = v001 + sx * (v101 - v001)
    v11 = v011 + sx * (v111 - v011)
    v0  = v00  + sy * (v10  - v00 )
    v1  = v01  + sy * (v11  - v01 )
    return v0 + sz * (v1 - v0)


def _fbm3(
    x: float, y: float, z: float,
    octaves: int = 6,
    lacunarity: float = 2.1,
    gain: float = 0.5,
) -> float:
    """Fractional Brownian motion built on _value_noise3, result ∈ approx [-1, 1]."""
    value = 0.0
    amp = 0.5
    freq = 1.0
    norm = 0.0
    for _ in range(octaves):
        value += _value_noise3(x * freq, y * freq, z * freq) * amp
        norm  += amp
        amp   *= gain
        freq  *= lacunarity
    return value / norm


# ---------------------------------------------------------------------------
# Crater radial profile
# ---------------------------------------------------------------------------

def _crater_delta(d_norm: float, depth: float, age: float) -> float:
    """
    Height contribution from a single crater at normalised distance
    d_norm = arc_distance / crater_radius.

    Profile: interior bowl → rim → ejecta blanket.
    Age ∈ [0,1] erodes the crater (1 = completely filled / smooth).
    """
    if d_norm > 2.8:
        return 0.0
    if d_norm < 0.8:
        h = -depth * (1.0 - (d_norm / 0.8) ** 2)
    elif d_norm < 1.2:
        t = (d_norm - 0.8) / 0.4
        h = depth * 0.55 * math.sin(t * math.pi)
    else:
        t = (d_norm - 1.2) / 1.6
        h = depth * 0.12 * math.exp(-t * 3.0)
    return h * (1.0 - age * 0.65)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HEIGHT_SCALE: float = 40.0    # peak amplitude in simulation units (metres equiv.)

_TERRACE_STEP: float = 0.08   # fractional-height quantisation step

# (count, r_min, r_max) in arc-angle radians
_CRATER_SPEC: list = [
    (100, 0.008, 0.035),   # small
    ( 30, 0.035, 0.090),   # medium
    (  8, 0.090, 0.220),   # large
]

_FAULT_COUNT: int = 7


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rand_sphere_dir(rng: random.Random) -> Vec3:
    """Uniform random direction on unit sphere (rejection sampling)."""
    while True:
        x = rng.uniform(-1.0, 1.0)
        y = rng.uniform(-1.0, 1.0)
        z = rng.uniform(-1.0, 1.0)
        ls = x * x + y * y + z * z
        if 0.001 < ls <= 1.0:
            s = math.sqrt(ls)
            return Vec3(x / s, y / s, z / s)


def _terrace(h: float, step: float = _TERRACE_STEP) -> float:
    """Soft terracing: smooth quantisation into horizontal layers."""
    layer = math.floor(h / step)
    frac  = (h - layer * step) / step
    smooth = frac * frac * (3.0 - 2.0 * frac)
    return (layer + smooth * 0.55) * step


# ---------------------------------------------------------------------------
# PlanetHeightProvider
# ---------------------------------------------------------------------------

class PlanetHeightProvider:
    """
    Deterministic procedural height provider for the planet surface.

    ``sample_height(unit_dir)`` and ``sample_normal_approx(unit_dir)`` are
    stable for a given seed across runs (no external state).
    """

    def __init__(self, seed: int) -> None:
        self._seed: int = seed

        # Noise-domain offset isolates seeds from each other.
        rng = random.Random(seed ^ 0x5A5A_5A5A)
        self._ox: float = rng.uniform(-500.0, 500.0)
        self._oy: float = rng.uniform(-500.0, 500.0)
        self._oz: float = rng.uniform(-500.0, 500.0)

        # Pre-generate craters: list of (cx, cy, cz, radius, depth, age)
        self._craters: list[tuple] = []
        for spec_idx, (count, r_min, r_max) in enumerate(_CRATER_SPEC):
            rng2 = random.Random(seed ^ (spec_idx * 0x1357 + 0xFACE))
            for _ in range(count):
                d = _rand_sphere_dir(rng2)
                r     = rng2.uniform(r_min, r_max)
                depth = rng2.uniform(0.25, 1.0)
                age   = rng2.uniform(0.0,  1.0)
                self._craters.append((d.x, d.y, d.z, r, depth, age))

        # Pre-generate fault lines: list of (nx, ny, nz, half_width, depth)
        self._faults: list[tuple] = []
        frng = random.Random(seed ^ 0xDEAD_BEEF)
        for _ in range(_FAULT_COUNT):
            n      = _rand_sphere_dir(frng)
            width  = frng.uniform(0.008, 0.028)
            fdepth = frng.uniform(0.15,  0.55)
            self._faults.append((n.x, n.y, n.z, width, fdepth))

    # ------------------------------------------------------------------
    @property
    def seed(self) -> int:
        return self._seed

    # ------------------------------------------------------------------
    def sample_height(self, unit_dir: Vec3) -> float:
        """
        Return height offset (simulation units) at *unit_dir*.
        Deterministic for a given (unit_dir, seed) pair.
        """
        dx, dy, dz   = unit_dir.x, unit_dir.y, unit_dir.z
        ox, oy, oz   = self._ox, self._oy, self._oz

        # 1. Large-scale fBm base
        h  = _fbm3(dx * 3.0 + ox, dy * 3.0 + oy, dz * 3.0 + oz,
                   octaves=6) * 0.65
        # 2. Medium-frequency roughness
        h += _fbm3(dx * 8.5 + ox * 0.7, dy * 8.5 + oy * 0.7,
                   dz * 8.5 + oz * 0.7,
                   octaves=4, gain=0.45) * 0.22
        h  = max(-1.0, min(1.0, h))

        # 3. Craters
        c_total = 0.0
        for cx, cy, cz, r, depth, age in self._craters:
            cos_a = max(-1.0, min(1.0, cx * dx + cy * dy + cz * dz))
            arc   = math.acos(cos_a)
            c_total += _crater_delta(arc / r, depth, age)
        h += max(-1.0, min(0.85, c_total * 0.35))

        # 4. Fault / canyon lines
        f_total = 0.0
        for nx, ny, nz, width, fdepth in self._faults:
            dist = abs(nx * dx + ny * dy + nz * dz)
            if dist < width * 3.5:
                t       = dist / width
                f_total -= fdepth * max(0.0, 1.0 - t * 0.85) * (1.0 - 0.25 * t)
        h += max(-0.7, min(0.0, f_total * 0.28))

        # 5. Soft terrace
        h = _terrace(h, _TERRACE_STEP)

        return h * HEIGHT_SCALE

    # ------------------------------------------------------------------
    def sample_normal_approx(self, unit_dir: Vec3, eps: float = 2e-3) -> Vec3:
        """Approximate surface normal via two-axis finite differences."""
        up  = unit_dir
        ref = Vec3(0.0, 1.0, 0.0) if abs(up.x) < 0.9 else Vec3(0.0, 0.0, 1.0)
        t1  = ref.cross(up).normalized()
        t2  = up.cross(t1).normalized()

        h0 = self.sample_height(unit_dir)
        h1 = self.sample_height((unit_dir + t1 * eps).normalized())
        h2 = self.sample_height((unit_dir + t2 * eps).normalized())

        # Tangent vectors in 3-D, scaled by height gradient
        v1 = t1 * eps + up * ((h1 - h0) / eps)
        v2 = t2 * eps + up * ((h2 - h0) / eps)
        n  = v2.cross(v1).normalized()
        if n.is_near_zero():
            return up
        return n
