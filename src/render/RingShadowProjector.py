"""RingShadowProjector — Stage 62 planetary ring shadow on surface (§5.2).

Computes the ring shadow attenuation for a surface point given:

* The ring plane normal (aligned with the planet's rotation axis).
* The primary sun's direction.
* The surface point's position in planet-local space.

The ring shadow has:
* A radial profile (inner/outer radius).
* Low-frequency banding (§5.2: тонкая полосатость).
* A ``coldBias`` visual tint (subtle cool shift under ring shadow).

No textures — all patterns are procedural (trigonometric band function).

Public API
----------
RingShadowProjector(config=None)
  .shadow_at(surface_point, sun_dir) → float   [0=no shadow, 1=max shadow]
  .cold_bias_at(shadow) → float                [0–1 cold-bias tint strength]
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

_Vec3 = Tuple[float, float, float]


def _dot(a: _Vec3, b: _Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


def _length(v: _Vec3) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _normalize(v: _Vec3) -> _Vec3:
    m = _length(v)
    if m < 1e-12:
        return (0.0, 1.0, 0.0)
    return (v[0] / m, v[1] / m, v[2] / m)


class RingShadowProjector:
    """Projects planetary ring shadow onto the surface (§5.2).

    Parameters
    ----------
    config :
        Optional dict; reads ``render.ring_shadow_strength``,
        ``render.ring_inner_radius``, ``render.ring_outer_radius``,
        ``render.ring_band_strength``.
    """

    # Default ring geometry (in units of planet_radius)
    _DEFAULT_INNER = 1.3   # ring starts at 1.3× planet radius
    _DEFAULT_OUTER = 2.2   # ring ends at 2.2× planet radius
    _DEFAULT_BAND  = 0.25  # low-frequency banding amplitude (§5.2)

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = (config or {}).get("render", {}) or {}

        self._strength: float = float(cfg.get("ring_shadow_strength", 0.35))
        self._inner_r: float = float(cfg.get("ring_inner_radius", self._DEFAULT_INNER))
        self._outer_r: float = float(cfg.get("ring_outer_radius", self._DEFAULT_OUTER))
        self._band: float = float(cfg.get("ring_band_strength", self._DEFAULT_BAND))

        # Ring plane normal = planet north pole (Y-axis by convention)
        self._ring_normal: _Vec3 = (0.0, 1.0, 0.0)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def shadow_at(
        self,
        surface_point: _Vec3,
        sun_dir: _Vec3,
    ) -> float:
        """Compute ring shadow attenuation at *surface_point*.

        Parameters
        ----------
        surface_point :
            Position in planet-local coordinates (origin = planet centre).
        sun_dir :
            Normalised direction **toward** the sun.

        Returns
        -------
        float
            Shadow in [0, 1]; 0 = no ring shadow, 1 = maximum attenuation.
        """
        sun = _normalize(sun_dir)

        # Project the sun ray from surface_point upward to the ring plane
        # Ring plane: dot(p, ring_normal) = 0 (equatorial plane)
        rn = self._ring_normal
        denom = _dot(sun, rn)

        if abs(denom) < 1e-6:
            # Sun ray nearly parallel to ring plane → no shadow
            return 0.0

        # t: how far along the sun ray until we hit the ring plane
        t = -_dot(surface_point, rn) / denom
        if t < 0.0:
            # Ring plane is behind the surface (sun below horizon)
            return 0.0

        # Intersection point on the ring plane
        ix = surface_point[0] + sun[0] * t
        iy = surface_point[1] + sun[1] * t
        iz = surface_point[2] + sun[2] * t

        # Radial distance from planet axis (XZ plane distance)
        r = math.sqrt(ix * ix + iz * iz)

        # Check if r falls within the ring annulus
        if r < self._inner_r or r > self._outer_r:
            return 0.0

        # Smooth ramp at inner and outer edges
        ring_width = self._outer_r - self._inner_r
        edge_blend = _clamp(
            min(r - self._inner_r, self._outer_r - r) / (ring_width * 0.15),
            0.0, 1.0,
        )

        # Low-frequency banding (§5.2: тонкая полосатость)
        # 3 bands across ring width
        band_phase = (r - self._inner_r) / ring_width * math.pi * 3.0
        band = 0.5 + 0.5 * math.cos(band_phase)
        banding = 1.0 - self._band * band  # [1-band, 1] range

        shadow = self._strength * edge_blend * banding
        return _clamp(shadow, 0.0, 1.0)

    def cold_bias_at(self, shadow: float) -> float:
        """Return a cold-bias tint strength proportional to ring shadow.

        The ring's cool shadow produces a subtle blue shift on the surface
        (§5.2: влияет на coldBias визуально).
        """
        return _clamp(shadow * 0.6, 0.0, 1.0)
