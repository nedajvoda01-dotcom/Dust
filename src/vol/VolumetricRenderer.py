"""VolumetricRenderer — Stage 65 LOD-aware volumetric ray-march renderer.

Implements spec §6: short near-field ray-march + far-field height-fog
approximation, executed in an internal buffer before the pixel pass (§6.3).

LOD tiers (spec §10)
---------------------
Tier 0 : height fog only (0 ray-march steps)
Tier 1 : 3-D density 48³ + 16 ray-march steps
Tier 2 : 64³ + 24 steps
Tier 3 : 96³ + 32 steps (high-end only)

Output
------
For each pixel/ray, the renderer accumulates transmittance and in-scattered
light using the Beer-Lambert law.  The result is a per-pixel
``(transmittance, scatter_r, scatter_g, scatter_b)`` tuple that can be
composited with the rasterised colour buffer **before** the PixelQuantizer
pass (Stage 62).

TAA after pixelisation is forbidden (spec §6.3); the renderer does not
implement or call any temporal AA.

Public API
----------
VolumetricRenderer(config=None)
  .render_ray(grid, ray_origin, ray_dir, max_distance)
      → RaymarchResult
  .set_light_params(phase_strength, absorption, anisotropy)
  .tier  → int  (0–3, read/write)

RaymarchResult (dataclass)
  .transmittance    float [0..1]
  .scatter          tuple[float, float, float]   (r, g, b)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

from src.vol.DensityGrid import DensityGrid


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


# ---------------------------------------------------------------------------
# LOD tier configuration
# ---------------------------------------------------------------------------

_TIER_STEPS = {0: 0, 1: 16, 2: 24, 3: 32}
_TIER_GRID_RES = {0: 0, 1: 48, 2: 64, 3: 96}


# ---------------------------------------------------------------------------
# RaymarchResult
# ---------------------------------------------------------------------------

@dataclass
class RaymarchResult:
    """Output of one ray-march pass through a volumetric domain."""
    transmittance: float = 1.0
    scatter: Tuple[float, float, float] = (0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# VolumetricRenderer
# ---------------------------------------------------------------------------

class VolumetricRenderer:
    """LOD-aware volumetric renderer for density grids.

    Parameters
    ----------
    config :
        Optional dict; reads ``vol.raymarch_steps``, ``vol.absorption``,
        ``vol.anisotropy``, ``vol.phase_function_strength``.
    """

    _DEFAULT_ABSORPTION        = 0.4
    _DEFAULT_ANISOTROPY        = 0.3
    _DEFAULT_PHASE_STRENGTH    = 0.6
    _DEFAULT_SCATTER_COLOR     = (0.85, 0.78, 0.60)  # warm dust/fog scatter

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg  = config or {}
        vcfg = cfg.get("vol", {}) or {}

        self._absorption:     float = float(vcfg.get("absorption",     self._DEFAULT_ABSORPTION))
        self._anisotropy:     float = float(vcfg.get("anisotropy",     self._DEFAULT_ANISOTROPY))
        self._phase_strength: float = float(vcfg.get("phase_function_strength", self._DEFAULT_PHASE_STRENGTH))
        self._scatter_color: Tuple[float, float, float] = self._DEFAULT_SCATTER_COLOR

        # Default raymarch steps; can be overridden by tier setter
        self._steps: int = int(vcfg.get("raymarch_steps", _TIER_STEPS[1]))
        self._tier:  int = 1

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def tier(self) -> int:
        return self._tier

    @tier.setter
    def tier(self, value: int) -> None:
        t = max(0, min(3, int(value)))
        self._tier  = t
        self._steps = _TIER_STEPS[t]

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def set_light_params(
        self,
        phase_strength: float,
        absorption:     float,
        anisotropy:     float,
    ) -> None:
        """Update phase/absorption/anisotropy coefficients (spec §7)."""
        self._phase_strength = _clamp(phase_strength)
        self._absorption     = _clamp(absorption)
        self._anisotropy     = _clamp(anisotropy)

    def render_ray(
        self,
        grid:         DensityGrid,
        ray_origin:   Tuple[float, float, float],
        ray_dir:      Tuple[float, float, float],
        max_distance: float,
    ) -> RaymarchResult:
        """Ray-march through *grid* and return transmittance + scatter.

        If tier == 0 (height-fog-only mode), the result is computed
        analytically from the mean column density without ray-marching.

        Parameters
        ----------
        grid :
            Source DensityGrid to sample.
        ray_origin :
            World-space ray origin in voxel-normalised coords [0..1]³.
        ray_dir :
            Normalised ray direction.
        max_distance :
            Maximum distance to march (same units as ray_origin).
        """
        if self._steps == 0:
            return self._height_fog_approx(grid, ray_origin)

        ox, oy, oz = ray_origin
        dx, dy, dz = ray_dir

        # Normalise direction
        length = math.sqrt(dx * dx + dy * dy + dz * dz)
        if length < 1e-12:
            return RaymarchResult()
        dx, dy, dz = dx / length, dy / length, dz / length

        step_size = max_distance / max(self._steps, 1)
        transmittance = 1.0
        scatter_r = scatter_g = scatter_b = 0.0

        for i in range(self._steps):
            t = (i + 0.5) * step_size
            px = ox + dx * t
            py = oy + dy * t
            pz = oz + dz * t

            # Map to voxel indices
            ix = int(_clamp(px, 0.0, 1.0 - 1e-9) * grid.width)
            iy = int(_clamp(py, 0.0, 1.0 - 1e-9) * grid.height)
            iz = int(_clamp(pz, 0.0, 1.0 - 1e-9) * grid.depth)

            d = grid.density(ix, iy, iz)
            if d <= 0.0:
                continue

            # Beer-Lambert extinction
            ext = self._absorption * d * step_size
            tr_step = math.exp(-ext)
            scatter_contrib = transmittance * (1.0 - tr_step) * self._phase_strength

            scatter_r += scatter_contrib * self._scatter_color[0]
            scatter_g += scatter_contrib * self._scatter_color[1]
            scatter_b += scatter_contrib * self._scatter_color[2]
            transmittance *= tr_step

            if transmittance < 1e-4:
                break

        return RaymarchResult(
            transmittance=_clamp(transmittance),
            scatter=(
                _clamp(scatter_r),
                _clamp(scatter_g),
                _clamp(scatter_b),
            ),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _height_fog_approx(
        self,
        grid:       DensityGrid,
        ray_origin: Tuple[float, float, float],
    ) -> RaymarchResult:
        """Tier-0: analytical height-fog from average column density."""
        ox, oy, _ = ray_origin
        ix = int(_clamp(ox, 0.0, 1.0 - 1e-9) * grid.width)
        iy = int(_clamp(oy, 0.0, 1.0 - 1e-9) * grid.height)

        col_density = sum(
            grid.density(ix, iy, iz) for iz in range(grid.depth)
        ) / max(grid.depth, 1)

        transmittance = math.exp(-self._absorption * col_density)
        scatter_val   = (1.0 - transmittance) * self._phase_strength
        return RaymarchResult(
            transmittance=_clamp(transmittance),
            scatter=(
                _clamp(scatter_val * self._scatter_color[0]),
                _clamp(scatter_val * self._scatter_color[1]),
                _clamp(scatter_val * self._scatter_color[2]),
            ),
        )
