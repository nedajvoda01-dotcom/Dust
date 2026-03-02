"""DeformDisplacementPass — Stage 35 render pass for surface deformation.

Consumes DeformationField data and translates it into render-layer hints:

* H(x,y) → vertex displacement or parallax-normal perturbation suggestion
* M(x,y) → material overlay tint (darker/lighter/matte) + micro-normal hint

All output is pixel-style (band-limited, no high-frequency noise).
GPU uploads are budgeted via DeformationIntegrator.consume_dirty_set().

This module is a render-side stub that exposes the interface without
requiring a real GPU context (safe in headless / CI environments).

DeformDisplacementPass
    update(integrator, dirty_set)
    sample_displacement(chunk_id, ix, iy) -> float   (metres, for vertex)
    sample_overlay(chunk_id, ix, iy)      -> dict    (tint, normal_perturb)
    debug_heatmap(chunk_id)               -> List[float]  (dev-only)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set

from src.surface.DeformationIntegrator import DeformationIntegrator


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


# ---------------------------------------------------------------------------
# DeformDisplacementPass
# ---------------------------------------------------------------------------

class DeformDisplacementPass:
    """Render pass that exposes deformation field values to the renderer.

    Parameters
    ----------
    config:
        Optional Config with ``deform.render.*`` keys.
    """

    def __init__(self, config=None) -> None:
        self._cfg        = config
        self._max_uploads: int = self._get_i("render.max_uploads_per_frame", 4)

        # Shadow copies of the most-recently-uploaded fields
        # chunk_id → (h_snapshot, m_snapshot) as flat float lists
        self._h_cache: Dict[object, List[float]] = {}
        self._m_cache: Dict[object, List[float]] = {}

        self._uploads_this_frame: int = 0

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def update(
        self,
        integrator: DeformationIntegrator,
        dirty_set: Optional[Set] = None,
    ) -> None:
        """Copy field data from *integrator* for dirty chunks.

        Call once per frame after DeformationIntegrator.apply_samples().

        Parameters
        ----------
        integrator:
            The authoritative integrator holding DeformationField instances.
        dirty_set:
            Set of dirty chunk IDs (from integrator.consume_dirty_set()).
            When None, all active chunks are refreshed (use sparingly).
        """
        to_update = dirty_set if dirty_set is not None else set(
            integrator._fields.keys())

        self._uploads_this_frame = 0
        budget = self._max_uploads

        for chunk_id in to_update:
            if self._uploads_this_frame >= budget:
                break
            if chunk_id not in integrator._fields:
                continue
            f    = integrator._fields[chunk_id]
            res  = f.grid_res
            n    = res * res
            self._h_cache[chunk_id] = [f.h_at(i % res, i // res) for i in range(n)]
            self._m_cache[chunk_id] = [f.m_at(i % res, i // res) for i in range(n)]
            self._uploads_this_frame += 1

    def sample_displacement(
        self,
        chunk_id: object,
        ix: int,
        iy: int,
    ) -> float:
        """Return cached H displacement [m] at grid cell (ix, iy).

        Returns 0.0 if the chunk is not yet uploaded.
        """
        cache = self._h_cache.get(chunk_id)
        if cache is None:
            return 0.0
        res = int(len(cache) ** 0.5)
        idx = iy * res + ix
        if idx < 0 or idx >= len(cache):
            return 0.0
        return cache[idx]

    def sample_overlay(
        self,
        chunk_id: object,
        ix: int,
        iy: int,
    ) -> dict:
        """Return material overlay hints for pixel-style rendering.

        Returns
        -------
        dict with keys:
            tint_factor    — [0, 1]; 0 = normal, 1 = very dark (deep indent)
            roughness_bump — [0, 1]; increases perceived roughness of loose M
            normal_perturb — [−1, 1]; micro-normal tilt toward berm
        """
        h_cache = self._h_cache.get(chunk_id)
        m_cache = self._m_cache.get(chunk_id)

        if h_cache is None or m_cache is None:
            return {"tint_factor": 0.0, "roughness_bump": 0.0, "normal_perturb": 0.0}

        res = int(len(h_cache) ** 0.5)
        idx = iy * res + ix
        if idx < 0 or idx >= len(h_cache):
            return {"tint_factor": 0.0, "roughness_bump": 0.0, "normal_perturb": 0.0}

        h_m = h_cache[idx]          # metres (negative = indent)
        m   = m_cache[idx]          # [0, 1]

        # Deep indents appear darker (10 m → full dark tint); berms lighter
        tint    = _clamp(-h_m * 10.0, 0.0, 1.0)   # >0 for indents
        roughness = _clamp(m * 0.8, 0.0, 1.0)       # more loose material = rougher
        # Normal perturb based on H gradient (very approximate)
        normal_perturb = _clamp(h_m * 5.0, -1.0, 1.0)

        return {
            "tint_factor":    tint,
            "roughness_bump": roughness,
            "normal_perturb": normal_perturb,
        }

    def debug_heatmap(self, chunk_id: object) -> List[float]:
        """Return flat H values [m] for dev heatmap visualisation."""
        return list(self._h_cache.get(chunk_id, []))

    @property
    def uploads_this_frame(self) -> int:
        return self._uploads_this_frame

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _get_i(self, key: str, default: int) -> int:
        if self._cfg is None:
            return default
        v = self._cfg.get("deform", key, default=None)
        return int(v) if v is not None else default
