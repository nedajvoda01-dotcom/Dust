"""MemoryToMaterialAdapter — Stage 51 memory → MaterialState bridge.

Translates WorldMemoryState fields into adjustments applied to
SurfaceMaterialState, bridging the emergent memory layer with the
material phase system (Stage 45).

Effects applied
---------------
* **compactionHistory** → boosts ``snow_compaction`` gain; reduces
  ``dust_thickness`` (the trail is packed / dust is displaced).
* **stressAccumulation** → reduces ``crust_hardness`` (stressed crust
  weakens); slightly increases ``roughness`` (micro-fractures).
* **acousticImprint** — no direct material change; consumed by audio layer.

The adapter does NOT write directly to MaterialState fields — it returns
delta values that the caller applies.  This keeps it stateless and
deterministic.

Public API
----------
MemoryToMaterialAdapter(config=None)
  .material_deltas(memory_state, tile_idx, dt) -> dict
      Returns dict with keys:
        "snow_compaction_delta"  : float
        "dust_thickness_delta"   : float   (negative = less dust)
        "crust_hardness_delta"   : float   (negative = weakened)
        "roughness_delta"        : float
"""
from __future__ import annotations

from src.memory.WorldMemoryState import WorldMemoryState


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


class MemoryToMaterialAdapter:
    """Produces material field deltas from WorldMemoryState.

    Parameters
    ----------
    config : dict or None
        Reads ``memory.max_influence``.
    """

    _DEFAULTS = {
        "max_influence":         0.3,   # cap on any single field delta per call
        "compaction_boost_k":    0.05,  # snow compaction boost coefficient
        "dust_displace_k":       0.04,  # dust thickness reduction coefficient
        "stress_crust_k":        0.06,  # crust hardness reduction per stress unit
        "stress_roughness_k":    0.03,  # roughness increase per stress unit
    }

    def __init__(self, config=None) -> None:
        cfg = dict(self._DEFAULTS)
        if isinstance(config, dict):
            src = config.get("memory", config)
            for k in self._DEFAULTS:
                if k in src:
                    cfg[k] = src[k]

        self._max_inf:       float = float(cfg["max_influence"])
        self._compact_k:     float = float(cfg["compaction_boost_k"])
        self._dust_k:        float = float(cfg["dust_displace_k"])
        self._stress_crust:  float = float(cfg["stress_crust_k"])
        self._stress_rough:  float = float(cfg["stress_roughness_k"])

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def material_deltas(
        self,
        memory_state: WorldMemoryState,
        tile_idx: int,
        dt: float,
    ) -> dict:
        """Compute per-dt material field adjustments for *tile_idx*.

        Parameters
        ----------
        memory_state : WorldMemoryState (read-only).
        tile_idx     : Flat tile index.
        dt           : Time step (seconds).

        Returns
        -------
        dict with keys ``snow_compaction_delta``, ``dust_thickness_delta``,
        ``crust_hardness_delta``, ``roughness_delta``.
        """
        ch      = memory_state.compactionHistoryField[tile_idx]
        stress  = memory_state.stressAccumulationField[tile_idx]
        cap     = self._max_inf

        snow_delta   = _clamp(self._compact_k  * ch     * dt, -cap, cap)
        dust_delta   = _clamp(-self._dust_k    * ch     * dt, -cap, 0.0)
        crust_delta  = _clamp(-self._stress_crust * stress * dt, -cap, 0.0)
        rough_delta  = _clamp(self._stress_rough  * stress * dt, 0.0, cap)

        return {
            "snow_compaction_delta": snow_delta,
            "dust_thickness_delta":  dust_delta,
            "crust_hardness_delta":  crust_delta,
            "roughness_delta":       rough_delta,
        }
