"""MemoryToEvolutionAdapter — Stage 51 memory → PlanetEvolutionState bridge.

Translates WorldMemoryState fields into adjustments applied to
PlanetEvolutionState, bridging the emergent memory layer with the
planetary evolution system (Stage 50).

Effects applied
---------------
* **stressAccumulation** → accelerates ``crustStabilityMap`` decay and can
  trigger localised crust collapse.
* **erosionBias** → boosts ``slopeCreepMap`` (erosion memory accelerates
  slow downhill mass flux).
* **compactionHistory** → reduces ``dustReservoirMap`` locally (dust
  displaced by packed trail → less available for wind transport).

The adapter is stateless and deterministic; it returns delta values
that the caller applies to PlanetEvolutionState in-place.

Public API
----------
MemoryToEvolutionAdapter(config=None)
  .evolution_deltas(memory_state, tile_idx, dt) -> dict
      Returns dict with keys:
        "crust_stability_delta"  : float   (negative = weakened)
        "slope_creep_delta"      : float   (positive = more creep)
        "dust_reservoir_delta"   : float   (negative = less dust)
"""
from __future__ import annotations

from src.memory.WorldMemoryState import WorldMemoryState


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


class MemoryToEvolutionAdapter:
    """Produces evolution field deltas from WorldMemoryState.

    Parameters
    ----------
    config : dict or None
        Reads ``memory.max_influence``.
    """

    _DEFAULTS = {
        "max_influence":          0.3,
        "stress_crust_decay_k":   0.04,  # extra crust decay per stress unit
        "erosion_creep_k":        0.05,  # slope creep boost per erosion bias unit
        "compact_dust_reduce_k":  0.03,  # dust reservoir reduction per compaction
    }

    def __init__(self, config=None) -> None:
        cfg = dict(self._DEFAULTS)
        if isinstance(config, dict):
            src = config.get("memory", config)
            for k in self._DEFAULTS:
                if k in src:
                    cfg[k] = src[k]

        self._max_inf:       float = float(cfg["max_influence"])
        self._stress_crust:  float = float(cfg["stress_crust_decay_k"])
        self._erosion_creep: float = float(cfg["erosion_creep_k"])
        self._compact_dust:  float = float(cfg["compact_dust_reduce_k"])

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def evolution_deltas(
        self,
        memory_state: WorldMemoryState,
        tile_idx: int,
        dt: float,
    ) -> dict:
        """Compute per-dt evolution field adjustments for *tile_idx*.

        Parameters
        ----------
        memory_state : WorldMemoryState (read-only).
        tile_idx     : Flat tile index.
        dt           : Time step (seconds).

        Returns
        -------
        dict with keys ``crust_stability_delta``, ``slope_creep_delta``,
        ``dust_reservoir_delta``.
        """
        stress   = memory_state.stressAccumulationField[tile_idx]
        eb       = memory_state.erosionBiasField[tile_idx]
        ch       = memory_state.compactionHistoryField[tile_idx]
        cap      = self._max_inf

        crust_delta = _clamp(-self._stress_crust  * stress * dt, -cap, 0.0)
        creep_delta = _clamp( self._erosion_creep * eb     * dt,  0.0, cap)
        dust_delta  = _clamp(-self._compact_dust  * ch     * dt, -cap, 0.0)

        return {
            "crust_stability_delta": crust_delta,
            "slope_creep_delta":     creep_delta,
            "dust_reservoir_delta":  dust_delta,
        }
