"""DustAvalancheModel — Stage 52 dust avalanching model.

Fires when shearStressField exceeds threshold AND dustLoadField is high.
On trigger:

* dustLoadField is partially discharged.
* shearStressField is reduced.
* Dust is redistributed: source tile loses dust, neighbouring downslope
  tiles gain (caller provides slope map for direction hint).

Public API
----------
DustAvalancheModel(config=None)
  .process(state, tile, slope_map) → DustAvalancheEvent | None
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.instability.InstabilityState import InstabilityState


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


@dataclass
class DustAvalancheEvent:
    """Fired when a dust avalanche is triggered.

    Attributes
    ----------
    tile              : Source tile index.
    intensity         : Discharge energy.
    dust_delta        : Dust removed from source tile [0..1].
    dust_redistrib    : {tile_idx: dust_gain} for neighbour tiles.
    """
    tile:           int
    intensity:      float
    dust_delta:     float
    dust_redistrib: Dict[int, float] = field(default_factory=dict)


class DustAvalancheModel:
    """Detects and processes dust avalanche events.

    Parameters
    ----------
    config :
        Optional dict; reads ``instability.*`` keys.
    """

    _DEFAULT_SHEAR_THRESHOLD = 0.6
    _DEFAULT_DUST_THRESHOLD  = 0.4
    _DEFAULT_DISCHARGE_K     = 0.6
    _DEFAULT_MATERIAL_K      = 0.3   # fraction of discharge to dust redistribution

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = (config or {}).get("instability", {}) or {}
        self._shear_thresh:  float = float(cfg.get("shear_threshold",  self._DEFAULT_SHEAR_THRESHOLD))
        self._dust_thresh:   float = float(cfg.get("dust_threshold",   self._DEFAULT_DUST_THRESHOLD))
        self._discharge_k:   float = float(cfg.get("dust_discharge_k", self._DEFAULT_DISCHARGE_K))
        self._material_k:    float = float(cfg.get("energy_to_material_k", self._DEFAULT_MATERIAL_K))

    def process(
        self,
        state:     InstabilityState,
        tile:      int,
        slope_map: Optional[List[float]] = None,
    ) -> Optional[DustAvalancheEvent]:
        """Test tile for dust avalanche and discharge if conditions met.

        Parameters
        ----------
        state     : InstabilityState (modified in-place on discharge).
        tile      : Flat tile index to test.
        slope_map : Per-tile slope [0..1]; used to weight redistribution.

        Returns
        -------
        DustAvalancheEvent if an avalanche occurred, else None.
        """
        shear = state.shearStressField[tile]
        dust  = state.dustLoadField[tile]

        if shear <= self._shear_thresh or dust <= self._dust_thresh:
            return None

        shear_excess = shear - self._shear_thresh
        intensity    = shear_excess * dust * self._discharge_k

        # Discharge source tile
        state.shearStressField[tile] = _clamp(shear - shear_excess * self._discharge_k)
        dust_removed = _clamp(intensity * self._material_k)
        state.dustLoadField[tile] = _clamp(dust - dust_removed)

        # Redistribute to neighbours weighted by their slope (higher slope = more)
        neighbours = state.neighbors(tile)
        total_slope = sum(
            (slope_map[nb] if slope_map is not None else 0.5)
            for nb in neighbours
        ) or 1.0

        redistrib: Dict[int, float] = {}
        for nb in neighbours:
            nb_slope = slope_map[nb] if slope_map is not None else 0.5
            share = dust_removed * (nb_slope / total_slope)
            if share > 1e-6:
                redistrib[nb] = share
                state.dustLoadField[nb] = _clamp(state.dustLoadField[nb] + share)

        return DustAvalancheEvent(
            tile=tile,
            intensity=intensity,
            dust_delta=dust_removed,
            dust_redistrib=redistrib,
        )
