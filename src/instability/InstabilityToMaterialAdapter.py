"""InstabilityToMaterialAdapter — Stage 52 instability → material state bridge.

Applies the physical consequences of instability events to
:class:`~material.SurfaceMaterialState.SurfaceMaterialState` cells.

Operations
----------
* ``apply_crust_failure(event, cell)``  — reduces crust_hardness, raises roughness.
* ``apply_dust_avalanche(event, grid, tile_to_cell)`` — redistributes dust thickness.

Public API
----------
InstabilityToMaterialAdapter(config=None)
  .apply_crust_failure(event, cell)                       → None
  .apply_dust_redistrib(dust_delta, dust_gain, src_cell, dst_cells) → None
"""
from __future__ import annotations

from typing import List, Optional

from src.instability.CrustFailureModel import CrustFailureEvent
from src.instability.DustAvalancheModel import DustAvalancheEvent
from src.material.SurfaceMaterialState import SurfaceMaterialState


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


class InstabilityToMaterialAdapter:
    """Applies instability events to SurfaceMaterialState cells.

    Parameters
    ----------
    config :
        Optional dict; reads ``instability.*`` keys.
    """

    _DEFAULT_MATERIAL_K = 0.4   # scale factor on event intensity → material delta

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = (config or {}).get("instability", {}) or {}
        self._k: float = float(cfg.get("energy_to_material_k", self._DEFAULT_MATERIAL_K))

    def apply_crust_failure(
        self,
        event: CrustFailureEvent,
        cell:  SurfaceMaterialState,
    ) -> None:
        """Reduce crust hardness and raise roughness on *cell*.

        Parameters
        ----------
        event : CrustFailureEvent from CrustFailureModel.
        cell  : SurfaceMaterialState to modify in-place.
        """
        cell.crust_hardness = _clamp(cell.crust_hardness - event.crust_delta * self._k)
        cell.roughness      = _clamp(cell.roughness      + event.roughness_gain * self._k)

    def apply_dust_redistrib(
        self,
        dust_delta: float,
        src_cell:   SurfaceMaterialState,
        dst_cells:  List[SurfaceMaterialState],
        shares:     Optional[List[float]] = None,
    ) -> None:
        """Move dust from *src_cell* to *dst_cells*.

        Parameters
        ----------
        dust_delta : Total dust thickness removed from source [0..1].
        src_cell   : Source SurfaceMaterialState.
        dst_cells  : Destination cells.
        shares     : Fraction of dust_delta sent to each dst cell.
                     If None, split equally.
        """
        if not dst_cells:
            return

        n = len(dst_cells)
        if shares is None:
            shares = [1.0 / n] * n

        src_cell.dust_thickness = _clamp(src_cell.dust_thickness - dust_delta * self._k)

        for cell, share in zip(dst_cells, shares):
            cell.dust_thickness = _clamp(cell.dust_thickness + dust_delta * self._k * share)
