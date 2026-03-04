"""CondensationModel — Stage 65 fog/steam condensation and evaporation.

Implements spec §3.2 and §4.2 for FogVolume and SteamVolume layers.

Physics model
-------------
Condensation rate:
    condense = condense_k × humidity_proxy × cold_bias × dt

Evaporation rate:
    evaporate = evap_k × (1 − humidity_proxy) × temperature_proxy × dt

Both rates are computed per-domain (not per-voxel) from ambient atmospheric
proxies; the result is applied uniformly to all voxels in the grid, with a
small vertical gradient (fog thicker near ground).

Mass conservation link (spec §5.2)
------------------------------------
The model returns ``(condensed, evaporated)`` floats representing the net
vapor exchange.  The authoritative server decrements the humidity/vapor
reservoir by ``condensed`` and increments it by ``evaporated``; the client
visual simulation does *not* write back to atmospheric state.

Public API
----------
CondensationModel(config=None)
  .step(grid, humidity_proxy, temperature_proxy, dt)
      → tuple[float, float]  (condensed, evaporated)
      In-place update of *grid*.
"""
from __future__ import annotations

from typing import Optional, Tuple

from src.vol.DensityGrid import DensityGrid


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


class CondensationModel:
    """Fog/steam condensation & evaporation for volumetric grids.

    Parameters
    ----------
    config :
        Optional dict; reads ``vol.condense_k`` and ``vol.evap_k``.
    """

    _DEFAULT_CONDENSE_K = 0.08  # per second
    _DEFAULT_EVAP_K     = 0.04  # per second

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg  = config or {}
        vcfg = cfg.get("vol", {}) or {}
        self._condense_k: float = float(
            vcfg.get("condense_k", self._DEFAULT_CONDENSE_K)
        )
        self._evap_k: float = float(
            vcfg.get("evap_k", self._DEFAULT_EVAP_K)
        )

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def step(
        self,
        grid:              DensityGrid,
        humidity_proxy:    float,
        temperature_proxy: float,
        dt:                float,
    ) -> Tuple[float, float]:
        """Update fog/steam density in-place.

        Parameters
        ----------
        grid :
            The :class:`~src.vol.DensityGrid.DensityGrid` to update.
        humidity_proxy :
            Ambient humidity proxy [0..1].
        temperature_proxy :
            Ambient temperature proxy [0..1]; higher = warmer = less fog.
        dt :
            Time step in seconds.

        Returns
        -------
        (condensed, evaporated) : float, float
            Net vapor exchanged with the atmospheric reservoir.
        """
        cold_bias   = _clamp(1.0 - temperature_proxy)
        warm_bias   = _clamp(temperature_proxy)

        condense_rate = self._condense_k * humidity_proxy * cold_bias * dt
        evap_rate     = self._evap_k     * (1.0 - humidity_proxy) * warm_bias * dt

        total_condensed  = 0.0
        total_evaporated = 0.0

        w, h, d = grid.width, grid.height, grid.depth

        for iz in range(d):
            # Fog thicker near ground → condense more at low layers
            ground_factor = 1.0 - iz / max(d - 1, 1)
            layer_condense = condense_rate * (0.5 + 0.5 * ground_factor)
            layer_evap     = evap_rate

            for iy in range(h):
                for ix in range(w):
                    current = grid.density(ix, iy, iz)

                    # Condensation: adds density (fog forms)
                    added = _clamp(layer_condense, 0.0, 1.0 - current)
                    grid.add_density(ix, iy, iz, added)
                    total_condensed += added

                    # Evaporation: removes density (fog disperses)
                    removed = _clamp(current * layer_evap, 0.0, grid.density(ix, iy, iz))
                    grid.add_density(ix, iy, iz, -removed)
                    total_evaporated += removed

        return (total_condensed, total_evaporated)
