"""SubsurfaceToVolumetrics — Stage 67 adapter: vent → volumetric layers.

Translates vent events into steam and dust density additions to a
:class:`~src.vol.DensityGrid.DensityGrid`.

Effects applied (§8)
--------------------
Vent (explosive) :
    * +SteamVolume density (vent steam plume)
    * +DustVolume  density (explosive ash/dust)

Lava cell (active) :
    * +SteamVolume density (off-gassing steam from hot lava surface)

Public API
----------
SubsurfaceToVolumetrics(config=None)
  .apply_vent(steam_grid, dust_grid, intensity, dt) → None
  .apply_lava_offgas(steam_grid, lava_volume, dt)   → None
"""
from __future__ import annotations

from typing import Optional

from src.vol.DensityGrid import DensityGrid


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


class SubsurfaceToVolumetrics:
    """Inject vent and lava off-gassing into volumetric density grids.

    Parameters
    ----------
    config :
        Optional dict; reads ``subsurface67.*`` keys.
    """

    _DEFAULT_STEAM_RATE  = 0.15   # steam added per unit intensity per second
    _DEFAULT_DUST_RATE   = 0.08   # dust added per unit intensity per second
    _DEFAULT_OFFGAS_RATE = 0.03   # steam from active lava surface

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = (config or {}).get("subsurface67", {}) or {}
        self._steam_rate  = float(cfg.get("vent_steam_rate",    self._DEFAULT_STEAM_RATE))
        self._dust_rate   = float(cfg.get("vent_dust_rate",     self._DEFAULT_DUST_RATE))
        self._offgas_rate = float(cfg.get("lava_offgas_rate",   self._DEFAULT_OFFGAS_RATE))

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def apply_vent(
        self,
        steam_grid: DensityGrid,
        dust_grid:  DensityGrid,
        intensity:  float,
        dt:         float,
    ) -> None:
        """Add steam and dust to the grids from a vent event.

        Density is added at voxel (0, 0, 0) as a point source; the
        advection solver spreads it over subsequent ticks.
        """
        steam_delta = self._steam_rate * intensity * dt
        dust_delta  = self._dust_rate  * intensity * dt
        steam_grid.add_density(0, 0, 0, steam_delta)
        dust_grid.add_density(0, 0, 0, dust_delta)

    def apply_lava_offgas(
        self,
        steam_grid:  DensityGrid,
        lava_volume: float,
        dt:          float,
    ) -> None:
        """Add steam from active lava surface off-gassing.

        Parameters
        ----------
        steam_grid :
            Steam DensityGrid to inject into.
        lava_volume :
            Normalised magma volume [0, 1] driving off-gassing.
        dt :
            Time step in seconds.
        """
        steam_delta = self._offgas_rate * lava_volume * dt
        steam_grid.add_density(0, 0, 0, steam_delta)
