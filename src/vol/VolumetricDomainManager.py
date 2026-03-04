"""VolumetricDomainManager — Stage 65 active domain lifecycle manager.

Manages a set of :class:`~src.vol.DensityGrid.DensityGrid` domains
(one per active player/source/fog-basin), each paired with its own
:class:`~src.vol.AdvectionSolver.AdvectionSolver`,
:class:`~src.vol.SettlingModel.SettlingModel`, and
:class:`~src.vol.CondensationModel.CondensationModel`.

Design (spec §2)
----------------
* Each domain has a world-space anchor (x, y) and a fixed voxel resolution
  that maps to the configurable domain_radius_m.
* Domains are created lazily when a player/source activates them and removed
  when the player moves away (beyond ``evict_radius_m``).
* The manager ticks all active domains at ``vol.tick_hz`` (5–15 Hz), not at
  the frame rate.

Scrolling (spec §2.3)
----------------------
When a domain anchor would need to shift, the existing density data is
translated (scrolled) by one-voxel steps rather than discarded.

Public API
----------
VolumetricDomainManager(config=None)
  .get_or_create(layer_type, anchor_x, anchor_y)  → DensityGrid
  .tick(dt, wind_x, wind_y, humidity, temperature) → dict[str, float]
      Returns per-domain settled_mass for authoritative consumption.
  .remove(layer_type, anchor_x, anchor_y)          → None
  .active_domains()                                → list[DomainHandle]
  .tick_hz                                         → float

DomainHandle (dataclass)
  .layer_type, .anchor_x, .anchor_y, .grid
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.vol.DensityGrid       import DensityGrid, VolumeLayerType
from src.vol.AdvectionSolver   import AdvectionSolver
from src.vol.SettlingModel     import SettlingModel
from src.vol.CondensationModel import CondensationModel


# ---------------------------------------------------------------------------
# DomainHandle
# ---------------------------------------------------------------------------

@dataclass
class DomainHandle:
    """Metadata + grid for one active volumetric domain."""
    layer_type: str
    anchor_x:   float
    anchor_y:   float
    grid:       DensityGrid


# ---------------------------------------------------------------------------
# VolumetricDomainManager
# ---------------------------------------------------------------------------

class VolumetricDomainManager:
    """Lifecycle manager for volumetric density domains.

    Parameters
    ----------
    config :
        Optional dict; reads ``vol.*`` keys.
    """

    _DEFAULT_TICK_HZ      = 10.0
    _DEFAULT_GRID_RES     = 64
    _DEFAULT_DOMAIN_R     = 300.0   # metres
    _DEFAULT_EVICT_RADIUS = 600.0   # metres

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg  = config or {}
        vcfg = cfg.get("vol", {}) or {}

        self._tick_hz:      float = float(vcfg.get("tick_hz",        self._DEFAULT_TICK_HZ))
        self._grid_res:     int   = int(  vcfg.get("grid_resolution", self._DEFAULT_GRID_RES))
        self._domain_r:     float = float(vcfg.get("domain_radius_m", self._DEFAULT_DOMAIN_R))
        self._evict_r:      float = float(vcfg.get("evict_radius_m",  self._DEFAULT_EVICT_RADIUS))

        self._tick_interval: float = 1.0 / max(self._tick_hz, 1e-6)
        self._accum:         float = 0.0

        self._domains: Dict[Tuple[str, float, float], DomainHandle] = {}
        self._advect   = AdvectionSolver(config)
        self._settling = SettlingModel(config)
        self._condense = CondensationModel(config)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def tick_hz(self) -> float:
        return self._tick_hz

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def get_or_create(
        self,
        layer_type: str,
        anchor_x:   float,
        anchor_y:   float,
    ) -> DensityGrid:
        """Return existing domain grid or create a new one.

        Parameters
        ----------
        layer_type :
            One of :class:`~src.vol.DensityGrid.VolumeLayerType` constants.
        anchor_x, anchor_y :
            World-space anchor coordinates.
        """
        key = (layer_type, float(anchor_x), float(anchor_y))
        if key not in self._domains:
            r   = self._grid_res
            grid = DensityGrid(r, r, max(r // 4, 8), layer_type=layer_type)
            self._domains[key] = DomainHandle(
                layer_type=layer_type,
                anchor_x=anchor_x,
                anchor_y=anchor_y,
                grid=grid,
            )
        return self._domains[key].grid

    def remove(
        self,
        layer_type: str,
        anchor_x:   float,
        anchor_y:   float,
    ) -> None:
        """Remove (evict) a domain."""
        key = (layer_type, float(anchor_x), float(anchor_y))
        self._domains.pop(key, None)

    def active_domains(self) -> List[DomainHandle]:
        """Return list of all active domain handles."""
        return list(self._domains.values())

    def tick(
        self,
        dt:          float,
        wind_x:      float = 0.0,
        wind_y:      float = 0.0,
        humidity:    float = 0.3,
        temperature: float = 0.5,
    ) -> Dict[str, float]:
        """Advance all active domains by *dt* seconds.

        Physics ticks at ``tick_hz``; multiple ticks may fire if *dt* is
        large.  Only dust-type layers use the settling model; only fog/steam
        layers use condensation.

        Returns
        -------
        dict
            Mapping ``"<layer_type>@(<ax>,<ay>)"`` → settled mass floats.
            Intended for authoritative server consumption via MassExchangeAPI.
        """
        self._accum += dt
        settled_report: Dict[str, float] = {}

        while self._accum >= self._tick_interval:
            sub_dt = self._tick_interval
            self._accum -= sub_dt

            for key, handle in self._domains.items():
                grid    = handle.grid
                ltype   = handle.layer_type
                lbl     = f"{ltype}@({handle.anchor_x},{handle.anchor_y})"

                # Advection: all layer types
                self._advect.step(grid, wind_x, wind_y, sub_dt)

                # Settling: dust and snow
                if ltype in (VolumeLayerType.DUST, VolumeLayerType.SNOW_DRIFT):
                    settled = self._settling.step(grid, sub_dt)
                    settled_report[lbl] = settled_report.get(lbl, 0.0) + settled

                # Condensation: fog and steam
                elif ltype in (VolumeLayerType.FOG, VolumeLayerType.STEAM):
                    self._condense.step(grid, humidity, temperature, sub_dt)

        return settled_report
