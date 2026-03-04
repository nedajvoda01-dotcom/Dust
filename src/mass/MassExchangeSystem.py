"""MassExchangeSystem — Stage 66 orchestrator for the mass exchange framework.

Implements the canonical tick order (spec §4):

1. Lift          surface → air
2. Advection     handled externally by Stage 65 VolumetricDomainManager
3. Settling      air → surface
4. Downhill flux surface ↔ surface (requires slope map)
5. Contact       player / instability impulses
6. Phase transitions handled externally by Stage 63 PlanetPhaseTransitions

This class owns the sub-models and exposes a ``tick()`` entry point that
drives a single active zone (one :class:`~src.material.PlanetChunkGrid.PlanetChunkGrid`
row + paired air densities).

LOD (spec §10)
--------------
Full computation runs only in active zones.  Inactive zones use a coarse
erosion / settling rate that approximates long-term behaviour without
voxel-level detail.

Config keys (``mass.*``)
-------------------------
See sub-model docstrings.  All keys are prefixed with ``mass.``.

Public API
----------
MassExchangeSystem(config=None)
  .tick(surface, air_dust, air_snow, wind_speed, slope_map,
        temperature, shelter, stress_map, dt) → MassExchangeStats
  .apply_contact(api, contact_impulse) → ContactResult
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from src.material.MassExchangeAPI import MassExchangeAPI
from src.material.PlanetChunkState import PlanetChunkState
from src.mass.LiftModel import LiftModel
from src.mass.SettlingModel import SettlingModel
from src.mass.DownhillFluxModel import DownhillFluxModel
from src.mass.ContactDisplacementModel import ContactDisplacementModel, ContactResult
from src.mass.FluxConservationChecker import FluxConservationChecker


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


@dataclass
class MassExchangeStats:
    """Per-tick summary of mass exchange activity."""
    total_dust_lifted:    float = 0.0
    total_snow_lifted:    float = 0.0
    total_dust_settled:   float = 0.0
    total_snow_settled:   float = 0.0
    total_downhill_flux:  float = 0.0
    conservation_ok:      bool  = True


class MassExchangeSystem:
    """Drive the Stage 66 mass exchange tick for one active zone.

    Parameters
    ----------
    config :
        Optional dict; reads ``mass.*`` sub-keys for each model.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        self._lift    = LiftModel(config)
        self._settle  = SettlingModel(config)
        self._downhill = DownhillFluxModel(config)
        self._contact  = ContactDisplacementModel(config)
        self._checker  = FluxConservationChecker(tolerance=1e-2)

        cfg = (config or {}).get("mass", {}) or {}
        self._max_flux: float = float(cfg.get("max_flux_per_tick", 0.05))

    # ------------------------------------------------------------------
    # Public — main tick
    # ------------------------------------------------------------------

    def tick(
        self,
        cells:         List[PlanetChunkState],
        air_dust:      List[float],
        air_snow:      List[float],
        wind_speed:    float,
        slope_map:     Optional[List[float]] = None,
        temperature:   float = 0.5,
        shelter:       float = 0.0,
        stress_map:    Optional[List[float]] = None,
        dt:            float = 1.0,
    ) -> MassExchangeStats:
        """Execute one mass-exchange tick over *cells*.

        Steps executed (spec §4):
          1. Lift       (surface → air_dust / air_snow)
          3. Settling   (air_dust / air_snow → surface)
          4. Downhill   (surface → neighbour surface)

        Steps 2 (advection) and 6 (phase transitions) are delegated
        to external systems and are not performed here.

        Parameters
        ----------
        cells :
            List of surface cells (one per terrain tile in the active zone).
        air_dust :
            Airborne dust density per cell [0..1].  Modified in-place.
        air_snow :
            Airborne snow-drift density per cell [0..1].  Modified in-place.
        wind_speed :
            Global wind speed for this zone [0..1].
        slope_map :
            Per-cell slope [0..1]; defaults to 0 for all cells if None.
        temperature :
            Zone temperature proxy [0..1].
        shelter :
            Zone topographic shelter [0..1].
        stress_map :
            Per-cell mechanical stress [0..1]; defaults to 0 if None.
        dt :
            Time step in seconds.

        Returns
        -------
        MassExchangeStats
        """
        n = len(cells)
        slopes  = slope_map  if slope_map  is not None else [0.0] * n
        stresses = stress_map if stress_map is not None else [0.0] * n

        stats = MassExchangeStats()

        # Snapshot total mass before
        before = self._checker.snapshot_total(cells, list(air_dust) + list(air_snow))

        # ------------------------------------------------------------------
        # Step 1: Lift — surface → air
        # ------------------------------------------------------------------
        for i, cell in enumerate(cells):
            rates = self._lift.compute_lift_rate(cell, wind_speed * dt, temperature)
            api = MassExchangeAPI(cell)
            api.apply_mass_delta("dustThickness", -rates.dust_lift)
            api.apply_mass_delta("snowMass",      -rates.snow_lift)
            air_dust[i] = _clamp(air_dust[i] + rates.dust_lift)
            air_snow[i] = _clamp(air_snow[i] + rates.snow_lift)
            stats.total_dust_lifted += rates.dust_lift
            stats.total_snow_lifted += rates.snow_lift

        # ------------------------------------------------------------------
        # Step 3: Settling — air → surface
        # ------------------------------------------------------------------
        for i, cell in enumerate(cells):
            rates = self._settle.compute_settling_rate(
                air_dust[i], air_snow[i],
                wind_speed, slopes[i], shelter,
            )
            api = MassExchangeAPI(cell)
            api.apply_mass_delta("dustThickness", rates.dust_settle)
            api.apply_mass_delta("snowMass",      rates.snow_settle)
            air_dust[i] = _clamp(air_dust[i] - rates.dust_settle)
            air_snow[i] = _clamp(air_snow[i] - rates.snow_settle)
            stats.total_dust_settled += rates.dust_settle
            stats.total_snow_settled += rates.snow_settle

        # ------------------------------------------------------------------
        # Step 4: Downhill flux — cells flow to the next cell (simple 1-D)
        # ------------------------------------------------------------------
        for i, cell in enumerate(cells):
            flux = self._downhill.compute_downhill_flux(cell, slopes[i], stresses[i])
            total_flux = flux.dust_flux + flux.snow_flux + flux.debris_flux
            if total_flux <= 0.0:
                continue
            api_src = MassExchangeAPI(cell)
            api_src.apply_mass_delta("dustThickness", -flux.dust_flux)
            api_src.apply_mass_delta("snowMass",      -flux.snow_flux)
            api_src.apply_mass_delta("debrisMass",    -flux.debris_flux)
            stats.total_downhill_flux += total_flux

            # Deposit into downslope neighbour (circular wrap for simplicity)
            dst_i = (i + 1) % n
            api_dst = MassExchangeAPI(cells[dst_i])
            api_dst.apply_mass_delta("dustThickness", flux.dust_flux)
            api_dst.apply_mass_delta("snowMass",      flux.snow_flux)
            api_dst.apply_mass_delta("debrisMass",    flux.debris_flux)

        # ------------------------------------------------------------------
        # Conservation check
        # ------------------------------------------------------------------
        after = self._checker.snapshot_total(cells, list(air_dust) + list(air_snow))
        stats.conservation_ok = self._checker.check_conserved(before, after)

        return stats

    # ------------------------------------------------------------------
    # Public — contact displacement
    # ------------------------------------------------------------------

    def apply_contact(
        self,
        api:             MassExchangeAPI,
        contact_impulse: float,
    ) -> ContactResult:
        """Apply contact-driven compaction/displacement to one cell.

        Parameters
        ----------
        api :
            MassExchangeAPI wrapping the target cell.
        contact_impulse :
            Normalised contact force [0..1].

        Returns
        -------
        ContactResult
            Displaced mass amounts (caller distributes to neighbours).
        """
        return self._contact.apply(api, contact_impulse)
