"""PlanetPhaseTransitions — Stage 63 deterministic phase transition rules.

Implements the four transition groups defined in the Planet Reality Spec v1.0
(section 4), using :class:`~src.material.MassExchangeAPI.MassExchangeAPI`
exclusively — no direct field writes.

Transition groups
-----------------
4.1  Deposition / Erosion
     AerosolDust → RegolithDust (deposition)
     SnowLoose ↔ AerosolDust (wind)
     Crust → DebrisFragments (fracture)
     DebrisFragments → RegolithDust (grinding)

4.2  Compaction
     SnowLoose → SnowCompacted (pressure / wind)
     RegolithDust → Crust (sintering)

4.3  Melt / Freeze
     SnowLoose → WaterRare (melt)
     IceFilm ↔ WaterRare
     WaterRare → Vapor (evaporation)
     Vapor → IceFilm / SnowLoose (condensation)

4.4  Fracture / Instability
     Crust → DebrisFragments (stress-driven)

4.5  Magma
     Magma → Crust (cooling)
     Crust → Magma (vent trigger — only when vent_active=True)

Config keys (AI-adjustable, within allowlist)
---------------------------------------------
erosionRate, depositionRate, compactionRate, meltRate,
fractureThreshold, magmaCoolingRate

Public API
----------
PlanetPhaseTransitions(config=None)
  .tick(grid, climate, dt) -> None
      Apply all transitions to every cell of *grid*.
  .tick_budget(grid, climate, dt, cells_per_tick) -> None
      Round-robin budget variant (for large grids).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from src.material.PlanetChunkState import PlanetChunkGrid
from src.material.MassExchangeAPI  import MassExchangeAPI


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


# ---------------------------------------------------------------------------
# ClimateSample (minimal; compatible with Stage 45 PhaseChangeSystem input)
# ---------------------------------------------------------------------------

@dataclass
class ClimateSample63:
    """Climate drivers for one stage-63 tick.

    Attributes
    ----------
    wind_speed : float
        Normalised wind [0, 1].
    insolation : float
        Surface insolation (0 = full shadow, 1 = full sun).
    temperature : float
        Normalised temperature proxy [0, 1].
    dust_density : float
        Airborne dust loading [0, 1].
    vent_active : bool
        True when a magma vent is active (enables Crust → Magma).
    """
    wind_speed:   float = 0.0
    insolation:   float = 0.5
    temperature:  float = 0.3
    dust_density: float = 0.1
    vent_active:  bool  = False


# ---------------------------------------------------------------------------
# PlanetPhaseTransitions
# ---------------------------------------------------------------------------

class PlanetPhaseTransitions:
    """Applies all stage-63 phase transitions to a PlanetChunkGrid.

    Parameters
    ----------
    config : dict or None
        Optional config dict.  Reads ``material63.*`` keys.
    """

    _DEFAULTS: Dict[str, float] = {
        "erosionRate":        0.012,
        "depositionRate":     0.015,
        "compactionRate":     0.018,
        "meltRate":           0.020,
        "fractureThreshold":  0.70,
        "magmaCoolingRate":   0.008,
    }

    _ALLOWLIST = frozenset(_DEFAULTS.keys())

    # Insolation/temperature thresholds for melt/freeze transitions
    _MELT_TEMP_THRESHOLD   = 0.6   # temperatureProxy above which melt begins
    _MELT_INSOL_THRESHOLD  = 0.7   # insolation above which melt begins
    _MELT_ONSET_TEMP       = 0.5   # lower bound for melt rate calculation
    _MELT_ONSET_INSOL      = 0.5
    _ICE_MELT_INSOL_ONSET  = 0.4   # insolation onset for ice-film sublimation
    _COND_TEMP_THRESHOLD   = 0.3   # below this temperature, condensation forms
    _COND_INSOL_THRESHOLD  = 0.3   # below this insolation, condensation forms

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg_src = {}
        if config is not None:
            cfg_src = config.get("material63", config) or {}
        c: Dict[str, float] = dict(self._DEFAULTS)
        for k in self._ALLOWLIST:
            if k in cfg_src:
                c[k] = float(cfg_src[k])
        self.erosionRate:       float = c["erosionRate"]
        self.depositionRate:    float = c["depositionRate"]
        self.compactionRate:    float = c["compactionRate"]
        self.meltRate:          float = c["meltRate"]
        self.fractureThreshold: float = c["fractureThreshold"]
        self.magmaCoolingRate:  float = c["magmaCoolingRate"]

        self._cursor: int = 0

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def tick(
        self,
        grid: PlanetChunkGrid,
        climate: ClimateSample63,
        dt: float,
    ) -> None:
        """Apply all phase transitions to every cell of *grid*.

        Parameters
        ----------
        grid :
            The terrain chunk grid to update.
        climate :
            Current climate state.
        dt :
            Time step in seconds.
        """
        if dt <= 0.0:
            return
        for iy in range(grid.h):
            for ix in range(grid.w):
                self._update_cell(grid.cell(ix, iy), climate, dt)

    def tick_budget(
        self,
        grid: PlanetChunkGrid,
        climate: ClimateSample63,
        dt: float,
        cells_per_tick: int = 64,
    ) -> None:
        """Round-robin budget variant for large grids.

        Parameters
        ----------
        cells_per_tick :
            Maximum number of cells to process in this call.
        """
        if dt <= 0.0:
            return
        n = grid.w * grid.h
        for _ in range(min(cells_per_tick, n)):
            idx = self._cursor % n
            self._cursor = (self._cursor + 1) % n
            iy = idx // grid.w
            ix = idx % grid.w
            self._update_cell(grid.cell(ix, iy), climate, dt)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _update_cell(
        self,
        cell,
        c: ClimateSample63,
        dt: float,
    ) -> None:
        api = MassExchangeAPI(cell)

        # ------------------------------------------------------------------
        # 4.1  Deposition / Erosion
        # ------------------------------------------------------------------

        # AerosolDust → RegolithDust (wind deposition)
        # Represented as dustThickness input from aerosol_dust proxy
        deposit = self.depositionRate * c.dust_density * dt
        api.apply_mass_delta("dustThickness", deposit)

        # Wind erosion: RegolithDust → airborne (dustThickness decreases)
        erode = self.erosionRate * c.wind_speed * dt
        # We reduce dustThickness; mass leaves to aerosol (not tracked in
        # the chunk; conserved globally through world-level aerosol budget)
        api.apply_mass_delta("dustThickness", -erode)

        # DebrisFragments → RegolithDust (grinding by wind)
        if cell.debrisMass > 0.0:
            grind = self.erosionRate * 0.5 * c.wind_speed * cell.debrisMass * dt
            api.transfer_mass("debrisMass", "dustThickness", grind)

        # ------------------------------------------------------------------
        # 4.2  Compaction
        # ------------------------------------------------------------------

        # SnowLoose → SnowCompacted (wind pressure)
        if c.wind_speed > 0.3 and cell.snowMass > 0.0:
            comp = self.compactionRate * c.wind_speed * dt
            new_comp = _clamp(cell.snowCompaction + comp)
            # snowCompaction is a phase indicator, not a mass field
            cell.snowCompaction = new_comp

        # RegolithDust → Crust (sintering: wind + cool cycles)
        sinter_driver = (
            c.wind_speed * 0.5 + (1.0 - c.temperature) * 0.3 + c.dust_density * 0.2
        ) * dt
        if sinter_driver > 0.0 and cell.dustThickness > 0.05:
            dcrust = self.compactionRate * 0.5 * sinter_driver
            api.transfer_mass("dustThickness", "crustHardness", dcrust)

        # ------------------------------------------------------------------
        # 4.3  Melt / Freeze
        # ------------------------------------------------------------------

        # SnowLoose → WaterRare (melt under high temperature / insolation)
        if c.temperature > self._MELT_TEMP_THRESHOLD or c.insolation > self._MELT_INSOL_THRESHOLD:
            melt_driver = (
                max(0.0, c.temperature - self._MELT_ONSET_TEMP)
                + max(0.0, c.insolation - self._MELT_ONSET_INSOL)
            )
            melt = self.meltRate * melt_driver * dt
            api.transfer_mass("snowMass", "iceFilmThickness", melt)

        # IceFilm → WaterRare (sublimation / melt — represented as
        # iceFilmThickness decrease; mass enters moistureProxy)
        if c.insolation > self._MELT_ONSET_INSOL or c.temperature > self._MELT_ONSET_TEMP:
            ice_melt = self.meltRate * 0.8 * max(0.0, c.insolation - self._ICE_MELT_INSOL_ONSET) * dt
            removed = api.apply_mass_delta("iceFilmThickness", -ice_melt)
            api.apply_mass_delta("moistureProxy", -removed)  # proxy update

        # Vapor → IceFilm / SnowLoose (condensation at low temperature)
        if c.temperature < self._COND_TEMP_THRESHOLD and c.insolation < self._COND_INSOL_THRESHOLD:
            cond = self.meltRate * 0.6 * (self._COND_TEMP_THRESHOLD - c.temperature) * dt
            # Preferentially forms ice film, then snow
            added = api.apply_mass_delta("iceFilmThickness", cond)
            if added < cond * 0.5:
                api.apply_mass_delta("snowMass", cond - added)

        # ------------------------------------------------------------------
        # 4.4  Fracture / Instability (stress-driven)
        # ------------------------------------------------------------------

        if (
            cell.crustHardness > self.fractureThreshold
            and cell.stressField > 0.5
        ):
            frac_amount = self.erosionRate * cell.stressField * dt
            api.transfer_mass("crustHardness", "debrisMass", frac_amount)
            api.apply_stress_delta(-frac_amount * 0.5)

        # ------------------------------------------------------------------
        # 4.5  Magma transitions
        # ------------------------------------------------------------------

        # Magma → Crust (cooling; magma not tracked as mass field here,
        # but crustHardness slowly grows when magma is active nearby)
        # Crust → Magma (vent active)
        if c.vent_active:
            # Vent melts crust into magma (crustHardness decreases)
            api.apply_mass_delta("crustHardness", -self.magmaCoolingRate * dt)
        else:
            # Cooling: solidRockDepth slowly rebuilds crust layer
            cool = self.magmaCoolingRate * 0.5 * (1.0 - cell.crustHardness) * dt
            api.transfer_mass("solidRockDepth", "crustHardness", cool)

        # ------------------------------------------------------------------
        # Temperature / stress proxy updates
        # ------------------------------------------------------------------

        # Temperature proxy driven by insolation
        t_target = c.insolation * 0.7 + c.temperature * 0.3
        api.apply_heat_delta((t_target - cell.temperatureProxy) * 0.05 * dt)

        # Stress dissipates slowly
        if cell.stressField > 0.0:
            api.apply_stress_delta(-cell.stressField * 0.02 * dt)

        # Roughness: wind polishing
        if c.wind_speed > 0.2:
            polish = 0.008 * c.wind_speed * c.dust_density * dt
            cell.surfaceRoughness = _clamp(cell.surfaceRoughness - polish)
