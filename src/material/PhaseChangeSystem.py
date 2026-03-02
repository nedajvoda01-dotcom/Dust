"""PhaseChangeSystem — Stage 45 deterministic surface phase evolution.

Advances a :class:`~src.material.SurfaceMaterialState.SurfaceMaterialStateGrid`
on a slow tick driven by climate / astro / contact inputs.

Phase transition rules
----------------------
Dust → Crust
    Strong wind + fine dust + nightly cooling builds ``crustHardness`` and
    reduces ``dustThickness``.

Snow → Compaction
    Contact load + elapsed time increases ``snowCompaction``.

IceFilm formation / melt
    Forms when in ring shadow or at night (low insolation);
    melts under direct sun or during storms.

Roughness polishing
    Wind-driven abrasion decreases ``roughness`` (0 = glassy smooth).

Contact contribution
    Each foot/wheel contact can increase ``snowCompaction`` or crack
    ``crustHardness`` (brittle fracture), generating audio impulses.

Tick budget
-----------
Only ``cells_per_tick`` cells are updated per call to :meth:`tick` (LRU
round-robin).  Background (distant) cells are updated at coarse resolution
via :meth:`tick_coarse`.

Public API
----------
PhaseChangeSystem(config=None)
  .tick(grid, climate_sample, dt) -> List[BrittleEvent]
  .apply_contact(grid, ix, iy, normal_force, area) -> Optional[BrittleEvent]
  .tick_coarse(grids, climate_sample, dt)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.material.SurfaceMaterialState import SurfaceMaterialState, SurfaceMaterialStateGrid


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


# ---------------------------------------------------------------------------
# ClimateSample — lightweight input struct
# ---------------------------------------------------------------------------

@dataclass
class ClimateSample:
    """Snapshot of climate drivers for one tick at one chunk location.

    Attributes
    ----------
    wind_speed : float
        Normalised wind speed [0, 1].
    dust_density : float
        Airborne dust loading [0, 1].
    insolation : float
        Surface insolation (0 = full shadow/night, 1 = full sunlight).
    temperature : float
        Normalised surface temperature proxy [0, 1]; 0 = coldest, 1 = hottest.
    moisture : float
        Rare moisture trace [0, 1].
    storm_active : bool
        True during a mega-storm event.
    """
    wind_speed:   float = 0.0
    dust_density: float = 0.0
    insolation:   float = 0.5
    temperature:  float = 0.5
    moisture:     float = 0.0
    storm_active: bool  = False


# ---------------------------------------------------------------------------
# BrittleEvent — audio notification from crust fracture
# ---------------------------------------------------------------------------

@dataclass
class BrittleEvent:
    """Emitted when crustHardness exceeds brittleThreshold under pressure.

    Parameters
    ----------
    ix, iy : int
        Grid cell that fractured.
    impulse_count : int
        Number of micro-impulses to inject into the audio system.
    hardness_before : float
        crustHardness value before the break.
    """
    ix: int
    iy: int
    impulse_count: int
    hardness_before: float


# ---------------------------------------------------------------------------
# PhaseChangeSystem
# ---------------------------------------------------------------------------

class PhaseChangeSystem:
    """Advances surface material phase fields on a slow tick.

    Parameters
    ----------
    config : dict or None
        Optional flat dict with keys from ``material.*`` config section.
        If None, built-in defaults are used.
    """

    # Default config values (overridable via config dict)
    _DEFAULTS = {
        "tick_hz":               0.5,    # update frequency
        "dust_to_crust_k":       0.015,  # dust sintering rate
        "snow_compaction_k":     0.020,  # compaction rate under wind/load
        "icefilm_form_k":        0.025,  # ice film formation rate
        "icefilm_melt_k":        0.040,  # ice film melt rate
        "roughness_polish_k":    0.010,  # wind polishing rate
        "brittle_threshold":     0.70,   # crustHardness above which breakage occurs
        "crust_break_k":         0.30,   # hardness reduction on break
        "cells_per_tick":        64,     # budget per tick call
    }

    def __init__(self, config=None) -> None:
        cfg = dict(self._DEFAULTS)
        if config is not None:
            for k in self._DEFAULTS:
                if k in config:
                    cfg[k] = float(config[k]) if k != "cells_per_tick" else int(config[k])
        self.tick_hz:             float = cfg["tick_hz"]
        self.dust_to_crust_k:     float = cfg["dust_to_crust_k"]
        self.snow_compaction_k:   float = cfg["snow_compaction_k"]
        self.icefilm_form_k:      float = cfg["icefilm_form_k"]
        self.icefilm_melt_k:      float = cfg["icefilm_melt_k"]
        self.roughness_polish_k:  float = cfg["roughness_polish_k"]
        self.brittle_threshold:   float = cfg["brittle_threshold"]
        self.crust_break_k:       float = cfg["crust_break_k"]
        self.cells_per_tick:      int   = cfg["cells_per_tick"]

        # LRU round-robin cursor per grid (keyed by id(grid))
        self._cursors: Dict[int, int] = {}

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def tick(
        self,
        grid: SurfaceMaterialStateGrid,
        climate: ClimateSample,
        dt: float,
    ) -> List[BrittleEvent]:
        """Advance phase fields for a budget slice of cells.

        Parameters
        ----------
        grid :
            The chunk grid to update.
        climate :
            Climate drivers for this tick.
        dt :
            Time step [s] (should match 1/tick_hz approximately).

        Returns
        -------
        list of BrittleEvent
            Any crust fracture events that occurred this tick.
        """
        if dt <= 0.0:
            return []

        n = grid.w * grid.h
        cursor = self._cursors.get(id(grid), 0)
        events: List[BrittleEvent] = []

        for _ in range(min(self.cells_per_tick, n)):
            idx = cursor % n
            cursor += 1
            iy = idx // grid.w
            ix = idx % grid.w
            cell = grid.cell(ix, iy)
            evt = self._update_cell(cell, ix, iy, climate, dt)
            if evt is not None:
                events.append(evt)

        self._cursors[id(grid)] = cursor % n
        return events

    def apply_contact(
        self,
        grid: SurfaceMaterialStateGrid,
        ix: int,
        iy: int,
        normal_force: float,
        area: float,
        dt: float = 0.016,
    ) -> Optional[BrittleEvent]:
        """Apply a contact event (step/wheel) to one cell.

        Increases ``snowCompaction`` proportional to pressure and may
        trigger crust fracture if ``crustHardness`` exceeds threshold.

        Parameters
        ----------
        normal_force : float
            Contact normal force [N].
        area : float
            Contact area [m²].
        dt : float
            Contact duration [s].

        Returns
        -------
        BrittleEvent or None
            If the crust fractures, returns a BrittleEvent.
        """
        cell = grid.cell(
            max(0, min(ix, grid.w - 1)),
            max(0, min(iy, grid.h - 1)),
        )
        pressure = normal_force / max(area, 1e-6)
        # Normalised pressure (threshold ~5000 Pa = full)
        p_norm = _clamp(pressure / 5000.0)

        # Snow compaction increases with pressure
        snow_delta = self.snow_compaction_k * p_norm * dt
        cell.snow_compaction = _clamp(cell.snow_compaction + snow_delta)

        # Dust pushed down → slight crust contribution
        if cell.dust_thickness > 0.1:
            cell.dust_thickness = _clamp(cell.dust_thickness - snow_delta * 0.3)

        # Brittle crust fracture check
        if cell.crust_hardness > self.brittle_threshold and p_norm > 0.3:
            hb = cell.crust_hardness
            cell.crust_hardness = _clamp(
                cell.crust_hardness - self.crust_break_k * p_norm
            )
            impulses = max(1, int(hb * 5))
            return BrittleEvent(ix=ix, iy=iy,
                                impulse_count=impulses,
                                hardness_before=hb)
        return None

    def tick_coarse(
        self,
        grids: List[SurfaceMaterialStateGrid],
        climate: ClimateSample,
        dt: float,
    ) -> None:
        """Apply a single-cell averaged update to each distant grid.

        Used for background chunks that are not in the player's interest
        zone.  Only the (0,0) cell of each grid is sampled.
        """
        for grid in grids:
            cell = grid.cell(0, 0)
            self._update_cell(cell, 0, 0, climate, dt)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _update_cell(
        self,
        cell: SurfaceMaterialState,
        ix: int,
        iy: int,
        c: ClimateSample,
        dt: float,
    ) -> Optional[BrittleEvent]:
        """Apply all phase transition rules to one cell."""

        # ----- IceFilm -----
        # Forms in shadow/night (low insolation), dissolves in sunlight/storms
        if c.insolation < 0.3:
            form = self.icefilm_form_k * (0.3 - c.insolation) / 0.3 * dt
            cell.ice_film = _clamp(cell.ice_film + form)
        if c.insolation > 0.5:
            melt = self.icefilm_melt_k * (c.insolation - 0.5) / 0.5 * dt
            cell.ice_film = _clamp(cell.ice_film - melt)
        if c.storm_active:
            cell.ice_film = _clamp(cell.ice_film - self.icefilm_melt_k * 2.0 * dt)

        # ----- Roughness polishing by wind -----
        if c.wind_speed > 0.2:
            polish = self.roughness_polish_k * c.wind_speed * c.dust_density * dt
            cell.roughness = _clamp(cell.roughness - polish)

        # ----- Dust → Crust sintering -----
        # Requires wind + fine dust + low-to-mid temperature (cooling cycles)
        # or moisture/evaporation trace
        crust_driver = (
            c.wind_speed * 0.5
            + c.dust_density * 0.3
            + (1.0 - c.temperature) * 0.2
            + c.moisture * 0.5
        ) * dt
        if crust_driver > 0.0 and cell.dust_thickness > 0.05:
            dcrust = self.dust_to_crust_k * crust_driver
            cell.crust_hardness = _clamp(cell.crust_hardness + dcrust)
            cell.dust_thickness = _clamp(cell.dust_thickness - dcrust * 0.5)

        # ----- Snow compaction by wind -----
        if c.wind_speed > 0.3 and cell.snow_compaction < 1.0:
            dsnow = self.snow_compaction_k * c.wind_speed * dt
            cell.snow_compaction = _clamp(cell.snow_compaction + dsnow)

        # ----- Storm: erodes crust, blows dust -----
        if c.storm_active:
            cell.crust_hardness = _clamp(
                cell.crust_hardness - self.dust_to_crust_k * 3.0 * dt
            )
            if cell.dust_thickness < 0.8:
                cell.dust_thickness = _clamp(
                    cell.dust_thickness + c.dust_density * 0.05 * dt
                )

        # ----- Brittle fracture check (climate-driven stress) -----
        if cell.crust_hardness > self.brittle_threshold and c.storm_active:
            hb = cell.crust_hardness
            cell.crust_hardness = _clamp(
                cell.crust_hardness - self.crust_break_k * dt
            )
            impulses = max(1, int(hb * 3))
            return BrittleEvent(ix=ix, iy=iy,
                                impulse_count=impulses,
                                hardness_before=hb)

        return None
