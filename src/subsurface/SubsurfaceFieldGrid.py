"""SubsurfaceFieldGrid — Stage 67 coarse subsurface field grid.

Stores per-tile subsurface proxy fields at low resolution (coarse grid).
All fields are deterministically seeded from ``worldSeed`` and updated at
low frequency (0.01–0.1 Hz).

Fields per coarse tile
----------------------
magmaPressureProxy  : [0, 1]  magma reservoir pressure indicator
thermalGradientProxy: [0, 1]  thermal gradient from below
crustWeakness       : [0, 1]  structural weakness of the crust
subsurfaceStress    : [0, 1]  accumulated mechanical stress
ventPotential       : [0, 1]  combined likelihood of a vent event

Design
------
The grid is a 1-D array of ``SubsurfaceTile`` objects indexed by tile index.
Grid size and tile scale are configurable; the default is suitable for a
planet with a 1 km tile spacing.

Public API
----------
SubsurfaceFieldGrid(config, world_seed, tile_count)
  .tile(idx)              → SubsurfaceTile
  .tile_count             → int
  .update(dt, game_time)  → None   (advances field dynamics)
  .snapshot()             → dict   (serialisable state)
  .restore(snapshot)      → None
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class SubsurfaceTile:
    """Coarse-grid subsurface state for one tile."""

    tile_idx:             int   = 0
    magmaPressureProxy:   float = 0.0
    thermalGradientProxy: float = 0.0
    crustWeakness:        float = 0.0
    subsurfaceStress:     float = 0.0
    ventPotential:        float = 0.0

    def compute_vent_potential(self) -> float:
        """Derived vent potential from the other fields (clamped [0, 1])."""
        raw = (
            self.magmaPressureProxy * 0.4
            + self.crustWeakness    * 0.35
            + self.subsurfaceStress * 0.25
        )
        return _clamp(raw)


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------

class SubsurfaceFieldGrid:
    """Coarse subsurface field grid for Stage 67.

    Parameters
    ----------
    config :
        Optional dict; reads ``subsurface67.*`` keys.
    world_seed :
        Deterministic seed for initial field generation.
    tile_count :
        Number of coarse tiles on the grid (default 64).
    """

    _DEFAULT_TILE_COUNT         = 64
    _DEFAULT_TICK_HZ            = 0.05    # 1 update per 20 s
    _DEFAULT_PRESSURE_GAIN      = 0.003   # per second
    _DEFAULT_PRESSURE_DECAY     = 0.001
    _DEFAULT_WEAKNESS_DECAY     = 0.002
    _DEFAULT_STRESS_GAIN        = 0.002
    _DEFAULT_STRESS_DECAY       = 0.001

    def __init__(
        self,
        config:     Optional[dict] = None,
        world_seed: int = 0,
        tile_count: int = 0,
    ) -> None:
        cfg  = (config or {}).get("subsurface67", {}) or {}

        self._tile_count   = int(cfg.get("tile_count",        tile_count or self._DEFAULT_TILE_COUNT))
        self._tick_hz      = float(cfg.get("tick_hz",         self._DEFAULT_TICK_HZ))
        self._pressure_gain  = float(cfg.get("magma_pressure_gain",   self._DEFAULT_PRESSURE_GAIN))
        self._pressure_decay = float(cfg.get("magma_pressure_decay",  self._DEFAULT_PRESSURE_DECAY))
        self._weakness_decay = float(cfg.get("crust_weakness_decay",  self._DEFAULT_WEAKNESS_DECAY))
        self._stress_gain    = float(cfg.get("subsurface_stress_gain",self._DEFAULT_STRESS_GAIN))
        self._stress_decay   = float(cfg.get("subsurface_stress_decay",self._DEFAULT_STRESS_DECAY))

        self._world_seed   = world_seed
        self._tick_accum   = 0.0
        self._tick_period  = 1.0 / max(self._tick_hz, 1e-6)

        self._tiles: List[SubsurfaceTile] = []
        self._initialise_tiles()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    @property
    def tile_count(self) -> int:
        return self._tile_count

    def tile(self, idx: int) -> SubsurfaceTile:
        """Return tile at *idx* (wrapped modulo tile_count)."""
        return self._tiles[idx % self._tile_count]

    def update(self, dt: float, game_time: float = 0.0) -> None:
        """Advance field dynamics; runs at coarse tick rate."""
        self._tick_accum += dt
        if self._tick_accum < self._tick_period:
            return
        step = self._tick_accum
        self._tick_accum = 0.0
        self._field_tick(step, game_time)

    def snapshot(self) -> dict:
        """Return serialisable state dict for save/restore."""
        return {
            "world_seed": self._world_seed,
            "tiles": [
                {
                    "tile_idx":             t.tile_idx,
                    "magmaPressureProxy":   t.magmaPressureProxy,
                    "thermalGradientProxy": t.thermalGradientProxy,
                    "crustWeakness":        t.crustWeakness,
                    "subsurfaceStress":     t.subsurfaceStress,
                    "ventPotential":        t.ventPotential,
                }
                for t in self._tiles
            ],
        }

    def restore(self, snap: dict) -> None:
        """Restore state from a snapshot dict."""
        tiles_data = snap.get("tiles", [])
        for td in tiles_data:
            idx = td.get("tile_idx", 0)
            if 0 <= idx < self._tile_count:
                t = self._tiles[idx]
                t.magmaPressureProxy   = float(td.get("magmaPressureProxy",   0.0))
                t.thermalGradientProxy = float(td.get("thermalGradientProxy", 0.0))
                t.crustWeakness        = float(td.get("crustWeakness",        0.0))
                t.subsurfaceStress     = float(td.get("subsurfaceStress",     0.0))
                t.ventPotential        = float(td.get("ventPotential",        0.0))

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _initialise_tiles(self) -> None:
        rng = random.Random(self._world_seed ^ 0xDEAD_BEEF)
        for i in range(self._tile_count):
            t = SubsurfaceTile(tile_idx=i)
            t.magmaPressureProxy   = rng.random() * 0.5
            t.thermalGradientProxy = rng.random() * 0.6
            t.crustWeakness        = rng.random() * 0.4
            t.subsurfaceStress     = rng.random() * 0.3
            t.ventPotential        = t.compute_vent_potential()
            self._tiles.append(t)

    def _field_tick(self, dt: float, game_time: float) -> None:
        """Advance all tile fields by *dt* seconds."""
        rng = random.Random(int(self._world_seed ^ int(game_time * 1000)) & 0xFFFF_FFFF)
        for t in self._tiles:
            # Magma pressure slowly builds; small random perturbation
            t.magmaPressureProxy = _clamp(
                t.magmaPressureProxy
                + self._pressure_gain * dt
                - self._pressure_decay * dt * t.magmaPressureProxy
                + (rng.random() - 0.5) * 0.001
            )
            # Thermal gradient changes slowly
            t.thermalGradientProxy = _clamp(
                t.thermalGradientProxy
                + (rng.random() - 0.5) * 0.001 * dt
            )
            # Crust weakness decays toward zero unless stressed
            t.crustWeakness = _clamp(
                t.crustWeakness
                - self._weakness_decay * dt
                + t.subsurfaceStress * 0.001 * dt
            )
            # Subsurface stress accumulates, partly released by crustWeakness
            t.subsurfaceStress = _clamp(
                t.subsurfaceStress
                + self._stress_gain * dt
                - self._stress_decay * dt * t.subsurfaceStress
                - t.crustWeakness * 0.002 * dt
            )
            # Vent potential is recomputed
            t.ventPotential = t.compute_vent_potential()
