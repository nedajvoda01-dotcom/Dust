"""LavaSurfaceFlow — Stage 67 slope-based lava flow model.

Simple proxy model for lava flowing downhill.  Does not implement
Navier–Stokes; instead uses a scalar flow-rate proxy:

    magmaFlowRate = f(slope, viscosityProxy, thermalState)

Each :class:`LavaCell` tracks surface magma volume and temperature
proxy.  Lava flows from higher to lower cells proportionally to slope.

Public API
----------
LavaCell (dataclass)
  .magma_volume   float  [0, 1]  normalised magma depth
  .temp_proxy     float  [0, 1]  thermal proxy

LavaSurfaceFlow(config=None)
  .spawn_lava(cell_idx, intensity)    → None
  .tick(cells, slope_map, dt)         → None
  .cell(idx)                          → LavaCell
  .cell_count                         → int
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class LavaCell:
    """Per-cell surface lava state."""

    cell_idx:     int   = 0
    magma_volume: float = 0.0   # [0, 1] normalised depth
    temp_proxy:   float = 0.0   # [0, 1] thermal proxy


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LavaSurfaceFlow:
    """Slope-based lava surface flow model.

    Parameters
    ----------
    config :
        Optional dict; reads ``lava67.*`` keys.
    cell_count :
        Number of surface cells in the grid.
    """

    _DEFAULT_VISCOSITY_PROXY  = 0.3    # lower = flows faster
    _DEFAULT_MAX_FLOW_RATE    = 0.05   # fraction per second
    _DEFAULT_SPAWN_TEMP       = 0.9    # initial temperature proxy on spawn

    def __init__(
        self,
        config:     Optional[dict] = None,
        cell_count: int = 64,
    ) -> None:
        cfg = (config or {}).get("lava67", {}) or {}
        self._viscosity    = float(cfg.get("lava_viscosity_proxy",  self._DEFAULT_VISCOSITY_PROXY))
        self._max_flow     = float(cfg.get("lava_max_flow_rate",    self._DEFAULT_MAX_FLOW_RATE))
        self._spawn_temp   = float(cfg.get("lava_spawn_temp_proxy", self._DEFAULT_SPAWN_TEMP))
        self._cell_count   = max(1, cell_count)

        self._cells: List[LavaCell] = [
            LavaCell(cell_idx=i) for i in range(self._cell_count)
        ]

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    @property
    def cell_count(self) -> int:
        return self._cell_count

    def cell(self, idx: int) -> LavaCell:
        return self._cells[idx % self._cell_count]

    def spawn_lava(self, cell_idx: int, intensity: float) -> None:
        """Spawn lava at *cell_idx* with given intensity [0, 1]."""
        c = self.cell(cell_idx)
        c.magma_volume = _clamp(c.magma_volume + intensity)
        c.temp_proxy   = _clamp(max(c.temp_proxy, self._spawn_temp * intensity))

    def tick(
        self,
        slope_map: Sequence[float],
        dt:        float,
        air_temp:  float = 0.0,
    ) -> None:
        """Advance lava flow one time step.

        Parameters
        ----------
        slope_map :
            Per-cell slope magnitude [0, 1].  Length must equal cell_count.
            Lava flows to the adjacent cell with lower notional height
            (approximated as the *next* cell index for a 1-D grid).
        dt :
            Time step in seconds.
        air_temp :
            Ambient air temperature proxy [0, 1]; higher = slower cooling.
        """
        n = self._cell_count
        for i in range(n):
            c = self._cells[i]
            if c.magma_volume <= 0.0:
                continue

            slope = slope_map[i] if i < len(slope_map) else 0.0
            flow_rate = self._flow_rate(slope, c.temp_proxy) * dt
            flow = min(flow_rate, c.magma_volume)

            if flow > 0.0:
                # Flow to the next cell (downhill proxy)
                dst = self._cells[(i + 1) % n]
                c.magma_volume   = _clamp(c.magma_volume - flow)
                dst.magma_volume = _clamp(dst.magma_volume + flow)
                # Temperature partially advects with flow
                if dst.temp_proxy < c.temp_proxy:
                    dst.temp_proxy = _clamp(
                        dst.temp_proxy + (c.temp_proxy - dst.temp_proxy) * 0.3
                    )

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _flow_rate(self, slope: float, temp_proxy: float) -> float:
        """Return magmaFlowRate = f(slope, viscosity, temp)."""
        effective_viscosity = self._viscosity * (1.0 - temp_proxy * 0.5)
        rate = slope * (1.0 - effective_viscosity)
        return _clamp(rate, 0.0, self._max_flow)
