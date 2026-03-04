"""CoolingModel — Stage 67 lava → crust cooling model.

Implements the ``Magma → Crust`` transition (§3.3):

    coolingRate depends on:
      * air temperature proxy
      * wind proxy
      * contact with cold surface

When magma_volume drops to zero (or temp_proxy falls below threshold)
the cell is considered solidified (CrustState.CRUST).

Public API
----------
CellCrustState (enum)
  MAGMA — still molten
  CRUST — solidified

CoolingModel(config=None)
  .tick(cells, air_temp, wind_speed, dt) → None
      Cool all LavaCell objects in place.
  .crust_state(cell) → CellCrustState
"""
from __future__ import annotations

from enum import Enum, auto
from typing import Optional, Sequence

from src.lava.LavaSurfaceFlow import LavaCell


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


class CellCrustState(Enum):
    MAGMA = auto()
    CRUST = auto()


class CoolingModel:
    """Cool lava cells toward crust state.

    Parameters
    ----------
    config :
        Optional dict; reads ``lava67.*`` keys.
    """

    _DEFAULT_COOLING_RATE    = 0.02    # base temp drop per second
    _DEFAULT_WIND_SCALE      = 0.01    # extra cooling per unit wind
    _DEFAULT_CRUST_THRESHOLD = 0.15    # temp_proxy below which → CRUST

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = (config or {}).get("lava67", {}) or {}
        self._base_rate  = float(cfg.get("lava_cooling_rate",     self._DEFAULT_COOLING_RATE))
        self._wind_scale = float(cfg.get("cooling_wind_scale",    self._DEFAULT_WIND_SCALE))
        self._threshold  = float(cfg.get("crust_temp_threshold",  self._DEFAULT_CRUST_THRESHOLD))

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def tick(
        self,
        cells:     Sequence[LavaCell],
        air_temp:  float = 0.0,
        wind_speed: float = 0.0,
        dt:        float = 1.0,
    ) -> None:
        """Cool all cells in place.

        Parameters
        ----------
        cells :
            Iterable of LavaCell objects to cool.
        air_temp :
            Ambient air temperature proxy [0, 1]; higher slows cooling.
        wind_speed :
            Wind speed proxy [0, 1]; higher accelerates cooling.
        dt :
            Time step in seconds.
        """
        cooling = (
            self._base_rate * (1.0 - air_temp * 0.5)
            + self._wind_scale * wind_speed
        ) * dt

        for cell in cells:
            if cell.temp_proxy <= 0.0:
                continue
            cell.temp_proxy = _clamp(cell.temp_proxy - cooling)
            # When solidified, magma_volume stays but is inert (crust formed)

    def crust_state(self, cell: LavaCell) -> CellCrustState:
        """Return whether the cell is still molten or has solidified."""
        if cell.temp_proxy > self._threshold and cell.magma_volume > 0.0:
            return CellCrustState.MAGMA
        return CellCrustState.CRUST
