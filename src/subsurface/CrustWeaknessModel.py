"""CrustWeaknessModel — Stage 67 crust weakness field management.

Maintains crust weakness proxy on a :class:`SubsurfaceFieldGrid` and
handles stress-driven crack events.

Public API
----------
CrustWeaknessModel(config=None)
  .tick(grid, dt, memory_stress=0.0)  → None
  .weakness(grid, tile_idx)           → float  [0, 1]
  .apply_crack(grid, tile_idx, amount) → None
      Increase weakness and reduce crustHardness proxy.
"""
from __future__ import annotations

from typing import Optional

from src.subsurface.SubsurfaceFieldGrid import SubsurfaceFieldGrid


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


class CrustWeaknessModel:
    """Update crust weakness fields in a :class:`SubsurfaceFieldGrid`.

    Parameters
    ----------
    config :
        Optional dict; reads ``subsurface67.*`` keys.
    """

    _DEFAULT_DECAY         = 0.002   # natural healing per second
    _DEFAULT_STRESS_SCALE  = 0.005   # stress-to-weakness coupling

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = (config or {}).get("subsurface67", {}) or {}
        self._decay        = float(cfg.get("crust_weakness_decay",   self._DEFAULT_DECAY))
        self._stress_scale = float(cfg.get("crust_stress_scale",     self._DEFAULT_STRESS_SCALE))

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def tick(
        self,
        grid:          SubsurfaceFieldGrid,
        dt:            float,
        memory_stress: float = 0.0,
    ) -> None:
        """Advance crust weakness on all tiles.

        Parameters
        ----------
        grid :
            The subsurface field grid to update.
        dt :
            Time step in seconds.
        memory_stress :
            Scalar [0, 1] from Stage 51 memory system — increases
            weakness when accumulated stress is high.
        """
        for i in range(grid.tile_count):
            t = grid.tile(i)
            dw = (
                t.subsurfaceStress * self._stress_scale * dt
                + memory_stress   * self._stress_scale * 0.5 * dt
                - self._decay     * t.crustWeakness * dt
            )
            t.crustWeakness = _clamp(t.crustWeakness + dw)

    def weakness(self, grid: SubsurfaceFieldGrid, tile_idx: int) -> float:
        """Return crust weakness for *tile_idx*."""
        return grid.tile(tile_idx).crustWeakness

    def apply_crack(
        self,
        grid:     SubsurfaceFieldGrid,
        tile_idx: int,
        amount:   float,
    ) -> None:
        """Increase crust weakness by *amount* at *tile_idx* (crack event)."""
        t = grid.tile(tile_idx)
        t.crustWeakness    = _clamp(t.crustWeakness + amount)
        t.ventPotential    = t.compute_vent_potential()
