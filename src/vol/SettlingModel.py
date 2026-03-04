"""SettlingModel — Stage 65 gravitational settling of dust.

Implements spec §3.1 and §4.2: dust density decreases in each voxel and
the settled mass is returned as a *surface deposition* value that the
authoritative server can apply to terrain via MassExchangeAPI (spec §5.1).

Key invariant (spec §5.1)
--------------------------
The settling model does **not** modify terrain state directly — it only
returns how much mass settled in this tick.  Authoritative terrain writes
are handled server-side via MassExchangeAPI.

Public API
----------
SettlingModel(config=None)
  .step(grid, dt)  → float
      In-place settling of *grid*.
      Returns total settled mass (sum removed from all voxels) so that
      authoritative code can apply it to terrain via MassExchangeAPI.
      dt : time step in seconds.
"""
from __future__ import annotations

from typing import Optional

from src.vol.DensityGrid import DensityGrid


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


class SettlingModel:
    """Gravitational settling for dust volumetric grids.

    Parameters
    ----------
    config :
        Optional dict; reads ``vol.settling_k`` (settling rate per second).
    """

    _DEFAULT_SETTLING_K = 0.05   # fraction of density settled per second

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg  = config or {}
        vcfg = cfg.get("vol", {}) or {}
        self._k: float = float(
            vcfg.get("settling_k", self._DEFAULT_SETTLING_K)
        )

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def step(self, grid: DensityGrid, dt: float) -> float:
        """Apply settling in-place.

        Parameters
        ----------
        grid :
            The :class:`~src.vol.DensityGrid.DensityGrid` to settle.
            Only the lowest layer (iz=0) settles to the ground.
        dt :
            Time step in seconds.

        Returns
        -------
        float
            Total density removed from the grid (settled to surface).
        """
        w, h = grid.width, grid.height
        settled = 0.0
        rate = self._k * dt

        for iz in range(grid.depth):
            # Settling rate is stronger at lower altitudes
            layer_factor = 1.0 - iz / max(grid.depth - 1, 1)
            effective_rate = rate * layer_factor

            for iy in range(h):
                for ix in range(w):
                    current = grid.density(ix, iy, iz)
                    delta   = current * effective_rate
                    grid.add_density(ix, iy, iz, -delta)
                    settled += delta

                    # Transfer downward: density falls to layer below
                    if iz > 0:
                        grid.add_density(ix, iy, iz - 1, delta * 0.5)
                        settled -= delta * 0.5   # net ground deposition reduced

        return max(0.0, settled)
