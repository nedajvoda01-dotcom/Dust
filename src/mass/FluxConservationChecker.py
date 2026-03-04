"""FluxConservationChecker — Stage 66 budget/conservation validator.

Verifies that mass is not created or destroyed during a tick by comparing
total mass before and after all exchange steps.

Usage
-----
checker = FluxConservationChecker()
before = checker.snapshot_total(surface_cells, air_densities)
# ... run exchange steps ...
after = checker.snapshot_total(surface_cells, air_densities)
checker.assert_conserved(before, after)  # raises if delta > tolerance

Public API
----------
FluxConservationChecker(tolerance=1e-3)
  .snapshot_total(surface_cells, air_densities) → float
  .check_conserved(before, after) → bool
  .assert_conserved(before, after) → None   (raises AssertionError)
"""
from __future__ import annotations

from typing import Iterable, Optional

from src.material.PlanetChunkState import PlanetChunkState


class FluxConservationChecker:
    """Validate mass conservation across surface and air reservoirs.

    Parameters
    ----------
    tolerance :
        Maximum allowed absolute deviation in total mass between two
        snapshots.  Default is 5e-3, which is larger than a single uint8
        quantisation step (1/255 ≈ 0.004) to account for rounding across
        a small number of cells.
    """

    def __init__(self, tolerance: float = 5e-3) -> None:
        self._tol = tolerance

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def snapshot_total(
        self,
        surface_cells:  Iterable[PlanetChunkState],
        air_densities:  Iterable[float] = (),
    ) -> float:
        """Compute combined surface + air mass.

        Parameters
        ----------
        surface_cells :
            Iterable of :class:`~src.material.PlanetChunkState.PlanetChunkState`
            objects.
        air_densities :
            Iterable of scalar density values from volumetric grids.

        Returns
        -------
        float
            Sum of all mass-carrying surface fields plus all air densities.
        """
        total = sum(c.total_mass() for c in surface_cells)
        total += sum(float(d) for d in air_densities)
        return total

    def check_conserved(self, before: float, after: float) -> bool:
        """Return True if mass difference is within tolerance."""
        return abs(after - before) <= self._tol

    def assert_conserved(self, before: float, after: float) -> None:
        """Raise AssertionError if mass is not conserved within tolerance.

        Parameters
        ----------
        before, after :
            Total mass snapshots from :meth:`snapshot_total`.
        """
        delta = abs(after - before)
        if delta > self._tol:
            raise AssertionError(
                f"Mass conservation violated: before={before:.6f}, "
                f"after={after:.6f}, delta={delta:.6f} > tol={self._tol:.6f}"
            )
