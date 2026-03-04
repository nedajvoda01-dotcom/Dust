"""MassExchangeAPI — Stage 63 mass-conservation interface.

All subsystems must modify material fields exclusively through this API.
No subsystem may write PlanetChunkState fields directly.

Design
------
* ``apply_mass_delta`` clamps the result to [0, 1]; it never creates
  negative fields or overflows.
* ``transfer_mass`` conserves mass: the amount actually removed from
  *from_field* is the amount actually added to *to_field*.  Clamping is
  applied independently, so conservation is exact only within uint8
  quantisation resolution.
* ``apply_heat_delta`` and ``apply_stress_delta`` modify the non-mass
  proxy fields (temperatureProxy, stressField).
* ``total_mass`` reads the six mass-carrying fields and returns their sum.

Public API
----------
MassExchangeAPI(chunk: PlanetChunkState)
  .apply_mass_delta(field: str, delta: float) -> float
      Actually-applied delta (after clamping).
  .transfer_mass(from_field: str, to_field: str, amount: float) -> float
      Actually-transferred amount.
  .apply_heat_delta(amount: float) -> None
  .apply_stress_delta(amount: float) -> None
  .total_mass() -> float

Module-level helpers
--------------------
MASS_FIELDS : frozenset[str]
    Names of the six mass-carrying chunk fields.
ALL_FIELDS : frozenset[str]
    All eleven mutable chunk fields.
"""
from __future__ import annotations

from src.material.PlanetChunkState import PlanetChunkState


# ---------------------------------------------------------------------------
# Field sets
# ---------------------------------------------------------------------------

MASS_FIELDS: frozenset = frozenset([
    "solidRockDepth",
    "crustHardness",
    "dustThickness",
    "snowMass",
    "iceFilmThickness",
    "debrisMass",
])
"""Fields that participate in mass conservation."""

ALL_FIELDS: frozenset = frozenset([
    "solidRockDepth",
    "crustHardness",
    "dustThickness",
    "snowMass",
    "snowCompaction",
    "iceFilmThickness",
    "debrisMass",
    "surfaceRoughness",
    "temperatureProxy",
    "moistureProxy",
    "stressField",
])
"""All eleven mutable fields of PlanetChunkState."""


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


# ---------------------------------------------------------------------------
# MassExchangeAPI
# ---------------------------------------------------------------------------

class MassExchangeAPI:
    """Controlled interface for modifying one :class:`PlanetChunkState`.

    Parameters
    ----------
    chunk :
        The chunk cell to operate on.
    """

    def __init__(self, chunk: PlanetChunkState) -> None:
        self._chunk = chunk

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def apply_mass_delta(self, field: str, delta: float) -> float:
        """Add *delta* to a named field, clamping to [0, 1].

        Parameters
        ----------
        field :
            Name of any field in ``ALL_FIELDS``.
        delta :
            Signed change.  Positive = add material, negative = remove.

        Returns
        -------
        float
            The amount actually applied (may be less than *delta* due to
            clamping).

        Raises
        ------
        ValueError
            If *field* is not in ``ALL_FIELDS``.
        """
        if field not in ALL_FIELDS:
            raise ValueError(f"Unknown field: {field!r}")
        before = getattr(self._chunk, field)
        after  = _clamp(before + delta)
        setattr(self._chunk, field, after)
        return after - before

    def transfer_mass(
        self,
        from_field: str,
        to_field: str,
        amount: float,
    ) -> float:
        """Move mass from one field to another, conserving total mass.

        The actual transferred amount is the minimum of *amount*, what can
        be removed from *from_field* without going negative, and what can
        be added to *to_field* without exceeding 1.0.

        Parameters
        ----------
        from_field, to_field :
            Names of mass-carrying fields (must be in ``MASS_FIELDS``).
        amount :
            Requested transfer amount [0..1].

        Returns
        -------
        float
            The amount actually transferred.

        Raises
        ------
        ValueError
            If either field is not in ``MASS_FIELDS``.
        """
        if from_field not in MASS_FIELDS:
            raise ValueError(f"{from_field!r} is not a mass-carrying field")
        if to_field not in MASS_FIELDS:
            raise ValueError(f"{to_field!r} is not a mass-carrying field")

        from_val = getattr(self._chunk, from_field)
        to_val   = getattr(self._chunk, to_field)

        # Compute maximum transferable without violating bounds
        can_remove = max(0.0, min(amount, from_val))
        can_add    = max(0.0, min(can_remove, 1.0 - to_val))
        actual     = can_add

        if actual > 0.0:
            setattr(self._chunk, from_field, _clamp(from_val - actual))
            setattr(self._chunk, to_field,   _clamp(to_val   + actual))

        return actual

    def apply_heat_delta(self, amount: float) -> None:
        """Adjust *temperatureProxy* by *amount* (clamped to [0, 1])."""
        self.apply_mass_delta("temperatureProxy", amount)

    def apply_stress_delta(self, amount: float) -> None:
        """Adjust *stressField* by *amount* (clamped to [0, 1])."""
        self.apply_mass_delta("stressField", amount)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def total_mass(self) -> float:
        """Return the sum of all mass-carrying fields for this cell."""
        return self._chunk.total_mass()
