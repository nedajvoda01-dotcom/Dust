"""ContactDisplacementModel — Stage 66 player/object contact displacement.

When a character steps, falls, or slides on a surface, it creates:
  * local compaction (snowCompaction / crustHardness change)
  * displacement of loose material sideways (dust/snow moved out of the footprint)
  * modification of surfaceRoughness (tracks)

All writes happen through :class:`~src.material.MassExchangeAPI.MassExchangeAPI`.

Contact model (spec §2.4, §5)
------------------------------
compaction_delta = contact_displacement_k × impulse × (1 − current_compaction)
displacement     = contact_displacement_k × impulse × available_loose_mass

Displaced mass is returned to the caller as a ``displaced_mass`` value so
that the caller can distribute it to neighbouring cells (conserving mass).

Config keys (``mass.*``)
-------------------------
contact_displacement_k : float (default 0.20)
max_flux_per_tick      : float (default 0.05)

Public API
----------
ContactDisplacementModel(config=None)
  .compute_contact_displacement(surface, contact_impulse) → ContactResult
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.material.MassExchangeAPI import MassExchangeAPI
from src.material.PlanetChunkState import PlanetChunkState


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


@dataclass
class ContactResult:
    """Result of a contact displacement step.

    Attributes
    ----------
    compaction_applied :
        Amount by which snowCompaction was increased.
    dust_displaced :
        Dust mass removed from this cell (to be distributed to neighbours).
    snow_displaced :
        Snow mass removed from this cell (to be distributed to neighbours).
    roughness_delta :
        Change applied to surfaceRoughness (positive = smoother depression).
    """
    compaction_applied: float = 0.0
    dust_displaced:     float = 0.0
    snow_displaced:     float = 0.0
    roughness_delta:    float = 0.0


class ContactDisplacementModel:
    """Apply contact-driven compaction and displacement to a surface cell.

    Parameters
    ----------
    config :
        Optional dict; reads ``mass.*`` keys (see module docstring).
    """

    _DEFAULT_CONTACT_K         = 0.20
    _DEFAULT_MAX_FLUX_PER_TICK = 0.05

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = (config or {}).get("mass", {}) or {}
        self._k:        float = float(cfg.get("contact_displacement_k", self._DEFAULT_CONTACT_K))
        self._max_flux: float = float(cfg.get("max_flux_per_tick",      self._DEFAULT_MAX_FLUX_PER_TICK))

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def compute_contact_displacement(
        self,
        surface:          PlanetChunkState,
        contact_impulse:  float,
    ) -> ContactResult:
        """Compute compaction and displacement from a single contact event.

        Parameters
        ----------
        surface :
            Current surface state (read-only; writes go through the API).
        contact_impulse :
            Normalised contact force [0..1]; 0 = gentle touch, 1 = hard impact.

        Returns
        -------
        ContactResult
            Callers must apply these via MassExchangeAPI to modify the cell
            and distribute displaced mass to neighbours.
        """
        imp = _clamp(contact_impulse)

        # Compaction increases in soft material
        comp_headroom = _clamp(1.0 - surface.snowCompaction)
        comp_delta = _clamp(
            min(self._k * imp * comp_headroom, self._max_flux),
            0.0, 1.0,
        )

        # Displacement of loose dust
        dust_disp = _clamp(
            min(self._k * imp * surface.dustThickness, self._max_flux, surface.dustThickness),
            0.0, 1.0,
        )

        # Displacement of loose snow (less when compacted)
        snow_looseness = _clamp(1.0 - surface.snowCompaction)
        snow_disp = _clamp(
            min(self._k * imp * surface.snowMass * snow_looseness, self._max_flux, surface.snowMass),
            0.0, 1.0,
        )

        # Roughness: contact creates a depression (roughness can increase slightly)
        rough_delta = _clamp(imp * self._k * 0.3, 0.0, self._max_flux)

        return ContactResult(
            compaction_applied=comp_delta,
            dust_displaced=dust_disp,
            snow_displaced=snow_disp,
            roughness_delta=rough_delta,
        )

    def apply(
        self,
        api:             MassExchangeAPI,
        contact_impulse: float,
    ) -> ContactResult:
        """Compute and immediately apply displacement to the cell via *api*.

        Parameters
        ----------
        api :
            :class:`~src.material.MassExchangeAPI.MassExchangeAPI` wrapping
            the target cell.
        contact_impulse :
            Normalised contact force [0..1].

        Returns
        -------
        ContactResult
            The displacement result (displaced mass removed from the cell;
            caller should distribute to neighbours).
        """
        result = self.compute_contact_displacement(api._chunk, contact_impulse)

        api.apply_mass_delta("snowCompaction",   result.compaction_applied)
        api.apply_mass_delta("dustThickness",   -result.dust_displaced)
        api.apply_mass_delta("snowMass",        -result.snow_displaced)
        api.apply_mass_delta("surfaceRoughness", result.roughness_delta)

        return result
