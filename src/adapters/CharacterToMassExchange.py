"""CharacterToMassExchange — Stage 66 adapter: character (69/74) → mass exchange.

Routes character-generated mass events through the Stage 66 API:

* Footsteps → ``apply_contact`` (compaction + displacement)
* Dust raised at each step → ``dustThickness`` → ``DustVolume`` (lift)
* Body deposition (74) → increments local surface field when character
  settles or shakes accumulated material off

All writes go through :class:`~src.material.MassExchangeAPI.MassExchangeAPI`
(spec §9).

Public API
----------
CharacterToMassExchange(config=None)
  .on_footstep(api, contact_impulse) → ContactResult
  .on_dust_raise(api, air_dust_list, cell_idx, wind_speed) → float
  .on_body_deposition_release(api, deposited_dust, deposited_snow) → None
"""
from __future__ import annotations

from typing import List, Optional

from src.material.MassExchangeAPI import MassExchangeAPI
from src.mass.ContactDisplacementModel import ContactDisplacementModel, ContactResult
from src.mass.LiftModel import LiftModel


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


class CharacterToMassExchange:
    """Route character events into the mass exchange framework.

    Parameters
    ----------
    config :
        Optional dict; reads ``mass.*`` keys for underlying models.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        self._contact = ContactDisplacementModel(config)
        self._lift    = LiftModel(config)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def on_footstep(
        self,
        api:             MassExchangeAPI,
        contact_impulse: float,
    ) -> ContactResult:
        """Apply footstep compaction and displacement.

        Parameters
        ----------
        api :
            MassExchangeAPI wrapping the cell under the foot.
        contact_impulse :
            Normalised foot contact force [0..1].

        Returns
        -------
        ContactResult
            Displaced mass (caller distributes to neighbouring cells).
        """
        return self._contact.apply(api, contact_impulse)

    def on_dust_raise(
        self,
        api:           MassExchangeAPI,
        air_dust_list: List[float],
        cell_idx:      int,
        wind_speed:    float = 0.0,
    ) -> float:
        """Lift dust kicked up by character movement.

        Uses LiftModel with the character's motion as a wind proxy.

        Parameters
        ----------
        api :
            MassExchangeAPI for the cell being stepped on.
        air_dust_list :
            Mutable list of per-cell air dust densities.  Modified in-place.
        cell_idx :
            Index into *air_dust_list* corresponding to this cell.
        wind_speed :
            Ambient wind speed [0..1]; combined with movement to estimate lift.

        Returns
        -------
        float
            Amount of dust lifted into the air.
        """
        rates = self._lift.compute_lift_rate(api._chunk, wind_speed)
        dust_lift = rates.dust_lift
        api.apply_mass_delta("dustThickness", -dust_lift)
        air_dust_list[cell_idx] = _clamp(air_dust_list[cell_idx] + dust_lift)
        return dust_lift

    def on_body_deposition_release(
        self,
        api:             MassExchangeAPI,
        deposited_dust:  float,
        deposited_snow:  float,
    ) -> None:
        """Return accumulated body deposition back to the surface.

        Called when the character shakes off or walks off accumulated
        material from its body (Stage 74).

        Parameters
        ----------
        api :
            MassExchangeAPI for the cell where the character is standing.
        deposited_dust :
            Amount of dust accumulated on the character body [0..1].
        deposited_snow :
            Amount of snow accumulated on the character body [0..1].
        """
        api.apply_mass_delta("dustThickness", _clamp(deposited_dust))
        api.apply_mass_delta("snowMass",      _clamp(deposited_snow))
