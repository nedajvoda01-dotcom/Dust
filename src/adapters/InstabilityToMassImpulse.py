"""InstabilityToMassImpulse — Stage 66 adapter: instability events → mass impulses.

Translates instability events (Stage 52) into mass-exchange impulses:

* ``CrustFailureEvent``   → debrisMass ↑, crustHardness ↓ (fracture)
* ``DustAvalancheEvent``  → dustThickness redistributed, local lift ↑
* Generic shock events    → stressField spike → downhill flux boost via stress_map

All writes happen through :class:`~src.material.MassExchangeAPI.MassExchangeAPI`;
this adapter never writes PlanetChunkState directly.

Public API
----------
InstabilityToMassImpulse(config=None)
  .on_crust_failure(api, event)     → float  (debris produced)
  .on_dust_avalanche(api, event)    → float  (dust redistributed)
  .on_vibration(api, intensity)     → float  (stress boost → use as stress input)
"""
from __future__ import annotations

from typing import Optional

from src.material.MassExchangeAPI import MassExchangeAPI


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


class InstabilityToMassImpulse:
    """Map instability events to mass-field impulses.

    Parameters
    ----------
    config :
        Optional dict; reads ``mass.contact_displacement_k`` and
        ``mass.max_flux_per_tick``.
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

    def on_crust_failure(self, api: MassExchangeAPI, event: object) -> float:
        """Handle a CrustFailureEvent: convert fractured crust to debris.

        Parameters
        ----------
        api :
            MassExchangeAPI for the affected cell.
        event :
            A ``CrustFailureEvent`` (duck-typed; needs ``.intensity`` and
            ``.crust_delta`` attributes).

        Returns
        -------
        float
            Amount of debris mass produced.
        """
        intensity  = _clamp(getattr(event, "intensity",   0.0))
        crust_delta = _clamp(getattr(event, "crust_delta", 0.0))

        debris_gain = _clamp(min(crust_delta * intensity * self._k, self._max_flux))
        api.apply_mass_delta("crustHardness", -debris_gain)
        api.apply_mass_delta("debrisMass",    +debris_gain)
        return debris_gain

    def on_dust_avalanche(self, api: MassExchangeAPI, event: object) -> float:
        """Handle a DustAvalancheEvent: boost local lift (dust already moved by Stage 52).

        Parameters
        ----------
        api :
            MassExchangeAPI for the source cell.
        event :
            A ``DustAvalancheEvent`` (duck-typed; needs ``.dust_delta``).

        Returns
        -------
        float
            Additional dust mass removed from the cell into the air.
        """
        dust_delta = _clamp(getattr(event, "dust_delta", 0.0))
        lift = _clamp(min(dust_delta * 0.3, self._max_flux, api._chunk.dustThickness))
        api.apply_mass_delta("dustThickness", -lift)
        return lift

    def on_vibration(self, api: MassExchangeAPI, intensity: float) -> float:
        """Handle a vibration event: return stress boost for downhill flux.

        The stress boost is clamped and does not modify the cell directly —
        callers pass it as ``stress_map`` input to
        :class:`~src.mass.DownhillFluxModel.DownhillFluxModel`.

        Parameters
        ----------
        api :
            MassExchangeAPI for the affected cell (currently read-only).
        intensity :
            Vibration intensity [0..1].

        Returns
        -------
        float
            Stress boost value [0..1].
        """
        return _clamp(intensity)
