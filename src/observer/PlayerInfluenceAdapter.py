"""PlayerInfluenceAdapter — Stage 53 main observer orchestrator.

Aggregates all player physical contributions and routes them to the
appropriate world systems.  Must be called from the server tick (or
authoritative simulation tick) because all field mutations are server-side.

Responsibilities
----------------
1. Contact stress → :class:`ContactStressInjector`
2. Thermal footprint → :class:`ThermalFootprintInjector`
3. Impulse shear → :class:`ImpulseToShearInjector`
4. Wind wake bias → applied directly to :class:`MicroclimateState`
5. Tick shared :class:`InfluenceLimiter`

Wind wake
---------
The player's body partially shelters the immediate downwind position and
creates a small turbulent wake behind it.  This is expressed as a tiny
``windShelter`` bias on the chunk the player occupies::

    windShelter += wind_wake_factor * (1 − existing_shelter)

The bias disappears instantly when the player moves to a different chunk
(no history is stored), reflecting the ephemeral nature of body-scale
aerodynamics.

Network / determinism
---------------------
All inputs (contact_force, body_heat, impulse, wind_wake_factor) must be
supplied by the server from authoritative physics data.  Clients send
aggregated contact impulses (as in Stage 48) and do not compute field
mutations directly.

Config keys (under ``observer.*``)
-----------------------------------
enable                 : bool  — master switch  (default True)
k_player_stress        : float — contact stress gain (default 0.02)
k_thermal_melt         : float — thermal melt rate  (default 0.005)
k_impulse_shear        : float — impulse shear gain  (default 0.015)
wind_wake_factor       : float — shelter bias per tick (default 0.02)
max_influence_per_tile : float — hard cap per tile  (default 0.05)
influence_radius       : float — radius in world units (default 2.0)
decay_tau              : float — limiter decay time (default 30.0)

Public API
----------
PlayerInfluenceAdapter(config=None)
  .apply_contact(memory_state, tile_idx, contact_force, dt) -> None
  .apply_thermal(material_state, tile_idx, body_heat, dt) -> None
  .apply_impulse(instability_state, tile_idx, impulse_magnitude) -> None
  .apply_wind_wake(microclimate_state) -> None
  .tick(dt) -> None   — advance limiter decay
"""
from __future__ import annotations

from typing import Optional

from src.memory.WorldMemoryState       import WorldMemoryState
from src.material.SurfaceMaterialState import SurfaceMaterialState
from src.instability.InstabilityState  import InstabilityState
from src.microclimate.MicroclimateState import MicroclimateState
from src.observer.InfluenceLimiter     import InfluenceLimiter
from src.observer.ContactStressInjector  import ContactStressInjector
from src.observer.ThermalFootprintInjector import ThermalFootprintInjector
from src.observer.ImpulseToShearInjector   import ImpulseToShearInjector


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


class PlayerInfluenceAdapter:
    """Orchestrates all Stage 53 player physical contributions.

    Parameters
    ----------
    config :
        Optional dict; reads ``observer.*`` keys.
    """

    _DEFAULTS = {
        "enable":                 True,
        "wind_wake_factor":       0.02,
        "influence_radius":       2.0,
    }

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = dict(self._DEFAULTS)
        if isinstance(config, dict):
            src = config.get("observer", config)
            for k in self._DEFAULTS:
                if k in src:
                    cfg[k] = src[k]

        self._enabled:           bool  = bool(cfg["enable"])
        self._wind_wake_factor:  float = float(cfg["wind_wake_factor"])
        self.influence_radius:   float = float(cfg["influence_radius"])

        # Shared limiter — all injectors share the same per-tile budget
        self._limiter = InfluenceLimiter(config)

        self._contact_inj = ContactStressInjector(config, self._limiter)
        self._thermal_inj = ThermalFootprintInjector(config, self._limiter)
        self._impulse_inj = ImpulseToShearInjector(config, self._limiter)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply_contact(
        self,
        memory_state:  WorldMemoryState,
        tile_idx:      int,
        contact_force: float,
        dt:            float,
    ) -> None:
        """Route a footstep / fall contact to the stress injector.

        Parameters
        ----------
        memory_state  : WorldMemoryState updated in-place.
        tile_idx      : Flat tile index.
        contact_force : Normalised contact force [0, 1].
        dt            : Time step (seconds).
        """
        if self._enabled:
            self._contact_inj.inject(memory_state, tile_idx, contact_force, dt)

    def apply_thermal(
        self,
        material_state: SurfaceMaterialState,
        tile_idx:       int,
        body_heat:      float,
        dt:             float,
    ) -> None:
        """Route body heat to the thermal footprint injector.

        Parameters
        ----------
        material_state : SurfaceMaterialState updated in-place.
        tile_idx       : Flat tile index.
        body_heat      : Normalised body heat [0, 1].
        dt             : Time step (seconds).
        """
        if self._enabled:
            self._thermal_inj.inject(material_state, tile_idx, body_heat, dt)

    def apply_impulse(
        self,
        instability_state: InstabilityState,
        tile_idx:          int,
        impulse_magnitude: float,
    ) -> None:
        """Route a jump/fall/grasp impulse to the shear injector.

        Parameters
        ----------
        instability_state : InstabilityState updated in-place.
        tile_idx          : Flat tile index.
        impulse_magnitude : Normalised impulse [0, 1].
        """
        if self._enabled:
            self._impulse_inj.inject(instability_state, tile_idx, impulse_magnitude)

    def apply_wind_wake(self, microclimate_state: MicroclimateState) -> None:
        """Bias *microclimate_state* to reflect the player's body as a wind break.

        The player partially shelters the immediate downwind position.  The
        effect is proportional to the remaining unsheltered fraction so that
        it saturates gracefully in already-sheltered spots.

        Parameters
        ----------
        microclimate_state : MicroclimateState modified in-place.
        """
        if not self._enabled:
            return
        bias = self._wind_wake_factor * (1.0 - microclimate_state.windShelter)
        microclimate_state.windShelter = _clamp(
            microclimate_state.windShelter + bias
        )

    def tick(self, dt: float) -> None:
        """Advance the shared influence limiter decay.

        Parameters
        ----------
        dt : Elapsed time in seconds.
        """
        self._limiter.tick(dt)
