"""SubsurfaceToMaterials — Stage 67 adapter: lava/vent → chunk materials.

Translates lava surface state and vent events into mass/heat delta calls
on :class:`~src.material.MassExchangeAPI.MassExchangeAPI`.

The adapter never writes PlanetChunkState fields directly.

Effects applied
---------------
Active lava cell (MAGMA) :
    * +temperatureProxy  (heat from lava)
    * -snowMass          (lava melts snow)
    * -dustThickness     (lava burns dust)
    * +debrisMass        (small debris from burning)

Solidified cell (CRUST) :
    * +crustHardness     (new crust forming)
    * +surfaceRoughness  (rough crust surface)

Crack event :
    * -crustHardness     (crack reduces hardness)
    * +debrisMass        (fragments released)

Public API
----------
SubsurfaceToMaterials(config=None)
  .apply_lava(chunk_api, lava_cell, cooling_model, dt) → None
  .apply_crack(chunk_api, crack_energy, dt)            → None
"""
from __future__ import annotations

from typing import Optional

from src.lava.LavaSurfaceFlow import LavaCell
from src.lava.CoolingModel    import CoolingModel, CellCrustState
from src.material.MassExchangeAPI import MassExchangeAPI


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


class SubsurfaceToMaterials:
    """Apply lava and crack effects to terrain chunks via MassExchangeAPI.

    Parameters
    ----------
    config :
        Optional dict; reads ``subsurface67.*`` keys.
    """

    _DEFAULT_HEAT_SCALE    = 0.05
    _DEFAULT_SNOW_MELT     = 0.04
    _DEFAULT_DUST_BURN     = 0.02
    _DEFAULT_DEBRIS_RATE   = 0.01
    _DEFAULT_CRUST_GAIN    = 0.03
    _DEFAULT_ROUGH_GAIN    = 0.02
    _DEFAULT_CRACK_DAMAGE  = 0.05

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = (config or {}).get("subsurface67", {}) or {}
        self._heat_scale   = float(cfg.get("lava_heat_scale",     self._DEFAULT_HEAT_SCALE))
        self._snow_melt    = float(cfg.get("lava_snow_melt_rate", self._DEFAULT_SNOW_MELT))
        self._dust_burn    = float(cfg.get("lava_dust_burn_rate", self._DEFAULT_DUST_BURN))
        self._debris_rate  = float(cfg.get("lava_debris_rate",    self._DEFAULT_DEBRIS_RATE))
        self._crust_gain   = float(cfg.get("lava_crust_gain",     self._DEFAULT_CRUST_GAIN))
        self._rough_gain   = float(cfg.get("lava_rough_gain",     self._DEFAULT_ROUGH_GAIN))
        self._crack_damage = float(cfg.get("crack_damage_rate",   self._DEFAULT_CRACK_DAMAGE))

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def apply_lava(
        self,
        chunk_api:     MassExchangeAPI,
        lava_cell:     LavaCell,
        cooling_model: CoolingModel,
        dt:            float,
    ) -> None:
        """Apply lava surface effects to *chunk_api*.

        Parameters
        ----------
        chunk_api :
            MassExchangeAPI for the terrain chunk under the lava cell.
        lava_cell :
            Current lava cell state.
        cooling_model :
            CoolingModel instance (used to determine crust/magma state).
        dt :
            Time step in seconds.
        """
        state = cooling_model.crust_state(lava_cell)
        v     = lava_cell.magma_volume

        if state == CellCrustState.MAGMA and v > 0.0:
            # Heat lava emits
            chunk_api.apply_heat_delta(self._heat_scale * v * dt)
            # Snow and ice melt
            chunk_api.apply_mass_delta("snowMass",        -self._snow_melt * v * dt)
            # Dust burns
            chunk_api.apply_mass_delta("dustThickness",   -self._dust_burn * v * dt)
            # Some debris released
            chunk_api.apply_mass_delta("debrisMass",      +self._debris_rate * v * dt)

        elif state == CellCrustState.CRUST and v > 0.0:
            # New crust forming on surface
            chunk_api.apply_mass_delta("crustHardness",   +self._crust_gain * v * dt)
            chunk_api.apply_mass_delta("surfaceRoughness",+self._rough_gain  * v * dt)

    def apply_crack(
        self,
        chunk_api:    MassExchangeAPI,
        crack_energy: float,
        dt:           float,
    ) -> None:
        """Apply crack event effects to *chunk_api*.

        Parameters
        ----------
        chunk_api :
            MassExchangeAPI for the cracked terrain chunk.
        crack_energy :
            Energy released by the crack [0, 1].
        dt :
            Time step in seconds.
        """
        damage = self._crack_damage * crack_energy * dt
        chunk_api.apply_mass_delta("crustHardness", -damage)
        chunk_api.apply_mass_delta("debrisMass",    +damage * 0.5)
        chunk_api.apply_stress_delta(crack_energy * 0.1 * dt)
