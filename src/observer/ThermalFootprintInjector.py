"""ThermalFootprintInjector — Stage 53 player body heat → local phase shift.

The player's body heat creates a tiny, localised thermal footprint that
accelerates ice melt and marginally increases snow compaction in the cell
directly under the player.

This is not real thermodynamics — it is a lightweight proxy model.

Model
-----
If the player stands on a tile for ``dt`` seconds::

    thermal_effect   = k_thermal_melt * body_heat * dt
    ice_film        -= thermal_effect
    snow_compaction += thermal_effect * 0.3    # minor secondary effect

Both values are clamped to [0, 1].  The effect is so small (k_thermal_melt
is tiny) that it only becomes visible after long exposure.  The underlying
material's own relaxation (Stage 45) erases the footprint quickly after the
player leaves.

The limiter is applied per *tile* so the player cannot melt all ice by
standing indefinitely on the same cell.

Config keys (under ``observer.*``)
-----------------------------------
enable          : bool  — master switch  (default True)
k_thermal_melt  : float — melt rate per body-heat unit per second (default 0.005)

Public API
----------
ThermalFootprintInjector(config=None, limiter=None)
  .inject(material_state, tile_idx, body_heat, dt) -> None
"""
from __future__ import annotations

from typing import Optional

from src.material.SurfaceMaterialState import SurfaceMaterialState
from src.observer.InfluenceLimiter     import InfluenceLimiter


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


class ThermalFootprintInjector:
    """Injects player body heat into SurfaceMaterialState locally.

    Parameters
    ----------
    config :
        Optional dict; reads ``observer.*`` keys.
    limiter :
        Shared :class:`InfluenceLimiter` instance.
    """

    _DEFAULTS = {
        "enable":         True,
        "k_thermal_melt": 0.005,
    }

    def __init__(
        self,
        config:  Optional[dict] = None,
        limiter: Optional[InfluenceLimiter] = None,
    ) -> None:
        cfg = dict(self._DEFAULTS)
        if isinstance(config, dict):
            src = config.get("observer", config)
            for k in self._DEFAULTS:
                if k in src:
                    cfg[k] = src[k]

        self._enabled:  bool  = bool(cfg["enable"])
        self._k_melt:   float = float(cfg["k_thermal_melt"])
        self._limiter:  InfluenceLimiter = limiter or InfluenceLimiter(config)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def inject(
        self,
        material_state: SurfaceMaterialState,
        tile_idx:       int,
        body_heat:      float,
        dt:             float,
    ) -> None:
        """Apply thermal footprint to *material_state*.

        Parameters
        ----------
        material_state : SurfaceMaterialState updated in-place.
        tile_idx       : Flat tile index (used only for limiter bookkeeping).
        body_heat      : Normalised body heat intensity [0, 1].
        dt             : Time step (seconds).
        """
        if not self._enabled or dt <= 0.0 or body_heat <= 0.0:
            return

        thermal_raw     = self._k_melt * body_heat * dt
        thermal_clipped = self._limiter.clip(tile_idx, thermal_raw)

        if thermal_clipped <= 0.0:
            return

        # Reduce ice film
        material_state.ice_film = _clamp(
            material_state.ice_film - thermal_clipped
        )

        # Slight snow compaction increase (pressure + melt → denser pack)
        compact_gain = thermal_clipped * 0.3
        material_state.snow_compaction = _clamp(
            material_state.snow_compaction + compact_gain
        )

        self._limiter.record(tile_idx, thermal_clipped)
