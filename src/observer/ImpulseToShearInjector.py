"""ImpulseToShearInjector — Stage 53 player impulse events → shear stress.

High-magnitude impulse events (jumps, hard landings, grasp rips, slide
arrests) deposit a small shear-stress contribution into the instability
field of the tile where the impulse occurs.

If the zone is already near the instability threshold, the player's
contribution may be the "last drop" that triggers an event.  In a stable
zone the contribution is absorbed harmlessly.

Model
-----
Per impulse event::

    shear_delta = k_impulse_shear * impulse_magnitude

The limiter caps the contribution per tile so that repeated small jumps on
the same spot cannot bootstrap a cascade in an otherwise stable zone.

Config keys (under ``observer.*``)
-----------------------------------
enable          : bool  — master switch   (default True)
k_impulse_shear : float — shear gain per unit impulse (default 0.015)

Public API
----------
ImpulseToShearInjector(config=None, limiter=None)
  .inject(instability_state, tile_idx, impulse_magnitude) -> None
"""
from __future__ import annotations

from typing import Optional

from src.instability.InstabilityState import InstabilityState
from src.observer.InfluenceLimiter    import InfluenceLimiter


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


class ImpulseToShearInjector:
    """Injects player impulse events into InstabilityState shear stress.

    Parameters
    ----------
    config :
        Optional dict; reads ``observer.*`` keys.
    limiter :
        Shared :class:`InfluenceLimiter` instance.
    """

    _DEFAULTS = {
        "enable":          True,
        "k_impulse_shear": 0.015,
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

        self._enabled:    bool  = bool(cfg["enable"])
        self._k_impulse:  float = float(cfg["k_impulse_shear"])
        self._limiter:    InfluenceLimiter = limiter or InfluenceLimiter(config)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def inject(
        self,
        instability_state: InstabilityState,
        tile_idx:          int,
        impulse_magnitude: float,
    ) -> None:
        """Apply an impulse contribution to *instability_state*.

        Parameters
        ----------
        instability_state : InstabilityState updated in-place.
        tile_idx          : Flat tile index.
        impulse_magnitude : Normalised impulse magnitude [0, 1].
        """
        if not self._enabled or impulse_magnitude <= 0.0:
            return

        shear_raw     = self._k_impulse * impulse_magnitude
        shear_clipped = self._limiter.clip(tile_idx, shear_raw)

        if shear_clipped <= 0.0:
            return

        f = instability_state.shearStressField
        f[tile_idx] = _clamp(f[tile_idx] + shear_clipped)
        self._limiter.record(tile_idx, shear_clipped)
