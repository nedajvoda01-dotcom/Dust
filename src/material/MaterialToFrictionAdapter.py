"""MaterialToFrictionAdapter — Stage 45 material state → effective friction.

Computes ``effectiveMu`` from the five surface material state fields:

    mu = baseMu(material)
       * f(roughness)
       * g(iceFilm)
       * h(dustThickness, snowCompaction, crustHardness)

Config-driven multipliers allow tuning without code changes.

Public API
----------
MaterialToFrictionAdapter(config=None)
  .effective_mu(state, base_mu=0.5) -> float
"""
from __future__ import annotations

from src.material.SurfaceMaterialState import SurfaceMaterialState


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


class MaterialToFrictionAdapter:
    """Computes effective friction coefficient from :class:`SurfaceMaterialState`.

    Parameters
    ----------
    config : dict or None
        Keys:
        ``mu_icefilm_multiplier`` — multiplier applied when ice_film = 1.0
        (default 0.15; at ice_film=0 no effect).
        ``mu_roughness_scale`` — how much roughness changes mu
        (default 0.4; 0 = no roughness effect).
        ``mu_dust_scale`` — mu reduction per unit of thick dust layer
        (default 0.10).
        ``mu_crust_hardness_scale`` — mu change when fully crusted
        (default −0.05; hard crust + polished = slightly slippery).
        ``mu_snow_compaction_scale`` — mu change per unit compaction
        (default 0.05; compact naст can be slightly higher friction).
    """

    _DEFAULTS = {
        "mu_icefilm_multiplier":    0.15,
        "mu_roughness_scale":       0.40,
        "mu_dust_scale":            0.10,
        "mu_crust_hardness_scale": -0.05,
        "mu_snow_compaction_scale": 0.05,
    }

    def __init__(self, config=None) -> None:
        cfg = dict(self._DEFAULTS)
        if config:
            for k in self._DEFAULTS:
                if k in config:
                    cfg[k] = float(config[k])
        self._ice_mult   = cfg["mu_icefilm_multiplier"]
        self._rough_k    = cfg["mu_roughness_scale"]
        self._dust_k     = cfg["mu_dust_scale"]
        self._crust_k    = cfg["mu_crust_hardness_scale"]
        self._snow_k     = cfg["mu_snow_compaction_scale"]

    def effective_mu(
        self,
        state: SurfaceMaterialState,
        base_mu: float = 0.5,
    ) -> float:
        """Return effective friction coefficient.

        Parameters
        ----------
        state :
            Current surface material state.
        base_mu :
            Baseline friction of the underlying material (e.g. from
            :class:`~src.physics.MaterialYieldModel.MaterialYieldModel`).
        """
        mu = base_mu

        # Roughness: more rough → higher friction
        mu += self._rough_k * (state.roughness - 0.5)

        # Ice film: major friction reduction (lerp toward ice_mult)
        if state.ice_film > 0.0:
            mu = mu * (1.0 - state.ice_film) + self._ice_mult * state.ice_film

        # Dust layer: thick soft dust reduces effective grip slightly
        mu -= self._dust_k * state.dust_thickness

        # Crust hardness (sign depends on config; default slightly slippery)
        mu += self._crust_k * state.crust_hardness

        # Snow compaction: compact naст can increase or decrease friction
        mu += self._snow_k * state.snow_compaction

        return _clamp(mu, 0.01, 1.0)
