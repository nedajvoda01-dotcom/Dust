"""EvolutionToMaterialAdapter — Stage 50 bridge from planetary evolution to
material phase parameters.

Maps :class:`~src.evolution.PlanetEvolutionState.PlanetEvolutionState` fields
to drift offsets applied to the fast-tick ``PhaseChangeSystem`` parameters.

The mapping is intentionally additive and bounded: evolution state provides
*baseline bias* that shifts formation/melt/hardness/roughness rates, while
``PhaseChangeSystem`` continues to handle the per-cell fast dynamics.

Mapping
-------
dustReservoir  → dust_deposition_bias  [0, +0.3]
crustStability → crust_hardness_bias   [−0.1, +0.1]
iceBelt        → ice_formation_bias    [0, +0.2]
crustStability → roughness_bias        [−0.05, 0]   (low stability = rougher)

Public API
----------
EvolutionToMaterialAdapter()
  .get_material_biases(dust_reservoir, crust_stability, ice_belt)
      -> MaterialBiases
"""
from __future__ import annotations

from dataclasses import dataclass


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


@dataclass
class MaterialBiases:
    """Additive rate biases applied on top of PhaseChangeSystem base rates.

    Attributes
    ----------
    dust_deposition_bias : float  — added to icefilm / crust formation rate
    crust_hardness_bias  : float  — added to crust_hardness drift
    ice_formation_bias   : float  — added to icefilm_form_k
    roughness_bias       : float  — added to roughness (negative = smoother)
    """
    dust_deposition_bias: float = 0.0
    crust_hardness_bias:  float = 0.0
    ice_formation_bias:   float = 0.0
    roughness_bias:       float = 0.0


class EvolutionToMaterialAdapter:
    """Converts coarse planetary evolution fields into per-cell material biases.

    No config required — mapping is fixed based on domain knowledge.
    """

    # Bias scale constants
    _DUST_DEP_MAX:    float = 0.3
    _CRUST_BIAS_MAX:  float = 0.1
    _ICE_BIAS_MAX:    float = 0.2
    _ROUGH_BIAS_MIN:  float = -0.05

    def get_material_biases(
        self,
        dust_reservoir:  float,
        crust_stability: float,
        ice_belt:        float,
    ) -> MaterialBiases:
        """Compute :class:`MaterialBiases` from evolution field values.

        Parameters
        ----------
        dust_reservoir  : dustReservoirMap value at the tile [0, 1].
        crust_stability : crustStabilityMap value at the tile [0, 1].
        ice_belt        : iceBeltDistribution value at the tile [0, 1].

        Returns
        -------
        MaterialBiases
        """
        dust_dep  = _clamp(dust_reservoir) * self._DUST_DEP_MAX
        # High stability → positive crust bias; low stability → negative
        crust_b   = (_clamp(crust_stability) - 0.5) * 2.0 * self._CRUST_BIAS_MAX
        ice_form  = _clamp(ice_belt) * self._ICE_BIAS_MAX
        # Low stability makes surface rougher
        rough_b   = (1.0 - _clamp(crust_stability)) * self._ROUGH_BIAS_MIN

        return MaterialBiases(
            dust_deposition_bias=dust_dep,
            crust_hardness_bias=crust_b,
            ice_formation_bias=ice_form,
            roughness_bias=rough_b,
        )
