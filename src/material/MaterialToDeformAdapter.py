"""MaterialToDeformAdapter — Stage 45 material state → deformation params.

Maps :class:`~src.material.SurfaceMaterialState.SurfaceMaterialState`
fields to modified :class:`~src.physics.MaterialYieldModel.MaterialParams`
used by :class:`~src.surface.DeformationField.DeformationField`.

Rules (section 5 of spec):
* crustHardness high → indent_k ↓, mass_push_k ↓
* snowCompaction high → indent_k ↓ (naст resists indentation)
* dustThickness high → indent_k ↑ (soft loose layer)

Public API
----------
MaterialToDeformAdapter(config=None)
  .apply(params, state) -> MaterialParams
"""
from __future__ import annotations

from src.material.SurfaceMaterialState import SurfaceMaterialState
from src.physics.MaterialYieldModel import MaterialParams


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


class MaterialToDeformAdapter:
    """Modifies :class:`MaterialParams` based on surface material state.

    Parameters
    ----------
    config : dict or None
        Keys:
        ``crust_indent_reduction``  — max fractional reduction of indent_k
        at crust_hardness=1.0 (default 0.80).
        ``snow_indent_reduction``   — max fractional reduction at
        snow_compaction=1.0 (default 0.60).
        ``dust_indent_gain``        — max fractional increase of indent_k
        at dust_thickness=1.0 (default 0.50).
        ``crust_push_reduction``    — max fractional reduction of
        mass_push_k at crust_hardness=1.0 (default 0.75).
    """

    _DEFAULTS = {
        "crust_indent_reduction": 0.80,
        "snow_indent_reduction":  0.60,
        "dust_indent_gain":       0.50,
        "crust_push_reduction":   0.75,
    }

    def __init__(self, config=None) -> None:
        cfg = dict(self._DEFAULTS)
        if config:
            for k in self._DEFAULTS:
                if k in config:
                    cfg[k] = float(config[k])
        self._crust_indent = cfg["crust_indent_reduction"]
        self._snow_indent  = cfg["snow_indent_reduction"]
        self._dust_gain    = cfg["dust_indent_gain"]
        self._crust_push   = cfg["crust_push_reduction"]

    def apply(
        self,
        params: MaterialParams,
        state: SurfaceMaterialState,
    ) -> MaterialParams:
        """Return a new :class:`MaterialParams` modified by *state*.

        The original *params* is not mutated.
        """
        # indent_k: reduced by crust hardness and snow compaction,
        #           increased by thick dust layer
        indent_k = params.indent_k
        indent_k *= (1.0 - self._crust_indent * state.crust_hardness)
        indent_k *= (1.0 - self._snow_indent  * state.snow_compaction)
        indent_k *= (1.0 + self._dust_gain    * state.dust_thickness)
        indent_k = max(0.0, indent_k)

        # mass_push_k: reduced by crust
        mass_push_k = params.mass_push_k * (
            1.0 - self._crust_push * state.crust_hardness
        )
        mass_push_k = max(0.0, mass_push_k)

        # yield_strength: crusted surface needs more force to deform
        yield_strength = params.yield_strength * (
            1.0 + state.crust_hardness * 4.0
            + state.snow_compaction   * 2.0
        )

        return MaterialParams(
            yield_strength=yield_strength,
            indent_k=indent_k,
            mass_push_k=mass_push_k,
            slip_track_k=params.slip_track_k,
            tau_h_sec=params.tau_h_sec,
            tau_m_sec=params.tau_m_sec,
            base_friction=params.base_friction,
            compaction_friction_gain=params.compaction_friction_gain,
        )
