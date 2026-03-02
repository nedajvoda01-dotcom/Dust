"""MaterialYieldModel — Stage 35 per-material yield and friction parameters.

MaterialClass
    Enum of deformable surface classes (BEDROCK, DUST, SNOW, ICE_FILM, SCREE).

MaterialParams
    Yield strength, plasticity, friction, relaxation tau, and mass-push
    characteristics for one material class.

MaterialYieldModel
    Look-up table keyed by MaterialClass.  Config-overridable parameters.

Public API
----------
MaterialYieldModel.get(material_class) -> MaterialParams
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# MaterialClass
# ---------------------------------------------------------------------------

class MaterialClass(Enum):
    BEDROCK   = auto()   # no deformation
    DUST      = auto()   # strong plastic
    SNOW      = auto()   # plastic + compaction
    ICE_FILM  = auto()   # almost no H, affects friction
    SCREE     = auto()   # loose: high M-transport, low H


# ---------------------------------------------------------------------------
# MaterialParams
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MaterialParams:
    """Per-material deformation parameters.

    Attributes
    ----------
    yield_strength:
        Pressure threshold [Pa or normalised units] above which plastic
        indentation begins.  0 = always deforms; large = very rigid.
    indent_k:
        Scale factor for depth increment per (pressure - yield) * dt.
    mass_push_k:
        Scale factor for tangential mass transport (bulldozing).
    slip_track_k:
        Scale factor for slip-track groove formation.
    tau_h_sec:
        Relaxation time constant [s] for H field (vertical displacement).
    tau_m_sec:
        Relaxation time constant [s] for M field (loose material layer).
    base_friction:
        Baseline friction coefficient on a clean, undeformed surface.
    compaction_friction_gain:
        Friction increase per unit of compaction (M decrease).
    """
    yield_strength:          float
    indent_k:                float
    mass_push_k:             float
    slip_track_k:            float
    tau_h_sec:               float
    tau_m_sec:               float
    base_friction:           float
    compaction_friction_gain: float


# ---------------------------------------------------------------------------
# Defaults per material
# ---------------------------------------------------------------------------

_DEFAULTS: Dict[MaterialClass, MaterialParams] = {
    MaterialClass.BEDROCK: MaterialParams(
        yield_strength=1e9,
        indent_k=0.0,
        mass_push_k=0.0,
        slip_track_k=0.0,
        tau_h_sec=1e9,
        tau_m_sec=1e9,
        base_friction=0.75,
        compaction_friction_gain=0.0,
    ),
    MaterialClass.DUST: MaterialParams(
        yield_strength=500.0,
        indent_k=0.012,
        mass_push_k=0.08,
        slip_track_k=0.06,
        tau_h_sec=120.0,
        tau_m_sec=90.0,
        base_friction=0.45,
        compaction_friction_gain=0.15,
    ),
    MaterialClass.SNOW: MaterialParams(
        yield_strength=800.0,
        indent_k=0.009,
        mass_push_k=0.06,
        slip_track_k=0.05,
        tau_h_sec=300.0,
        tau_m_sec=200.0,
        base_friction=0.35,
        compaction_friction_gain=0.20,
    ),
    MaterialClass.ICE_FILM: MaterialParams(
        yield_strength=5000.0,
        indent_k=0.001,
        mass_push_k=0.01,
        slip_track_k=0.02,
        tau_h_sec=600.0,
        tau_m_sec=400.0,
        base_friction=0.10,
        compaction_friction_gain=0.05,
    ),
    MaterialClass.SCREE: MaterialParams(
        yield_strength=300.0,
        indent_k=0.005,
        mass_push_k=0.15,
        slip_track_k=0.04,
        tau_h_sec=60.0,
        tau_m_sec=45.0,
        base_friction=0.55,
        compaction_friction_gain=0.08,
    ),
}


# ---------------------------------------------------------------------------
# MaterialYieldModel
# ---------------------------------------------------------------------------

class MaterialYieldModel:
    """Look-up table of per-material deformation parameters.

    Parameters
    ----------
    config:
        Optional Config object.  When provided, ``deform.*`` keys override
        the built-in defaults.
    """

    def __init__(self, config=None) -> None:
        self._params: Dict[MaterialClass, MaterialParams] = {}
        for mat, defaults in _DEFAULTS.items():
            self._params[mat] = self._load(mat, defaults, config)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def get(self, material: MaterialClass) -> MaterialParams:
        """Return parameters for *material*.  Falls back to BEDROCK if unknown."""
        return self._params.get(material, self._params[MaterialClass.BEDROCK])

    def effective_friction(
        self,
        material: MaterialClass,
        m_value: float,
        m_base: float,
        ice_film: float = 0.0,
    ) -> float:
        """Compute effective friction coefficient given current field values.

        Parameters
        ----------
        material:
            Surface material class.
        m_value:
            Current M field value at contact point [0, 1].
        m_base:
            Baseline M (undeformed).  Compaction = m_base - m_value when
            m_value < m_base.
        ice_film:
            Ice film overlay [0, 1] from evolution system.
        """
        params = self.get(material)
        compaction = max(0.0, m_base - m_value)
        mu = (
            params.base_friction
            + compaction * params.compaction_friction_gain
            - ice_film * 0.30
        )
        return max(0.01, min(1.0, mu))

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    @staticmethod
    def _load(
        mat: MaterialClass,
        defaults: MaterialParams,
        config,
    ) -> MaterialParams:
        """Merge config overrides into defaults for *mat*."""
        if config is None:
            return defaults

        def _cfg(key: str, fallback: float) -> float:
            v = config.get("deform", key, default=None)
            if v is not None:
                return float(v)
            return fallback

        mat_name = mat.name.lower()
        return MaterialParams(
            yield_strength=_cfg(
                f"yield_{mat_name}", defaults.yield_strength),
            indent_k=_cfg("indent_k", defaults.indent_k),
            mass_push_k=_cfg("mass_push_k", defaults.mass_push_k),
            slip_track_k=_cfg("slip_track_k", defaults.slip_track_k),
            tau_h_sec=_cfg("relax_tau_h_sec", defaults.tau_h_sec),
            tau_m_sec=_cfg("relax_tau_m_sec", defaults.tau_m_sec),
            base_friction=defaults.base_friction,
            compaction_friction_gain=defaults.compaction_friction_gain,
        )
