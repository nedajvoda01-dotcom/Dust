"""EnergyNormalizer — Stage 54 auto-normalization of over/under-budget reservoirs.

Checks each reservoir against configured bounds and scales its value (and
optionally a paired field list) back toward the allowed range.  Called by
GlobalEnergySystem or externally once per balance tick.

Public API
----------
EnergyNormalizer(config=None)
  .normalize(ledger, dust_totals=None, ice_total=None)
      → dict[str, float]   (adjustment deltas applied)
"""
from __future__ import annotations

from typing import Optional

from src.energy.EnergyLedger import EnergyLedger

_DEFAULT_MAX_MECH       = 0.9
_DEFAULT_MAX_DUST_MASS  = 1.0
_DEFAULT_MAX_ICE_MASS   = 1.0
_DEFAULT_NORMALIZE_K    = 0.05


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


class EnergyNormalizer:
    """Keeps energy reservoirs within configured bounds.

    Parameters
    ----------
    config : optional dict; reads ``energy.*`` keys.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = (config or {}).get("energy", {}) or {}
        self._max_mech:      float = float(cfg.get("max_mech_stress",  _DEFAULT_MAX_MECH))
        self._max_dust_mass: float = float(cfg.get("max_dust_mass",    _DEFAULT_MAX_DUST_MASS))
        self._max_ice_mass:  float = float(cfg.get("max_ice_mass",     _DEFAULT_MAX_ICE_MASS))
        self._k:             float = float(cfg.get("auto_normalize_k", _DEFAULT_NORMALIZE_K))

    def normalize(
        self,
        ledger: EnergyLedger,
        dust_total: Optional[float] = None,
        ice_total:  Optional[float] = None,
    ) -> dict:
        """Apply normalization pass.

        If ``dust_total`` or ``ice_total`` are provided they are compared
        against their configured maxima; any excess is logged as a delta
        (callers are responsible for redistributing the excess in tile fields).

        Parameters
        ----------
        ledger     : EnergyLedger to adjust in-place.
        dust_total : Aggregate dust mass across all tiles [0..max_dust_mass].
        ice_total  : Aggregate ice mass across all tiles  [0..max_ice_mass].

        Returns
        -------
        dict mapping reservoir/key name → float delta applied.
        """
        deltas: dict = {}

        # Mechanical stress cap
        mech = ledger.get("mechanical")
        if mech > self._max_mech:
            excess = mech - self._max_mech
            consumed = ledger.consume("mechanical", excess * self._k)
            deltas["mechanical"] = -consumed

        # Dust mass conservation hint
        if dust_total is not None and dust_total > self._max_dust_mass:
            excess = dust_total - self._max_dust_mass
            deltas["dust_excess"] = -excess  # caller must redistribute

        # Ice mass cap hint
        if ice_total is not None and ice_total > self._max_ice_mass:
            excess = ice_total - self._max_ice_mass
            deltas["ice_excess"] = -excess   # caller must redistribute

        return deltas
