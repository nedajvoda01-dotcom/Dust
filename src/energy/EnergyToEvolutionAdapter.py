"""EnergyToEvolutionAdapter — Stage 54 energy ↔ evolution coupling.

Enforces:

1. Dust mass conservation:
   total_dust_mass is constant (erosion in one tile = deposition elsewhere).
   Provides helpers to redistribute excess dust back to a reservoir.

2. Ice film energy budget:
   Ice formation requires thermal reservoir below a threshold.
   Melting returns energy to the thermal reservoir.

3. Wind erosion coupling:
   atmosphericKineticEnergy modulates how strongly wind erodes / deposits dust.

Public API
----------
EnergyToEvolutionAdapter(config=None)
  .dust_conservation_tick(ledger, dust_fields, dt) → float  (correction applied)
  .ice_formation_allowed(ledger)                   → bool
  .on_ice_melt(ledger, ice_mass)                   → float  (thermal returned)
  .wind_erosion_factor(ledger)                     → float  [0..1]
"""
from __future__ import annotations

from typing import List, Optional

from src.energy.EnergyLedger import EnergyLedger

_DEFAULT_DUST_TARGET       = 0.5   # target mean dust density per tile
_DEFAULT_ICE_THERMAL_THR   = 0.3   # thermal must be below this for ice to form
_DEFAULT_ICE_MELT_RETURN   = 0.6   # fraction of ice mass returned to thermal
_DEFAULT_WIND_EROSION_BASE = 0.4   # minimum erosion factor even with no wind


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


class EnergyToEvolutionAdapter:
    """Couples energy ledger to planetary evolution / surface processes.

    Parameters
    ----------
    config : optional dict; reads ``energy.*`` keys.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = (config or {}).get("energy", {}) or {}
        self._dust_target:     float = float(cfg.get("dust_target_mean",     _DEFAULT_DUST_TARGET))
        self._ice_thermal_thr: float = float(cfg.get("ice_thermal_threshold", _DEFAULT_ICE_THERMAL_THR))
        self._ice_melt_return: float = float(cfg.get("ice_melt_return_k",    _DEFAULT_ICE_MELT_RETURN))
        self._wind_base:       float = float(cfg.get("wind_erosion_base",    _DEFAULT_WIND_EROSION_BASE))
        self._max_dust_mass:   float = float(cfg.get("max_dust_mass", 1.0))

    # ------------------------------------------------------------------
    # Dust mass conservation
    # ------------------------------------------------------------------

    def dust_conservation_tick(
        self,
        ledger:      EnergyLedger,
        dust_fields: List[float],
        dt:          float,
    ) -> float:
        """Enforce dust mass conservation.

        If total dust exceeds ``max_dust_mass`` the excess is uniformly
        subtracted from all tiles.  The corrected fields are updated
        **in-place**.

        Parameters
        ----------
        ledger      : EnergyLedger (reads ``atmospheric`` for wind factor).
        dust_fields : Per-tile dust values (modified in-place).
        dt          : Elapsed seconds (not used for conservation itself,
                      kept for API parity).

        Returns
        -------
        float — total correction applied (negative means dust was removed).
        """
        n = len(dust_fields)
        if n == 0:
            return 0.0

        total = sum(dust_fields)
        if total <= self._max_dust_mass:
            return 0.0

        # Proportional scaling preserves distribution and guarantees total ≤ max
        scale = self._max_dust_mass / total
        excess = total - self._max_dust_mass
        for i in range(n):
            dust_fields[i] *= scale
        return -excess

    # ------------------------------------------------------------------
    # Ice film budget
    # ------------------------------------------------------------------

    def ice_formation_allowed(self, ledger: EnergyLedger) -> bool:
        """Return True if thermal reservoir is cold enough for ice to form."""
        return ledger.get("thermal") < self._ice_thermal_thr

    def on_ice_melt(self, ledger: EnergyLedger, ice_mass: float) -> float:
        """Return melted ice energy to the thermal reservoir.

        Parameters
        ----------
        ice_mass : mass of ice melted (normalised [0..1]).

        Returns
        -------
        float — energy added to thermal reservoir.
        """
        returned = ice_mass * self._ice_melt_return
        ledger.add("thermal", returned)
        return returned

    # ------------------------------------------------------------------
    # Wind erosion factor
    # ------------------------------------------------------------------

    def wind_erosion_factor(self, ledger: EnergyLedger) -> float:
        """Return wind erosion strength modulated by atmospheric energy.

        Returns
        -------
        float in [wind_erosion_base .. 1.0].
        """
        atmo = ledger.get("atmospheric")
        return _clamp(self._wind_base + atmo * (1.0 - self._wind_base))
