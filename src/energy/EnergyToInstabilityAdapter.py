"""EnergyToInstabilityAdapter — Stage 54 energy ↔ instability coupling.

Translates instability events into energy-ledger operations:

    CrustFailureEvent   → consume mechanicalStressEnergy, lower shearStressField
    DustAvalancheEvent  → consume mechanicalStressEnergy + atmosphericKineticEnergy
    ThermalFractureEvent→ consume thermalReservoir, produce mechanicalStressEnergy

Also provides a pre-tick gate: if ``mechanicalStressEnergy`` is below a
minimum threshold the instability system should not trigger new cascades.

Public API
----------
EnergyToInstabilityAdapter(config=None)
  .on_crust_failure(ledger, event)   → float  (mechanical consumed)
  .on_dust_avalanche(ledger, event)  → float  (total consumed)
  .on_thermal_fracture(ledger, event)→ float  (thermal consumed)
  .can_trigger_cascade(ledger)       → bool
"""
from __future__ import annotations

from typing import Optional, Any

from src.energy.EnergyLedger import EnergyLedger

_DEFAULT_MECH_PER_CRUST     = 0.08
_DEFAULT_MECH_PER_DUST      = 0.04
_DEFAULT_ATMO_PER_DUST      = 0.03
_DEFAULT_THERMAL_PER_FRAC   = 0.06
_DEFAULT_MECH_FROM_FRAC     = 0.04
_DEFAULT_MIN_MECH_TRIGGER   = 0.02


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


class EnergyToInstabilityAdapter:
    """Couples energy ledger to instability events.

    Parameters
    ----------
    config : optional dict; reads ``energy.*`` keys.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = (config or {}).get("energy", {}) or {}
        self._mech_per_crust:    float = float(cfg.get("mech_per_crust_event",  _DEFAULT_MECH_PER_CRUST))
        self._mech_per_dust:     float = float(cfg.get("mech_per_dust_event",   _DEFAULT_MECH_PER_DUST))
        self._atmo_per_dust:     float = float(cfg.get("atmo_per_dust_event",   _DEFAULT_ATMO_PER_DUST))
        self._thermal_per_frac:  float = float(cfg.get("thermal_per_frac_event",_DEFAULT_THERMAL_PER_FRAC))
        self._mech_from_frac:    float = float(cfg.get("mech_from_frac_event",  _DEFAULT_MECH_FROM_FRAC))
        self._min_mech_trigger:  float = float(cfg.get("min_mech_trigger",      _DEFAULT_MIN_MECH_TRIGGER))

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def on_crust_failure(self, ledger: EnergyLedger, event: Any) -> float:
        """Consume mechanicalStressEnergy proportional to event intensity.

        Parameters
        ----------
        ledger : EnergyLedger (modified in-place).
        event  : CrustFailureEvent (needs .intensity).

        Returns
        -------
        float — mechanical energy consumed.
        """
        intensity = float(getattr(event, "intensity", 0.0))
        consumed = ledger.consume("mechanical", intensity * self._mech_per_crust)
        # Mechanical → acoustic (sound of collapse)
        ledger.transfer("mechanical", "acoustic", intensity * 0.02)
        return consumed

    def on_dust_avalanche(self, ledger: EnergyLedger, event: Any) -> float:
        """Consume mechanical + atmospheric energy on dust avalanche.

        Returns
        -------
        float — total energy consumed.
        """
        intensity = float(getattr(event, "intensity", 0.0))
        c1 = ledger.consume("mechanical",   intensity * self._mech_per_dust)
        c2 = ledger.consume("atmospheric",  intensity * self._atmo_per_dust)
        return c1 + c2

    def on_thermal_fracture(self, ledger: EnergyLedger, event: Any) -> float:
        """Consume thermal energy and produce mechanical stress on fracture.

        Returns
        -------
        float — thermal energy consumed.
        """
        intensity = float(getattr(event, "intensity", 0.0))
        consumed = ledger.consume("thermal", intensity * self._thermal_per_frac)
        ledger.add("mechanical", intensity * self._mech_from_frac)
        return consumed

    # ------------------------------------------------------------------
    # Cascade gate
    # ------------------------------------------------------------------

    def can_trigger_cascade(self, ledger: EnergyLedger) -> bool:
        """Return True only if mechanical energy is sufficient for a cascade."""
        return ledger.get("mechanical") >= self._min_mech_trigger
