"""SubsurfaceToEnergy — Stage 67 adapter: vent/crack → energy budget.

Translates vent events and crack events into energy ledger operations on
:class:`~src.energy.EnergyLedger.EnergyLedger` (Stage 54).

Energy flows (§6)
-----------------
Vent event :
    * Consume mechanicalStressEnergy proportional to vent intensity
    * Add thermalReservoir (magma heat release)
    * Add acousticEnergy (eruption sound)

Crack event :
    * Consume mechanicalStressEnergy (stress relief)
    * Add acousticEnergy (fracture sound — §5.1)
    * Add thermalReservoir (friction heat)

Constraints (§6):
    * Cannot generate infinite lava: mechanical energy caps consumption
    * Cannot infinitely raise temperature: thermal reservoir is clamped

Public API
----------
SubsurfaceToEnergy(config=None)
  .apply_vent(ledger, intensity)              → None
  .apply_crack(ledger, crack_energy)          → None
"""
from __future__ import annotations

from typing import Optional

from src.energy.EnergyLedger import EnergyLedger


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


class SubsurfaceToEnergy:
    """Apply vent and crack energy flows to the planetary energy ledger.

    Parameters
    ----------
    config :
        Optional dict; reads ``subsurface67.*`` keys.
    """

    _DEFAULT_VENT_MECH_COST     = 0.10   # mechanical energy consumed per vent unit
    _DEFAULT_VENT_THERMAL_GAIN  = 0.06   # thermal energy released per vent unit
    _DEFAULT_VENT_ACOUSTIC_GAIN = 0.03   # acoustic energy generated per vent unit
    _DEFAULT_CRACK_MECH_COST    = 0.05   # mechanical stress released per crack unit
    _DEFAULT_CRACK_ACOUSTIC     = 0.04   # acoustic energy from fracture
    _DEFAULT_CRACK_THERMAL      = 0.02   # friction heat from crack

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = (config or {}).get("subsurface67", {}) or {}
        self._vent_mech     = float(cfg.get("vent_mech_cost",      self._DEFAULT_VENT_MECH_COST))
        self._vent_thermal  = float(cfg.get("vent_thermal_gain",   self._DEFAULT_VENT_THERMAL_GAIN))
        self._vent_acoustic = float(cfg.get("vent_acoustic_gain",  self._DEFAULT_VENT_ACOUSTIC_GAIN))
        self._crack_mech    = float(cfg.get("crack_mech_cost",     self._DEFAULT_CRACK_MECH_COST))
        self._crack_acoustic = float(cfg.get("crack_acoustic_gain", self._DEFAULT_CRACK_ACOUSTIC))
        self._crack_thermal = float(cfg.get("crack_thermal_gain",  self._DEFAULT_CRACK_THERMAL))

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def apply_vent(self, ledger: EnergyLedger, intensity: float) -> None:
        """Apply energy flows for a vent event of given *intensity*.

        Parameters
        ----------
        ledger :
            The planetary energy ledger to modify.
        intensity :
            Vent intensity [0, 1].
        """
        ledger.consume("mechanical", self._vent_mech     * intensity)
        ledger.add(    "thermal",    self._vent_thermal  * intensity)
        ledger.add(    "acoustic",   self._vent_acoustic * intensity)

    def apply_crack(self, ledger: EnergyLedger, crack_energy: float) -> None:
        """Apply energy flows for a crack event of given *crack_energy*.

        Parameters
        ----------
        ledger :
            The planetary energy ledger to modify.
        crack_energy :
            Energy released by fracture [0, 1].
        """
        ledger.consume("mechanical", self._crack_mech     * crack_energy)
        ledger.add(    "acoustic",   self._crack_acoustic * crack_energy)
        ledger.add(    "thermal",    self._crack_thermal  * crack_energy)
