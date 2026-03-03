"""EnergyLedger — Stage 54 energy-flow accounting.

Provides the three canonical operations required by all major systems:

    AddEnergy(type, amount)       — injects energy into a reservoir
    ConsumeEnergy(type, amount)   — removes energy from a reservoir
    TransferEnergy(from, to, amount) — moves energy between reservoirs
                                       applying transfer_efficiency loss

Energy types match the GlobalEnergyState reservoir names:

    "solar"         → solarInputEnergy
    "atmospheric"   → atmosphericKineticEnergy
    "gravitational" → gravitationalPotentialProxy
    "thermal"       → thermalReservoir
    "mechanical"    → mechanicalStressEnergy
    "phase"         → materialPhaseEnergy
    "acoustic"      → acousticEnergy

Public API
----------
EnergyLedger(config=None)
  .add(reservoir, amount)              → float  (new value)
  .consume(reservoir, amount)          → float  (actual consumed, ≤ amount)
  .transfer(src, dst, amount)          → float  (actual transferred)
  .get(reservoir)                      → float
  .set(reservoir, value)               → None
  .reservoirs()                        → dict[str, float]
"""
from __future__ import annotations

from typing import Optional

_RESERVOIR_NAMES = (
    "solar",
    "atmospheric",
    "gravitational",
    "thermal",
    "mechanical",
    "phase",
    "acoustic",
)

_DEFAULT_EFFICIENCY = 0.85


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


class EnergyLedger:
    """Tracks the seven planetary energy reservoirs.

    Parameters
    ----------
    config : optional dict; reads ``energy.*`` keys.
    """

    __slots__ = ("_store", "_efficiency")

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = (config or {}).get("energy", {}) or {}
        self._efficiency: float = float(
            cfg.get("transfer_efficiency", _DEFAULT_EFFICIENCY)
        )
        self._store: dict[str, float] = {k: 0.0 for k in _RESERVOIR_NAMES}

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def add(self, reservoir: str, amount: float) -> float:
        """Add *amount* to *reservoir* (clamps at 1.0).

        Returns
        -------
        float — new reservoir value.
        """
        if reservoir not in self._store:
            raise KeyError(f"EnergyLedger: unknown reservoir '{reservoir}'")
        self._store[reservoir] = _clamp(self._store[reservoir] + max(0.0, amount))
        return self._store[reservoir]

    def consume(self, reservoir: str, amount: float) -> float:
        """Remove up to *amount* from *reservoir* (cannot go below 0).

        Returns
        -------
        float — actual amount consumed (≤ amount).
        """
        if reservoir not in self._store:
            raise KeyError(f"EnergyLedger: unknown reservoir '{reservoir}'")
        available = self._store[reservoir]
        actual = min(available, max(0.0, amount))
        self._store[reservoir] = available - actual
        return actual

    def transfer(self, src: str, dst: str, amount: float) -> float:
        """Move *amount* × efficiency from *src* to *dst*.

        The efficiency loss (1 - efficiency) is silently dissipated.

        Returns
        -------
        float — actual amount added to *dst*.
        """
        consumed = self.consume(src, amount)
        delivered = consumed * self._efficiency
        self.add(dst, delivered)
        return delivered

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get(self, reservoir: str) -> float:
        """Return current value of *reservoir*."""
        if reservoir not in self._store:
            raise KeyError(f"EnergyLedger: unknown reservoir '{reservoir}'")
        return self._store[reservoir]

    def set(self, reservoir: str, value: float) -> None:
        """Directly set *reservoir* to *value* (clamped to [0, 1])."""
        if reservoir not in self._store:
            raise KeyError(f"EnergyLedger: unknown reservoir '{reservoir}'")
        self._store[reservoir] = _clamp(value)

    def reservoirs(self) -> dict:
        """Return a shallow copy of the reservoir dict."""
        return dict(self._store)
