"""GlobalEnergySystem — Stage 54 planetary energy budget controller.

Manages the seven energy reservoirs described in the problem statement and
exposes an ``EnergyBalanceTick`` that runs at low frequency (0.05–0.5 Hz).

Reservoir semantics
-------------------
solar           : energy injected by the two suns each tick
atmospheric     : kinetic energy driving wind, erosion, dust migration
gravitational   : potential energy from slope; converts to mechanical on collapse
thermal         : heat reservoir; drives ice formation/melt and fracture
mechanical      : stress/collapse energy; dissipates via friction + fractures
phase           : energy locked in material phase transitions (ice ↔ liquid)
acoustic        : short-lived energy from fractures, avalanches; decays rapidly

Entropy scalar (0..1)
---------------------
``planetEntropy`` measures world "structuredness":
  • instability events raise it
  • erosion smoothing lowers it
  • extreme phase gradients raise it
When entropy exceeds ``entropy_upper_bound`` the system scales up dissipation
coefficients; when it falls below ``entropy_lower_bound`` it scales them down.

Public API
----------
GlobalEnergySystem(config=None)
  .ledger                 → EnergyLedger
  .planet_entropy         → float  [0..1]
  .inject_solar(amount)
  .wind_tick(wind_strength, dt)
  .record_instability_event(intensity)
  .record_erosion_smoothing(amount)
  .record_phase_gradient(gradient)
  .energy_balance_tick(dt)
  .get_dissipation_scale() → float
  .state_dict()           → dict
  .load_state_dict(d)
"""
from __future__ import annotations

from typing import Optional

from src.energy.EnergyLedger import EnergyLedger

_DEFAULT_TICK_HZ         = 0.1
_DEFAULT_MAX_MECH        = 0.9
_DEFAULT_MAX_DUST_MASS   = 1.0
_DEFAULT_MAX_ICE_MASS    = 1.0
_DEFAULT_ENTROPY_UPPER   = 0.8
_DEFAULT_ENTROPY_LOWER   = 0.2
_DEFAULT_NORMALIZE_K     = 0.05
_DEFAULT_ACOUSTIC_DECAY  = 0.35   # per balance tick
_DEFAULT_SOLAR_K         = 0.12   # fraction of solar that reaches thermal


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


class GlobalEnergySystem:
    """Planetary energy budget for Stage 54.

    Parameters
    ----------
    config : optional dict; reads ``energy.*`` keys.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = (config or {}).get("energy", {}) or {}

        self._enabled:        bool  = bool(cfg.get("enable", True))
        self._tick_hz:        float = float(cfg.get("tick_hz", _DEFAULT_TICK_HZ))
        self._max_mech:       float = float(cfg.get("max_mech_stress",   _DEFAULT_MAX_MECH))
        self._max_dust_mass:  float = float(cfg.get("max_dust_mass",     _DEFAULT_MAX_DUST_MASS))
        self._max_ice_mass:   float = float(cfg.get("max_ice_mass",      _DEFAULT_MAX_ICE_MASS))
        self._entropy_upper:  float = float(cfg.get("entropy_upper_bound", _DEFAULT_ENTROPY_UPPER))
        self._entropy_lower:  float = float(cfg.get("entropy_lower_bound", _DEFAULT_ENTROPY_LOWER))
        self._normalize_k:    float = float(cfg.get("auto_normalize_k",  _DEFAULT_NORMALIZE_K))

        self._ledger = EnergyLedger(config)

        # Entropy scalar
        self._planet_entropy: float = 0.5

        # Accumulator for balance tick scheduling
        self._tick_accumulator: float = 0.0

        # Dissipation scale (modified by entropy feedback)
        self._dissipation_scale: float = 1.0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def ledger(self) -> EnergyLedger:
        return self._ledger

    @property
    def planet_entropy(self) -> float:
        return self._planet_entropy

    # ------------------------------------------------------------------
    # Energy injection helpers
    # ------------------------------------------------------------------

    def inject_solar(self, amount: float) -> None:
        """Inject solar energy (fraction passes to thermal reservoir).

        Parameters
        ----------
        amount : normalised solar irradiance [0..1].
        """
        if not self._enabled:
            return
        self._ledger.add("solar", amount)
        # Solar → thermal conversion
        self._ledger.transfer("solar", "thermal", amount * _DEFAULT_SOLAR_K)

    def wind_tick(self, wind_strength: float, dt: float) -> None:
        """Update atmospheric kinetic energy from wind.

        Parameters
        ----------
        wind_strength : normalised [0..1].
        dt            : elapsed seconds.
        """
        if not self._enabled:
            return
        gain = wind_strength * 0.08 * dt
        self._ledger.add("atmospheric", gain)

    def record_slope_collapse(self, slope: float) -> None:
        """Convert gravitational potential to mechanical stress on collapse.

        Parameters
        ----------
        slope : normalised slope [0..1].
        """
        if not self._enabled:
            return
        potential = slope * 0.15
        self._ledger.add("gravitational", potential)
        self._ledger.transfer("gravitational", "mechanical", potential * 0.9)

    # ------------------------------------------------------------------
    # Entropy update helpers
    # ------------------------------------------------------------------

    def record_instability_event(self, intensity: float) -> None:
        """Raise entropy on an instability event.

        Also converts mechanical stress → acoustic energy (dissipation path).
        """
        if not self._enabled:
            return
        self._planet_entropy = _clamp(self._planet_entropy + intensity * 0.04)
        # mechanical → acoustic (short-lived dissipation)
        self._ledger.transfer("mechanical", "acoustic", intensity * 0.10)

    def record_erosion_smoothing(self, amount: float) -> None:
        """Lower entropy when erosion smooths the landscape."""
        if not self._enabled:
            return
        self._planet_entropy = _clamp(self._planet_entropy - amount * 0.02)

    def record_phase_gradient(self, gradient: float) -> None:
        """Raise entropy on extreme phase gradients (ice/liquid boundaries)."""
        if not self._enabled:
            return
        if gradient > 0.5:
            self._planet_entropy = _clamp(
                self._planet_entropy + (gradient - 0.5) * 0.03
            )

    # ------------------------------------------------------------------
    # Energy balance tick
    # ------------------------------------------------------------------

    def energy_balance_tick(self, dt: float) -> None:
        """Low-frequency global balance pass.

        Schedules itself via ``tick_hz``; call every simulation frame with
        the frame dt — the actual balance logic only runs when enough time
        has accumulated.
        """
        if not self._enabled:
            return

        self._tick_accumulator += dt
        tick_period = 1.0 / max(self._tick_hz, 1e-6)

        if self._tick_accumulator < tick_period:
            return

        self._tick_accumulator -= tick_period
        self._run_balance()

    def _run_balance(self) -> None:
        """Internal: execute one balance pass."""
        # 1. Acoustic energy decays rapidly
        acoustic = self._ledger.get("acoustic")
        self._ledger.consume("acoustic", acoustic * _DEFAULT_ACOUSTIC_DECAY)

        # 2. Solar dissipates gradually (re-radiated)
        solar = self._ledger.get("solar")
        self._ledger.consume("solar", solar * 0.20)

        # 3. Entropy feedback → dissipation scale
        if self._planet_entropy > self._entropy_upper:
            excess = self._planet_entropy - self._entropy_upper
            self._dissipation_scale = _clamp(1.0 + excess * 2.0, 1.0, 3.0)
            # Force mechanical stress reduction
            mech = self._ledger.get("mechanical")
            self._ledger.consume("mechanical", mech * self._normalize_k * self._dissipation_scale)
        elif self._planet_entropy < self._entropy_lower:
            self._dissipation_scale = _clamp(
                1.0 - (self._entropy_lower - self._planet_entropy), 0.2, 1.0
            )
        else:
            self._dissipation_scale = 1.0

        # 4. Clamp mechanical stress to max
        mech = self._ledger.get("mechanical")
        if mech > self._max_mech:
            excess = mech - self._max_mech
            self._ledger.consume("mechanical", excess * 0.5)
            # excess → entropy rise
            self._planet_entropy = _clamp(self._planet_entropy + excess * 0.05)

        # 5. Entropy naturally relaxes toward 0.5
        self._planet_entropy += (0.5 - self._planet_entropy) * 0.01

        # 6. Clamp entropy
        self._planet_entropy = _clamp(self._planet_entropy)

    # ------------------------------------------------------------------
    # Dissipation scale accessor
    # ------------------------------------------------------------------

    def get_dissipation_scale(self) -> float:
        """Return current dissipation scale coefficient (≥ 1.0 when entropy is high)."""
        return self._dissipation_scale

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Return a serialisable snapshot of the energy system state."""
        return {
            "type":              "GLOBAL_ENERGY_STATE_54",
            "planet_entropy":    self._planet_entropy,
            "dissipation_scale": self._dissipation_scale,
            "tick_accumulator":  self._tick_accumulator,
            "reservoirs":        self._ledger.reservoirs(),
        }

    def load_state_dict(self, d: dict) -> None:
        """Restore state from a dict produced by :meth:`state_dict`."""
        if d.get("type") != "GLOBAL_ENERGY_STATE_54":
            raise ValueError("GlobalEnergySystem.load_state_dict: unexpected type")
        self._planet_entropy    = float(d["planet_entropy"])
        self._dissipation_scale = float(d["dissipation_scale"])
        self._tick_accumulator  = float(d.get("tick_accumulator", 0.0))
        for name, val in d.get("reservoirs", {}).items():
            try:
                self._ledger.set(name, float(val))
            except KeyError:
                pass
