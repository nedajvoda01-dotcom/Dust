"""DeterminismContract — Stage 42 formal determinism specification.

This module acts both as *documentation* (the contract) and as the *runtime
enforcement* point that other systems import to verify they are operating
within the rules.

Contract summary
----------------
1. **Time model** — all deterministic systems run on fixed ticks defined by
   :attr:`TICK_RATES`.  Floating ``dt`` is forbidden inside deterministic
   systems; only ``sim_tick_dt`` (``1 / sim_tick_hz``) may be used.

2. **RNG rules** — ``random`` / ``os.urandom`` / time-seeded generators are
   forbidden.  Use :class:`src.core.DetRng.DetRng` with the canonical
   five-tuple seed.

3. **Number quantisation** — network-transmitted floats must be quantised:
   positions → fixed-point millimetres (int32), forces/directions → int16.
   Internal computations must use a consistent summation order and the shared
   :data:`EPSILON` / :func:`clamp` helpers.

4. **Deterministic ordering** — any list of events must be sorted by
   ``(tick_index, entity_id)`` before processing.  Use
   :func:`sort_events` to enforce this.

Systems covered by this contract
---------------------------------
``COVERED_SYSTEMS`` lists the system identifiers that are obligated to follow
every rule.  Registration happens at import time via
:func:`register_system`.

Violation reporting
-------------------
Violations are logged as ERRORs (dev) or raise ``DeterminismViolation`` in
test/strict mode.  Enable strict mode with
``DeterminismContract.strict_mode = True``.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

from src.core.Logger import Logger

_TAG = "DetContract"

# ---------------------------------------------------------------------------
# Contract constants
# ---------------------------------------------------------------------------

#: Fixed-tick rates (Hz) for each deterministic time domain.
TICK_RATES: Dict[str, float] = {
    "sim":        60.0,   # physics / motor / IK
    "perception": 20.0,   # PerceptionSystem
    "intent":     10.0,   # IntentArbitrator
    "social":      5.0,   # SocialCoupler
    "evolution":   1.0 / 60.0,  # LongHorizonEvolution (~1/60 Hz = once per sim-min)
}

#: Derived fixed-step dt values (seconds).
TICK_DT: Dict[str, float] = {name: 1.0 / hz for name, hz in TICK_RATES.items()}

#: Shared epsilon for float comparisons inside deterministic modules.
EPSILON: float = 1e-6

# ---------------------------------------------------------------------------
# Systems obligated by this contract
# ---------------------------------------------------------------------------

#: Canonical list of systems that must be deterministic.
COVERED_SYSTEMS: List[str] = [
    "motor",
    "ik_solver",
    "deformation",
    "audio_resonator",
    "perception",
    "micro_intent",
    "social",
    "grasp",
    "camera",
    "astro_climate",
]

# Runtime registry populated via register_system()
_registered: Dict[str, Any] = {}

# Strict mode: raise instead of just logging
strict_mode: bool = False


class DeterminismViolation(RuntimeError):
    """Raised in strict mode when a determinism rule is violated."""


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_system(system_id: str, instance: Any = None) -> None:
    """Record that *system_id* is operating under this contract."""
    if system_id not in COVERED_SYSTEMS:
        Logger.warn(_TAG, f"Unknown system '{system_id}' registered; add it to COVERED_SYSTEMS.")
    _registered[system_id] = instance
    Logger.debug(_TAG, f"System '{system_id}' registered under DeterminismContract.")


def registered_systems() -> List[str]:
    """Return list of currently registered system ids."""
    return list(_registered.keys())


# ---------------------------------------------------------------------------
# Number helpers
# ---------------------------------------------------------------------------

def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp *value* to [lo, hi].  Uses the contract's consistent ordering."""
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def quantise_position_mm(metres: float) -> int:
    """Quantise a position value in metres to millimetres (int32)."""
    return int(round(metres * 1000.0))


def dequantise_position_mm(mm: int) -> float:
    """Restore a millimetre-quantised position back to metres."""
    return mm / 1000.0


def quantise_direction_int16(component: float) -> int:
    """Quantise a unit-vector component [-1, 1] to int16 range [-32767, 32767]."""
    clamped = clamp(component, -1.0, 1.0)
    return int(round(clamped * 32767.0))


def dequantise_direction_int16(raw: int) -> float:
    """Restore a direction component from int16."""
    return clamp(raw / 32767.0, -1.0, 1.0)


def quantise_force_int16(force: float, max_force: float = 2000.0) -> int:
    """Quantise a force value to int16 relative to *max_force*."""
    normalised = clamp(force / max_force, -1.0, 1.0)
    return int(round(normalised * 32767.0))


def dequantise_force_int16(raw: int, max_force: float = 2000.0) -> float:
    """Restore a force value from int16."""
    return (raw / 32767.0) * max_force


# ---------------------------------------------------------------------------
# Deterministic ordering
# ---------------------------------------------------------------------------

def sort_events(events: List[Any], tick_key: str = "tick_index", id_key: str = "entity_id") -> List[Any]:
    """Return *events* sorted deterministically by ``(tick_index, entity_id)``.

    Events that lack the tick or id key are sorted last and then by their
    position in the original list (stable).
    """
    def _key(e: Any) -> Tuple[int, Any]:
        tick = getattr(e, tick_key, None) or (e.get(tick_key, 0) if isinstance(e, dict) else 0)
        eid  = getattr(e, id_key,   None) or (e.get(id_key,   0) if isinstance(e, dict) else 0)
        return (tick, eid)

    return sorted(events, key=_key)


# ---------------------------------------------------------------------------
# Violation reporting
# ---------------------------------------------------------------------------

def report_violation(rule: str, detail: str = "") -> None:
    """Log (or raise) a determinism contract violation."""
    msg = f"VIOLATION [{rule}]: {detail}"
    Logger.error(_TAG, msg)
    if strict_mode:
        raise DeterminismViolation(msg)


# ---------------------------------------------------------------------------
# Tick-rate validation helper
# ---------------------------------------------------------------------------

def assert_fixed_tick(domain: str, dt: float) -> None:
    """Assert that *dt* matches the expected fixed tick duration for *domain*.

    Logs a violation if the tolerance is exceeded.  Called inside
    deterministic update methods to catch floating-dt bugs.
    """
    expected = TICK_DT.get(domain)
    if expected is None:
        report_violation("unknown_domain", f"domain={domain!r} not in TICK_DT")
        return
    if not math.isclose(dt, expected, rel_tol=1e-4, abs_tol=1e-6):
        report_violation(
            "floating_dt",
            f"domain={domain!r} expected dt={expected:.6f} got dt={dt:.6f}",
        )
