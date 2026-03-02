"""DeterminismContract — Stage 42 §2.

Centralises all determinism rules:

* canonical tick-rate constants
* RNG rules (forbidden / allowed)
* fixed-point quantisation helpers
* deterministic event-ordering key

Tick-rate model
---------------
All simulation subsystems run at **fixed ticks** derived from these constants.
Floating-point ``dt`` is forbidden inside deterministic systems; use tick
counters instead.

    SIM_TICK_HZ         = 60   # physics / motor / IK
    PERCEPTION_TICK_HZ  = 20   # perception / micro-intent
    INTENT_TICK_HZ      = 10   # intent planner
    SOCIAL_TICK_HZ      = 5    # social negotiation
    EVOLUTION_TICK_HZ   = 1    # long-horizon evolution (Stage 30)

RNG rules
---------
* **Forbidden**: ``random.random()``, ``time``-seeded generators, any entropy
  source that may vary between runs.
* **Required**: :class:`~src.core.DetRng.DetRng` with a fully-qualified seed
  ``hash(worldSeed, playerId, systemId, tickIndex, regionId)``.

Quantisation
------------
Network-transmitted values must be quantised before hashing / comparison:

* positions  → **int32 mm**  (millimetre fixed-point)
* directions → **int16**     (scaled to ±32 767)
* forces     → **int16 dN**  (deci-Newtons, ±32 767)

Ordering
--------
Event lists processed inside deterministic systems **must** be sorted by
``(tick_index, entity_id)`` before iteration.

Public API
----------
DeterminismContract          — constants + static helpers (no state)
  .SIM_TICK_HZ               — int
  .PERCEPTION_TICK_HZ        — int
  .INTENT_TICK_HZ            — int
  .SOCIAL_TICK_HZ            — int
  .EVOLUTION_TICK_HZ         — int
  .quantise_pos_mm(v)        — float → int (millimetres)
  .dequantise_pos_mm(i)      — int → float (metres)
  .quantise_dir_i16(v)       — float [-1,1] → int [-32767,32767]
  .dequantise_dir_i16(i)     — int → float
  .quantise_force_dn(v)      — float (N) → int (deci-Newtons)
  .dequantise_force_dn(i)    — int → float (N)
  .event_sort_key(e)         — (tick_index, entity_id) tuple for sorting
  .sort_events(events)       — sorted list
"""
from __future__ import annotations

from typing import Any, List


class DeterminismContract:
    """Static determinism rules and quantisation helpers.

    All members are class-level constants or static methods; instantiation
    is never needed but harmless.
    """

    # ------------------------------------------------------------------
    # Tick rates (Hz)
    # ------------------------------------------------------------------

    SIM_TICK_HZ: int = 60
    """Physics / motor / IK subsystems run at this rate."""

    PERCEPTION_TICK_HZ: int = 20
    """Perception and micro-intent update rate."""

    INTENT_TICK_HZ: int = 10
    """Intent-planner update rate."""

    SOCIAL_TICK_HZ: int = 5
    """Social-negotiation update rate."""

    EVOLUTION_TICK_HZ: int = 1
    """Long-horizon evolution update rate (Stage 30)."""

    # ------------------------------------------------------------------
    # Derived tick durations (seconds)
    # ------------------------------------------------------------------

    SIM_TICK_DT: float = 1.0 / SIM_TICK_HZ
    PERCEPTION_TICK_DT: float = 1.0 / PERCEPTION_TICK_HZ
    INTENT_TICK_DT: float = 1.0 / INTENT_TICK_HZ
    SOCIAL_TICK_DT: float = 1.0 / SOCIAL_TICK_HZ
    EVOLUTION_TICK_DT: float = 1.0 / EVOLUTION_TICK_HZ

    # ------------------------------------------------------------------
    # Quantisation: position (mm fixed-point)
    # ------------------------------------------------------------------

    _POS_SCALE: int = 1000  # 1 metre = 1000 mm

    @staticmethod
    def quantise_pos_mm(metres: float) -> int:
        """Convert position in metres to integer millimetres."""
        return int(round(metres * DeterminismContract._POS_SCALE))

    @staticmethod
    def dequantise_pos_mm(mm: int) -> float:
        """Convert integer millimetres back to metres."""
        return mm / DeterminismContract._POS_SCALE

    # ------------------------------------------------------------------
    # Quantisation: direction (int16, ±32 767)
    # ------------------------------------------------------------------

    _DIR_SCALE: int = 32767

    @staticmethod
    def quantise_dir_i16(v: float) -> int:
        """Quantise a scalar direction component in [-1, 1] to int16 range."""
        clamped = max(-1.0, min(1.0, v))
        return int(round(clamped * DeterminismContract._DIR_SCALE))

    @staticmethod
    def dequantise_dir_i16(i: int) -> float:
        """Restore direction component from int16 representation."""
        return max(-1.0, min(1.0, i / DeterminismContract._DIR_SCALE))

    # ------------------------------------------------------------------
    # Quantisation: force (deci-Newtons, int16)
    # ------------------------------------------------------------------

    _FORCE_SCALE: float = 10.0  # 1 N = 10 dN

    @staticmethod
    def quantise_force_dn(newtons: float) -> int:
        """Convert force in Newtons to integer deci-Newtons, clamped to int16."""
        raw = int(round(newtons * DeterminismContract._FORCE_SCALE))
        return max(-32767, min(32767, raw))

    @staticmethod
    def dequantise_force_dn(dn: int) -> float:
        """Convert deci-Newton integer back to Newtons."""
        return dn / DeterminismContract._FORCE_SCALE

    # ------------------------------------------------------------------
    # Deterministic event ordering
    # ------------------------------------------------------------------

    @staticmethod
    def event_sort_key(event: Any) -> tuple:
        """Return ``(tick_index, entity_id)`` sort key for an event dict/object.

        Accepts both attribute-style objects and dict-style mappings.
        Missing keys default to 0.
        """
        if isinstance(event, dict):
            return (int(event.get("tick_index", 0)), int(event.get("entity_id", 0)))
        return (int(getattr(event, "tick_index", 0)), int(getattr(event, "entity_id", 0)))

    @staticmethod
    def sort_events(events: List[Any]) -> List[Any]:
        """Return a new list sorted by ``(tick_index, entity_id)``."""
        return sorted(events, key=DeterminismContract.event_sort_key)

    # ------------------------------------------------------------------
    # Shared epsilon / clamp
    # ------------------------------------------------------------------

    EPSILON: float = 1e-7
    """Minimum meaningful float difference inside deterministic modules."""
