"""FallbackLadders — Stage 42 quality-tier definitions for LOD degradation.

Each subsystem has a named ladder of quality tiers.  When the
:class:`src.runtime.BudgetManager.BudgetManager` detects that a subsystem
is over budget, it calls :meth:`FallbackLadder.degrade`.  When the subsystem
comes back within budget, :meth:`FallbackLadder.recover` steps quality back up.

Ladders are purely descriptive enumerations; the actual implementation
of each tier lives in the respective subsystem.

Pre-defined ladders
-------------------
* ``AUDIO``  — full modal → reduced modes → noise proxy → drop quiet
* ``DEFORM`` — H+M field → H only → material overlay → disabled
* ``IK``     — full-body → reduced constraints → legs only → stabilize only

Usage
-----
ladder = FallbackLadders.get("audio")
ladder.degrade()           # step down one tier
current = ladder.current   # AudioTier.REDUCED_MODES (for example)
ladder.recover()           # step back up
"""
from __future__ import annotations

from enum import IntEnum
from typing import Dict, Optional

from src.core.Logger import Logger

_TAG = "FallbackLadder"


# ---------------------------------------------------------------------------
# Tier enumerations (lower value = higher quality)
# ---------------------------------------------------------------------------

class AudioTier(IntEnum):
    FULL_MODAL     = 0   # full modal synthesis, many modes
    REDUCED_MODES  = 1   # fewer modes per resonator
    NOISE_PROXY    = 2   # broadband noise proxy
    DROP_QUIET     = 3   # drop below-threshold sources entirely


class DeformTier(IntEnum):
    FULL_HM         = 0  # H-field + M-field active
    H_ONLY          = 1  # only H (coarse) field
    MATERIAL_OVERLAY = 2  # only material-type overlay
    DISABLED        = 3  # no deformation near player


class IKTier(IntEnum):
    FULL_BODY        = 0  # all constraints + full-body solve
    REDUCED          = 1  # reduced constraint set
    LEGS_ONLY        = 2  # only lower-limb IK
    STABILIZE_ONLY   = 3  # ground-contact stabilisation only


# ---------------------------------------------------------------------------
# Generic ladder
# ---------------------------------------------------------------------------

class FallbackLadder:
    """A quality ladder with degrade / recover step logic.

    Parameters
    ----------
    name:       Human-readable identifier (e.g. ``"audio"``, ``"ik"``).
    tier_count: Number of quality tiers (0 = best, tier_count-1 = worst).
    """

    def __init__(self, name: str, tier_count: int) -> None:
        self._name = name
        self._max_tier = tier_count - 1
        self._current = 0

    @property
    def current(self) -> int:
        """Current tier index (0 = best quality)."""
        return self._current

    @property
    def is_degraded(self) -> bool:
        """True when not at the best-quality tier."""
        return self._current > 0

    @property
    def is_worst(self) -> bool:
        """True when at the lowest quality tier."""
        return self._current >= self._max_tier

    def degrade(self) -> bool:
        """Step down one tier.  Returns True if tier changed."""
        if self._current < self._max_tier:
            self._current += 1
            Logger.warn(_TAG, f"[{self._name}] degraded to tier {self._current}")
            return True
        return False

    def recover(self) -> bool:
        """Step up one tier.  Returns True if tier changed."""
        if self._current > 0:
            self._current -= 1
            Logger.info(_TAG, f"[{self._name}] recovered to tier {self._current}")
            return True
        return False

    def reset(self) -> None:
        """Return to the highest quality tier immediately."""
        if self._current != 0:
            Logger.info(_TAG, f"[{self._name}] reset to tier 0")
        self._current = 0


# ---------------------------------------------------------------------------
# Typed sub-classes (convenience)
# ---------------------------------------------------------------------------

class AudioFallbackLadder(FallbackLadder):
    def __init__(self) -> None:
        super().__init__("audio", len(AudioTier))

    @property
    def audio_tier(self) -> AudioTier:
        return AudioTier(self.current)


class DeformFallbackLadder(FallbackLadder):
    def __init__(self) -> None:
        super().__init__("deform", len(DeformTier))

    @property
    def deform_tier(self) -> DeformTier:
        return DeformTier(self.current)


class IKFallbackLadder(FallbackLadder):
    def __init__(self) -> None:
        super().__init__("ik", len(IKTier))

    @property
    def ik_tier(self) -> IKTier:
        return IKTier(self.current)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, FallbackLadder] = {}


def _build_defaults() -> None:
    _REGISTRY["audio"]  = AudioFallbackLadder()
    _REGISTRY["deform"] = DeformFallbackLadder()
    _REGISTRY["ik"]     = IKFallbackLadder()


_build_defaults()


class FallbackLadders:
    """Global accessor for named fallback ladders."""

    @staticmethod
    def get(name: str) -> Optional[FallbackLadder]:
        """Return the ladder for *name*, or None if not found."""
        return _REGISTRY.get(name)

    @staticmethod
    def register(name: str, ladder: FallbackLadder) -> None:
        """Register a custom ladder under *name*."""
        _REGISTRY[name] = ladder

    @staticmethod
    def all_ladders() -> Dict[str, FallbackLadder]:
        """Return a copy of the full registry."""
        return dict(_REGISTRY)

    @staticmethod
    def reset_all() -> None:
        """Reset all ladders to best quality."""
        for ladder in _REGISTRY.values():
            ladder.reset()
