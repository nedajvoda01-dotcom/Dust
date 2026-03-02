"""FallbackLadders — Stage 42 §4.2  Quality fallback ladders per subsystem.

Each subsystem exposes a *ladder*: an ordered list of quality tiers from
full fidelity (tier 0) down to minimal / disabled (tier 3).

Ladders
-------
AudioLadder
  0  full_modal      — Full modal synthesis (many modes)
  1  reduced_modes   — Reduced modes
  2  noise_proxy     — Band-filtered noise proxy
  3  drop_quiet      — Drop quiet sources (silence)

DeformLadder
  0  h_and_m_field   — H + M deformation field
  1  h_field_only    — H field only
  2  material_overlay — Material overlay only
  3  no_deform       — No deformation (far/disabled)

IKLadder
  0  full_body             — Full-body constraints
  1  reduced_constraints   — Reduced constraints
  2  legs_only             — Legs-only IK
  3  stabilise_only        — Stabilise-only (minimal)

Public API
----------
AudioLadder / DeformLadder / IKLadder
  .quality_for(tier)  → str descriptor
  .max_tier           → FallbackTier (always DISABLED)
  .all_tiers()        → list[(FallbackTier, str)]
"""
from __future__ import annotations

from typing import List, Tuple

from src.runtime.BudgetManager import FallbackTier


class _BaseLadder:
    """Abstract quality ladder — subclasses fill ``_RUNGS``."""

    _RUNGS: List[Tuple[FallbackTier, str]] = []

    def quality_for(self, tier: FallbackTier) -> str:
        """Return the quality descriptor for the given tier."""
        for rung_tier, desc in reversed(self._RUNGS):
            if tier >= rung_tier:
                return desc
        return self._RUNGS[0][1]

    @property
    def max_tier(self) -> FallbackTier:
        return self._RUNGS[-1][0]

    def all_tiers(self) -> List[Tuple[FallbackTier, str]]:
        return list(self._RUNGS)


class AudioLadder(_BaseLadder):
    """Fallback ladder for procedural audio (modal synthesis)."""

    _RUNGS = [
        (FallbackTier.FULL,     "full_modal"),
        (FallbackTier.REDUCED,  "reduced_modes"),
        (FallbackTier.PROXY,    "noise_proxy"),
        (FallbackTier.DISABLED, "drop_quiet"),
    ]


class DeformLadder(_BaseLadder):
    """Fallback ladder for surface deformation."""

    _RUNGS = [
        (FallbackTier.FULL,     "h_and_m_field"),
        (FallbackTier.REDUCED,  "h_field_only"),
        (FallbackTier.PROXY,    "material_overlay"),
        (FallbackTier.DISABLED, "no_deform"),
    ]


class IKLadder(_BaseLadder):
    """Fallback ladder for inverse kinematics."""

    _RUNGS = [
        (FallbackTier.FULL,     "full_body"),
        (FallbackTier.REDUCED,  "reduced_constraints"),
        (FallbackTier.PROXY,    "legs_only"),
        (FallbackTier.DISABLED, "stabilise_only"),
    ]
