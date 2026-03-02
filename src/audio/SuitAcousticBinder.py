"""SuitAcousticBinder — Stage 43 suit-to-audio parameter binding.

Links a player's :class:`~src.character.SuitKitAssembler.SuitKit` to the
generative audio system (Stage 36) by producing an
:class:`AcousticProfile` that the audio layer reads each frame.

Each suit module contributes additive parameters to the profile:
* Cloth / strap modules → friction noise weight (rustle amplitude).
* Metal / plastic modules → modal resonance frequency and decay.
* Backpack → low-frequency thump weight on impact.

The profile is cheap to compute (it is assembled once at spawn and cached
until the player's suit changes, which never happens during a session).

Public API
----------
SuitAcousticBinder()
  .bind(suit_kit) → AcousticProfile
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from src.character.SuitKitAssembler import SuitKit, SuitModule


# ---------------------------------------------------------------------------
# AcousticProfile
# ---------------------------------------------------------------------------

@dataclass
class AcousticProfile:
    """Aggregated acoustic parameters derived from a SuitKit.

    All values are [0..1] or Hz; the audio system uses them to adjust
    generative sounds without per-module branching.
    """
    friction_noise:  float = 0.0   # cloth/strap rustle amplitude [0..1]
    impact_lf:       float = 0.0   # low-freq thump on movement impact [0..1]
    modal_freqs_hz:  List[float] = field(default_factory=list)
    # Dominant modal frequency (Hz) — most prominent metallic ring
    primary_modal_hz: float = 0.0


# ---------------------------------------------------------------------------
# SuitAcousticBinder
# ---------------------------------------------------------------------------

class SuitAcousticBinder:
    """Computes an :class:`AcousticProfile` from a :class:`SuitKit`."""

    def bind(self, suit_kit: SuitKit) -> AcousticProfile:
        """Aggregate acoustic parameters across all suit modules.

        Parameters
        ----------
        suit_kit :
            Assembled suit kit (output of SuitKitAssembler.assemble).
        """
        total_friction = 0.0
        total_lf       = 0.0
        modal_freqs: List[float] = []

        for mod in suit_kit.modules:
            total_friction += mod.friction_noise_weight
            total_lf       += mod.impact_lf_weight
            if mod.modal_freq_hz > 0.0:
                modal_freqs.append(mod.modal_freq_hz)

        n = max(len(suit_kit.modules), 1)
        friction_norm = min(total_friction / n, 1.0)
        lf_norm       = min(total_lf       / n, 1.0)

        primary_hz = max(modal_freqs) if modal_freqs else 0.0

        return AcousticProfile(
            friction_noise   = friction_norm,
            impact_lf        = lf_norm,
            modal_freqs_hz   = sorted(set(modal_freqs)),
            primary_modal_hz = primary_hz,
        )
