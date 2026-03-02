"""PerceptionNetInputs — Stage 37 decoder for replicated motor state flags.

Translates the compact ``motorFlags`` bitmask (carried in PLAYER_STATE
network messages) into typed booleans used by :class:`PresenceField`.

Bit layout of ``motorFlags`` (uint8)
-------------------------------------
Bit 0  — is_slipping   (character in SLIDING / STUMBLING state)
Bit 1  — is_stumbling  (CharacterState.STUMBLING)
Bit 2  — support_ok    (at least one confirmed ground contact)
Bits 3-7 — reserved

Public API
----------
PerceptionNetInputs
  .decode_flags(motor_flags: int) → MotorFlagsDecoded
  .encode_flags(is_slipping, is_stumbling, support_ok) → int
"""
from __future__ import annotations

from dataclasses import dataclass


# Bit positions
_BIT_SLIPPING   = 0
_BIT_STUMBLING  = 1
_BIT_SUPPORT_OK = 2


@dataclass(frozen=True)
class MotorFlagsDecoded:
    """Decoded motor state flags for one remote player.

    Attributes
    ----------
    is_slipping :
        Character is in SLIDING or STUMBLING state.
    is_stumbling :
        Character is specifically in STUMBLING state.
    support_ok :
        At least one ground support contact is confirmed.
    """
    is_slipping:  bool
    is_stumbling: bool
    support_ok:   bool


class PerceptionNetInputs:
    """Encoder / decoder for the compact ``motorFlags`` bitmask."""

    @staticmethod
    def decode_flags(motor_flags: int) -> MotorFlagsDecoded:
        """Decode a ``motorFlags`` bitmask into typed booleans.

        Parameters
        ----------
        motor_flags :
            Raw uint8 bitmask from the network message.
        """
        flags = int(motor_flags) & 0xFF
        return MotorFlagsDecoded(
            is_slipping  = bool(flags & (1 << _BIT_SLIPPING)),
            is_stumbling = bool(flags & (1 << _BIT_STUMBLING)),
            support_ok   = bool(flags & (1 << _BIT_SUPPORT_OK)),
        )

    @staticmethod
    def encode_flags(
        is_slipping:  bool = False,
        is_stumbling: bool = False,
        support_ok:   bool = True,
    ) -> int:
        """Encode motor state booleans into a compact bitmask.

        Parameters
        ----------
        is_slipping :
            True when SLIDING or STUMBLING.
        is_stumbling :
            True specifically when STUMBLING.
        support_ok :
            True when ground contact confirmed.

        Returns
        -------
        int
            uint8 bitmask suitable for inclusion in a PLAYER_STATE message.
        """
        flags = 0
        if is_slipping:
            flags |= 1 << _BIT_SLIPPING
        if is_stumbling:
            flags |= 1 << _BIT_STUMBLING
        if support_ok:
            flags |= 1 << _BIT_SUPPORT_OK
        return flags
