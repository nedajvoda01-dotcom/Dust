"""SocialNetInputs — Stage 39 replicated remote-agent state for SocialCoupler.

Provides a richer description of other players than the bare
:class:`~src.perception.PresenceField.OtherPlayerState` used by Stage 37.

Also supplies a compact encode/decode codec (2-byte payload) for network
replication of the social-relevant flags.

Public API
----------
SocialAgentState
  .position      Vec3
  .velocity      Vec3
  .is_slipping   bool
  .is_stumbling  bool
  .global_risk   float  [0..1]

SocialNetInputs.encode_flags(is_slipping, is_stumbling, global_risk) → bytes (2)
SocialNetInputs.decode_flags(data) → (is_slipping, is_stumbling, global_risk)
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Tuple

from src.math.Vec3 import Vec3


@dataclass
class SocialAgentState:
    """Replicated state of a remote player consumed by SocialCoupler (§6)."""
    position:    Vec3  = field(default_factory=Vec3.zero)
    velocity:    Vec3  = field(default_factory=Vec3.zero)
    is_slipping: bool  = False
    is_stumbling: bool = False
    global_risk: float = 0.0   # [0..1] — how unsafe the other player is


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


class SocialNetInputs:
    """Encode / decode social-relevant flags for network replication.

    Payload format (2 bytes):

    * Byte 0 — flag bits: bit0=is_slipping, bit1=is_stumbling
    * Byte 1 — global_risk quantised to uint8 (0..255 → 0.0..1.0)
    """

    _PAYLOAD_SIZE = 2

    @staticmethod
    def encode_flags(
        is_slipping:  bool,
        is_stumbling: bool,
        global_risk:  float = 0.0,
    ) -> bytes:
        """Encode social flags into 2 bytes.

        Parameters
        ----------
        is_slipping :
            Player is currently slipping.
        is_stumbling :
            Player is currently stumbling.
        global_risk :
            Aggregated personal risk [0..1].
        """
        flags = (0x01 if is_slipping else 0x00) | (0x02 if is_stumbling else 0x00)
        risk_q = int(_clamp(global_risk, 0.0, 1.0) * 255.0 + 0.5)
        return struct.pack("BB", flags, risk_q)

    @staticmethod
    def decode_flags(data: bytes) -> Tuple[bool, bool, float]:
        """Decode 2-byte payload back to social flags.

        Returns
        -------
        (is_slipping, is_stumbling, global_risk)

        Raises
        ------
        ValueError
            If *data* is not exactly 2 bytes.
        """
        if len(data) != SocialNetInputs._PAYLOAD_SIZE:
            raise ValueError(
                f"SocialNetInputs: expected {SocialNetInputs._PAYLOAD_SIZE} bytes, "
                f"got {len(data)}"
            )
        flags, risk_q = struct.unpack("BB", data)
        return (
            bool(flags & 0x01),
            bool(flags & 0x02),
            risk_q / 255.0,
        )
