"""InjuryReplicator — Stage 48 network replication of injury state.

The server is **authoritative** for the player's injury state.
Clients receive coarse (1–2 Hz) quantised snapshots and apply the values
directly (injury state is already a slowly-changing signal).

Encoding (compact wire format)
-------------------------------
Each snapshot encodes the 13-joint injury state as 2 bytes per joint:
  Byte 0 : strain   uint8 (0–255 → 0.0–1.0)
  Byte 1 : acute    uint8 (0–255 → 0.0–1.0)

Total: 26 bytes for 13 joints + 1 byte globalInjuryIndex + 1 byte checksum
     = 28 bytes per snapshot.

Public API
----------
InjurySnapshot (dataclass)
InjuryReplicator(config=None)
  .encode(state)                → bytes (28 B)
  .decode(raw)                  → InjurySnapshot
  .should_send(sim_time)        → bool
  .apply_server_snapshot(snap)  → InjuryState
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Dict, Optional

from src.injury.InjurySystem import InjuryState, JointInjury, JOINT_NAMES


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


def _to_uint8(v: float) -> int:
    return int(_clamp(v, 0.0, 1.0) * 255.0 + 0.5) & 0xFF


def _from_uint8(b: int) -> float:
    return (b & 0xFF) / 255.0


# ---------------------------------------------------------------------------
# InjurySnapshot — decoded wire payload
# ---------------------------------------------------------------------------

@dataclass
class InjurySnapshot:
    """Coarse injury snapshot received from the server.

    Attributes
    ----------
    joint_strain :
        Per-joint strain, keyed by joint name [0..1].
    joint_acute :
        Per-joint acute, keyed by joint name [0..1].
    globalInjuryIndex :
        Aggregate injury index [0..1].
    """
    joint_strain:      Dict[str, float] = field(
        default_factory=lambda: {n: 0.0 for n in JOINT_NAMES}
    )
    joint_acute:       Dict[str, float] = field(
        default_factory=lambda: {n: 0.0 for n in JOINT_NAMES}
    )
    globalInjuryIndex: float = 0.0


# ---------------------------------------------------------------------------
# InjuryReplicator
# ---------------------------------------------------------------------------

class InjuryReplicator:
    """Handles encoding, decoding, and replication of injury state.

    Parameters
    ----------
    config :
        Optional dict; reads ``injury.*`` keys.
    """

    _N_JOINTS         = len(JOINT_NAMES)   # 13
    # Layout: 2 bytes/joint × 13 joints + 1 byte globalIdx + 1 byte checksum
    _PACKET_SIZE      = _N_JOINTS * 2 + 2  # 28 bytes

    _DEFAULT_REPL_HZ  = 2.0

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg  = config or {}
        icfg = cfg.get("injury", {}) or {}

        repl_hz = float(icfg.get("repl_hz", self._DEFAULT_REPL_HZ))
        self._repl_interval = 1.0 / max(0.1, repl_hz)
        self._next_send_time: float = 0.0

    # ------------------------------------------------------------------
    # Server side
    # ------------------------------------------------------------------

    def should_send(self, sim_time: float) -> bool:
        """Return True when the server should send a snapshot this tick."""
        if sim_time >= self._next_send_time:
            self._next_send_time = sim_time + self._repl_interval
            return True
        return False

    @staticmethod
    def encode(state: InjuryState) -> bytes:
        """Encode authoritative injury state into a compact wire format.

        Parameters
        ----------
        state :
            Server-authoritative :class:`InjuryState`.

        Returns
        -------
        bytes
            28 bytes: 2 per joint (strain+acute), 1 globalIdx, 1 checksum.
        """
        data: list[int] = []
        checksum = 0
        for name in JOINT_NAMES:
            j  = state.joints[name]
            b0 = _to_uint8(j.strain)
            b1 = _to_uint8(j.acute)
            data.append(b0)
            data.append(b1)
            checksum ^= b0 ^ b1

        g_byte = _to_uint8(state.globalInjuryIndex)
        checksum ^= g_byte
        data.append(g_byte)
        data.append(checksum & 0xFF)

        return struct.pack(f"{len(data)}B", *data)

    @staticmethod
    def decode(raw: bytes) -> InjurySnapshot:
        """Decode a wire payload into an :class:`InjurySnapshot`.

        Parameters
        ----------
        raw :
            Bytes as produced by :meth:`encode`.

        Returns
        -------
        InjurySnapshot
            Neutral snapshot on checksum failure or short packet.
        """
        expected_size = len(JOINT_NAMES) * 2 + 2
        if len(raw) < expected_size:
            return InjurySnapshot()

        values = struct.unpack(f"{expected_size}B", raw[:expected_size])
        # Verify checksum
        checksum = 0
        for b in values[:-1]:
            checksum ^= b
        if (checksum & 0xFF) != values[-1]:
            return InjurySnapshot()

        joint_strain: Dict[str, float] = {}
        joint_acute:  Dict[str, float] = {}
        idx = 0
        for name in JOINT_NAMES:
            joint_strain[name] = _from_uint8(values[idx])
            joint_acute[name]  = _from_uint8(values[idx + 1])
            idx += 2

        global_idx = _from_uint8(values[idx])
        return InjurySnapshot(
            joint_strain=joint_strain,
            joint_acute=joint_acute,
            globalInjuryIndex=global_idx,
        )

    # ------------------------------------------------------------------
    # Client side
    # ------------------------------------------------------------------

    @staticmethod
    def apply_server_snapshot(snapshot: InjurySnapshot) -> InjuryState:
        """Reconstruct an :class:`InjuryState` from a server snapshot.

        Parameters
        ----------
        snapshot :
            Decoded server snapshot.

        Returns
        -------
        InjuryState
            Reconstructed injury state (painAvoidance inferred from strain).
        """
        joints: Dict[str, JointInjury] = {}
        for name in JOINT_NAMES:
            strain = snapshot.joint_strain.get(name, 0.0)
            acute  = snapshot.joint_acute.get(name, 0.0)
            joints[name] = JointInjury(
                strain=strain,
                acute=acute,
                # painAvoidance not transmitted; infer from strain+acute
                painAvoidance=_clamp(strain + acute * 0.5, 0.0, 1.0),
                recoveryRate=0.0,
            )
        return InjuryState(
            joints=joints,
            globalInjuryIndex=snapshot.globalInjuryIndex,
        )
