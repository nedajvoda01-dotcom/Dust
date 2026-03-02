"""FatigueReplicator — Stage 44 network replication of fatigue state.

The server is **authoritative** for the player's fatigue state.
Clients receive coarse (1–2 Hz) quantised snapshots and apply smoothing
locally.

Encoding
--------
Each snapshot is 4 bytes:

  Byte 0 : energy          uint8  (0–255 → 0.0–1.0)
  Byte 1 : tremor          uint8  (0–255 → 0.0–1.0)
  Byte 2 : coordination    uint8  (0–255 → 0.0–1.0)
  Byte 3 : reserved / checksum nibble

Public API
----------
FatigueSnapshot (dataclass)
FatigueReplicator(config=None)
  .encode(state)            → bytes (4 B)
  .decode(raw)              → FatigueSnapshot
  .apply_server_snapshot(snapshot, dt) → FatigueState (smoothed)
  .should_send(sim_time)    → bool
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Optional

from src.fatigue.FatigueSystem import FatigueState


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * _clamp(t, 0.0, 1.0)


def _to_uint8(v: float) -> int:
    return int(_clamp(v, 0.0, 1.0) * 255.0 + 0.5) & 0xFF


def _from_uint8(b: int) -> float:
    return (b & 0xFF) / 255.0


# ---------------------------------------------------------------------------
# FatigueSnapshot
# ---------------------------------------------------------------------------

@dataclass
class FatigueSnapshot:
    """Coarse fatigue snapshot received from the server.

    Attributes
    ----------
    energy :
        Server-authoritative energy [0..1].
    tremor :
        Server-authoritative tremor [0..1].
    coordination :
        Server-authoritative coordination [0..1].
    """
    energy:       float = 1.0
    tremor:       float = 0.0
    coordination: float = 1.0


# ---------------------------------------------------------------------------
# FatigueReplicator
# ---------------------------------------------------------------------------

class FatigueReplicator:
    """Handles encoding, decoding, and client-side smoothing of fatigue state.

    Parameters
    ----------
    config :
        Optional dict; reads ``fatigue.*`` keys.
    """

    _DEFAULT_REPL_HZ     = 2.0    # send rate (server → clients)
    _DEFAULT_SMOOTH_TAU  = 0.5    # client smoothing time constant [s]

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg  = config or {}
        fcfg = cfg.get("fatigue", {}) or {}

        self._repl_interval = 1.0 / max(0.1, float(fcfg.get("repl_hz", self._DEFAULT_REPL_HZ)))
        self._smooth_tau    = float(fcfg.get("repl_smooth_tau", self._DEFAULT_SMOOTH_TAU))

        self._next_send_time: float = 0.0

        # Client-side smoothed state
        self._client_state = FatigueState()

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
    def encode(state: FatigueState) -> bytes:
        """Encode authoritative fatigue state into a 4-byte wire format.

        Parameters
        ----------
        state :
            Server-authoritative :class:`FatigueState`.

        Returns
        -------
        bytes
            4 bytes: energy, tremor, coordination, reserved.
        """
        b0 = _to_uint8(state.energy)
        b1 = _to_uint8(state.tremor)
        b2 = _to_uint8(state.coordination)
        b3 = (b0 ^ b1 ^ b2) & 0xFF   # simple XOR checksum nibble
        return struct.pack("BBBB", b0, b1, b2, b3)

    @staticmethod
    def decode(raw: bytes) -> FatigueSnapshot:
        """Decode a 4-byte wire payload into a :class:`FatigueSnapshot`.

        Parameters
        ----------
        raw :
            Exactly 4 bytes as produced by :meth:`encode`.

        Returns
        -------
        FatigueSnapshot
        """
        if len(raw) < 4:
            return FatigueSnapshot()
        b0, b1, b2, b3 = struct.unpack("BBBB", raw[:4])
        # Verify checksum
        expected = (b0 ^ b1 ^ b2) & 0xFF
        if b3 != expected:
            # Corrupted packet — return neutral state
            return FatigueSnapshot()
        return FatigueSnapshot(
            energy       = _from_uint8(b0),
            tremor       = _from_uint8(b1),
            coordination = _from_uint8(b2),
        )

    # ------------------------------------------------------------------
    # Client side
    # ------------------------------------------------------------------

    def apply_server_snapshot(
        self,
        snapshot: FatigueSnapshot,
        dt:       float,
    ) -> FatigueState:
        """Apply a server snapshot and return the smoothed client state.

        Parameters
        ----------
        snapshot :
            Decoded server snapshot.
        dt :
            Elapsed time since last call [s]; used for exponential smoothing.

        Returns
        -------
        FatigueState
            Smoothed local state (used by client motor params).
        """
        import math
        alpha = 1.0 - math.exp(-dt / max(self._smooth_tau, 1e-6))
        s     = self._client_state

        self._client_state = FatigueState(
            energy             = _lerp(s.energy,      snapshot.energy,       alpha),
            neuromuscularNoise = s.neuromuscularNoise,   # not replicated; local
            coordination       = _lerp(s.coordination, snapshot.coordination, alpha),
            tremor             = _lerp(s.tremor,       snapshot.tremor,       alpha),
            gripReserve        = s.gripReserve,          # not replicated; local
            thermalLoad        = s.thermalLoad,          # not replicated; local
        )
        return self._client_state

    @property
    def client_state(self) -> FatigueState:
        """Current client-side smoothed fatigue state."""
        return self._client_state
