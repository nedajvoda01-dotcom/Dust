"""InputSender — Stage 58 client-side input frame sender.

Packages raw input (movement direction, speed intent, look yaw/pitch) into
compact ``InputFrame`` dicts and exposes a ready-to-send queue that the
network layer drains at ``net.input_hz``.

Quantisation follows the Stage 43/47 convention so that server-side
``InputReceiver`` can reconstruct the same values deterministically.

Public API
----------
InputFrame
    Dataclass representing one input snapshot.

quantise_dir(x, z, bits) → (int, int)
    Quantise a 2-D direction to ``bits``-bit fixed-point integers.

dequantise_dir(qx, qz, bits) → (float, float)
    Inverse of quantise_dir.

quantise_angle(radians, bits) → int
    Map an angle to a fixed-point integer (full-circle).

dequantise_angle(q, bits) → float
    Inverse of quantise_angle.

InputSender(config)
    .apply_input(move_x, move_z, speed_intent, yaw_rad, pitch_rad)
        Record the latest raw input; call every frame.
    .tick(now_s) → InputFrame | None
        Advance internal clock; returns a ready InputFrame at the configured
        rate, or *None* if it is not yet time.
    .pending() → InputFrame | None
        Return the last frame produced by tick() without consuming it.
    .ack(last_processed_seq)
        Discard input history up to and including *last_processed_seq*.
    .history_len() → int
        Number of unacknowledged frames in the history buffer.
"""
from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Quantisation helpers
# ---------------------------------------------------------------------------

def quantise_dir(x: float, z: float, bits: int = 8) -> Tuple[int, int]:
    """Map a 2-D direction component pair to ``bits``-bit signed integers.

    Each component is clamped to [-1, 1] then mapped to [-127, 127] for
    8-bit (adjusting for the chosen *bits*).
    """
    half = (1 << (bits - 1)) - 1  # e.g. 127 for bits=8
    qx = int(round(max(-1.0, min(float(x), 1.0)) * half))
    qz = int(round(max(-1.0, min(float(z), 1.0)) * half))
    return (qx, qz)


def dequantise_dir(qx: int, qz: int, bits: int = 8) -> Tuple[float, float]:
    """Inverse of :func:`quantise_dir`."""
    half = (1 << (bits - 1)) - 1
    return (qx / half, qz / half)


def quantise_angle(radians: float, bits: int = 12) -> int:
    """Map an angle (radians) to a *bits*-bit unsigned integer (full circle)."""
    normalised = (float(radians) % (2.0 * math.pi)) / (2.0 * math.pi)
    scale = (1 << bits) - 1
    return int(round(normalised * scale)) & scale


def dequantise_angle(q: int, bits: int = 12) -> float:
    """Map a quantised angle back to radians (0 … 2π)."""
    scale = (1 << bits) - 1
    return (q / scale) * 2.0 * math.pi


# ---------------------------------------------------------------------------
# InputFrame
# ---------------------------------------------------------------------------

@dataclass
class InputFrame:
    """One input snapshot sent from client to server.

    Fields
    ------
    sequence_id : int
        Monotonically increasing per-client identifier.
    client_tick : int
        Client-side tick counter when this frame was captured.
    move_dir_qx : int
        Quantised (8-bit) local move direction X.
    move_dir_qz : int
        Quantised (8-bit) local move direction Z.
    speed_intent : float
        Desired speed fraction [0..1].
    look_yaw_q : int
        Quantised (12-bit) yaw angle.
    look_pitch_q : int
        Quantised (12-bit) pitch angle.
    """
    sequence_id:   int
    client_tick:   int
    move_dir_qx:   int
    move_dir_qz:   int
    speed_intent:  float
    look_yaw_q:    int
    look_pitch_q:  int

    def to_dict(self) -> dict:
        return {
            "seq":          self.sequence_id,
            "cTick":        self.client_tick,
            "mvX":          self.move_dir_qx,
            "mvZ":          self.move_dir_qz,
            "spd":          round(self.speed_intent, 4),
            "yaw":          self.look_yaw_q,
            "pitch":        self.look_pitch_q,
        }

    @staticmethod
    def from_dict(d: dict) -> "InputFrame":
        return InputFrame(
            sequence_id  = int(d["seq"]),
            client_tick  = int(d["cTick"]),
            move_dir_qx  = int(d["mvX"]),
            move_dir_qz  = int(d["mvZ"]),
            speed_intent = float(d["spd"]),
            look_yaw_q   = int(d["yaw"]),
            look_pitch_q = int(d["pitch"]),
        )


# ---------------------------------------------------------------------------
# InputSender
# ---------------------------------------------------------------------------

class InputSender:
    """Client-side input frame packetiser.

    Collects raw input, packs it into :class:`InputFrame` objects at the
    configured send rate, and maintains an unacknowledged history buffer so
    that the reconciliation layer can re-simulate missed frames.

    Parameters
    ----------
    config : dict
        Full game config dict; reads ``net.input_hz`` (default 30).
    """

    _HISTORY_MAX = 256  # hard cap on unacknowledged frames kept in memory

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = (config or {}).get("net", {}) or {}
        self._send_hz: float = float(cfg.get("input_hz", 30))
        self._send_interval: float = 1.0 / max(1.0, self._send_hz)

        self._seq:          int   = 0
        self._client_tick:  int   = 0
        self._last_send_t:  float = 0.0

        # Latest raw input
        self._move_x:      float = 0.0
        self._move_z:      float = 0.0
        self._speed:       float = 0.0
        self._yaw_rad:     float = 0.0
        self._pitch_rad:   float = 0.0

        # Unacknowledged frame history: seq_id → InputFrame
        self._history: OrderedDict[int, InputFrame] = OrderedDict()
        self._last_frame: Optional[InputFrame] = None

    # ------------------------------------------------------------------
    # Input accumulation
    # ------------------------------------------------------------------

    def apply_input(
        self,
        move_x:      float,
        move_z:      float,
        speed_intent: float,
        yaw_rad:     float,
        pitch_rad:   float,
    ) -> None:
        """Record the latest raw input from game logic or the player."""
        self._move_x    = float(move_x)
        self._move_z    = float(move_z)
        self._speed     = max(0.0, min(float(speed_intent), 1.0))
        self._yaw_rad   = float(yaw_rad)
        self._pitch_rad = float(pitch_rad)

    # ------------------------------------------------------------------
    # Tick / frame production
    # ------------------------------------------------------------------

    def tick(self, now_s: float) -> Optional[InputFrame]:
        """Advance the sender clock; return a new :class:`InputFrame` when due.

        Parameters
        ----------
        now_s : float
            Current time in seconds (monotonic).

        Returns
        -------
        InputFrame or None
        """
        self._client_tick += 1
        if now_s - self._last_send_t < self._send_interval - 1e-9:
            return None

        self._last_send_t = now_s
        self._seq += 1

        qx, qz = quantise_dir(self._move_x, self._move_z)
        yaw_q  = quantise_angle(self._yaw_rad)
        pitch_q = quantise_angle(self._pitch_rad)

        frame = InputFrame(
            sequence_id  = self._seq,
            client_tick  = self._client_tick,
            move_dir_qx  = qx,
            move_dir_qz  = qz,
            speed_intent = self._speed,
            look_yaw_q   = yaw_q,
            look_pitch_q = pitch_q,
        )

        # Maintain history (trim if at cap)
        self._history[self._seq] = frame
        if len(self._history) > self._HISTORY_MAX:
            self._history.popitem(last=False)

        self._last_frame = frame
        return frame

    def pending(self) -> Optional[InputFrame]:
        """Return the most recently produced frame without consuming it."""
        return self._last_frame

    # ------------------------------------------------------------------
    # Acknowledgement / history management
    # ------------------------------------------------------------------

    def ack(self, last_processed_seq: int) -> None:
        """Discard history for frames already processed by the server.

        Parameters
        ----------
        last_processed_seq : int
            The server's most recently acknowledged input sequence id.
        """
        to_remove = [s for s in self._history if s <= last_processed_seq]
        for s in to_remove:
            del self._history[s]

    def history_len(self) -> int:
        """Number of unacknowledged frames held in the history buffer."""
        return len(self._history)

    def get_history(self) -> Dict[int, InputFrame]:
        """Return a snapshot of the current unacknowledged history."""
        return dict(self._history)
