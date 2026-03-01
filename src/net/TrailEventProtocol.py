"""TrailEventProtocol — Stage 25 network encoding for diegetic trail events.

Trail events are generated on the sender's client and relayed by the server
only to players in adjacent sectors (interest-based).  They are cheap: each
event is a small fixed-schema dict rather than a mesh or texture.

Message format  (JSON, Server→Client)
--------------------------------------
::

    {
      "type":     "TRAIL_EVENT",
      "events": [
        {
          "type":     "footprint" | "slide" | "dustpuff",
          "pos":      [x, y, z],        // fixed-point int32 encoded as float
          "dir":      [dx, dy, dz],     // quantised to 1/255 per axis
          "strength": 0..127,           // int8 mapped to 0..1
          "material": "Dust" | "LooseDebris" | "IceFilm" | "Rock",
          "tick":     <int>             // server tick index for ordering
        },
        ...
      ],
      "playerId": "<sender id>"
    }

Client→Server
-------------
::

    {
      "type":   "TRAIL_BATCH",
      "events": [ <same schema as above> ]
    }

Interest management
-------------------
The server relays a batch only to clients in the same or adjacent sectors as
the originating player.  Use :func:`should_relay` to test this condition.

Batching
--------
:class:`TrailBatchAccumulator` aggregates outgoing events on the sender side
within a configurable time window (default 300 ms) before flushing them as
one ``TRAIL_BATCH`` message.

Public API
----------
encode_trail_event(trail_type, pos, direction, strength, material, tick)
  → dict

decode_trail_events(msg) → list[dict]
  Decode a ``TRAIL_EVENT`` or ``TRAIL_BATCH`` server/client message.

should_relay(sender_pos, recipient_pos, sector_deg) → bool
  Interest-management test: True when the recipient is within adjacent sectors.

TrailBatchAccumulator(batch_ms, max_batch_size)
  .add(trail_type, pos, direction, strength, material, tick)
  .flush() → dict | None  — returns a TRAIL_BATCH msg or None if empty
  .maybe_flush(now_ms) → dict | None  — flush if batch window elapsed
"""
from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def _quantise_dir(d: List[float]) -> List[float]:
    """Clamp direction vector components to [-1, 1] and round to 1/127."""
    result = []
    for v in d:
        clamped = max(-1.0, min(float(v), 1.0))
        result.append(round(clamped * 127.0) / 127.0)
    return result


def _encode_strength(strength: float) -> int:
    """Map [0..1] float to int8 range [0..127]."""
    return int(max(0.0, min(float(strength), 1.0)) * 127.0)


def _decode_strength(value: int) -> float:
    """Map int8 [0..127] back to [0..1] float."""
    return max(0.0, min(int(value), 127)) / 127.0


# ---------------------------------------------------------------------------
# Public encoding / decoding
# ---------------------------------------------------------------------------

def encode_trail_event(
    trail_type: str,
    pos:        List[float],
    direction:  List[float],
    strength:   float,
    material:   str,
    tick:       int = 0,
) -> Dict[str, Any]:
    """Encode one trail occurrence into a compact network dict.

    Parameters
    ----------
    trail_type : str
        ``"footprint"``, ``"slide"``, or ``"dustpuff"``.
    pos : list[float]
        World-space [x, y, z].
    direction : list[float]
        Normalised movement direction [dx, dy, dz].
    strength : float
        Visual intensity [0..1].
    material : str
        One of the :class:`~src.systems.DiegeticMultiplayerPresence.MaterialClass`
        string values.
    tick : int
        Server tick index for ordering.

    Returns
    -------
    dict
        Compact event dict suitable for inclusion in a ``TRAIL_BATCH``.
    """
    return {
        "type":     trail_type,
        "pos":      [float(pos[0]), float(pos[1]), float(pos[2])],
        "dir":      _quantise_dir(list(direction)),
        "strength": _encode_strength(strength),
        "material": material,
        "tick":     int(tick),
    }


def decode_trail_events(msg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Decode a ``TRAIL_EVENT`` or ``TRAIL_BATCH`` message into event dicts.

    Strength values are converted back from int8 to float.
    Unknown/malformed events are skipped.

    Parameters
    ----------
    msg : dict
        Parsed JSON message with ``"type"`` of ``"TRAIL_EVENT"`` or
        ``"TRAIL_BATCH"`` and an ``"events"`` list.

    Returns
    -------
    list[dict]
        Each dict has keys: ``type``, ``pos``, ``dir``, ``strength``
        (float 0..1), ``material``, ``tick``.
    """
    raw_events = msg.get("events", [])
    result: List[Dict[str, Any]] = []
    for ev in raw_events:
        try:
            decoded = {
                "type":     str(ev["type"]),
                "pos":      [float(v) for v in ev["pos"]],
                "dir":      [float(v) for v in ev.get("dir", [0.0, 0.0, 1.0])],
                "strength": _decode_strength(ev.get("strength", 64)),
                "material": str(ev.get("material", "Dust")),
                "tick":     int(ev.get("tick", 0)),
            }
            result.append(decoded)
        except (KeyError, TypeError, ValueError):
            continue
    return result


# ---------------------------------------------------------------------------
# Interest management
# ---------------------------------------------------------------------------

def should_relay(
    sender_pos:    List[float],
    recipient_pos: List[float],
    sector_deg:    float = 5.0,
    sector_radius: int   = 2,
) -> bool:
    """Return True when *recipient_pos* is within the interest zone of *sender_pos*.

    Uses the same angular-sector model as :class:`~src.net.PlayerRegistry`
    interest management.

    Parameters
    ----------
    sender_pos, recipient_pos : list[float]
        World-space [x, y, z] positions.
    sector_deg : float
        Angular width of one sector tile (degrees).
    sector_radius : int
        Number of adjacent sector rings included in the subscription.
    """
    if len(sender_pos) < 3 or len(recipient_pos) < 3:
        return True

    sx, sy, sz = sender_pos[0],    sender_pos[1],    sender_pos[2]
    rx, ry, rz = recipient_pos[0], recipient_pos[1], recipient_pos[2]

    sl = math.sqrt(sx * sx + sy * sy + sz * sz)
    rl = math.sqrt(rx * rx + ry * ry + rz * rz)
    if sl < 1e-9 or rl < 1e-9:
        return True

    # Angular distance between the two positions
    dot = (sx * rx + sy * ry + sz * rz) / (sl * rl)
    dot = max(-1.0, min(dot, 1.0))
    angle_deg = math.degrees(math.acos(dot))

    max_deg = sector_deg * (sector_radius + 0.5)  # +0.5 for boundary overlap
    return angle_deg <= max_deg


# ---------------------------------------------------------------------------
# TrailBatchAccumulator
# ---------------------------------------------------------------------------

class TrailBatchAccumulator:
    """Accumulates outgoing trail events and flushes them in batches.

    Reduces the number of small network messages by aggregating events within
    a configurable time window.

    Parameters
    ----------
    batch_ms : float
        Flush window in milliseconds (default 300).
    max_batch_size : int
        Maximum events per batch before an early flush is triggered.
    """

    def __init__(
        self,
        batch_ms:       float = 300.0,
        max_batch_size: int   = 32,
    ) -> None:
        self._batch_ms        = batch_ms
        self._max_size        = max_batch_size
        self._pending: List[Dict[str, Any]] = []
        self._window_start_ms = _now_ms()

    # ------------------------------------------------------------------

    def add(
        self,
        trail_type: str,
        pos:        List[float],
        direction:  List[float],
        strength:   float,
        material:   str,
        tick:       int = 0,
    ) -> Optional[Dict[str, Any]]:
        """Queue one event; returns a flush batch if the window just filled."""
        self._pending.append(
            encode_trail_event(trail_type, pos, direction, strength, material, tick)
        )
        if len(self._pending) >= self._max_size:
            return self.flush()
        return None

    def flush(self) -> Optional[Dict[str, Any]]:
        """Return a ``TRAIL_BATCH`` message and reset the accumulator.

        Returns *None* when there are no pending events.
        """
        if not self._pending:
            return None
        msg = {
            "type":   "TRAIL_BATCH",
            "events": list(self._pending),
        }
        self._pending.clear()
        self._window_start_ms = _now_ms()
        return msg

    def maybe_flush(self, now_ms: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Flush if the batch window has elapsed; otherwise return *None*."""
        now = now_ms if now_ms is not None else _now_ms()
        if now - self._window_start_ms >= self._batch_ms:
            return self.flush()
        return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _now_ms() -> float:
    return time.monotonic() * 1000.0
