"""InstabilityEventReplicator — Stage 52 network replication for instability events.

Serialises instability events (CrustFailureEvent, DustAvalancheEvent,
ThermalFractureEvent) into compact wire-format dicts and deserialises them.

Also handles periodic broadcast of the full InstabilityState snapshot.

Public API
----------
InstabilityEventReplicator(config=None)
  .serialise_event(event)           → dict
  .deserialise_event(record)        → event object | None
  .build_snapshot(state)            → dict
  .apply_snapshot(state, msg)       → bool
  .should_broadcast(sim_time)       → bool
  .record_broadcast(sim_time)       → None
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from src.instability.InstabilityState     import InstabilityState
from src.instability.CrustFailureModel    import CrustFailureEvent
from src.instability.DustAvalancheModel   import DustAvalancheEvent
from src.instability.ThermalFractureModel import ThermalFractureEvent


_TYPE_CRUST   = "INSTABILITY_CRUST_FAILURE"
_TYPE_DUST    = "INSTABILITY_DUST_AVALANCHE"
_TYPE_THERMAL = "INSTABILITY_THERMAL_FRACTURE"
_TYPE_STATE   = "INSTABILITY_STATE_52"


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


def _q8(v: float) -> int:
    return int(_clamp(v) * 255 + 0.5)


def _dq8(b: int) -> float:
    return b / 255.0


class InstabilityEventReplicator:
    """Serialises instability events and state for network transport.

    Parameters
    ----------
    config :
        Optional dict; reads ``instability.*`` keys.
    """

    _DEFAULT_BROADCAST_INTERVAL = 60.0   # seconds

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = (config or {}).get("instability", {}) or {}
        self._broadcast_interval: float = float(
            cfg.get("broadcast_interval_s", self._DEFAULT_BROADCAST_INTERVAL)
        )
        self.last_broadcast_time: float = -1e9

    # ------------------------------------------------------------------
    # Per-event serialisation
    # ------------------------------------------------------------------

    def serialise_event(self, event: Any) -> Dict[str, Any]:
        """Encode one instability event to a wire-format dict."""
        if isinstance(event, CrustFailureEvent):
            return {
                "type":       _TYPE_CRUST,
                "tile":       event.tile,
                "intensity":  _q8(event.intensity),
                "crust_delta": _q8(event.crust_delta),
                "rough_gain": _q8(event.roughness_gain),
            }
        if isinstance(event, DustAvalancheEvent):
            return {
                "type":      _TYPE_DUST,
                "tile":      event.tile,
                "intensity": _q8(event.intensity),
                "dust_delta":_q8(event.dust_delta),
            }
        if isinstance(event, ThermalFractureEvent):
            return {
                "type":      _TYPE_THERMAL,
                "tile":      event.tile,
                "intensity": _q8(event.intensity),
                "crust_gain":_q8(event.crust_potential_gain),
            }
        raise TypeError(f"InstabilityEventReplicator: unknown event type {type(event)}")

    def deserialise_event(self, record: Dict[str, Any]) -> Optional[Any]:
        """Decode a wire-format dict back to an event object."""
        t = record.get("type")
        if t == _TYPE_CRUST:
            return CrustFailureEvent(
                tile=int(record["tile"]),
                intensity=_dq8(record["intensity"]),
                crust_delta=_dq8(record["crust_delta"]),
                roughness_gain=_dq8(record["rough_gain"]),
            )
        if t == _TYPE_DUST:
            return DustAvalancheEvent(
                tile=int(record["tile"]),
                intensity=_dq8(record["intensity"]),
                dust_delta=_dq8(record["dust_delta"]),
            )
        if t == _TYPE_THERMAL:
            return ThermalFractureEvent(
                tile=int(record["tile"]),
                intensity=_dq8(record["intensity"]),
                crust_potential_gain=_dq8(record["crust_gain"]),
            )
        return None

    # ------------------------------------------------------------------
    # Full-state snapshot
    # ------------------------------------------------------------------

    def build_snapshot(self, state: InstabilityState) -> Dict[str, Any]:
        """Serialise the full InstabilityState for broadcast."""
        return state.to_dict()

    def apply_snapshot(self, state: InstabilityState, msg: Dict[str, Any]) -> bool:
        """Apply a received snapshot to *state*.

        Returns True on success, False if the message type is wrong.
        """
        if msg.get("type") != _TYPE_STATE:
            return False
        try:
            state.from_dict(msg)
        except (ValueError, KeyError):
            return False
        return True

    # ------------------------------------------------------------------
    # Broadcast scheduling
    # ------------------------------------------------------------------

    def should_broadcast(self, sim_time: float) -> bool:
        """Return True if a full-state broadcast is due."""
        return (sim_time - self.last_broadcast_time) >= self._broadcast_interval

    def record_broadcast(self, sim_time: float) -> None:
        """Record that a broadcast happened at *sim_time*."""
        self.last_broadcast_time = sim_time
