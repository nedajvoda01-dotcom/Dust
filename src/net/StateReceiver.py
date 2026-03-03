"""StateReceiver — Stage 58 client-side authoritative/remote state receiver.

Decodes incoming server state messages and routes them to either
:class:`~src.net.Reconciliation.Reconciliation` (own player) or
:class:`~src.net.RemoteInterpolation.RemoteInterpolation` (remote players).

Message schemas (JSON)
-----------------------
AuthoritativeState (server → client, own player)::

    {
      "type": "AUTH_STATE",
      "pos":  [x, y, z],
      "vel":  [vx, vy, vz],
      "yaw":  <float>,
      "contact": <int>,
      "sTick": <int>,
      "lastSeq": <int>
    }

RemoteState (server → client, other players)::

    {
      "type": "REMOTE_STATE",
      "playerId": "<str>",
      "ts": <float>,
      "pos":  [x, y, z],
      "vel":  [vx, vy, vz],
      "yaw":  <float>,
      "contact": <int>,
      "sTick": <int>
    }

Public API
----------
StateReceiver(own_player_id, reconciliation, remote_interpolation)
    .receive(msg) → bool
        Decode and dispatch one parsed JSON message dict.
        Returns True if the message was handled.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from src.net.JitterBuffer import StateFrame
from src.net.Reconciliation import Reconciliation
from src.net.RemoteInterpolation import RemoteInterpolation

_MSG_AUTH   = "AUTH_STATE"
_MSG_REMOTE = "REMOTE_STATE"


class StateReceiver:
    """Dispatch server state messages to the appropriate client subsystem.

    Parameters
    ----------
    own_player_id : str
        The local player's identifier (used to ignore own RemoteState echoes).
    reconciliation : Reconciliation
        Own-player reconciliation instance.
    remote_interpolation : RemoteInterpolation
        Remote-player interpolation manager.
    """

    def __init__(
        self,
        own_player_id:        str,
        reconciliation:       Reconciliation,
        remote_interpolation: RemoteInterpolation,
    ) -> None:
        self._own_id    = own_player_id
        self._reconcile = reconciliation
        self._remote    = remote_interpolation

        # Predicted state that the reconciliation module compares against.
        # Updated by the caller via set_predicted_state() each tick.
        self._pred_pos:  tuple = (0.0, 0.0, 0.0)
        self._pred_vel:  tuple = (0.0, 0.0, 0.0)
        self._pred_yaw:  float = 0.0

        self._auth_count:   int = 0
        self._remote_count: int = 0

    # ------------------------------------------------------------------
    # Predicted state registration
    # ------------------------------------------------------------------

    def set_predicted_state(
        self,
        pos: tuple,
        vel: tuple,
        yaw: float,
    ) -> None:
        """Register the current predicted state for reconciliation comparison."""
        self._pred_pos = tuple(pos)
        self._pred_vel = tuple(vel)
        self._pred_yaw = float(yaw)

    # ------------------------------------------------------------------
    # Message dispatch
    # ------------------------------------------------------------------

    def receive(self, msg: Dict[str, Any]) -> bool:
        """Decode and dispatch a parsed JSON message.

        Returns
        -------
        bool
            True when the message type was recognised and handled.
        """
        msg_type = msg.get("type", "")

        if msg_type == _MSG_AUTH:
            self._handle_auth(msg)
            return True

        if msg_type == _MSG_REMOTE:
            pid = str(msg.get("playerId", ""))
            if pid and pid != self._own_id:
                self._handle_remote(msg, pid)
            return True

        return False

    # ------------------------------------------------------------------
    # Internal handlers
    # ------------------------------------------------------------------

    def _handle_auth(self, msg: Dict[str, Any]) -> None:
        """Process an AuthoritativeState message for the own player."""
        pos = _parse_vec3(msg.get("pos"))
        vel = _parse_vec3(msg.get("vel"))
        yaw = float(msg.get("yaw", 0.0))
        last_seq  = int(msg.get("lastSeq",  -1))
        server_tick = int(msg.get("sTick", 0))

        self._reconcile.receive_authoritative(
            server_pos         = pos,
            server_vel         = vel,
            server_yaw         = yaw,
            last_processed_seq = last_seq,
            server_tick        = server_tick,
            predicted_pos      = self._pred_pos,
            predicted_vel      = self._pred_vel,
            predicted_yaw      = self._pred_yaw,
        )
        self._auth_count += 1

    def _handle_remote(self, msg: Dict[str, Any], player_id: str) -> None:
        """Process a RemoteState message for a remote player."""
        ts  = float(msg.get("ts", 0.0))
        pos = _parse_vec3(msg.get("pos"))
        vel = _parse_vec3(msg.get("vel"))
        yaw = float(msg.get("yaw", 0.0))
        contact = int(msg.get("contact", 0))
        server_tick = int(msg.get("sTick", 0))

        frame = StateFrame(
            timestamp_s   = ts,
            pos           = pos,
            vel           = vel,
            yaw           = yaw,
            contact_flags = contact,
            server_tick   = server_tick,
        )
        self._remote.push(player_id, frame)
        self._remote_count += 1

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @property
    def auth_messages_received(self) -> int:
        """Total authoritative state messages processed."""
        return self._auth_count

    @property
    def remote_messages_received(self) -> int:
        """Total remote state messages processed."""
        return self._remote_count


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_vec3(v: Any) -> tuple:
    try:
        return (float(v[0]), float(v[1]), float(v[2]))
    except (TypeError, IndexError, ValueError):
        return (0.0, 0.0, 0.0)
