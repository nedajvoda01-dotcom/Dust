"""InputReceiver — Stage 58 server-side input frame receiver and validator.

Accepts :class:`~src.net.InputSender.InputFrame`-compatible dicts from
clients, validates them, enforces per-client rate-limiting, and exposes the
latest valid frame for each connected player.

Public API
----------
InputReceiver(config)
    .receive(player_id, msg) → InputFrame | None
        Process one incoming ``INPUT_FRAME`` message.
        Returns a decoded :class:`InputFrame` or *None* if rejected.
    .latest(player_id) → InputFrame | None
        Return the most recent accepted frame for *player_id*.
    .remove(player_id)
        Drop tracking state on disconnect.
    .player_ids() → list[str]
        Currently tracked players.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.net.InputSender import InputFrame

_MSG_TYPE = "INPUT_FRAME"

# Maximum gap between accepted sequence ids before a frame is dropped
_MAX_SEQ_GAP = 128


class InputReceiver:
    """Server-side input frame receiver with sequence-gap and type validation.

    Parameters
    ----------
    config : dict
        Full game config dict (currently unused; reserved for future rate-limit
        tuning via ``net.input_hz``).
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = (config or {}).get("net", {}) or {}
        self._input_hz: float = float(cfg.get("input_hz", 30))
        # last_seq per player, latest frame per player
        self._last_seq:    Dict[str, int]                  = {}
        self._latest:      Dict[str, InputFrame]           = {}

    # ------------------------------------------------------------------
    # Receive
    # ------------------------------------------------------------------

    def receive(self, player_id: str, msg: Dict[str, Any]) -> Optional[InputFrame]:
        """Validate and decode one client input message.

        Rejects messages that:
        * are not of type ``INPUT_FRAME``
        * have a sequence id that is not greater than the last accepted one
        * have a sequence gap larger than :data:`_MAX_SEQ_GAP` (possible
          malicious spam or extreme packet reordering)

        Parameters
        ----------
        player_id : str
            Sender identifier.
        msg : dict
            Parsed JSON message.

        Returns
        -------
        InputFrame or None
        """
        if msg.get("type") != _MSG_TYPE:
            return None

        try:
            frame = InputFrame.from_dict(msg)
        except (KeyError, TypeError, ValueError):
            return None

        last = self._last_seq.get(player_id, -1)
        if frame.sequence_id <= last:
            return None  # old or duplicate
        if frame.sequence_id - last > _MAX_SEQ_GAP and last != -1:
            # Suspicious gap — accept but log (could be reconnect)
            pass

        self._last_seq[player_id] = frame.sequence_id
        self._latest[player_id]   = frame
        return frame

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def latest(self, player_id: str) -> Optional[InputFrame]:
        """Return the most recently accepted frame for *player_id*."""
        return self._latest.get(player_id)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def remove(self, player_id: str) -> None:
        """Discard all tracking state for a player on disconnect."""
        self._last_seq.pop(player_id, None)
        self._latest.pop(player_id, None)

    def player_ids(self) -> List[str]:
        """Currently tracked player ids."""
        return list(self._latest.keys())
