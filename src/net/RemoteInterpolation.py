"""RemoteInterpolation — Stage 58 smooth interpolation for remote players.

Maintains a per-player :class:`~src.net.JitterBuffer.JitterBuffer` and exposes
an interpolated :class:`~src.net.JitterBuffer.StateFrame` for each remote
entity.

Public API
----------
RemoteInterpolation(config)
    .push(player_id, frame)
        Feed a new state snapshot for a remote player.
    .get(player_id, now_s) → StateFrame | None
        Return the interpolated state for *player_id* at the current render
        time.
    .update_delay(rtt_s, jitter_s)
        Update the adaptive delay for all existing jitter buffers.
    .remove(player_id)
        Drop a player's buffer (on disconnect).
    .player_ids() → list[str]
        Currently tracked player ids.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from src.net.JitterBuffer import JitterBuffer, StateFrame


class RemoteInterpolation:
    """Per-player jitter-buffered interpolation manager.

    Parameters
    ----------
    config : dict
        Full game config dict; forwarded to each :class:`JitterBuffer`.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        self._config = config or {}
        self._buffers: Dict[str, JitterBuffer] = {}

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def push(self, player_id: str, frame: StateFrame) -> None:
        """Insert a new state snapshot for a remote player.

        A :class:`JitterBuffer` is created automatically on first push.
        """
        if player_id not in self._buffers:
            self._buffers[player_id] = JitterBuffer(self._config)
        self._buffers[player_id].push(frame)

    # ------------------------------------------------------------------
    # Render-time query
    # ------------------------------------------------------------------

    def get(self, player_id: str, now_s: float) -> Optional[StateFrame]:
        """Return the interpolated (or extrapolated) state at render time.

        Parameters
        ----------
        player_id : str
            Target player identifier.
        now_s : float
            Current monotonic render time (seconds).

        Returns
        -------
        StateFrame or None if no frames are buffered for this player.
        """
        buf = self._buffers.get(player_id)
        if buf is None:
            return None
        return buf.interpolate(now_s)

    # ------------------------------------------------------------------
    # Adaptive delay
    # ------------------------------------------------------------------

    def update_delay(self, rtt_s: float, jitter_s: float) -> None:
        """Propagate updated RTT/jitter to all player buffers."""
        for buf in self._buffers.values():
            buf.update_delay(rtt_s, jitter_s)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def remove(self, player_id: str) -> None:
        """Remove a player's jitter buffer on disconnect."""
        self._buffers.pop(player_id, None)

    def player_ids(self) -> List[str]:
        """List of currently tracked remote player ids."""
        return list(self._buffers.keys())
