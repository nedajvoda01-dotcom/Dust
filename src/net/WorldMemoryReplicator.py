"""WorldMemoryReplicator — Stage 51 server-authoritative network replication
of :class:`~src.memory.WorldMemoryState.WorldMemoryState`.

The server is authoritative over WorldMemoryState.  Clients receive coarse
updates infrequently (every ``broadcast_interval_s`` seconds of simTime).

Wire format
-----------
JSON-serialisable dict (same format as :meth:`WorldMemoryState.to_dict`)::

    { "type": "WORLD_MEMORY_STATE_51",
      "width": <int>, "height": <int>,
      "state_hash": <str>,
      "fields": { ... 8-bit quantised ... }
    }

Config keys (under ``memory.*``)
----------------------------------
broadcast_interval_s : float — simTime interval between broadcasts (default 60)

Public API
----------
WorldMemoryReplicator(config=None)
  .should_broadcast(sim_time: float) -> bool
  .build_snapshot(memory_state) -> dict
  .record_broadcast(sim_time: float) -> None
  .apply_snapshot(memory_state, msg: dict) -> bool
  .last_broadcast_time -> float
"""
from __future__ import annotations

from src.memory.WorldMemoryState import WorldMemoryState


class WorldMemoryReplicator:
    """Manages server → client replication of the world memory state.

    Parameters
    ----------
    config : dict or None
        Keys read: ``memory.broadcast_interval_s``.
    """

    _DEFAULTS = {
        "broadcast_interval_s": 60.0,
    }

    def __init__(self, config=None) -> None:
        cfg = dict(self._DEFAULTS)
        if isinstance(config, dict):
            src = config.get("memory", config)
            for k in self._DEFAULTS:
                if k in src:
                    cfg[k] = src[k]

        self._interval:       float = float(cfg["broadcast_interval_s"])
        self._last_broadcast: float = float("-inf")  # force first broadcast immediately

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def last_broadcast_time(self) -> float:
        """simTime of the last broadcast (−∞ if none yet)."""
        return self._last_broadcast

    # ------------------------------------------------------------------
    # Server side
    # ------------------------------------------------------------------

    def should_broadcast(self, sim_time: float) -> bool:
        """Return True if enough simTime has elapsed for a new broadcast."""
        return (sim_time - self._last_broadcast) >= self._interval

    def build_snapshot(self, memory_state: WorldMemoryState) -> dict:
        """Serialise *memory_state* to a network message dict."""
        return memory_state.to_dict()

    def record_broadcast(self, sim_time: float) -> None:
        """Mark *sim_time* as the last broadcast time."""
        self._last_broadcast = sim_time

    # ------------------------------------------------------------------
    # Client side
    # ------------------------------------------------------------------

    def apply_snapshot(
        self,
        memory_state: WorldMemoryState,
        msg: dict,
    ) -> bool:
        """Apply a received snapshot to *memory_state*.

        Returns True on success, False on format / size mismatch.
        """
        try:
            memory_state.from_dict(msg)
            return True
        except (ValueError, KeyError):
            return False
