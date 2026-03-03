"""EvolutionReplicator — Stage 50 server-authoritative network replication of
:class:`~src.evolution.PlanetEvolutionState.PlanetEvolutionState`.

The server is authoritative over ``planetTime`` and
``PlanetEvolutionState``.  Clients receive coarse updates infrequently
(every ``broadcast_interval_s`` seconds of simTime) and are expected to
interpolate locally between snapshots.

Wire format
-----------
The replicator produces / consumes JSON-serialisable dicts (same format as
:meth:`PlanetEvolutionState.to_dict`), tagged with:

    { "type": "EVOLUTION_STATE_50",
      "world_seed": <int>,
      "planet_time": <float>,
      "width": <int>, "height": <int>,
      "state_hash": <str>,
      "seasonal_insolation_phase": <float>,
      "fields": { ... 8-bit quantised ... }
    }

Config keys (under ``evolution.*``)
------------------------------------
broadcast_interval_s  : float — simTime interval between broadcasts (default 30)

Public API
----------
EvolutionReplicator(config=None)
  .should_broadcast(sim_time: float) -> bool
  .build_snapshot(state, world_seed, planet_time) -> dict
  .apply_snapshot(state, msg: dict) -> bool
  .last_broadcast_time -> float
"""
from __future__ import annotations

from src.evolution.PlanetEvolutionState import PlanetEvolutionState


class EvolutionReplicator:
    """Manages server → client replication of the planetary evolution state.

    Parameters
    ----------
    config : dict or None
        Keys read: ``evolution.broadcast_interval_s``.
    """

    _DEFAULTS = {
        "broadcast_interval_s": 30.0,
    }

    def __init__(self, config=None) -> None:
        cfg = dict(self._DEFAULTS)
        if isinstance(config, dict):
            src = config.get("evolution", config)
            for k in self._DEFAULTS:
                if k in src:
                    cfg[k] = src[k]

        self._interval:       float = float(cfg["broadcast_interval_s"])
        self._last_broadcast: float = -1e9  # force first broadcast immediately

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

    def build_snapshot(
        self,
        state: PlanetEvolutionState,
        world_seed: int,
        planet_time: float,
    ) -> dict:
        """Serialise *state* to a network message dict.

        Records the current simTime of the broadcast in internal state.
        Callers should pass the current ``sim_time`` as *planet_time* if
        they want to include it; or the actual ``planet_time`` value.
        """
        msg = state.to_dict(world_seed=world_seed, planet_time=planet_time)
        # Record the broadcast (callers should pass sim_time separately, but we
        # use planet_time as a proxy since simTime is not stored here)
        # Proper usage: call record_broadcast(sim_time) after calling this method.
        return msg

    def record_broadcast(self, sim_time: float) -> None:
        """Mark *sim_time* as the last broadcast time."""
        self._last_broadcast = sim_time

    # ------------------------------------------------------------------
    # Client side
    # ------------------------------------------------------------------

    def apply_snapshot(
        self,
        state: PlanetEvolutionState,
        msg: dict,
    ) -> bool:
        """Apply a received snapshot to *state*.

        Returns True on success, False on format / size mismatch.
        """
        try:
            state.from_dict(msg)
            return True
        except (ValueError, KeyError):
            return False
