"""GlobalAuthorityService — Stage 60 Global Authority component.

Holds the single source of truth for planet-wide state that must remain
consistent regardless of how many Region Shards (RS) are running.

Responsibilities
----------------
* worldId / epoch / seed
* planetTime (coarse clock shared with all RS)
* seasonal insolation phase
* global energy ledger summary (energy in/out totals)
* coarse evolution field snapshots (width × height grid of summary scalars)
* region → RS node mapping

RS nodes register themselves on start-up (``register_region``) and send
periodic heartbeats (``heartbeat``).  When a RS falls silent for longer than
``dead_timeout_s`` the GA marks that region ``dead``.

Cross-region events are relayed to all live neighbouring RS via the
``post_cross_region_event`` / ``drain_events_for_region`` interface so that
RS can create local far-emitters without GA knowing the internal simulation
details.

Public API
----------
GlobalAuthorityService(config=None)
  .tick(dt_s)                              — advance planet time
  .register_region(region_id, node_addr)   — RS announces itself
  .deregister_region(region_id)            — RS graceful shutdown
  .heartbeat(region_id)                    — RS keepalive
  .live_regions() → list[int]              — currently alive region IDs
  .region_node(region_id) → str | None     — RS address for region
  .post_cross_region_event(event)          — broadcast an infra/storm event
  .drain_events_for_region(region_id) → list — pending events for RS
  .global_snapshot() → dict               — serialisable snapshot of GA state
  .load_snapshot(snap: dict)               — restore from snapshot
"""
from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_REGION_DEAD_TIMEOUT_S = 60.0
_DEFAULT_TIMESCALE = 0.0001           # same default as planet.timescale


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class _RegionEntry:
    region_id: int
    node_addr: str
    last_seen: float = field(default_factory=time.monotonic)
    alive: bool = True


# ---------------------------------------------------------------------------
# GlobalAuthorityService
# ---------------------------------------------------------------------------

class GlobalAuthorityService:
    """Authoritative holder of all planet-wide shared state."""

    def __init__(self, config=None) -> None:
        self._cfg = config

        # World identity
        self.world_id:  str   = str(uuid.uuid4())
        self.epoch:     int   = 0
        self.seed:      int   = 42

        # Planet clock
        self.planet_time:        float = 0.0
        self.sim_time:           float = 0.0
        self.seasonal_phase:     float = 0.0
        self._timescale:         float = self._cfg_float(
            ("planet", "timescale"), _DEFAULT_TIMESCALE
        ) if config else _DEFAULT_TIMESCALE

        # Global energy ledger (summary scalars — not full physics)
        self.energy_in:  float = 0.0
        self.energy_out: float = 0.0

        # Coarse evolution fields (flat list of floats, width × height tiles)
        self.evolution_width:  int        = 0
        self.evolution_height: int        = 0
        self.evolution_fields: List[float] = []

        # Region registry
        self._regions: Dict[int, _RegionEntry] = {}
        self._dead_timeout_s: float = _DEFAULT_REGION_DEAD_TIMEOUT_S

        # Cross-region event queues  {region_id → [event, …]}
        self._event_queues: Dict[int, List[Dict[str, Any]]] = {}

    # ------------------------------------------------------------------
    # Clock
    # ------------------------------------------------------------------

    def tick(self, dt_s: float) -> None:
        """Advance the global clock by *dt_s* simulation seconds."""
        self.sim_time   += dt_s
        self.planet_time += dt_s * self._timescale
        # Mark stale regions dead
        now = time.monotonic()
        for entry in self._regions.values():
            if entry.alive and (now - entry.last_seen) > self._dead_timeout_s:
                entry.alive = False

    # ------------------------------------------------------------------
    # Region registry
    # ------------------------------------------------------------------

    def register_region(self, region_id: int, node_addr: str) -> None:
        """A RS announces that it is handling *region_id* at *node_addr*."""
        self._regions[region_id] = _RegionEntry(
            region_id=region_id,
            node_addr=node_addr,
        )
        if region_id not in self._event_queues:
            self._event_queues[region_id] = []

    def deregister_region(self, region_id: int) -> None:
        """RS graceful shutdown — remove from live registry."""
        self._regions.pop(region_id, None)
        self._event_queues.pop(region_id, None)

    def heartbeat(self, region_id: int) -> None:
        """RS keepalive — reset the dead-timeout for *region_id*."""
        entry = self._regions.get(region_id)
        if entry is not None:
            entry.last_seen = time.monotonic()
            entry.alive = True

    def live_regions(self) -> List[int]:
        """Return IDs of currently alive regions."""
        return [rid for rid, e in self._regions.items() if e.alive]

    def region_node(self, region_id: int) -> Optional[str]:
        """Return the RS node address for *region_id*, or ``None``."""
        entry = self._regions.get(region_id)
        return entry.node_addr if entry is not None and entry.alive else None

    # ------------------------------------------------------------------
    # Cross-region events
    # ------------------------------------------------------------------

    def post_cross_region_event(self, event: Dict[str, Any]) -> None:
        """Broadcast *event* to all live regions except the source.

        *event* must contain at least ``{"type": str, "source_region": int}``.
        All other fields are passed through to the RS intact.
        """
        source = event.get("source_region", -1)
        for rid, entry in self._regions.items():
            if entry.alive and rid != source:
                self._event_queues.setdefault(rid, []).append(event)

    def drain_events_for_region(self, region_id: int) -> List[Dict[str, Any]]:
        """Return and clear the pending cross-region events for *region_id*."""
        q = self._event_queues.get(region_id)
        if not q:
            return []
        events = list(q)
        q.clear()
        return events

    # ------------------------------------------------------------------
    # Snapshot serialisation
    # ------------------------------------------------------------------

    def global_snapshot(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict representing the full GA state."""
        return {
            "world_id":         self.world_id,
            "epoch":            self.epoch,
            "seed":             self.seed,
            "planet_time":      self.planet_time,
            "sim_time":         self.sim_time,
            "seasonal_phase":   self.seasonal_phase,
            "energy_in":        self.energy_in,
            "energy_out":       self.energy_out,
            "evolution_width":  self.evolution_width,
            "evolution_height": self.evolution_height,
            "evolution_fields": list(self.evolution_fields),
            "live_regions": {
                str(rid): e.node_addr
                for rid, e in self._regions.items()
                if e.alive
            },
        }

    def load_snapshot(self, snap: Dict[str, Any]) -> None:
        """Restore GA state from a dict produced by :meth:`global_snapshot`."""
        self.world_id        = str(snap.get("world_id", self.world_id))
        self.epoch           = int(snap.get("epoch", self.epoch))
        self.seed            = int(snap.get("seed", self.seed))
        self.planet_time     = float(snap.get("planet_time", self.planet_time))
        self.sim_time        = float(snap.get("sim_time", self.sim_time))
        self.seasonal_phase  = float(snap.get("seasonal_phase", self.seasonal_phase))
        self.energy_in       = float(snap.get("energy_in", self.energy_in))
        self.energy_out      = float(snap.get("energy_out", self.energy_out))
        self.evolution_width  = int(snap.get("evolution_width", self.evolution_width))
        self.evolution_height = int(snap.get("evolution_height", self.evolution_height))
        self.evolution_fields = [float(v) for v in snap.get("evolution_fields", [])]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _cfg_float(self, path: tuple, default: float) -> float:
        if self._cfg is None:
            return default
        return float(self._cfg.get(*path, default=default))
