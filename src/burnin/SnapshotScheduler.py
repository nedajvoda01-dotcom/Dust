"""SnapshotScheduler — Stage 56 periodic state-snapshot planner.

Determines *when* snapshots should be taken during a headless burn-in run
and accumulates the snapshot blobs produced by individual subsystems.

A snapshot bundles:
* planetTime
* evolution fields (Stage 50)
* memory fields    (Stage 51)
* instability fields (Stage 52)
* energy state     (Stage 54)
* material state for sample regions (Stage 45)

Usage
-----
    scheduler = SnapshotScheduler(interval_hours=6.0)

    # Each simulated tick:
    if scheduler.should_snap(planet_time_hours=t):
        blob = scheduler.build_snapshot(planet_time=t, subsystem_states=states)
        scheduler.record(blob)

    snapshots = scheduler.all_snapshots()
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.core.Logger import Logger

_TAG = "SnapshotSched"

# Keys expected from subsystem_states (all optional)
_SUBSYSTEM_KEYS = (
    "evolution",
    "memory",
    "instability",
    "energy",
    "material",
)


# ---------------------------------------------------------------------------
# Snapshot blob
# ---------------------------------------------------------------------------

@dataclass
class WorldSnapshot:
    """A single point-in-time world snapshot."""
    index: int = 0
    planet_time: float = 0.0          # simulated hours
    subsystems: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index":       self.index,
            "planet_time": self.planet_time,
            "subsystems":  dict(self.subsystems),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "WorldSnapshot":
        return WorldSnapshot(
            index=int(d.get("index", 0)),
            planet_time=float(d.get("planet_time", 0.0)),
            subsystems=dict(d.get("subsystems", {})),
        )


# ---------------------------------------------------------------------------
# SnapshotScheduler
# ---------------------------------------------------------------------------

class SnapshotScheduler:
    """Triggers periodic snapshots based on simulated planet-time.

    Parameters
    ----------
    interval_hours: Simulated-hour interval between snapshots (default 6 h).
    """

    def __init__(self, interval_hours: float = 6.0) -> None:
        self._interval = max(0.0001, interval_hours)
        self._next_snap_time: float = 0.0
        self._snapshots: List[WorldSnapshot] = []
        self._snap_index: int = 0

    # ------------------------------------------------------------------
    # Scheduling
    # ------------------------------------------------------------------

    def should_snap(self, planet_time_hours: float) -> bool:
        """Return True if it is time to take a snapshot."""
        return planet_time_hours >= self._next_snap_time

    def build_snapshot(
        self,
        planet_time: float,
        subsystem_states: Optional[Dict[str, Any]] = None,
    ) -> WorldSnapshot:
        """Construct a :class:`WorldSnapshot` from provided subsystem states.

        Missing keys in *subsystem_states* are silently omitted.
        """
        states: Dict[str, Any] = {}
        if subsystem_states:
            for key in _SUBSYSTEM_KEYS:
                if key in subsystem_states:
                    states[key] = subsystem_states[key]
        snap = WorldSnapshot(
            index=self._snap_index,
            planet_time=planet_time,
            subsystems=states,
        )
        return snap

    def record(self, snapshot: WorldSnapshot) -> None:
        """Store *snapshot* and advance the next-snap time."""
        self._snapshots.append(snapshot)
        self._snap_index += 1
        self._next_snap_time = snapshot.planet_time + self._interval
        Logger.debug(
            _TAG,
            f"Snapshot #{snapshot.index} at planetTime={snapshot.planet_time:.2f}h"
            f" — total stored: {len(self._snapshots)}",
        )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def all_snapshots(self) -> List[WorldSnapshot]:
        """Return all stored snapshots."""
        return list(self._snapshots)

    def latest(self) -> Optional[WorldSnapshot]:
        """Return the most recent snapshot, or None."""
        return self._snapshots[-1] if self._snapshots else None

    def count(self) -> int:
        """Return the number of stored snapshots."""
        return len(self._snapshots)

    def reset(self) -> None:
        """Clear all stored snapshots and reset the timer."""
        self._snapshots.clear()
        self._snap_index = 0
        self._next_snap_time = 0.0
