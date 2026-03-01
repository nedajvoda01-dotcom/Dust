"""WorldState — shared authoritative world state stored in ``world_state/``.

All connected clients share this single source of truth.  The directory
layout is:

    world_state/
        world.json          — seed, worldId, simTime, timeScale, epoch
        geo_events.jsonl    — newline-delimited JSON geo-event records
        climate.json        — periodic climate snapshot (storms, dust)

Reset by deleting the ``world_state/`` directory (or call :meth:`reset`).

Public API
----------
WorldState(state_dir="world_state")
  .load_or_create(default_seed, reset_on_missing) — load from disk or create
  .save()                                         — flush world.json
  .reset()                                        — wipe and recreate
  .seed / .world_id / .sim_time / .time_scale / .epoch  — live fields
  .append_geo_event(record)                       — add event to log
  .geo_events() → list                            — all recorded events
  .save_climate_snapshot(snap)                    — write climate.json
  .load_climate_snapshot() → dict | None
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_WORLD_FILE    = "world.json"
_GEO_FILE      = "geo_events.jsonl"
_CLIMATE_FILE  = "climate.json"
_SCHEMA_VERSION = 1


class WorldState:
    """Persistent shared world state for the multiplayer server."""

    def __init__(self, state_dir: str = "world_state") -> None:
        self._dir: Path = Path(state_dir)

        self.seed:        int   = 42
        self.world_id:    str   = ""
        self.sim_time:    float = 0.0
        self.time_scale:  float = 1.0
        self.epoch:       int   = 0

        self._geo_events: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load_or_create(
        self,
        default_seed: int = 42,
        reset_on_missing: bool = True,
    ) -> None:
        """Load state from disk, or create a fresh world if none exists.

        Parameters
        ----------
        default_seed:
            Seed to use when generating a new world identity.
        reset_on_missing:
            When *True* (default) a missing state dir triggers a fresh
            world.  When *False* an existing state dir is required.
        """
        self._dir.mkdir(parents=True, exist_ok=True)
        world_file = self._dir / _WORLD_FILE

        if world_file.exists():
            try:
                with open(world_file, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if data.get("schemaVersion", 0) == _SCHEMA_VERSION:
                    self.seed       = int(data.get("seed",      default_seed))
                    self.world_id   = str(data.get("worldId",   ""))
                    self.sim_time   = float(data.get("simTime", 0.0))
                    self.time_scale = float(data.get("timeScale", 1.0))
                    self.epoch      = int(data.get("epoch",     0))
                    self._load_geo_events()
                    return
            except Exception as exc:
                # Treat a corrupt file like a missing file and create a fresh world.
                import warnings
                warnings.warn(
                    f"WorldState: failed to load {world_file}: {type(exc).__name__} — "
                    "creating fresh world",
                    RuntimeWarning,
                    stacklevel=2,
                )

        # Fresh world
        self.seed       = default_seed
        self.world_id   = str(uuid.uuid4())
        self.sim_time   = 0.0
        self.time_scale = 1.0
        self.epoch      = 0
        self._geo_events = []
        self.save()

    def save(self) -> None:
        """Persist ``world.json`` atomically."""
        self._dir.mkdir(parents=True, exist_ok=True)
        self._write_atomic(
            _WORLD_FILE,
            json.dumps({
                "schemaVersion": _SCHEMA_VERSION,
                "seed":          self.seed,
                "worldId":       self.world_id,
                "simTime":       self.sim_time,
                "timeScale":     self.time_scale,
                "epoch":         self.epoch,
                "savedAt":       datetime.now(timezone.utc).isoformat(),
            }).encode("utf-8"),
        )

    def reset(self) -> None:
        """Delete ``world_state/`` and initialise a fresh world."""
        if self._dir.exists():
            shutil.rmtree(self._dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self.seed       = 42
        self.world_id   = str(uuid.uuid4())
        self.sim_time   = 0.0
        self.time_scale = 1.0
        self.epoch      = 0
        self._geo_events = []
        self.save()

    # ------------------------------------------------------------------
    # Geo events
    # ------------------------------------------------------------------

    def append_geo_event(self, record: Dict[str, Any]) -> None:
        """Append *record* to the geo-event log (disk + memory)."""
        self._geo_events.append(record)
        self._save_geo_events()

    def geo_events(self) -> List[Dict[str, Any]]:
        """Return a snapshot of all recorded geo-event dicts."""
        return list(self._geo_events)

    def _save_geo_events(self) -> None:
        lines = b"\n".join(
            json.dumps(ev).encode("utf-8") for ev in self._geo_events
        ) + b"\n"
        self._write_atomic(_GEO_FILE, lines)

    def _load_geo_events(self) -> None:
        path = self._dir / _GEO_FILE
        self._geo_events = []
        if not path.exists():
            return
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        self._geo_events.append(json.loads(line))
                    except Exception:
                        pass

    # ------------------------------------------------------------------
    # Climate snapshot
    # ------------------------------------------------------------------

    def save_climate_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """Persist *snapshot* to ``climate.json``."""
        self._write_atomic(
            _CLIMATE_FILE,
            json.dumps(snapshot).encode("utf-8"),
        )

    def load_climate_snapshot(self) -> Optional[Dict[str, Any]]:
        """Return the last saved climate snapshot, or *None*."""
        path = self._dir / _CLIMATE_FILE
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _write_atomic(self, filename: str, data: bytes) -> None:
        """Write *data* to *filename* inside ``self._dir`` atomically."""
        target = self._dir / filename
        fd, tmp_path = tempfile.mkstemp(dir=self._dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            shutil.move(tmp_path, str(target))
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
