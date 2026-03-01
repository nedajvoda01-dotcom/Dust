"""AutosaveManager — periodic autosave with crash-safe atomic writes.

Saves the following files to PersistentStorage:
  player_state.json    — lat/lon, velocity, simTime
  geo_event_log.json   — event log entries
  climate_state.json   — sparse temperature / dust snapshot

Writes are atomic (write-to-temp + rename) so a crash mid-save never
corrupts existing data.

Public API
----------
AutosaveManager(storage, autosave_minutes, save_on_geo_impact, schema_version)
  .notify_geo_impact()                       — flag next save as urgent
  .maybe_save(sim_time, ctrl, geo, climate)  — save if interval elapsed
  .load_player_state(schema_version)         — load saved player state dict
  .load_geo_log_count()                      — count of logged geo events
  .load_climate_state(schema_version)        — load saved climate snapshot dict
"""
from __future__ import annotations

import math
from typing import Any, Optional

from src.core.Logger import Logger
from src.core.PersistentStorage import PersistentStorage
from src.math.PlanetMath import PlanetMath

_TAG = "Autosave"

_PLAYER_FILE  = "player_state.json"
_GEO_LOG_FILE = "geo_event_log.json"
_CLIMATE_FILE = "climate_state.json"


class AutosaveManager:
    """Manages periodic and event-driven autosaving of game state.

    Parameters
    ----------
    storage:
        Initialised PersistentStorage instance.
    autosave_minutes:
        Interval (in *simulated* minutes) between automatic saves.
    save_on_geo_impact:
        When True, the next ``maybe_save()`` after a geo-impact saves
        immediately regardless of the interval timer.
    schema_version:
        Written into all save files and checked on load.
    """

    def __init__(
        self,
        storage:            PersistentStorage,
        autosave_minutes:   float = 2.0,
        save_on_geo_impact: bool  = True,
        schema_version:     int   = 1,
    ) -> None:
        self._storage          = storage
        self._interval_s       = autosave_minutes * 60.0
        self._on_geo_impact    = save_on_geo_impact
        self._schema_version   = schema_version
        self._last_save_sim_s: float = -1e9
        self._pending_impact:  bool  = False

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def notify_geo_impact(self) -> None:
        """Signal that a major geo event just occurred (trigger early save)."""
        if self._on_geo_impact:
            self._pending_impact = True

    def maybe_save(
        self,
        sim_time:            float,
        character_controller = None,
        geo_event_system     = None,
        climate_system       = None,
    ) -> bool:
        """Save if the autosave interval has elapsed (or a geo impact occurred).

        Parameters
        ----------
        sim_time:
            Current accumulated simulation time (game-seconds).
        character_controller:
            Optional ``CharacterPhysicalController``.
        geo_event_system:
            Optional ``GeoEventSystem`` (provides the event log).
        climate_system:
            Optional ``ClimateSystem`` (provides the field snapshot).

        Returns
        -------
        True if a save was performed, False otherwise.
        """
        due = sim_time - self._last_save_sim_s >= self._interval_s
        if not (due or self._pending_impact):
            return False

        self._do_save(sim_time, character_controller, geo_event_system, climate_system)
        self._last_save_sim_s = sim_time
        self._pending_impact  = False
        return True

    def force_save(
        self,
        sim_time:            float,
        character_controller = None,
        geo_event_system     = None,
        climate_system       = None,
    ) -> None:
        """Unconditionally save right now (e.g. on shutdown)."""
        self._do_save(sim_time, character_controller, geo_event_system, climate_system)
        self._last_save_sim_s = sim_time
        self._pending_impact  = False

    # ------------------------------------------------------------------
    # Load helpers
    # ------------------------------------------------------------------

    def load_player_state(self, schema_version: Optional[int] = None) -> Optional[dict]:
        """Return the stored player-state dict or None.

        Returns None when the file is missing or the schema version
        does not match *schema_version* (or ``self._schema_version`` when
        *schema_version* is None).
        """
        ver = schema_version if schema_version is not None else self._schema_version
        data = self._storage.read_json(_PLAYER_FILE)
        if data is None:
            return None
        if data.get("schemaVersion") != ver:
            Logger.warn(_TAG, "Player state schema mismatch — ignoring saved state")
            return None
        return data

    def load_geo_log_count(self) -> int:
        """Return the number of entries in the stored geo event log (0 if absent)."""
        data = self._storage.read_json(_GEO_LOG_FILE)
        if data is None:
            return 0
        return len(data.get("entries", []))

    def load_climate_state(self, schema_version: Optional[int] = None) -> Optional[dict]:
        """Return the stored climate snapshot dict or None."""
        ver = schema_version if schema_version is not None else self._schema_version
        data = self._storage.read_json(_CLIMATE_FILE)
        if data is None:
            return None
        if data.get("schemaVersion") != ver:
            return None
        return data

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _do_save(
        self,
        sim_time:   float,
        ctrl       = None,
        geo        = None,
        climate    = None,
    ) -> None:
        Logger.info(_TAG, f"Autosave at simTime={sim_time:.1f}s")
        self._save_player(sim_time, ctrl)
        self._save_geo_log(geo)
        self._save_climate(sim_time, climate)

    def _save_player(self, sim_time: float, ctrl=None) -> None:
        data: dict = {
            "schemaVersion": self._schema_version,
            "simTime":       sim_time,
        }
        if ctrl is not None:
            pos = ctrl.position
            if _is_finite_vec3(pos):
                ll = PlanetMath.from_direction(pos.normalized())
                data["lat_rad"]   = ll.lat_rad
                data["lon_rad"]   = ll.lon_rad
                data["world_pos"] = [pos.x, pos.y, pos.z]
            vel = ctrl.velocity
            if _is_finite_vec3(vel):
                data["velocity"] = [vel.x, vel.y, vel.z]
        self._storage.write_json_atomic(_PLAYER_FILE, data)

    def _save_geo_log(self, geo=None) -> None:
        entries: list = []
        if geo is not None and hasattr(geo, "_event_log"):
            event_log = geo._event_log
            if hasattr(event_log, "_records"):
                for rec in event_log._records:
                    entry: dict = {
                        "event_id":  rec.event_id,
                        "event_type": rec.event_type.name,
                        "start_time": rec.start_time,
                    }
                    pos = rec.direction
                    if _is_finite_vec3(pos):
                        entry["direction"] = [pos.x, pos.y, pos.z]
                    entries.append(entry)
        self._storage.write_json_atomic(
            _GEO_LOG_FILE,
            {"schemaVersion": self._schema_version, "entries": entries},
        )

    def _save_climate(self, sim_time: float, climate=None) -> None:
        data: dict = {
            "schemaVersion": self._schema_version,
            "simTime":       sim_time,
        }
        if climate is not None:
            # Attempt sparse snapshot of temperature and dust grids
            if (
                hasattr(climate, "_temp")
                and hasattr(climate, "_w")
                and hasattr(climate, "_h")
            ):
                W, H = climate._w, climate._h
                step = 4
                data["grid_W"] = W
                data["grid_H"] = H
                data["step"]   = step
                data["temp_sparse"] = [
                    round(climate._temp[r * W + c], 2)
                    for r in range(0, H, step)
                    for c in range(0, W, step)
                ]
                if hasattr(climate, "_dust"):
                    data["dust_sparse"] = [
                        round(climate._dust[r * W + c], 4)
                        for r in range(0, H, step)
                        for c in range(0, W, step)
                    ]
        self._storage.write_json_atomic(_CLIMATE_FILE, data)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _is_finite_vec3(v) -> bool:
    """Return True if all components of *v* are finite (not NaN/Inf)."""
    try:
        return math.isfinite(v.x) and math.isfinite(v.y) and math.isfinite(v.z)
    except (AttributeError, TypeError):
        return False
