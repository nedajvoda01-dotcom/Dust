"""PlanetTimeController — Stage 50 dual-time model.

Manages two independent time axes:

  simTime    — physical simulation time (seconds, driven by game tick)
  planetTime — slow planetary time (hours / days / seasons)

``planetTime`` advances at a configurable scale factor::

    planetTime += dt * planetTimeScale

``planetTimeScale`` is intentionally very small (default 1e-4) so that one
real simulated day of player gameplay corresponds to only a small fraction of
a planetary season.  In dev mode the scale can be increased to preview long-
term changes.

Config keys (under ``planet.*``)
---------------------------------
timescale       : float  — ratio of planetTime to simTime  (default 1e-4)
day_length_s    : float  — planet day in simTime seconds   (default 5400)
season_length_s : float  — one season in planetTime units  (default 86400)

Public API
----------
PlanetTimeController(config=None)
  .advance(dt: float) -> None
  .sim_time   -> float   (total simTime elapsed)
  .planet_time -> float  (total planetTime elapsed)
  .season_phase -> float  (fractional position within current season [0, 1))
  .day_phase    -> float  (fractional position within current planet day [0, 1))
"""
from __future__ import annotations

import math


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


class PlanetTimeController:
    """Manages the dual sim-time / planet-time axes.

    Parameters
    ----------
    config : dict or None
        Flat or nested dict.  Keys read: ``planet.timescale``,
        ``planet.day_length_s``, ``planet.season_length_s``.
        If *None*, built-in defaults are used.
    """

    _DEFAULTS = {
        "timescale":       1e-4,    # planetTime / simTime ratio
        "day_length_s":    5400.0,  # 90 min planet day in simTime seconds
        "season_length_s": 86400.0, # one season in planetTime units
    }

    def __init__(self, config=None) -> None:
        cfg = dict(self._DEFAULTS)
        if isinstance(config, dict):
            planet_cfg = config.get("planet", config)
            for k in self._DEFAULTS:
                if k in planet_cfg:
                    cfg[k] = float(planet_cfg[k])

        self._timescale:       float = cfg["timescale"]
        self._day_length_s:    float = cfg["day_length_s"]
        self._season_length_s: float = cfg["season_length_s"]

        self._sim_time:    float = 0.0
        self._planet_time: float = 0.0

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def sim_time(self) -> float:
        """Total elapsed simulation time (seconds)."""
        return self._sim_time

    @property
    def planet_time(self) -> float:
        """Total elapsed planetary time (planetTime units)."""
        return self._planet_time

    @property
    def season_phase(self) -> float:
        """Fractional progress through the current season [0, 1)."""
        raw = self._planet_time / self._season_length_s
        return raw - math.floor(raw)

    @property
    def day_phase(self) -> float:
        """Fractional progress through the current planet day [0, 1).

        Based on simTime so that the visual day/night cycle is crisp.
        """
        raw = self._sim_time / self._day_length_s
        return raw - math.floor(raw)

    # ------------------------------------------------------------------
    # Advance
    # ------------------------------------------------------------------

    def advance(self, dt: float) -> None:
        """Advance both clocks by *dt* simTime seconds."""
        if dt <= 0.0:
            return
        self._sim_time    += dt
        self._planet_time += dt * self._timescale

    # ------------------------------------------------------------------
    # State serialisation (for snapshots)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a compact dict for snapshot persistence."""
        return {
            "sim_time":    self._sim_time,
            "planet_time": self._planet_time,
        }

    def from_dict(self, d: dict) -> None:
        """Restore state from a dict produced by :meth:`to_dict`."""
        self._sim_time    = float(d.get("sim_time",    0.0))
        self._planet_time = float(d.get("planet_time", 0.0))
