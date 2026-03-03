"""CrustStabilityModel — Stage 50 long-term crust stability evolution.

Tracks how heat/cool cycles (dual suns + ring shadow) and dust loading
affect the structural stability of the surface crust over planetary time.

When stability drops below a threshold, a localised *crust collapse event*
is emitted.  The PhaseChangeSystem (Stage 45) then applies the MaterialPhase
update.

Model
-----
Per tick, for each processed tile::

    stability -= thermal_cycling_k * cycle_amplitude * dt
    stability -= dust_load_k       * dustReservoir   * dt
    stability += recovery_k        * dt               # slow self-healing

    if stability < collapse_threshold:
        emit CrustCollapseEvent
        stability = collapse_threshold + recovery_boost

Config keys (under ``evolution.*``)
------------------------------------
crust_decay_k         : float — thermal cycling decay rate    (default 0.02)
crust_dust_load_k     : float — dust loading decay rate       (default 0.01)
crust_recovery_k      : float — passive recovery rate         (default 0.005)
crust_collapse_threshold : float — stability level for collapse (default 0.2)
max_tiles_per_tick    : int   — budget                        (default 256)

Public API
----------
CrustCollapseEvent(tile_idx, planet_time, intensity)

CrustStabilityModel(config=None)
  .tick(state, thermal_cycle_amp, dt) -> List[CrustCollapseEvent]
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from src.evolution.PlanetEvolutionState import PlanetEvolutionState


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


@dataclass
class CrustCollapseEvent:
    """Emitted when crustStabilityMap drops below the collapse threshold.

    Attributes
    ----------
    tile_idx     : Flat tile index in the evolution grid.
    planet_time  : planetTime at the moment of collapse.
    intensity    : How far below threshold (0 = just at threshold, 1 = max).
    """
    tile_idx:    int
    planet_time: float
    intensity:   float


class CrustStabilityModel:
    """Advances the ``crustStabilityMap`` field of a :class:`PlanetEvolutionState`.

    Parameters
    ----------
    config : dict or None
        See module docstring for keys.
    """

    _DEFAULTS = {
        "crust_decay_k":          0.02,
        "crust_dust_load_k":      0.01,
        "crust_recovery_k":       0.005,
        "crust_collapse_threshold": 0.2,
        "crust_recovery_boost":   0.15,
        "max_tiles_per_tick":     256,
    }

    def __init__(self, config=None) -> None:
        cfg = dict(self._DEFAULTS)
        if isinstance(config, dict):
            src = config.get("evolution", config)
            for k in self._DEFAULTS:
                if k in src:
                    cfg[k] = src[k]

        self._decay_k:     float = float(cfg["crust_decay_k"])
        self._dust_k:      float = float(cfg["crust_dust_load_k"])
        self._recovery_k:  float = float(cfg["crust_recovery_k"])
        self._threshold:   float = float(cfg["crust_collapse_threshold"])
        self._boost:       float = float(cfg["crust_recovery_boost"])
        self._budget:      int   = int(cfg["max_tiles_per_tick"])
        self._cursor:      int   = 0

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def tick(
        self,
        state: PlanetEvolutionState,
        thermal_cycle_amp: float,
        dt: float,
        planet_time: float = 0.0,
    ) -> List[CrustCollapseEvent]:
        """Advance crust stability for a budget slice of tiles.

        Parameters
        ----------
        state             : PlanetEvolutionState to update in-place.
        thermal_cycle_amp : Amplitude of the thermal cycling stress [0, 1].
                            Derived from dual-sun orbital geometry.
        dt                : Evolution time step (planetTime units).
        planet_time       : Current planetTime (for event timestamps).

        Returns
        -------
        list of CrustCollapseEvent
        """
        if dt <= 0.0:
            return []

        n = state.size()
        crust = state.crustStabilityMap
        dust  = state.dustReservoirMap
        events: List[CrustCollapseEvent] = []

        count = min(self._budget, n)
        for _ in range(count):
            idx = self._cursor % n
            self._cursor = (self._cursor + 1) % n

            # Stability reduction from thermal cycling
            stab = crust[idx]
            stab -= self._decay_k * thermal_cycle_amp * dt
            # Stability reduction from dust loading (adds weight / weakens cement)
            stab -= self._dust_k * dust[idx] * dt
            # Passive recovery
            stab += self._recovery_k * dt
            stab = _clamp(stab)

            if stab < self._threshold:
                intensity = _clamp((self._threshold - stab) / self._threshold)
                events.append(CrustCollapseEvent(
                    tile_idx=idx,
                    planet_time=planet_time,
                    intensity=intensity,
                ))
                # Partial recovery after collapse
                stab = _clamp(stab + self._boost)

            crust[idx] = stab

        return events
