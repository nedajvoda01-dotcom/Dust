"""EmitterAggregator — Stage 46 acoustic emitter registry and clustering.

Manages a budget-limited set of :class:`AcousticEmitterRecord` objects that
represent world sound sources.  Each tick callers submit raw emitters; the
aggregator clusters nearby/simultaneous emitters, ranks by energy and retains
only the top-N, rolling the rest into a per-sector background energy value.

Emitter types
-------------
* ``LOCAL_CONTACT``  — footsteps/friction; only audible very close.
* ``IMPACT_MEDIUM``  — moderate-range impacts (falling rock etc.).
* ``STRUCTURAL``     — far-range events (cave-in, rift); also infrasound.
* ``ATMOSPHERIC``    — storm/dust wall; omnidirectional background.

Public API
----------
EmitterAggregator(config=None)
  .add(record)                    → None
  .tick(sim_tick)                 → None
  .active_emitters                → List[AcousticEmitterRecord]
  .background_energy(sector_id)   → float
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

from src.math.Vec3 import Vec3


# ---------------------------------------------------------------------------
# Emitter type
# ---------------------------------------------------------------------------

class EmitterType(IntEnum):
    LOCAL_CONTACT = 0
    IMPACT_MEDIUM = 1
    STRUCTURAL    = 2
    ATMOSPHERIC   = 3


# ---------------------------------------------------------------------------
# AcousticEmitterRecord
# ---------------------------------------------------------------------------

@dataclass
class AcousticEmitterRecord:
    """One logical sound source in the acoustic world model.

    Attributes
    ----------
    id :
        Unique identifier (caller-managed; 0 means unassigned).
    pos :
        World-space position (quantised to 4 m grid when stored).
    band_energy_audible :
        Energy in the audible band (80 Hz – 12 kHz) [0..1+].
    band_energy_infra :
        Energy in the infrasound band (5–80 Hz) [0..1+].
    directivity :
        0 = omnidirectional, 1 = fully directional.
    emitter_type :
        One of :class:`EmitterType`.
    created_tick :
        Sim tick at creation (for TTL bookkeeping).
    ttl :
        Life-time in sim ticks before the record is removed.
    """
    id:                   int         = 0
    pos:                  Vec3        = field(default_factory=Vec3.zero)
    band_energy_audible:  float       = 0.0
    band_energy_infra:    float       = 0.0
    directivity:          float       = 0.0
    emitter_type:         EmitterType = EmitterType.IMPACT_MEDIUM
    created_tick:         int         = 0
    ttl:                  int         = 60   # ~1 s at 60 Hz

    @property
    def total_energy(self) -> float:
        return self.band_energy_audible + self.band_energy_infra


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_QUANTISE_GRID = 4.0  # metres


def _quantise(pos: Vec3) -> Vec3:
    """Snap a position to a coarse grid for clustering."""
    return Vec3(
        round(pos.x / _QUANTISE_GRID) * _QUANTISE_GRID,
        round(pos.y / _QUANTISE_GRID) * _QUANTISE_GRID,
        round(pos.z / _QUANTISE_GRID) * _QUANTISE_GRID,
    )


def _sector_key(pos: Vec3, sector_size: float = 100.0) -> Tuple[int, int]:
    """Return a 2-D integer sector index for *pos*."""
    return (int(math.floor(pos.x / sector_size)),
            int(math.floor(pos.z / sector_size)))


# ---------------------------------------------------------------------------
# EmitterAggregator
# ---------------------------------------------------------------------------

class EmitterAggregator:
    """Budget-limited emitter registry with spatial clustering.

    Parameters
    ----------
    config :
        Optional dict; reads ``audio_world.*`` keys.
    """

    _DEFAULT_MAX_EMITTERS = 256
    _DEFAULT_SECTOR_SIZE  = 100.0  # metres

    def __init__(self, config: Optional[dict] = None) -> None:
        awcfg = (config or {}).get("audio_world", {}) or {}
        self._max_emitters: int   = int(awcfg.get("max_emitters", self._DEFAULT_MAX_EMITTERS))
        self._sector_size:  float = float(awcfg.get("sector_size", self._DEFAULT_SECTOR_SIZE))

        self._emitters:   List[AcousticEmitterRecord]  = []
        self._bg_energy:  Dict[Tuple[int, int], float] = {}
        self._next_id:    int                          = 1

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def add(self, record: AcousticEmitterRecord) -> None:
        """Register a new emitter record.

        Position is quantised to the internal clustering grid.
        If the record's ``id`` is 0, a unique id is assigned.
        """
        if record.id == 0:
            record.id = self._next_id
            self._next_id += 1
        record.pos = _quantise(record.pos)
        self._emitters.append(record)

    def tick(self, sim_tick: int) -> None:
        """Advance one sim tick: age emitters, apply budget, cluster.

        Parameters
        ----------
        sim_tick :
            Current absolute simulation tick counter.
        """
        # 1. Expire old emitters
        self._emitters = [
            e for e in self._emitters
            if (sim_tick - e.created_tick) < e.ttl
        ]

        # 2. If within budget, done
        if len(self._emitters) <= self._max_emitters:
            return

        # 3. Sort by total energy descending, keep top-N
        self._emitters.sort(key=lambda e: e.total_energy, reverse=True)
        to_drop = self._emitters[self._max_emitters:]
        self._emitters = self._emitters[:self._max_emitters]

        # 4. Accumulate dropped emitters into per-sector background
        for e in to_drop:
            key = _sector_key(e.pos, self._sector_size)
            self._bg_energy[key] = self._bg_energy.get(key, 0.0) + e.total_energy

        # 5. Gently decay background energy each tick
        decay = 0.99
        self._bg_energy = {k: v * decay for k, v in self._bg_energy.items() if v * decay > 1e-6}

    @property
    def active_emitters(self) -> List[AcousticEmitterRecord]:
        """Currently active (budget-culled) emitter records."""
        return list(self._emitters)

    def background_energy(self, sector_id: Tuple[int, int]) -> float:
        """Accumulated energy from culled emitters in *sector_id*."""
        return self._bg_energy.get(sector_id, 0.0)
