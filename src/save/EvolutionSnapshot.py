"""EvolutionSnapshot — Stage 50 binary save / restore for PlanetEvolutionState
and PlanetTimeController.

Format
------
Magic:        b"EVS1"  (4 bytes)
Header:       world_seed(4) + planet_time(8, double) + sim_time(8, double)
              + seasonal_phase(4, float) + width(2) + height(2)
Fields:       4 × (width × height) bytes — one byte per tile per field,
              in order: dustReservoir, crustStability, slopeCreep, iceBelt

Total size for a 64 × 32 grid:
  4 + (4 + 8 + 8 + 4 + 2 + 2) + 4 × 2048 = 4 + 28 + 8192 = 8224 bytes

Public API
----------
EvolutionSnapshot()
  .save(state, time_ctrl, world_seed=0) -> bytes
  .load(data: bytes) -> (PlanetEvolutionState, dict)
      dict keys: world_seed, planet_time, sim_time
"""
from __future__ import annotations

import struct
from typing import Dict, Tuple

from src.evolution.PlanetEvolutionState import PlanetEvolutionState

_MAGIC     = b"EVS1"
_HDR_FMT   = "!IddfHH"    # world_seed(I) + planet_time(d) + sim_time(d)
                            # + seasonal_phase(f) + width(H) + height(H)
_HDR_SIZE  = struct.calcsize(_HDR_FMT)
_PREAMBLE  = 4 + _HDR_SIZE   # magic + header


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


class EvolutionSnapshot:
    """Binary serialiser for :class:`PlanetEvolutionState`.

    Usage::

        snap = EvolutionSnapshot()
        blob = snap.save(state, time_ctrl, world_seed=42)
        state2, meta = snap.load(blob)
        # meta["planet_time"], meta["sim_time"], meta["world_seed"]
    """

    def save(
        self,
        state: PlanetEvolutionState,
        time_ctrl=None,
        world_seed: int = 0,
    ) -> bytes:
        """Serialise *state* (and optionally *time_ctrl*) to bytes.

        Parameters
        ----------
        state      : PlanetEvolutionState to persist.
        time_ctrl  : Optional PlanetTimeController; if None, times default to 0.
        world_seed : Integer world seed to embed in the snapshot.
        """
        planet_time = time_ctrl.planet_time if time_ctrl is not None else 0.0
        sim_time    = time_ctrl.sim_time    if time_ctrl is not None else 0.0

        header = struct.pack(
            _HDR_FMT,
            world_seed & 0xFFFF_FFFF,
            planet_time,
            sim_time,
            state.seasonalInsolationPhase,
            state.width,
            state.height,
        )

        n = state.size()
        fields = bytearray(4 * n)
        for i in range(n):
            fields[i]         = int(_clamp(state.dustReservoirMap[i])    * 255)
            fields[n + i]     = int(_clamp(state.crustStabilityMap[i])   * 255)
            fields[2 * n + i] = int(_clamp(state.slopeCreepMap[i])       * 255)
            fields[3 * n + i] = int(_clamp(state.iceBeltDistribution[i]) * 255)

        return _MAGIC + header + bytes(fields)

    def load(
        self,
        data: bytes,
    ) -> Tuple[PlanetEvolutionState, Dict]:
        """Deserialise a blob produced by :meth:`save`.

        Returns
        -------
        (state, meta)
            ``state`` is a newly-created :class:`PlanetEvolutionState`.
            ``meta`` is a dict with keys ``world_seed``, ``planet_time``,
            ``sim_time``.
        """
        if data[:4] != _MAGIC:
            raise ValueError("EvolutionSnapshot: bad magic bytes")

        (world_seed, planet_time, sim_time,
         seasonal_phase, width, height) = struct.unpack_from(
            _HDR_FMT, data, offset=4
        )

        state = PlanetEvolutionState(width=int(width), height=int(height))
        state.seasonalInsolationPhase = float(seasonal_phase)

        n = int(width) * int(height)
        offset = _PREAMBLE

        for i in range(n):
            state.dustReservoirMap[i]    = data[offset + i]           / 255.0
            state.crustStabilityMap[i]   = data[offset + n + i]       / 255.0
            state.slopeCreepMap[i]       = data[offset + 2 * n + i]   / 255.0
            state.iceBeltDistribution[i] = data[offset + 3 * n + i]   / 255.0

        meta = {
            "world_seed":   int(world_seed),
            "planet_time":  float(planet_time),
            "sim_time":     float(sim_time),
        }
        return state, meta
