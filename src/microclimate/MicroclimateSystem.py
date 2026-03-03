"""MicroclimateSystem — Stage 49 main orchestrator.

Manages per-chunk :class:`~src.microclimate.MicroclimateState.MicroclimateState`
computation with a tick budget, LRU caching, and LOD support.

Design
------
* A ``microclimateTick`` runs at ``micro.tick_hz`` (0.5–2 Hz).
* Only ``max_chunks_per_tick`` LRU-active chunks are recomputed per tick.
* Chunks beyond ``lod.far_dist`` receive reduced sampling counts.
* Shelter cache is keyed per (chunk_key, wind_bucket) to avoid recomputing
  when only the wind direction bucket changes.

Public API
----------
MicroclimateSystem(config=None, height_fn=None)
  .tick(active_chunks, player_pos, wind_dir_2d, insolation, dt) → None
  .get_state(chunk_key) → MicroclimateState
  .debug_info() → dict
"""
from __future__ import annotations

import math
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple

from src.math.Vec3 import Vec3
from src.microclimate.MicroclimateState      import MicroclimateState
from src.microclimate.ShelterEstimator       import ShelterEstimator
from src.microclimate.ChannelEstimator       import ChannelEstimator
from src.microclimate.DustTrapEstimator      import DustTrapEstimator
from src.microclimate.ColdBiasEstimator      import ColdBiasEstimator
from src.microclimate.ThermalInertiaEstimator import ThermalInertiaEstimator
from src.microclimate.EchoPotentialEstimator  import EchoPotentialEstimator


HeightFn = Callable[[float, float], float]
ChunkKey = Tuple  # e.g. (int, int)


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


class MicroclimateSystem:
    """Orchestrates microclimate state for all active chunks.

    Parameters
    ----------
    config :
        Optional dict; reads ``micro.*`` keys.
    height_fn :
        ``(x: float, z: float) -> float`` terrain height query shared by
        all sub-estimators.
    """

    _DEFAULT_TICK_HZ          = 1.0
    _DEFAULT_MAX_CHUNKS_TICK  = 8
    _DEFAULT_NEAR_DIST        = 100.0
    _DEFAULT_FAR_SAMPLE_REDUCE = 0.25

    def __init__(
        self,
        config:    Optional[dict] = None,
        height_fn: Optional[HeightFn] = None,
    ) -> None:
        cfg  = config or {}
        mcfg = cfg.get("micro", {}) or {}
        lod  = mcfg.get("lod", {}) or {}

        self._tick_hz:    float = float(mcfg.get("tick_hz",          self._DEFAULT_TICK_HZ))
        self._max_chunks: int   = int(mcfg.get("max_chunks_per_tick", self._DEFAULT_MAX_CHUNKS_TICK))
        self._near_dist:  float = float(lod.get("near_dist",          self._DEFAULT_NEAR_DIST))
        self._far_reduce: float = float(lod.get("far_sample_reduce",  self._DEFAULT_FAR_SAMPLE_REDUCE))

        hfn = height_fn or (lambda x, z: 0.0)

        self._shelter_est  = ShelterEstimator(config, hfn)
        self._channel_est  = ChannelEstimator(config, hfn)
        self._dusttrap_est = DustTrapEstimator(config, hfn)
        self._coldbias_est = ColdBiasEstimator(config)
        self._inertia_est  = ThermalInertiaEstimator(config, hfn)
        self._echo_est     = EchoPotentialEstimator(config, hfn)

        # LRU cache of computed states: OrderedDict keeps insertion order
        self._states: OrderedDict[ChunkKey, MicroclimateState] = OrderedDict()

        # Tick accumulator
        self._acc: float = 0.0
        self._tick_interval: float = 1.0 / max(self._tick_hz, 1e-3)

        # Internal counters for debug
        self._ticks_fired: int = 0
        self._samples_last_tick: int = 0

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def tick(
        self,
        active_chunks:  List[Tuple[ChunkKey, Vec3]],
        player_pos:     Vec3,
        wind_dir_2d:    Vec3,
        insolation:     float = 0.5,
        dt:             float = 1.0 / 20.0,
    ) -> None:
        """Advance the microclimate system.

        Parameters
        ----------
        active_chunks :
            List of (chunk_key, chunk_centre_world_pos) for all chunks
            that should be kept up to date.
        player_pos :
            Player world-space position (for LOD distance calculation).
        wind_dir_2d :
            Current macro wind direction (x, z used).
        insolation :
            Current macro insolation [0..1].
        dt :
            Elapsed real/simulation time [s].
        """
        self._acc += dt
        if self._acc < self._tick_interval:
            return
        self._acc -= self._tick_interval
        self._ticks_fired += 1

        # Sort chunks by distance to player (nearest first → LRU priority)
        def _dist(item: Tuple) -> float:
            _, cpos = item
            dx = cpos.x - player_pos.x
            dz = cpos.z - player_pos.z
            return math.sqrt(dx * dx + dz * dz)

        sorted_chunks = sorted(active_chunks, key=_dist)
        budget = self._max_chunks
        sample_count = 0

        for chunk_key, chunk_pos in sorted_chunks:
            if budget <= 0:
                break
            dist = _dist((chunk_key, chunk_pos))
            state = self._compute_state(
                chunk_key, chunk_pos, wind_dir_2d, insolation, dist
            )
            # Move to end (most-recently-used)
            if chunk_key in self._states:
                self._states.move_to_end(chunk_key)
            self._states[chunk_key] = state
            budget -= 1
            sample_count += 1

        self._samples_last_tick = sample_count

        # Evict old (unused) entries beyond twice the active set
        max_cache = max(len(active_chunks) * 2, 16)
        while len(self._states) > max_cache:
            self._states.popitem(last=False)

    def get_state(self, chunk_key: ChunkKey) -> MicroclimateState:
        """Return the most-recently-computed state for a chunk.

        Returns a neutral (all-zero) state if the chunk has not been computed.
        """
        return self._states.get(chunk_key, MicroclimateState())

    def debug_info(self) -> dict:
        """Return diagnostic counters."""
        return {
            "ticks_fired":        self._ticks_fired,
            "cached_chunks":      len(self._states),
            "samples_last_tick":  self._samples_last_tick,
        }

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _compute_state(
        self,
        chunk_key:   ChunkKey,
        pos:         Vec3,
        wind_dir_2d: Vec3,
        insolation:  float,
        dist:        float,
    ) -> MicroclimateState:
        """Compute a full MicroclimateState for one chunk position."""
        shelter_result = self._shelter_est.estimate(pos, wind_dir_2d, chunk_key)
        shelter        = shelter_result.shelter

        channel = self._channel_est.estimate(pos, wind_dir_2d)
        dust_trap = self._dusttrap_est.estimate(pos, shelter, channel)
        cold_bias = self._coldbias_est.estimate(insolation)

        # Far chunks get simplified estimates (skip sky samples)
        if dist > self._near_dist:
            inertia = shelter * 0.8   # rough proxy without sampling
            echo    = shelter * 0.7
        else:
            inertia = self._inertia_est.estimate(pos)
            echo    = self._echo_est.estimate(pos)

        return MicroclimateState(
            windShelter=shelter,
            windChannel=channel,
            dustTrap=dust_trap,
            coldBias=cold_bias,
            thermalInertia=inertia,
            echoPotential=echo,
        )
