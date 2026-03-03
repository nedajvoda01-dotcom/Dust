"""InstabilitySystem — Stage 52 main self-organizing instability controller.

Orchestrates:

1. Ingestion of external fields (slope, dust, stress, insolation).
2. Field update via ShearStressEstimator and ThermalFractureModel.
3. Per-tile threshold checks via CrustFailureModel and DustAvalancheModel.
4. Deterministic BFS cascade via CascadeProcessor.
5. Output of events to callers (material adapter, audio adapter).

Budget
------
``max_tiles_per_tick`` caps how many tiles are processed per :meth:`tick`
call.  Remaining tiles are carried over to the next tick via an internal
offset pointer (round-robin, fixed order for determinism).

Public API
----------
InstabilitySystem(config=None)
  .ingest_fields(state, slope_map, dust_map, stress_map,
                 evolution_state, memory_state, dt)   → None
  .tick(state, slope_map, sim_tick, dt)
      → List[instability event objects]
"""
from __future__ import annotations

from typing import Any, List, Optional

from src.instability.InstabilityState    import InstabilityState
from src.instability.ShearStressEstimator import ShearStressEstimator
from src.instability.CrustFailureModel   import CrustFailureModel
from src.instability.DustAvalancheModel  import DustAvalancheModel
from src.instability.ThermalFractureModel import ThermalFractureModel
from src.instability.CascadeProcessor    import CascadeProcessor


class InstabilitySystem:
    """Main instability tick controller.

    Parameters
    ----------
    config :
        Optional dict; reads ``instability.*`` keys.
    """

    _DEFAULT_MAX_TILES   = 64
    _DEFAULT_CASCADE_INCR = 0.12

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = (config or {}).get("instability", {}) or {}
        self._enabled:     bool  = bool(cfg.get("enable", True))
        self._max_tiles:   int   = int( cfg.get("max_tiles_per_tick", self._DEFAULT_MAX_TILES))
        self._cascade_incr:float = float(cfg.get("cascade_stress_incr", self._DEFAULT_CASCADE_INCR))

        self._shear_est    = ShearStressEstimator(config)
        self._crust_model  = CrustFailureModel(config)
        self._dust_model   = DustAvalancheModel(config)
        self._thermal_model= ThermalFractureModel(config)
        self._cascade      = CascadeProcessor(config)

        self._tile_offset: int = 0   # round-robin budget cursor

    # ------------------------------------------------------------------
    # Field ingestion
    # ------------------------------------------------------------------

    def ingest_fields(
        self,
        state:      InstabilityState,
        slope_map:  Optional[List[float]],
        dust_map:   Optional[List[float]],
        stress_map: Optional[List[float]],
        insolation_map:     Optional[List[float]] = None,
        prev_insolation_map:Optional[List[float]] = None,
        dt: float = 1.0,
    ) -> None:
        """Update instability fields from external inputs.

        Parameters
        ----------
        state               : InstabilityState to update.
        slope_map           : Per-tile slope [0..1].
        dust_map            : Per-tile dust thickness [0..1].
        stress_map          : Per-tile memory stress accumulation [0..1].
        insolation_map      : Current insolation per tile [0..1].
        prev_insolation_map : Previous tick insolation (for gradient).
        dt                  : Elapsed time in seconds.
        """
        if not self._enabled:
            return

        # 1. Shear stress update
        self._shear_est.tick(state, slope_map, dust_map, stress_map, dt)

        # 2. Dust load: mirror dust_map into the field (slow tracking)
        if dust_map is not None:
            n = state.size()
            for i in range(n):
                delta = (dust_map[i] - state.dustLoadField[i]) * 0.05 * dt
                state.dustLoadField[i] = _clamp(state.dustLoadField[i] + delta)

        # 3. Mass overhang: combine slope_map + stress_map
        if slope_map is not None and stress_map is not None:
            n = state.size()
            for i in range(n):
                gain = slope_map[i] * stress_map[i] * 0.02 * dt
                state.massOverhangField[i] = _clamp(state.massOverhangField[i] + gain)

        # 4. Thermal gradient from insolation change
        if insolation_map is not None and prev_insolation_map is not None:
            n = state.size()
            for i in range(n):
                delta = abs(insolation_map[i] - prev_insolation_map[i])
                self._thermal_model.update_gradient(state, i, delta, dt)

    # ------------------------------------------------------------------
    # Tick
    # ------------------------------------------------------------------

    def tick(
        self,
        state:     InstabilityState,
        slope_map: Optional[List[float]] = None,
        sim_tick:  int = 0,
        dt:        float = 1.0,
    ) -> List[Any]:
        """Process one instability tick.

        Checks up to ``max_tiles_per_tick`` tiles (round-robin) for
        threshold crossings.  Cascade events are applied deterministically.

        Parameters
        ----------
        state     : InstabilityState (modified in-place).
        slope_map : Per-tile slope [0..1] for avalanche model.
        sim_tick  : Current simulation tick counter.
        dt        : Elapsed time (unused internally; kept for API parity).

        Returns
        -------
        List of event objects (CrustFailureEvent | DustAvalancheEvent |
        ThermalFractureEvent) that fired this tick.
        """
        if not self._enabled:
            return []

        n = state.size()
        events: List[Any] = []
        cascade_seeds: List[int] = []

        processed = 0
        i = self._tile_offset

        while processed < self._max_tiles:
            tile = i % n

            # --- CrustFailure ---
            ev = self._crust_model.process(state, tile)
            if ev is not None:
                events.append(ev)
                cascade_seeds.append(tile)

            # --- DustAvalanche ---
            ev2 = self._dust_model.process(state, tile, slope_map)
            if ev2 is not None:
                events.append(ev2)
                cascade_seeds.append(tile)

            # --- ThermalFracture ---
            ev3 = self._thermal_model.process(state, tile)
            if ev3 is not None:
                events.append(ev3)
                # thermal fracture raises crust potential → cascade via crust field
                cascade_seeds.append(tile)

            processed += 1
            i += 1

        # Advance round-robin cursor
        self._tile_offset = i % n

        # --- Cascade ---
        if cascade_seeds:
            crust_threshold = self._crust_model._threshold
            self._cascade.run(
                state, cascade_seeds,
                "shearStressField",
                self._cascade_incr,
                crust_threshold,
            )

        return events
