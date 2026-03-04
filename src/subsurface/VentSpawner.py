"""VentSpawner — Stage 67 vent event spawner.

Iterates over a :class:`SubsurfaceFieldGrid`, uses :class:`VentDetector`
to find active tiles, and produces :class:`VentEvent` objects consumed by
lava surface flow and the adapter layer.

VentEvent is server-authoritative (deterministic from tile_idx + game_time).

Public API
----------
VentSpawner(config=None)
  .tick(grid, detector, game_time, dt) → list[VentEvent]

VentEvent (dataclass)
  .tile_idx   int
  .intensity  float  [0, 1]
  .game_time  float
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from src.subsurface.SubsurfaceFieldGrid import SubsurfaceFieldGrid
from src.subsurface.VentDetector        import VentDetector


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


@dataclass
class VentEvent:
    """A single vent activation event."""

    tile_idx:  int   = 0
    intensity: float = 0.0
    game_time: float = 0.0


class VentSpawner:
    """Scan the subsurface grid and spawn vent events where thresholds are met.

    Parameters
    ----------
    config :
        Optional dict; reads ``subsurface67.*`` keys.
    """

    _DEFAULT_MAX_VENTS_PER_TICK = 4
    _DEFAULT_COOLDOWN           = 30.0   # seconds before same tile can vent again

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = (config or {}).get("subsurface67", {}) or {}
        self._max_vents  = int(cfg.get("max_vents_per_tick",    self._DEFAULT_MAX_VENTS_PER_TICK))
        self._cooldown   = float(cfg.get("vent_cooldown_sec",   self._DEFAULT_COOLDOWN))

        self._last_vent_time: dict = {}   # tile_idx → last vent game_time

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def tick(
        self,
        grid:       SubsurfaceFieldGrid,
        detector:   VentDetector,
        game_time:  float,
        dt:         float,
    ) -> List[VentEvent]:
        """Return list of new vent events this tick.

        After a vent fires on a tile, the magma pressure on that tile is
        reduced (pressure release) and a cooldown is set.
        """
        events: List[VentEvent] = []
        for i in range(grid.tile_count):
            if len(events) >= self._max_vents:
                break
            # Cooldown guard
            last = self._last_vent_time.get(i, -1e9)
            if game_time - last < self._cooldown:
                continue

            t = grid.tile(i)
            if detector.should_activate(t):
                intensity = detector.activation_intensity(t)
                events.append(VentEvent(tile_idx=i, intensity=intensity, game_time=game_time))
                self._last_vent_time[i] = game_time
                # Pressure released by vent
                t.magmaPressureProxy = _clamp(t.magmaPressureProxy - intensity * 0.5)
                t.subsurfaceStress   = _clamp(t.subsurfaceStress   - intensity * 0.3)
                t.ventPotential      = t.compute_vent_potential()

        return events
