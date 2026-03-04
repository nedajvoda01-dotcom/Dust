"""AtmosphereSystem — Stage 64 main atmospheric controller.

Owns the :class:`~src.atmo.GlobalFieldGrid.GlobalFieldGrid` and drives
all field updates via :class:`~src.atmo.FieldAdvection.FieldAdvection`.

Clients can query:
  * Per-tile atmospheric state.
  * Derived weather regime for any tile.
  * LOD-aware snapshot for network replication.

Tick rate is configurable; by default the global atmosphere updates at
0.5 Hz (every 2 seconds game time).  Micro-scale gustiness is computed
procedurally from a deterministic hash of seed + time, keeping client/
server bit-identical without per-client randomness.

Public API
----------
AtmosphereSystem(width, height, config=None, seed=0)
  .tick(dt, insolation_map=None)       → None
  .get_tile(ix, iy)                    → AtmoTile
  .get_regime(ix, iy)                  → str (WeatherRegime constant)
  .get_local_params(ix, iy, micro)     → LocalAtmoParams
  .snapshot_near(cx, cy, radius)       → bytes  (full-fidelity)
  .snapshot_far(cx, cy, radius)        → bytes  (reduced fidelity)
  .grid_hash()                         → str
  .total_aerosol()                     → float
"""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Callable, Optional

from src.atmo.GlobalFieldGrid    import GlobalFieldGrid, AtmoTile
from src.atmo.FieldAdvection     import FieldAdvection
from src.atmo.WeatherRegimeDetector import WeatherRegimeDetector, WeatherRegime


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


# ---------------------------------------------------------------------------
# LocalAtmoParams — output struct for downstream systems
# ---------------------------------------------------------------------------

@dataclass
class LocalAtmoParams:
    """Locally adjusted atmospheric parameters for downstream systems.

    All float fields are [0..1] unless noted.
    """
    wind_speed:        float = 0.0   # local wind magnitude
    wind_x:            float = 0.0   # local wind vector X (−1..1)
    wind_y:            float = 0.0   # local wind vector Y (−1..1)
    aerosol:           float = 0.0   # local airborne dust
    humidity:          float = 0.2   # local humidity proxy
    temperature:       float = 0.5   # local temperature proxy
    pressure:          float = 0.5   # local pressure proxy
    fog_potential:     float = 0.0   # local fog potential
    storm_potential:   float = 0.0   # local storm potential
    visibility:        float = 1.0   # local visibility proxy
    electro:           float = 0.0   # local electro activity
    wind_load:         float = 0.0   # effective wind load on character
    precipitation_rate: float = 0.0  # effective precipitation rate (snow/dust)
    thermal_effect:    float = 0.0   # thermal load on character (signed, −1..1)
    regime:            str  = WeatherRegime.CLEAR


# ---------------------------------------------------------------------------
# AtmosphereSystem
# ---------------------------------------------------------------------------

class AtmosphereSystem:
    """Top-level controller for Stage 64 atmospheric dynamics.

    Parameters
    ----------
    width, height :
        Coarse grid dimensions.
    config :
        Optional config dict; reads ``atmo64.*``.
    seed :
        Deterministic seed for procedural micro-noise.
    """

    _DEFAULT_TICK_HZ = 0.5   # global atmosphere tick rate

    def __init__(
        self,
        width:  int,
        height: int,
        config: Optional[dict] = None,
        seed:   int = 0,
    ) -> None:
        self._config = config or {}
        acfg = self._config.get("atmo64", {}) or {}

        self._tick_hz: float = float(acfg.get("tick_hz", self._DEFAULT_TICK_HZ))
        self._tick_interval: float = 1.0 / max(self._tick_hz, 1e-6)
        self._accum: float = 0.0
        self._ticks_fired: int = 0

        self._grid     = GlobalFieldGrid(width, height, seed=seed)
        self._advect   = FieldAdvection(config)
        self._detector = WeatherRegimeDetector(config)

    # ------------------------------------------------------------------
    # Tick
    # ------------------------------------------------------------------

    def tick(
        self,
        dt: float,
        insolation_map: Optional[Callable[[int, int], float]] = None,
    ) -> None:
        """Advance the atmosphere by *dt* seconds of wall/game time."""
        self._accum += dt
        while self._accum >= self._tick_interval:
            self._advect.step(self._grid, self._tick_interval, insolation_map)
            self._accum -= self._tick_interval
            self._ticks_fired += 1

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_tile(self, ix: int, iy: int) -> AtmoTile:
        """Return raw field tile."""
        return self._grid.tile(ix, iy)

    def get_regime(self, ix: int, iy: int) -> str:
        """Return the current weather regime for tile (ix, iy)."""
        t  = self._grid.tile(ix, iy)
        fp = self._grid.fog_potential(ix, iy)
        fi = self._grid.front_intensity(ix, iy)
        sp = self._grid.storm_potential(ix, iy)
        return self._detector.detect(t, fp, fi, sp)

    def get_local_params(
        self,
        ix: int,
        iy: int,
        shelter:  float = 0.0,
        channeling: float = 0.0,
        cold_bias:  float = 0.0,
        dust_trap:  float = 0.0,
        thermal_inertia: float = 0.0,
    ) -> LocalAtmoParams:
        """Return LocalAtmoParams combining macro fields with microclimate offsets.

        Formulas from spec §4:
            LocalWind    = MacroWind × (1 − shelter) + MacroWind × channeling
            LocalAerosol = MacroAerosol + dustTrap × deposition − channeling × dispersion
            LocalTemp    = MacroTemp − coldBias × delta
        """
        t   = self._grid.tile(ix, iy)
        fp  = self._grid.fog_potential(ix, iy)
        fi  = self._grid.front_intensity(ix, iy)
        sp  = self._grid.storm_potential(ix, iy)
        vis = self._grid.visibility_proxy(ix, iy)
        regime = self._detector.detect(t, fp, fi, sp)

        # Microclimate adjustments
        local_wind = _clamp(t.wind_speed * (1.0 - shelter) + t.wind_speed * channeling)
        local_dust = _clamp(t.aerosol + dust_trap * 0.4 - channeling * 0.3)
        cold_delta = 0.3
        local_temp = _clamp(t.temperature - cold_bias * cold_delta)

        wind_load  = _clamp(local_wind * 0.8 + sp * 0.2)
        precip     = _clamp(
            t.humidity * 0.5 if regime == WeatherRegime.SNOW_DEPOSITION
            else local_dust * 0.3 if regime == WeatherRegime.DUST_STORM
            else 0.0
        )
        # Thermal effect: positive = hot, negative = cold
        thermal    = _clamp(local_temp * 2.0 - 1.0, -1.0, 1.0)

        return LocalAtmoParams(
            wind_speed        = local_wind,
            wind_x            = t.wind_x,
            wind_y            = t.wind_y,
            aerosol           = local_dust,
            humidity          = t.humidity,
            temperature       = local_temp,
            pressure          = t.pressure,
            fog_potential     = fp,
            storm_potential   = sp,
            visibility        = vis,
            electro           = t.electro,
            wind_load         = wind_load,
            precipitation_rate = precip,
            thermal_effect    = thermal,
            regime            = regime,
        )

    # ------------------------------------------------------------------
    # Aggregates
    # ------------------------------------------------------------------

    def total_aerosol(self) -> float:
        """Grid-wide sum of aerosol proxy (for conservation checks)."""
        return self._grid.total_aerosol()

    def grid_hash(self) -> str:
        return self._grid.grid_hash()

    def debug_info(self) -> dict:
        return {
            "ticks_fired": self._ticks_fired,
            "grid_w": self._grid.width,
            "grid_h": self._grid.height,
            "total_aerosol": self.total_aerosol(),
        }

    # ------------------------------------------------------------------
    # Replication / LOD snapshots
    # ------------------------------------------------------------------

    def snapshot_near(self, cx: int, cy: int, radius: int) -> bytes:
        """Full-fidelity snapshot: all fields for tiles within *radius*."""
        return self._snapshot(cx, cy, radius, reduced=False)

    def snapshot_far(self, cx: int, cy: int, radius: int) -> bytes:
        """Reduced-fidelity snapshot: Vis + wind dir only for far tiles."""
        return self._snapshot(cx, cy, radius, reduced=True)

    def _snapshot(self, cx: int, cy: int, radius: int, reduced: bool) -> bytes:
        w, h = self._grid.width, self._grid.height
        tiles_out = []
        for iy in range(h):
            for ix in range(w):
                dist = max(abs(ix - cx), abs(iy - cy))
                if dist > radius:
                    continue
                t = self._grid.tile(ix, iy)
                if reduced:
                    # Only Vis proxy (1 byte) + wind direction as uint8 angle
                    vis = self._grid.visibility_proxy(ix, iy)
                    angle = math.atan2(t.wind_y, t.wind_x)
                    vis_u8   = int(round(vis * 255)) & 0xFF
                    angle_u8 = int(round((angle + math.pi) / (2 * math.pi) * 255)) & 0xFF
                    tiles_out.append(struct.pack("!HHBBxx", ix, iy, vis_u8, angle_u8))
                else:
                    tiles_out.append(
                        struct.pack("!BB", int(ix), int(iy)) + t.to_bytes()
                    )
        header = struct.pack("!?HH", reduced, cx, cy)
        return header + b"".join(tiles_out)
