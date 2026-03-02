"""LongHorizonEvolutionSystem — Stage 30 slow planetary evolution.

Implements "world lives for months": dust migration, ice-film growth/melt,
rock fracture fatigue, and slope relaxation on a low-res planetary grid.
All processes are server-authoritative, deterministic, and tile-budgeted so
they cannot blow up server performance.

Architecture
------------
EvolutionFields
    Seven flat float arrays (one entry per tile) at low resolution
    (default 64 × 32).  All values are clamped to [0, 1].

LongHorizonEvolutionSystem
    Owns the fields and advances them on a slow fixed tick (default 60 s).
    Per tick only ``tiles_per_tick`` tiles are updated to spread the cost.
    Provides snapshot/delta serialisation for network sync and persistence.

Process summary
---------------
1. Dust transport  (DustThickness ↔ wind + DustLiftPotential)
2. Ice film        (IceFilm ↔ temperature + ring-shadow + night)
3. Fracture fatigue (FractureFatigue ← thermal cycling + stress)
4. Slope relaxation (SlopeStability → DebrisOverlay events)

Config keys (under ``evo.*``)
-----------------------------
enable, tick_seconds, tiles_per_tick, snapshot_interval_sec,
delta_interval_sec, dust_lift_k, dust_dep_k, ice_form_k, ice_melt_k,
fatigue_tempCycle_k, fatigue_stress_k, slope_relax_k,
max_slope_events_per_hour, network_quantization_bits, grid_w, grid_h
"""
from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.core.Config import Config

_TWO_PI  = 2.0 * math.pi
_HALF_PI = math.pi * 0.5


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


# ---------------------------------------------------------------------------
# EvolutionFields — low-res planetary state grid
# ---------------------------------------------------------------------------

@dataclass
class EvolutionFields:
    """Seven evolution fields on a W × H tile grid.

    All values are normalised floats in [0, 1].
    The flat index convention is ``idx = y * width + x``.
    """
    width:  int
    height: int

    dust_thickness:    List[float] = field(default_factory=list)
    dust_mobility:     List[float] = field(default_factory=list)
    ice_film:          List[float] = field(default_factory=list)
    surface_freshness: List[float] = field(default_factory=list)
    fracture_fatigue:  List[float] = field(default_factory=list)
    slope_stability:   List[float] = field(default_factory=list)
    regolith_cohesion: List[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        n = self.width * self.height
        defaults = (
            ("dust_thickness",    0.2),
            ("dust_mobility",     0.5),
            ("ice_film",          0.0),
            ("surface_freshness", 1.0),
            ("fracture_fatigue",  0.0),
            ("slope_stability",   1.0),
            ("regolith_cohesion", 0.5),
        )
        for attr, init_val in defaults:
            if not getattr(self, attr):
                setattr(self, attr, [init_val] * n)

    # ------------------------------------------------------------------
    def size(self) -> int:
        return self.width * self.height

    def clamp_all(self) -> None:
        """Clamp every field element to [0, 1]."""
        for attr in (
            "dust_thickness", "dust_mobility", "ice_film",
            "surface_freshness", "fracture_fatigue",
            "slope_stability", "regolith_cohesion",
        ):
            lst = getattr(self, attr)
            for i in range(len(lst)):
                if lst[i] < 0.0:
                    lst[i] = 0.0
                elif lst[i] > 1.0:
                    lst[i] = 1.0

    # ------------------------------------------------------------------
    def _field_hash(self, lst: List[float]) -> int:
        """Fast integer hash over a float list (for state verification)."""
        # Pack as int8 (quantised to 0-255) for speed
        packed = bytes(int(_clamp(v) * 255) for v in lst)
        return int.from_bytes(hashlib.md5(packed).digest()[:4], "little")

    def state_hash(self) -> str:
        """A short hex digest covering all 7 fields (for sync checks)."""
        combined = (
            self._field_hash(self.dust_thickness)
            ^ self._field_hash(self.dust_mobility)
            ^ self._field_hash(self.ice_film)
            ^ self._field_hash(self.surface_freshness)
            ^ self._field_hash(self.fracture_fatigue)
            ^ self._field_hash(self.slope_stability)
            ^ self._field_hash(self.regolith_cohesion)
        )
        return format(combined & 0xFFFF_FFFF, "08x")

    # ------------------------------------------------------------------
    def to_snapshot_dict(self, tick_index: int, bits: int = 8) -> dict:
        """Serialise to a dict suitable for JSON / network transport.

        Parameters
        ----------
        tick_index: Current evolution tick counter.
        bits:       Quantisation depth (8 = int8, 16 = int16).
        """
        scale = (1 << bits) - 1

        def _quant(lst: List[float]) -> List[int]:
            return [int(_clamp(v) * scale) for v in lst]

        return {
            "type":       "EVOLUTION_SNAPSHOT",
            "tickIndex":  tick_index,
            "width":      self.width,
            "height":     self.height,
            "bits":       bits,
            "stateHash":  self.state_hash(),
            "fields": {
                "dustThickness":    _quant(self.dust_thickness),
                "dustMobility":     _quant(self.dust_mobility),
                "iceFilm":          _quant(self.ice_film),
                "surfaceFreshness": _quant(self.surface_freshness),
                "fractureFatigue":  _quant(self.fracture_fatigue),
                "slopeStability":   _quant(self.slope_stability),
                "regolithCohesion": _quant(self.regolith_cohesion),
            },
        }

    @classmethod
    def from_snapshot_dict(cls, snap: dict) -> "EvolutionFields":
        """Reconstruct from a snapshot dict (reverse of *to_snapshot_dict*)."""
        w    = int(snap["width"])
        h    = int(snap["height"])
        bits = int(snap.get("bits", 8))
        scale = (1 << bits) - 1
        f = snap["fields"]

        def _dequant(lst: List[int]) -> List[float]:
            return [v / scale for v in lst]

        obj = cls.__new__(cls)
        obj.width             = w
        obj.height            = h
        obj.dust_thickness    = _dequant(f["dustThickness"])
        obj.dust_mobility     = _dequant(f["dustMobility"])
        obj.ice_film          = _dequant(f["iceFilm"])
        obj.surface_freshness = _dequant(f["surfaceFreshness"])
        obj.fracture_fatigue  = _dequant(f["fractureFatigue"])
        obj.slope_stability   = _dequant(f["slopeStability"])
        obj.regolith_cohesion = _dequant(f["regolithCohesion"])
        return obj

    def to_delta_dict(
        self, tick_index: int, dirty: List[int], bits: int = 8
    ) -> dict:
        """Serialise only the *dirty* tile indices as a sparse delta."""
        scale = (1 << bits) - 1

        def _q(lst: List[float], idx: int) -> int:
            return int(_clamp(lst[idx]) * scale)

        tiles = []
        for idx in dirty:
            tiles.append({
                "i":  idx,
                "dt": _q(self.dust_thickness,    idx),
                "dm": _q(self.dust_mobility,     idx),
                "ic": _q(self.ice_film,          idx),
                "sf": _q(self.surface_freshness, idx),
                "ff": _q(self.fracture_fatigue,  idx),
                "ss": _q(self.slope_stability,   idx),
                "rc": _q(self.regolith_cohesion, idx),
            })
        return {
            "type":      "EVOLUTION_DELTA",
            "tickIndex": tick_index,
            "bits":      bits,
            "tiles":     tiles,
        }

    def apply_delta_dict(self, delta: dict) -> None:
        """Apply a sparse delta received from the server."""
        bits  = int(delta.get("bits", 8))
        scale = (1 << bits) - 1
        for t in delta.get("tiles", []):
            idx = int(t["i"])
            if not (0 <= idx < self.size()):
                continue
            self.dust_thickness[idx]    = t["dt"] / scale
            self.dust_mobility[idx]     = t["dm"] / scale
            self.ice_film[idx]          = t["ic"] / scale
            self.surface_freshness[idx] = t["sf"] / scale
            self.fracture_fatigue[idx]  = t["ff"] / scale
            self.slope_stability[idx]   = t["ss"] / scale
            self.regolith_cohesion[idx] = t["rc"] / scale


# ---------------------------------------------------------------------------
# SlopeEvent — a server-side event emitted when slope stability collapses
# ---------------------------------------------------------------------------

@dataclass
class SlopeEvent:
    tile_idx:  int
    sim_time:  float
    intensity: float   # 0..1


# ---------------------------------------------------------------------------
# LongHorizonEvolutionSystem
# ---------------------------------------------------------------------------

class LongHorizonEvolutionSystem:
    """Slow-tick planetary evolution: dust, ice, fatigue, slope relaxation.

    Parameters
    ----------
    config:
        Game config.  Reads keys under ``evo.*``.
    width, height:
        Override evolution grid size (defaults from config or 64×32).
    """

    def __init__(
        self,
        config: Config,
        width:  Optional[int] = None,
        height: Optional[int] = None,
    ) -> None:
        def _cfg(key: str, default: Any) -> Any:
            return config.get("evo", key, default=default)

        self.enabled: bool = bool(_cfg("enable", True))

        # Grid dimensions
        self._w: int = int(width  if width  is not None else _cfg("grid_w", 64))
        self._h: int = int(height if height is not None else _cfg("grid_h", 32))

        # Tick schedule
        self._tick_seconds: float = float(_cfg("tick_seconds",   60.0))
        self._tiles_per_tick: int = int(_cfg("tiles_per_tick", 256))

        # Snapshot / delta intervals
        self._snapshot_interval: float = float(_cfg("snapshot_interval_sec", 120.0))
        self._delta_interval:    float = float(_cfg("delta_interval_sec",     20.0))

        # Process coefficients
        self._dust_lift_k:        float = float(_cfg("dust_lift_k",         0.02))
        self._dust_dep_k:         float = float(_cfg("dust_dep_k",          0.03))
        self._ice_form_k:         float = float(_cfg("ice_form_k",          0.02))
        self._ice_melt_k:         float = float(_cfg("ice_melt_k",          0.03))
        self._fatigue_temp_k:     float = float(_cfg("fatigue_tempCycle_k", 0.001))
        self._fatigue_stress_k:   float = float(_cfg("fatigue_stress_k",    0.0005))
        self._slope_relax_k:      float = float(_cfg("slope_relax_k",       0.005))
        self._max_slope_per_hour: int   = int(_cfg("max_slope_events_per_hour", 20))
        self._quant_bits: int           = int(_cfg("network_quantization_bits", 8))

        # Fracture fatigue threshold for triggering fracture growth
        self._fatigue_fracture_threshold: float = 0.75

        # Fields
        self.fields: EvolutionFields = EvolutionFields(self._w, self._h)

        # Internal state
        self._tick_index:   int   = 0
        self._tile_cursor:  int   = 0   # round-robin tile offset
        self._sim_accum:    float = 0.0
        self._snap_accum:   float = 0.0
        self._delta_accum:  float = 0.0

        # Dirty tiles since last delta (set of indices)
        self._dirty: set = set()

        # Slope events emitted this hour
        self._slope_events_this_hour:  int   = 0
        self._slope_hour_start_sim:    float = 0.0
        self._pending_slope_events:    List[SlopeEvent] = []

        # Debug fast-forward state (dev only)
        self._ff_days_remaining: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        dt:      float,
        coupler: Any = None,
        climate: Any = None,
        sim_time: float = 0.0,
    ) -> None:
        """Advance the evolution system by *dt* game-seconds.

        Parameters
        ----------
        dt:
            Game-seconds elapsed since last call.
        coupler:
            AstroClimateCoupler (optional) — provides dust_lift_potential,
            ice_form_rate, T_target_at.
        climate:
            ClimateSystem (optional) — provides sample_wind, sample_temperature.
        sim_time:
            Current simulation time (for slope-event rate limiting).
        """
        if not self.enabled or dt <= 0.0:
            return

        self._sim_accum  += dt
        self._snap_accum += dt
        self._delta_accum += dt

        # Fast-forward accumulation (dev)
        if self._ff_days_remaining > 0.0:
            ff_dt = min(self._ff_days_remaining * 86400.0, self._tick_seconds * 10)
            self._sim_accum += ff_dt
            self._ff_days_remaining -= ff_dt / 86400.0
            if self._ff_days_remaining < 0.0:
                self._ff_days_remaining = 0.0

        # Evolution tick (fixed step)
        while self._sim_accum >= self._tick_seconds:
            self._sim_accum -= self._tick_seconds
            self._do_evolution_tick(coupler, climate, sim_time)
            self._tick_index += 1

    def should_send_snapshot(self) -> bool:
        """True when a full snapshot should be broadcast."""
        if self._snap_accum >= self._snapshot_interval:
            self._snap_accum = 0.0
            return True
        return False

    def should_send_delta(self) -> bool:
        """True when a sparse delta should be broadcast."""
        if self._delta_accum >= self._delta_interval and self._dirty:
            self._delta_accum = 0.0
            return True
        return False

    def get_snapshot(self) -> dict:
        """Return a full EVOLUTION_SNAPSHOT dict."""
        return self.fields.to_snapshot_dict(self._tick_index, self._quant_bits)

    def get_delta(self) -> dict:
        """Return an EVOLUTION_DELTA for dirty tiles; clears the dirty set."""
        dirty_list = sorted(self._dirty)
        self._dirty.clear()
        return self.fields.to_delta_dict(self._tick_index, dirty_list, self._quant_bits)

    def apply_snapshot(self, snap: dict) -> None:
        """Apply a full snapshot (client-side)."""
        self.fields     = EvolutionFields.from_snapshot_dict(snap)
        self._tick_index = int(snap.get("tickIndex", 0))

    def apply_delta(self, delta: dict) -> None:
        """Apply a sparse delta (client-side)."""
        self.fields.apply_delta_dict(delta)
        self._tick_index = int(delta.get("tickIndex", self._tick_index))

    def state_hash(self) -> str:
        """Short hex digest of all evolution fields (for sync checks)."""
        return self.fields.state_hash()

    def pop_slope_events(self) -> List[SlopeEvent]:
        """Return and clear pending slope events."""
        evts = list(self._pending_slope_events)
        self._pending_slope_events.clear()
        return evts

    def fast_forward_days(self, days: float) -> None:
        """Dev helper: schedule extra sim-time to run through on next update calls."""
        self._ff_days_remaining += max(0.0, days)

    # ------------------------------------------------------------------
    # Internal: evolution tick
    # ------------------------------------------------------------------

    def _do_evolution_tick(
        self, coupler: Any, climate: Any, sim_time: float
    ) -> None:
        """Advance ``tiles_per_tick`` tiles through all 4 processes."""
        n = self._w * self._h
        count = min(self._tiles_per_tick, n)

        # Reset slope-event hour counter if needed
        if sim_time - self._slope_hour_start_sim >= 3600.0:
            self._slope_events_this_hour = 0
            self._slope_hour_start_sim   = sim_time

        for step in range(count):
            tile_idx = (self._tile_cursor + step) % n
            self._update_tile(tile_idx, coupler, climate, sim_time)
            self._dirty.add(tile_idx)

        self._tile_cursor = (self._tile_cursor + count) % n

    def _tile_lat_lon(self, idx: int) -> Tuple[float, float]:
        """Return (lat_rad, lon_rad) for the centre of tile *idx*."""
        y = idx // self._w
        x = idx  % self._w
        lat = -_HALF_PI + math.pi * (y + 0.5) / self._h
        lon = -math.pi  + _TWO_PI  * (x + 0.5) / self._w
        return lat, lon

    def _tile_world_pos(self, idx: int):
        """Return a unit Vec3 for tile *idx* (lazy import to avoid circular)."""
        lat, lon = self._tile_lat_lon(idx)
        cos_lat = math.cos(lat)
        return _FakeVec3(cos_lat * math.sin(lon), math.sin(lat), cos_lat * math.cos(lon))

    def _sample_coupler_wind(self, coupler, pos) -> float:
        """Scalar wind speed from coupler heat-wind delta (proxy)."""
        if coupler is None:
            return 0.0
        try:
            dw = coupler.heat_wind_delta(pos)
            return math.sqrt(dw.x * dw.x + dw.y * dw.y + dw.z * dw.z)
        except Exception:
            return 0.0

    def _sample_climate_wind(self, climate, pos) -> float:
        if climate is None:
            return 0.0
        try:
            w = climate.sample_wind(pos)
            return math.sqrt(w.x * w.x + w.y * w.y + w.z * w.z)
        except Exception:
            return 0.0

    def _update_tile(
        self, idx: int, coupler: Any, climate: Any, sim_time: float
    ) -> None:
        """Run all 4 evolution processes for a single tile."""
        f   = self.fields
        W   = self._w
        pos = self._tile_world_pos(idx)

        # --- Sample drivers ---
        dust_lift_potential = (
            _clamp(coupler.dust_lift_potential(pos)) if coupler is not None else 0.0
        )
        ice_form_rate_driver = (
            _clamp(coupler.ice_form_rate(pos)) if coupler is not None else 0.0
        )
        T_target = (
            coupler.T_target_at(pos) if coupler is not None else 260.0
        )
        wind_speed = (
            self._sample_climate_wind(climate, pos)
            or self._sample_coupler_wind(coupler, pos)
        )

        # Normalise wind_speed to [0, 1] (cap at 40 m/s)
        wind_norm = _clamp(wind_speed / 40.0)
        # Temperature factor: how hot is the tile (K above freezing / 50)
        temp_factor = _clamp((T_target - 270.0) / 50.0, -1.0, 1.0)

        # ----------------------------------------------------------------
        # Process 1: Dust Transport
        # ----------------------------------------------------------------
        dt_tile = self._tick_seconds
        dust = f.dust_thickness[idx]
        ice  = f.ice_film[idx]
        mob  = f.dust_mobility[idx]
        coh  = f.regolith_cohesion[idx]

        lift = self._dust_lift_k * dust_lift_potential * dust * (1.0 - ice) * (1.0 - coh * 0.5)
        # Deposit in this tile (fraction of lifted dust that settles locally)
        deposit = self._dust_dep_k * (1.0 - wind_norm) * lift
        # Net dust change from lift/deposit (transport to neighbours handled implicitly via
        # the round-robin tile order; nearby tiles will gain from this lift)
        dust_new = _clamp(dust - lift + deposit)

        # Advect a fraction of lifted dust to the downwind neighbour
        if lift > 0.0:
            x_tile  = idx % W
            y_tile  = idx // W
            # Simple eastward advection (dominant wind direction proxy)
            x_next  = (x_tile + 1) % W
            nbr_idx = y_tile * W + x_next
            f.dust_thickness[nbr_idx] = _clamp(
                f.dust_thickness[nbr_idx] + lift * wind_norm * 0.5
            )
            self._dirty.add(nbr_idx)

        f.dust_thickness[idx] = dust_new

        # Dust mobility: high wind increases mobility, cohesion reduces it
        f.dust_mobility[idx] = _clamp(mob + wind_norm * 0.01 - coh * 0.005)

        # ----------------------------------------------------------------
        # Process 2: Ice Film Evolution
        # ----------------------------------------------------------------
        ice_thresh = 270.0
        cold_factor = _clamp((ice_thresh - T_target) / 30.0)
        ice_growth  = self._ice_form_k * ice_form_rate_driver * cold_factor * (1.0 - dust * 0.5)
        ice_melt    = self._ice_melt_k * max(0.0, temp_factor) * (1.0 + wind_norm * 0.3)
        f.ice_film[idx] = _clamp(ice + ice_growth - ice_melt)

        # ----------------------------------------------------------------
        # Process 3: Fracture Fatigue
        # ----------------------------------------------------------------
        # dT is approximated from the temperature factor variation
        dT_cycle = abs(temp_factor) * 30.0  # Kelvin amplitude proxy
        geo_stress = 0.0  # could be driven by GeoEventSystem in future
        fatigue_inc = (
            self._fatigue_temp_k  * dT_cycle
            + self._fatigue_stress_k * geo_stress
        )
        f.fracture_fatigue[idx] = _clamp(f.fracture_fatigue[idx] + fatigue_inc)

        # Freshness fades with fatigue (erosion effect)
        f.surface_freshness[idx] = _clamp(
            f.surface_freshness[idx] - fatigue_inc * 0.5
        )

        # Regolith cohesion: ice increases it, fatigue decreases it slightly
        f.regolith_cohesion[idx] = _clamp(
            f.regolith_cohesion[idx]
            + f.ice_film[idx] * 0.002
            - f.fracture_fatigue[idx] * 0.001
        )

        # ----------------------------------------------------------------
        # Process 4: Slope Relaxation
        # ----------------------------------------------------------------
        stability = f.slope_stability[idx]
        stab_loss = self._slope_relax_k * (
            wind_norm * 0.5
            + f.ice_film[idx] * 0.3
            + (1.0 - f.regolith_cohesion[idx]) * 0.2
        )
        stability_new = _clamp(stability - stab_loss)
        f.slope_stability[idx] = stability_new

        # Emit slope event when stability crosses the threshold
        if stability >= 0.3 > stability_new:
            if (
                self._slope_events_this_hour < self._max_slope_per_hour
            ):
                intensity = _clamp(1.0 - stability_new / 0.3)
                self._pending_slope_events.append(
                    SlopeEvent(tile_idx=idx, sim_time=sim_time, intensity=intensity)
                )
                self._slope_events_this_hour += 1
                # Reset stability after event so it can recover
                f.slope_stability[idx] = _clamp(stability_new + 0.4)


# ---------------------------------------------------------------------------
# _FakeVec3 — minimal Vec3-like object to avoid circular imports
# ---------------------------------------------------------------------------

class _FakeVec3:
    """Minimal duck-typed Vec3 for tile world-position sampling."""
    __slots__ = ("x", "y", "z")

    def __init__(self, x: float, y: float, z: float) -> None:
        self.x = x
        self.y = y
        self.z = z

    def length(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)
