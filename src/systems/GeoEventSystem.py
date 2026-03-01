"""GeoEventSystem — Stage 9 geological event simulation.

Periodically evaluates geological risk from tectonic + climate fields
and fires rare but impactful world-modification events through the
SDFPatchSystem:

  * FaultCrackEvent  — разлом/трещина  (CapsuleCarve series along fault line)
  * LandslideEvent   — осыпь/обвал     (SphereCarve source + AdditiveDeposit runout)
  * CollapseEvent    — провал/обрушение (underground chamber + shaft + deposit ring)

Each event has three phases (signalled without UI):
  PRE    — precursor signals (rumble, micro-dust, wind-gust data)
  IMPACT — SDF patches applied, shock signal emitted
  POST   — elevated signals, local field updates

All generation is deterministic from (seed, game_time) pair.

Public API
----------
update(game_time)                              — advance simulation (IGeoEventSystem compat)
update_with_dt(dt, game_time, player_pos)      — full update
query_signals_near(world_pos, range) → list    — query GeoEventSignals
query_ground_stability(pos) → float            — IGeoEventSystem interface
event_log → GeoEventLog                        — access to replay log
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

from src.math.Vec3 import Vec3
from src.planet.SDFPatchSystem import (
    AdditiveDeposit,
    CapsuleCarve,
    SDFPatch,
    SDFPatchLog,
    SphereCarve,
)
from src.planet.TectonicPlatesSystem import BoundaryType
from src.systems.GeoEventSystemStub import IGeoEventSystem

# ---------------------------------------------------------------------------
# Seed XOR constants (plain ints — no non-hex letters in literals)
# ---------------------------------------------------------------------------

_FAULT_SEED_XOR    = 0xFA011212
_SLIDE_SEED_XOR    = 0x511DEAB0
_COLLAPSE_SEED_XOR = 0xC011A75E
_SCHED_SEED_XOR    = 0xEEEE5CCE


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class GeoEventType(Enum):
    FAULT_CRACK = auto()   # разлом / трещина
    LANDSLIDE   = auto()   # осыпь / обвал
    COLLAPSE    = auto()   # провал / обрушение


class GeoEventPhase(Enum):
    PRE    = auto()   # precursor — world not yet changed
    IMPACT = auto()   # SDF patches applied, shock signal
    POST   = auto()   # aftermath — elevated signals


# ---------------------------------------------------------------------------
# GeoEventSignal — public API for other systems (sound, atmosphere, character)
# ---------------------------------------------------------------------------

@dataclass
class GeoEventSignal:
    """Emitted each update tick while an event is active.

    Other systems (character, sound, atmosphere) query these signals via
    ``GeoEventSystem.query_signals_near(world_pos, range)``.
    """
    type:            GeoEventType
    position:        Vec3
    radius:          float    # influence radius in world units
    phase:           GeoEventPhase
    intensity:       float    # 0..1
    time_to_impact:  float    # seconds until IMPACT (negative after)


# ---------------------------------------------------------------------------
# GeoRiskScores
# ---------------------------------------------------------------------------

@dataclass
class GeoRiskScores:
    """Combined risk scores for a surface location (all values 0..1)."""
    fault_risk:     float
    landslide_risk: float
    collapse_risk:  float

    @property
    def max_risk(self) -> float:
        return max(self.fault_risk, self.landslide_risk, self.collapse_risk)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


# ---------------------------------------------------------------------------
# GeoRiskEvaluator
# ---------------------------------------------------------------------------

class GeoRiskEvaluator:
    """Computes risk scores from geological and (optionally) climate fields.

    Parameters
    ----------
    geo_sampler:
        Provides ``GeoSample`` for a unit-sphere direction.
        Must implement ``sample(direction: Vec3) -> GeoSample``.
    climate:
        Optional climate system (``IClimateSystem``).  When *None* all
        climate contributions are treated as zero.
    planet_radius:
        Used to derive the approximate world-space position from direction.
    """

    def __init__(
        self,
        geo_sampler,
        climate=None,
        planet_radius: float = 1000.0,
    ) -> None:
        self._geo     = geo_sampler
        self._climate = climate
        self._r       = planet_radius

    # ------------------------------------------------------------------
    def evaluate(self, direction: Vec3) -> GeoRiskScores:
        """Return composite risk scores for *direction* (unit sphere vector)."""
        geo = self._geo.sample(direction)

        # Derive void_risk from available fields:
        # — high fracture  → porous rock prone to collapse
        # — divergent zone → rift / subsidence prone
        void_risk = _clamp(
            geo.fracture * 0.6
            + (geo.boundary_strength * 0.4
               if geo.boundary_type == BoundaryType.DIVERGENT
               else 0.0),
            0.0, 1.0,
        )

        # Climate factors (default neutral = 0)
        wind_force  = 0.0
        freeze_thaw = 0.0
        wetness     = 0.0

        if self._climate is not None:
            world_pos   = direction * self._r
            wind_force  = _clamp(self._climate.get_wind_force_factor(world_pos), 0.0, 1.0)
            freeze_thaw = _clamp(self._climate.get_freeze_thaw_factor(world_pos), 0.0, 1.0)
            wetness     = _clamp(self._climate.get_wetness(world_pos), 0.0, 1.0)

        # A) Fault risk — active plate boundaries, high stress + fracture
        boundary_factor = geo.boundary_strength if geo.boundary_type != BoundaryType.NONE else 0.0
        fault_risk = _clamp(
            boundary_factor * (0.5 * geo.stress + 0.5 * geo.fracture),
            0.0, 1.0,
        )

        # B) Landslide risk — slope instability + climate triggers
        instability    = 1.0 - geo.stability
        landslide_risk = _clamp(
            instability * 0.5 + freeze_thaw * 0.3 + wind_force * 0.2,
            0.0, 1.0,
        )

        # C) Collapse risk — void zones + fracture + moisture acceleration
        collapse_risk = _clamp(
            void_risk * 0.6 + geo.fracture * 0.3 + wetness * 0.1,
            0.0, 1.0,
        )

        return GeoRiskScores(
            fault_risk     = fault_risk,
            landslide_risk = landslide_risk,
            collapse_risk  = collapse_risk,
        )


# ---------------------------------------------------------------------------
# GeoEventRecord — log entry (immutable after creation)
# ---------------------------------------------------------------------------

@dataclass
class GeoEventRecord:
    """Immutable record of one geological event; used for deterministic replay."""
    event_id:   int
    event_type: GeoEventType
    start_time: float
    direction:  Vec3                 # unit surface direction (encodes lat/lon)
    seed_local: int                  # per-event RNG seed for micro-variation
    params:     Dict[str, float]     # type-specific geometry parameters
    patches:    List[SDFPatch]       # SDF patches generated at IMPACT


# ---------------------------------------------------------------------------
# GeoEventLog
# ---------------------------------------------------------------------------

class GeoEventLog:
    """Ordered, append-only log of geo-event records.

    Supports deterministic replay: given the same sequence of records
    the same SDF patches are re-applied in insertion order.
    """

    def __init__(self) -> None:
        self._records:  List[GeoEventRecord] = []
        self._next_id:  int = 0

    # ------------------------------------------------------------------
    def add(self, record: GeoEventRecord) -> None:
        """Append *record* to the log."""
        self._records.append(record)
        self._next_id = max(self._next_id, record.event_id + 1)

    def records(self) -> List[GeoEventRecord]:
        """Read-only view of all records in insertion order."""
        return list(self._records)

    def __len__(self) -> int:
        return len(self._records)

    # ------------------------------------------------------------------
    def replay_to_patch_log(self, patch_log: SDFPatchLog) -> None:
        """Replay all recorded patches into *patch_log* in insertion order."""
        for record in self._records:
            for patch in record.patches:
                patch_log.add(patch)

    def allocate_id(self) -> int:
        """Return the next available event id and advance the counter."""
        idx = self._next_id
        self._next_id += 1
        return idx


# ---------------------------------------------------------------------------
# GeoEventExecutor — produces concrete SDF patches for each event type
# ---------------------------------------------------------------------------

class GeoEventExecutor:
    """Converts event parameters into SDF patches applied at IMPACT.

    Parameters
    ----------
    planet_radius:
        Planet surface radius in simulation units.
    height_provider:
        Optional ``PlanetHeightProvider``.  When provided, surface
        positions are computed accurately; otherwise the bare sphere
        radius is used.
    """

    def __init__(
        self,
        planet_radius:   float = 1000.0,
        height_provider=None,
    ) -> None:
        self._r  = planet_radius
        self._hp = height_provider

    # ------------------------------------------------------------------
    def _surface_pos(self, direction: Vec3) -> Vec3:
        """World-space point on the planet surface at *direction*."""
        h = self._hp.sample_height(direction) if self._hp is not None else 0.0
        return direction * (self._r + h)

    def _tangent_basis(self, direction: Vec3, rng: random.Random) -> Tuple[Vec3, Vec3]:
        """Two orthogonal unit tangents at *direction*, rotated randomly."""
        up  = direction.normalized()
        ref = Vec3(0.0, 1.0, 0.0) if abs(up.x) < 0.9 else Vec3(1.0, 0.0, 0.0)
        t1  = ref.cross(up).normalized()
        t2  = up.cross(t1).normalized()
        angle = rng.uniform(0.0, 2.0 * math.pi)
        c, s  = math.cos(angle), math.sin(angle)
        return t1 * c + t2 * s, t1 * (-s) + t2 * c

    # ------------------------------------------------------------------
    def execute_fault_crack(
        self,
        direction: Vec3,
        params:    Dict[str, float],
        seed:      int,
    ) -> List[SDFPatch]:
        """Generate a series of CapsuleCarve patches along a fault line.

        Creates a narrow, deep crack in the rock surface.

        Params
        ------
        fault_length : total fault line length (world units), default 60
        fault_width  : carve radius per segment, default 3
        fault_depth  : radial depth into the rock, default 10
        n_segments   : number of capsule segments, default 5
        """
        rng    = random.Random(seed ^ _FAULT_SEED_XOR)
        length = params.get("fault_length", 60.0)
        width  = params.get("fault_width",   3.0)
        depth  = params.get("fault_depth",  10.0)
        n_seg  = max(1, int(params.get("n_segments", 5)))

        surface_pos = self._surface_pos(direction)
        tangent, _  = self._tangent_basis(direction, rng)
        radial      = direction.normalized()

        patches: List[SDFPatch] = []
        for i in range(n_seg):
            frac_a = (i      / n_seg) - 0.5   # range −0.5..+0.5
            frac_b = ((i+1)  / n_seg) - 0.5
            jitter = rng.uniform(-0.05, 0.05) * length
            a = surface_pos + tangent * (frac_a * length + jitter) - radial * (depth * 0.3)
            b = surface_pos + tangent * (frac_b * length + jitter) - radial * depth
            patches.append(CapsuleCarve(a=a, b=b, radius=width))
        return patches

    # ------------------------------------------------------------------
    def execute_landslide(
        self,
        direction: Vec3,
        params:    Dict[str, float],
        seed:      int,
    ) -> List[SDFPatch]:
        """Carve a source hillside and deposit runout material down-slope.

        Params
        ------
        carve_radius   : radius of removed material at source, default 15
        deposit_radius : radius of deposit sphere below, default 10
        runout_length  : distance from source to deposit centre, default 50
        """
        rng       = random.Random(seed ^ _SLIDE_SEED_XOR)
        carve_r   = params.get("carve_radius",   15.0)
        deposit_r = params.get("deposit_radius", 10.0)
        runout    = params.get("runout_length",  50.0)

        surface_pos = self._surface_pos(direction)
        tangent, _  = self._tangent_basis(direction, rng)
        radial      = direction.normalized()
        sign        = 1.0 if rng.random() < 0.5 else -1.0
        down_slope  = tangent * sign

        src_carve = SphereCarve(
            centre=surface_pos - radial * (carve_r * 0.4),
            radius=carve_r,
        )
        deposit = AdditiveDeposit(
            centre=surface_pos + down_slope * runout,
            radius=deposit_r,
        )
        return [src_carve, deposit]

    # ------------------------------------------------------------------
    def execute_collapse(
        self,
        direction: Vec3,
        params:    Dict[str, float],
        seed:      int,
    ) -> List[SDFPatch]:
        """Carve underground chamber + shaft to surface + rim debris deposit.

        Params
        ------
        chamber_radius : radius of underground void, default 18
        shaft_radius   : radius of surface-connecting shaft, default 5
        shaft_depth    : depth of chamber below surface, default 20
        rim_radius     : radius of debris ring at surface, default 12
        """
        rng         = random.Random(seed ^ _COLLAPSE_SEED_XOR)
        chamber_r   = params.get("chamber_radius", 18.0)
        shaft_r     = params.get("shaft_radius",    5.0)
        shaft_depth = params.get("shaft_depth",    20.0)
        rim_r       = params.get("rim_radius",     12.0)

        surface_pos  = self._surface_pos(direction)
        radial       = direction.normalized()
        tangent, _   = self._tangent_basis(direction, rng)

        chamber_pos  = surface_pos - radial * shaft_depth
        chamber      = SphereCarve(centre=chamber_pos, radius=chamber_r)
        shaft        = CapsuleCarve(a=surface_pos, b=chamber_pos, radius=shaft_r)
        rim_deposit  = AdditiveDeposit(
            centre=surface_pos + tangent * (chamber_r * 0.5),
            radius=rim_r * 0.7,
        )
        return [chamber, shaft, rim_deposit]

    # ------------------------------------------------------------------
    def execute(
        self,
        event_type: GeoEventType,
        direction:  Vec3,
        params:     Dict[str, float],
        seed:       int,
    ) -> List[SDFPatch]:
        """Dispatch to the correct executor method."""
        if event_type == GeoEventType.FAULT_CRACK:
            return self.execute_fault_crack(direction, params, seed)
        if event_type == GeoEventType.LANDSLIDE:
            return self.execute_landslide(direction, params, seed)
        if event_type == GeoEventType.COLLAPSE:
            return self.execute_collapse(direction, params, seed)
        return []


# ---------------------------------------------------------------------------
# GeoEvent — active event with phase lifecycle
# ---------------------------------------------------------------------------

@dataclass
class GeoEvent:
    """One active event progressing through PRE → IMPACT → POST."""
    record:         GeoEventRecord
    phase:          GeoEventPhase = GeoEventPhase.PRE
    phase_elapsed:  float         = 0.0
    pre_duration:   float         = 5.0
    post_duration:  float         = 30.0
    impact_applied: bool          = False

    # ------------------------------------------------------------------
    def time_to_impact(self) -> float:
        """Seconds until IMPACT (negative once past it)."""
        if self.phase == GeoEventPhase.PRE:
            return self.pre_duration - self.phase_elapsed
        return -self.phase_elapsed

    def signal_intensity(self) -> float:
        """Signal intensity [0,1] based on current phase and elapsed time."""
        if self.phase == GeoEventPhase.PRE:
            return _clamp(self.phase_elapsed / max(1.0, self.pre_duration), 0.0, 1.0)
        if self.phase == GeoEventPhase.IMPACT:
            return 1.0
        return _clamp(1.0 - self.phase_elapsed / max(1.0, self.post_duration), 0.0, 1.0)


# ---------------------------------------------------------------------------
# GeoEventScheduler — hazard accumulation and stochastic event firing
# ---------------------------------------------------------------------------

class GeoEventScheduler:
    """Maintains candidate cells, accumulates hazard, and stochastically fires events.

    Uses an equirectangular grid of ``grid_w × grid_h`` cells as candidate
    locations.  On each ``update()`` a random sample of cells is evaluated;
    hazard accumulates from risk until an event is triggered.

    Parameters
    ----------
    evaluator:
        :class:`GeoRiskEvaluator` instance.
    seed:
        Deterministic seed (separate from geo/climate seeds).
    grid_w, grid_h:
        Candidate grid resolution (default 32×16 = 512 cells).
    hazard_threshold_minor:
        Hazard level at which minor events can fire.
    hazard_threshold_major:
        Hazard level at which major events can fire.
    rate_minor_per_hour, rate_major_per_hour:
        Maximum events per simulated hour (rate-limits burst spawning).
    cooldown_seconds_per_cell:
        Per-cell cooldown after an event fires.
    cells_per_tick:
        Cells evaluated per ``update()`` call (performance budget).
    pre_seconds, post_seconds:
        Default PRE / POST phase durations for spawned events.
    """

    def __init__(
        self,
        evaluator:                 GeoRiskEvaluator,
        seed:                      int   = 42,
        grid_w:                    int   = 32,
        grid_h:                    int   = 16,
        hazard_threshold_minor:    float = 0.4,
        hazard_threshold_major:    float = 0.8,
        rate_minor_per_hour:       float = 20.0,
        rate_major_per_hour:       float = 2.0,
        cooldown_seconds_per_cell: float = 300.0,
        cells_per_tick:            int   = 8,
        pre_seconds:               float = 5.0,
        post_seconds:              float = 30.0,
    ) -> None:
        self._evaluator      = evaluator
        self._rng            = random.Random(seed ^ _SCHED_SEED_XOR)
        self._W              = grid_w
        self._H              = grid_h
        self._n              = grid_w * grid_h
        self._threshold_min  = hazard_threshold_minor
        self._threshold_maj  = hazard_threshold_major
        self._rate_minor     = rate_minor_per_hour
        self._rate_major     = rate_major_per_hour
        self._cooldown_secs  = cooldown_seconds_per_cell
        self._cells_per_tick = cells_per_tick
        self._pre_sec        = pre_seconds
        self._post_sec       = post_seconds

        self._hazard:          List[float]       = [0.0] * self._n
        self._cooldown_timers: Dict[int, float]  = {}
        self._minor_bucket:    float             = 0.0
        self._major_bucket:    float             = 0.0
        self._next_id:         int               = 0

    # ------------------------------------------------------------------
    def _cell_direction(self, idx: int) -> Vec3:
        """Equirectangular cell index → unit sphere direction (Y-up convention)."""
        row     = idx // self._W
        col     = idx %  self._W
        lat     = -math.pi * 0.5 + math.pi * (row + 0.5) / self._H
        lon     = -math.pi + 2.0 * math.pi * (col + 0.5) / self._W
        cos_lat = math.cos(lat)
        return Vec3(
            cos_lat * math.sin(lon),
            math.sin(lat),
            cos_lat * math.cos(lon),
        )

    # ------------------------------------------------------------------
    def update(
        self,
        dt:                   float,
        game_time:            float,
        player_pos:           Optional[Vec3]          = None,
        min_dist_to_player:   float                   = 0.0,
        max_dist_to_player:   float                   = math.inf,
        event_log:            Optional[GeoEventLog]   = None,
        executor:             Optional[GeoEventExecutor] = None,
    ) -> List[GeoEvent]:
        """Advance scheduler state and return any newly spawned events.

        Parameters
        ----------
        dt:
            Game-time seconds elapsed since last call.
        game_time:
            Absolute game time (for event log timestamps).
        player_pos:
            When provided, events are filtered by distance.
        min_dist_to_player, max_dist_to_player:
            Distance bounds for event spawning (world units).
        event_log:
            If provided, spawned events are appended to the log.
        executor:
            If provided, SDF patches are generated and stored in records.
        """
        # Refill rate-limit buckets proportionally to dt
        _bucket_period = 3600.0   # one simulated hour
        self._minor_bucket = min(
            self._rate_minor,
            self._minor_bucket + self._rate_minor * (dt / _bucket_period),
        )
        self._major_bucket = min(
            self._rate_major,
            self._major_bucket + self._rate_major * (dt / _bucket_period),
        )

        # Tick down per-cell cooldown timers
        expired = [k for k, v in self._cooldown_timers.items() if v <= dt]
        for k in expired:
            del self._cooldown_timers[k]
        for k in list(self._cooldown_timers):
            self._cooldown_timers[k] -= dt

        spawned: List[GeoEvent] = []
        sample_n = min(self._cells_per_tick, self._n)
        indices  = self._rng.sample(range(self._n), sample_n)

        for idx in indices:
            if idx in self._cooldown_timers:
                continue

            direction = self._cell_direction(idx)

            # Optional player-distance filter
            if player_pos is not None:
                approx_pos = direction * 1000.0
                dist = (approx_pos - player_pos).length()
                if dist < min_dist_to_player or dist > max_dist_to_player:
                    continue

            scores    = self._evaluator.evaluate(direction)
            cell_risk = scores.max_risk
            self._hazard[idx] = _clamp(
                self._hazard[idx] + cell_risk * dt, 0.0, 2.0
            )

            hazard   = self._hazard[idx]
            is_major = hazard >= self._threshold_maj

            if is_major and self._major_bucket >= 1.0:
                use_major    = True
                threshold_ok = True
            elif hazard >= self._threshold_min and self._minor_bucket >= 1.0:
                use_major    = False
                threshold_ok = True
            else:
                continue

            # Stochastic gate — probability proportional to excess hazard
            thresh = self._threshold_maj if is_major else self._threshold_min
            prob   = _clamp((hazard - thresh) * 0.1, 0.0, 0.5)
            if self._rng.random() > prob:
                continue

            event_type = self._select_type(scores)
            seed_local = self._rng.randint(0, 0xFFFFFF)
            params     = self._default_params(event_type)
            patches: List[SDFPatch] = []
            if executor is not None:
                patches = executor.execute(event_type, direction, params, seed_local)

            event_id = self._next_id
            self._next_id += 1

            record = GeoEventRecord(
                event_id   = event_id,
                event_type = event_type,
                start_time = game_time,
                direction  = direction,
                seed_local = seed_local,
                params     = params,
                patches    = patches,
            )
            geo_event = GeoEvent(
                record        = record,
                phase         = GeoEventPhase.PRE,
                phase_elapsed = 0.0,
                pre_duration  = self._pre_sec,
                post_duration = self._post_sec,
            )
            spawned.append(geo_event)

            if event_log is not None:
                event_log.add(record)

            if use_major:
                self._major_bucket -= 1.0
            else:
                self._minor_bucket -= 1.0

            self._hazard[idx] = 0.0
            self._cooldown_timers[idx] = self._cooldown_secs

        return spawned

    # ------------------------------------------------------------------
    def _select_type(self, scores: GeoRiskScores) -> GeoEventType:
        """Choose the event type whose risk score is highest."""
        pairs = [
            (scores.fault_risk,     GeoEventType.FAULT_CRACK),
            (scores.landslide_risk, GeoEventType.LANDSLIDE),
            (scores.collapse_risk,  GeoEventType.COLLAPSE),
        ]
        return max(pairs, key=lambda p: p[0])[1]

    def _default_params(self, event_type: GeoEventType) -> Dict[str, float]:
        """Return default geometry parameters for the given event type."""
        if event_type == GeoEventType.FAULT_CRACK:
            return {"fault_length": 60.0, "fault_width": 3.0,
                    "fault_depth": 10.0,  "n_segments": 5.0}
        if event_type == GeoEventType.LANDSLIDE:
            return {"carve_radius": 15.0, "deposit_radius": 10.0,
                    "runout_length": 50.0}
        # COLLAPSE
        return {"chamber_radius": 18.0, "shaft_radius": 5.0,
                "shaft_depth": 20.0,    "rim_radius": 12.0}


# ---------------------------------------------------------------------------
# GeoEventSystem — top-level integration
# ---------------------------------------------------------------------------

class GeoEventSystem(IGeoEventSystem):
    """Top-level geological event simulation system (Stage 9).

    Integrates :class:`GeoRiskEvaluator`, :class:`GeoEventScheduler`,
    :class:`GeoEventExecutor`, and :class:`GeoEventLog` into a single
    system driven by a periodic ``update_with_dt()`` call.

    Parameters
    ----------
    geo_sampler:
        ``GeoFieldSampler`` or duck-type with ``sample(Vec3) → GeoSample``.
    climate:
        Optional climate system (``IClimateSystem``).
    sdf_world:
        Optional ``SDFWorld``.  When provided, IMPACT patches are applied
        immediately to loaded chunks.
    planet_radius:
        Planet surface radius in simulation units.
    height_provider:
        Optional ``PlanetHeightProvider`` for accurate surface positioning.
    seed:
        Deterministic seed for all stochastic elements.
    pre_seconds, post_seconds:
        Default phase durations for spawned events.
    min_dist_to_player, max_dist_to_player:
        Distance filter for event spawning (world units).
    rate_minor_per_hour, rate_major_per_hour:
        Maximum event rates (per simulated hour).
    cooldown_minutes_per_tile:
        Per-cell cooldown in game minutes.
    """

    def __init__(
        self,
        geo_sampler,
        climate                = None,
        sdf_world              = None,
        planet_radius:  float  = 1000.0,
        height_provider        = None,
        seed:           int    = 42,
        pre_seconds:    float  = 5.0,
        post_seconds:   float  = 30.0,
        min_dist_to_player: float = 0.0,
        max_dist_to_player: float = math.inf,
        rate_minor_per_hour: float = 20.0,
        rate_major_per_hour: float = 2.0,
        cooldown_minutes_per_tile: float = 5.0,
    ) -> None:
        self._sdf_world = sdf_world
        self._min_dist  = min_dist_to_player
        self._max_dist  = max_dist_to_player

        self._evaluator  = GeoRiskEvaluator(
            geo_sampler   = geo_sampler,
            climate       = climate,
            planet_radius = planet_radius,
        )
        self._executor   = GeoEventExecutor(
            planet_radius   = planet_radius,
            height_provider = height_provider,
        )
        self._scheduler  = GeoEventScheduler(
            evaluator                 = self._evaluator,
            seed                      = seed,
            pre_seconds               = pre_seconds,
            post_seconds              = post_seconds,
            rate_minor_per_hour       = rate_minor_per_hour,
            rate_major_per_hour       = rate_major_per_hour,
            cooldown_seconds_per_cell = cooldown_minutes_per_tile * 60.0,
        )
        self._event_log:     GeoEventLog    = GeoEventLog()
        self._active_events: List[GeoEvent] = []

    # ------------------------------------------------------------------
    # IGeoEventSystem interface
    # ------------------------------------------------------------------

    def update(self, game_time: float) -> None:
        """Advance simulation by one nominal second (IGeoEventSystem compat)."""
        self.update_with_dt(dt=1.0, game_time=game_time)

    def query_ground_stability(self, pos: Vec3) -> float:
        """Return ground stability [0,1] at world-space position *pos*."""
        unit_dir = pos.normalized()
        if unit_dir.is_near_zero():
            return 1.0
        geo = self._evaluator._geo.sample(unit_dir)
        return geo.stability

    # ------------------------------------------------------------------
    # Extended update
    # ------------------------------------------------------------------

    def update_with_dt(
        self,
        dt:          float,
        game_time:   float,
        player_pos:  Optional[Vec3] = None,
    ) -> None:
        """Full update: spawn events, advance phases, apply IMPACT patches.

        Parameters
        ----------
        dt:
            Game-time seconds elapsed since last call.
        game_time:
            Absolute game time (seconds).
        player_pos:
            Optional player world position for distance filtering.
        """
        # 1. Spawn new events
        new_events = self._scheduler.update(
            dt                 = dt,
            game_time          = game_time,
            player_pos         = player_pos,
            min_dist_to_player = self._min_dist,
            max_dist_to_player = self._max_dist,
            event_log          = self._event_log,
            executor           = self._executor,
        )
        self._active_events.extend(new_events)

        # 2. Advance active event phases
        finished: List[GeoEvent] = []
        for evt in self._active_events:
            self._advance_event(evt, dt)
            if (evt.phase == GeoEventPhase.POST
                    and evt.phase_elapsed >= evt.post_duration):
                finished.append(evt)
        for evt in finished:
            self._active_events.remove(evt)

    # ------------------------------------------------------------------
    # GeoEventSignal API
    # ------------------------------------------------------------------

    def query_signals_near(
        self,
        world_pos: Vec3,
        radius:    float,
    ) -> List[GeoEventSignal]:
        """Return all active event signals within *radius* of *world_pos*.

        Parameters
        ----------
        world_pos:
            Query position in world space.
        radius:
            Search radius in world units.
        """
        signals: List[GeoEventSignal] = []
        for evt in self._active_events:
            evt_pos = evt.record.direction * 1000.0   # approx surface pos
            dist    = (evt_pos - world_pos).length()
            evt_r   = _event_influence_radius(evt.record)
            if dist <= radius + evt_r:
                signals.append(GeoEventSignal(
                    type           = evt.record.event_type,
                    position       = evt_pos,
                    radius         = evt_r,
                    phase          = evt.phase,
                    intensity      = evt.signal_intensity(),
                    time_to_impact = evt.time_to_impact(),
                ))
        return signals

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    @property
    def event_log(self) -> GeoEventLog:
        """Access to the full event log for serialisation / replay."""
        return self._event_log

    @property
    def active_events(self) -> List[GeoEvent]:
        """Read-only copy of currently active events."""
        return list(self._active_events)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _advance_event(self, evt: GeoEvent, dt: float) -> None:
        """Advance phase timers; apply IMPACT patches at the right moment."""
        if evt.phase == GeoEventPhase.PRE:
            evt.phase_elapsed += dt
            if evt.phase_elapsed >= evt.pre_duration:
                evt.phase         = GeoEventPhase.IMPACT
                evt.phase_elapsed = 0.0

        if evt.phase == GeoEventPhase.IMPACT and not evt.impact_applied:
            if self._sdf_world is not None:
                for patch in evt.record.patches:
                    self._sdf_world.apply_patch(patch)
            evt.impact_applied = True
            evt.phase          = GeoEventPhase.POST
            evt.phase_elapsed  = 0.0

        elif evt.phase == GeoEventPhase.POST:
            evt.phase_elapsed += dt


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------

def _event_influence_radius(record: GeoEventRecord) -> float:
    """Approximate influence radius for signal queries."""
    p = record.params
    if record.event_type == GeoEventType.FAULT_CRACK:
        return p.get("fault_length", 60.0)
    if record.event_type == GeoEventType.LANDSLIDE:
        return p.get("runout_length", 50.0)
    return p.get("chamber_radius", 18.0) * 2.0
