"""DiegeticMultiplayerPresence — Stage 25 presence layer.

Makes other players *feel* present in the world through physics, light, and
traces — without any UI, markers, names, or outlines.

Components
----------
TrailSegment        — one unit of a player-movement trace in world space.
TrailDecalSystem    — stores, ages, and procedurally renders trail segments.
HeadlampProfile     — computed headlamp/glow parameters for one player/frame.
HeadlampSystem      — converts environment context to headlamp profiles.
RemotePlayerLOD     — selects animation detail level by distance.
DiegeticMultiplayerPresence — top-level orchestrator integrating all of the above.

Trail material classes
----------------------
DUST          — soft dust surface; footprints live up to TTL, erased by storm.
LOOSE_DEBRIS  — similar to dust, slightly coarser.
ICE_FILM      — scratches/gouges; long TTL, unaffected by storm wind.
ROCK          — barely any trace; very short TTL.

Trail event types
-----------------
FOOTPRINT     — one foot-plant stamp.
SLIDE_MARK    — continuous groove from sliding locomotion.
DUST_PUFF     — brief dispersal cloud (rendered as procedural haze, no mesh).

LOD levels
----------
LOD_FULL      — full procedural animation + IK  (distance < lod_distance_1)
LOD_REDUCED   — simplified pose, fewer bones, no IK
LOD_MINIMAL   — cheapest rig, just root + orientation  (distance > lod_distance_2)

Headlamp profiles
-----------------
In daylight the beam is weak and only adds subtle rim contrast.
At night the cone is clearly visible from far away.
In whiteout, the far cone collapses and a scatter halo forms instead.

Public API
----------
DiegeticMultiplayerPresence(config=None)
  .update_trails(dt, storm_intensity)          — age segments, evict LRU
  .add_trail_event(event)                      — ingest a TRAIL_EVENT dict
  .get_active_segments_near(pos, radius)       — list[TrailSegment] for render
  .compute_headlamp(day_fraction, visibility)  — HeadlampProfile
  .lod_for_distance(dist)                      — LOD level string
  .debug_info()                                — dict for dev-mode gizmos

TrailDecalSystem (standalone)
  .add_segment(seg)
  .update(dt, storm_intensity)
  .segments_near(pos, radius) → list[TrailSegment]
  .active_count() → int
"""
from __future__ import annotations

import collections
import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Tuple

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants / enumerations
# ---------------------------------------------------------------------------

class TrailType(str, Enum):
    FOOTPRINT  = "footprint"
    SLIDE_MARK = "slide"
    DUST_PUFF  = "dustpuff"


class MaterialClass(str, Enum):
    DUST         = "Dust"
    LOOSE_DEBRIS = "LooseDebris"
    ICE_FILM     = "IceFilm"
    ROCK         = "Rock"


class LODLevel(str, Enum):
    FULL    = "full"     # complete procedural rig + IK
    REDUCED = "reduced"  # simplified pose, fewer bones
    MINIMAL = "minimal"  # cheapest root-only rig


# ---------------------------------------------------------------------------
# Default configuration values (overridden by CONFIG_DEFAULTS.json §presence)
# ---------------------------------------------------------------------------

_DEFAULTS: Dict[str, Any] = {
    "headlamp_intensity_day":       0.15,
    "headlamp_intensity_night":     1.0,
    "headlamp_scatter_in_whiteout": 0.6,
    "trails_ttl_sec":               90.0,
    "trails_ttl_storm_multiplier":  0.25,
    "trails_max_segments_near":     512,
    "trails_sector_cache_size":     64,
    "net_trail_batch_ms":           300,
    "audio_remote_foot_gain":       0.6,
    "audio_remote_cutoff_storm":    400.0,
    "remote_lod_distance_1":        40.0,
    "remote_lod_distance_2":        120.0,
}

# Base TTL per material class (seconds) — may be scaled by storm
_MATERIAL_TTL_SCALE: Dict[str, float] = {
    MaterialClass.DUST:         1.0,
    MaterialClass.LOOSE_DEBRIS: 0.8,
    MaterialClass.ICE_FILM:     2.0,   # scratches last longer
    MaterialClass.ROCK:         0.1,   # barely any trace
}

# Storm wind only affects soft surfaces (dust/debris), not ice/rock
_STORM_ERODES: Dict[str, bool] = {
    MaterialClass.DUST:         True,
    MaterialClass.LOOSE_DEBRIS: True,
    MaterialClass.ICE_FILM:     False,
    MaterialClass.ROCK:         False,
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TrailSegment:
    """One segment of a player-movement trail in world space.

    Attributes
    ----------
    pos          : [x, y, z] world-space position of the segment centre.
    direction    : [dx, dy, dz] normalised movement direction.
    width        : footprint / groove width in sim units.
    strength     : opacity/depth [0..1].
    trail_type   : one of :class:`TrailType`.
    material     : one of :class:`MaterialClass`.
    ttl          : remaining lifetime in seconds (counts down each update).
    created_at   : monotonic clock when the segment was added (for LRU).
    """
    pos:        List[float]
    direction:  List[float]
    width:      float
    strength:   float
    trail_type: TrailType
    material:   MaterialClass
    ttl:        float
    created_at: float = field(default_factory=time.monotonic)


@dataclass
class HeadlampProfile:
    """Computed headlamp parameters for one frame.

    cone_intensity     : 0..1 — beam brightness.
    cone_half_angle    : radians — half-angle of the headlamp cone.
    scatter_radius     : 0..1 — whiteout halo radius (0 = no halo).
    body_glow_strength : 0..1 — soft omnidirectional suit glow.
    """
    cone_intensity:     float
    cone_half_angle:    float
    scatter_radius:     float
    body_glow_strength: float


# ---------------------------------------------------------------------------
# TrailDecalSystem
# ---------------------------------------------------------------------------

class TrailDecalSystem:
    """Manages active trail segments with TTL and LRU eviction.

    Segments are stored in a flat :class:`collections.deque` (LRU order)
    and also indexed by a coarse sector key ``(int(x/sector_size),
    int(z/sector_size))`` for fast spatial queries.

    Parameters
    ----------
    max_segments_near:
        Maximum number of segments kept in the near-camera pool.
    sector_cache_size:
        Number of distinct sector buckets retained; LRU eviction applies
        when this is exceeded.
    base_ttl_sec:
        Default TTL (seconds) for a DUST segment at rest.
    ttl_storm_multiplier:
        Multiply effective TTL by this when storm is active.
    sector_size:
        Spatial bucket size in sim units (default 50).
    """

    _SECTOR_SIZE = 50.0

    def __init__(
        self,
        max_segments_near:   int   = 512,
        sector_cache_size:   int   = 64,
        base_ttl_sec:        float = 90.0,
        ttl_storm_multiplier: float = 0.25,
    ) -> None:
        self._max_near    = max_segments_near
        self._max_sectors = sector_cache_size
        self._base_ttl    = base_ttl_sec
        self._storm_mul   = ttl_storm_multiplier

        # All active segments ordered oldest→newest (LRU: pop left)
        self._segments: Deque[TrailSegment] = collections.deque()

        # Sector → list[TrailSegment] spatial index
        self._sectors: Dict[Tuple[int, int], List[TrailSegment]] = {}

        # LRU order for sector eviction
        self._sector_lru: Deque[Tuple[int, int]] = collections.deque()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sector_key(self, pos: List[float]) -> Tuple[int, int]:
        return (
            int(math.floor(pos[0] / self._SECTOR_SIZE)),
            int(math.floor(pos[2] / self._SECTOR_SIZE)),
        )

    def _effective_ttl(self, seg: TrailSegment, storm_intensity: float) -> float:
        """Return TTL in seconds, accounting for material and storm erosion."""
        mat_scale = _MATERIAL_TTL_SCALE.get(seg.material, 1.0)
        base      = self._base_ttl * mat_scale
        if storm_intensity > 0.0 and _STORM_ERODES.get(seg.material, False):
            storm_scale = 1.0 - (1.0 - self._storm_mul) * min(storm_intensity, 1.0)
            base        = base * storm_scale
        return max(base, 0.0)

    def _add_to_sector(self, seg: TrailSegment) -> None:
        key = self._sector_key(seg.pos)
        if key not in self._sectors:
            if len(self._sectors) >= self._max_sectors:
                # Evict oldest sector
                old_key = self._sector_lru.popleft()
                self._sectors.pop(old_key, None)
            self._sectors[key] = []
            self._sector_lru.append(key)
        self._sectors[key].append(seg)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_segment(self, seg: TrailSegment) -> None:
        """Add a trail segment, evicting oldest if the near-pool is full."""
        # LRU eviction on the flat deque — remove evicted segment from sector index
        while len(self._segments) >= self._max_near:
            old = self._segments.popleft()
            key = self._sector_key(old.pos)
            bucket = self._sectors.get(key)
            if bucket is not None:
                try:
                    bucket.remove(old)
                except ValueError:
                    pass

        self._segments.append(seg)
        self._add_to_sector(seg)

    def update(self, dt: float, storm_intensity: float = 0.0) -> None:
        """Age all segments and remove expired ones.

        In storm conditions, erosion-sensitive surfaces (dust, loose debris)
        lose TTL faster: the effective remaining lifetime shrinks relative to
        the calm baseline, so each real-time second removes more TTL.

        Parameters
        ----------
        dt:
            Elapsed time in seconds since last update.
        storm_intensity:
            0..1 storm level used to scale TTL for erosion-sensitive materials.
        """
        alive: Deque[TrailSegment] = collections.deque()
        for seg in self._segments:
            mat_scale = _MATERIAL_TTL_SCALE.get(seg.material, 1.0)
            base_eff  = self._base_ttl * mat_scale
            storm_eff = max(self._effective_ttl(seg, storm_intensity), 1e-6)
            # Erosion rate: how many stored-TTL seconds are consumed per real second.
            # In calm storm_eff == base_eff → rate 1.0 (subtract dt normally).
            # In full storm on dust storm_eff = base_eff * storm_mul → rate > 1.
            erosion_rate = max(base_eff, 1e-6) / storm_eff
            seg.ttl -= dt * erosion_rate
            if seg.ttl > 0.0:
                alive.append(seg)
        self._segments = alive

        # Rebuild sector index from alive segments
        self._sectors.clear()
        self._sector_lru.clear()
        for seg in self._segments:
            self._add_to_sector(seg)

    def segments_near(
        self,
        pos:    List[float],
        radius: float = 100.0,
    ) -> List[TrailSegment]:
        """Return segments whose position is within *radius* of *pos*.

        Uses the sector index for a coarse pre-filter, then checks exact
        distance. Rock-surface segments with nearly-zero strength are excluded.
        """
        if len(pos) < 3:
            return list(self._segments)

        cx, cy, cz     = pos[0], pos[1], pos[2]
        sector_reach   = int(math.ceil(radius / self._SECTOR_SIZE)) + 1
        base_key       = self._sector_key(pos)
        result: List[TrailSegment] = []

        for dx in range(-sector_reach, sector_reach + 1):
            for dz in range(-sector_reach, sector_reach + 1):
                key = (base_key[0] + dx, base_key[1] + dz)
                for seg in self._sectors.get(key, []):
                    if seg.ttl <= 0.0:
                        continue
                    sx, sy, sz = seg.pos[0], seg.pos[1], seg.pos[2]
                    dist_sq = (sx - cx) ** 2 + (sy - cy) ** 2 + (sz - cz) ** 2
                    if dist_sq <= radius * radius:
                        result.append(seg)

        return result

    def active_count(self) -> int:
        """Return number of currently alive segments."""
        return len(self._segments)


# ---------------------------------------------------------------------------
# HeadlampSystem
# ---------------------------------------------------------------------------

class HeadlampSystem:
    """Computes headlamp / body-glow parameters from environment context.

    Intensity transitions are smooth (no hard on/off switch).  The lamp is
    always on — it is part of the suit, not a player toggle.

    Parameters
    ----------
    intensity_day:
        Multiplier for the cone beam in bright daylight.
    intensity_night:
        Multiplier for the cone beam at night.
    scatter_in_whiteout:
        Maximum scatter-halo radius multiplier (0 = no halo).
    """

    _CONE_HALF_ANGLE_DEFAULT = math.radians(20.0)  # 20° half-angle
    _BODY_GLOW_BASE          = 0.08               # barely visible in daylight

    def __init__(
        self,
        intensity_day:       float = 0.15,
        intensity_night:     float = 1.0,
        scatter_in_whiteout: float = 0.6,
    ) -> None:
        self._int_day     = max(0.0, min(intensity_day,       1.0))
        self._int_night   = max(0.0, min(intensity_night,     1.0))
        self._scatter_max = max(0.0, min(scatter_in_whiteout, 1.0))

    def compute(
        self,
        day_fraction:  float,
        visibility:    float = 1.0,
    ) -> HeadlampProfile:
        """Return a :class:`HeadlampProfile` for the given environment.

        Parameters
        ----------
        day_fraction:
            0.0 = full night, 1.0 = peak day (matches PlanetTimeSystem output).
        visibility:
            0.0 = complete whiteout, 1.0 = clear sky.
        """
        day_fraction = max(0.0, min(day_fraction,  1.0))
        visibility   = max(0.0, min(visibility,    1.0))

        # Blend between night and day intensity
        cone_intensity = (
            self._int_night * (1.0 - day_fraction)
            + self._int_day * day_fraction
        )

        # In whiteout the forward cone collapses and a scatter halo expands
        whiteout_amount  = 1.0 - visibility
        cone_compression = 1.0 - 0.7 * whiteout_amount  # cone narrows
        scatter_radius   = self._scatter_max * whiteout_amount

        # Subtle body glow — slightly stronger at night
        body_glow = self._BODY_GLOW_BASE * (0.5 + 0.5 * (1.0 - day_fraction))

        return HeadlampProfile(
            cone_intensity     = cone_intensity,
            cone_half_angle    = self._CONE_HALF_ANGLE_DEFAULT * cone_compression,
            scatter_radius     = scatter_radius,
            body_glow_strength = body_glow,
        )


# ---------------------------------------------------------------------------
# RemotePlayerLOD
# ---------------------------------------------------------------------------

class RemotePlayerLOD:
    """Selects animation LOD level for a remote player by distance.

    LOD_FULL    — full procedural animation + IK  (dist < lod_distance_1)
    LOD_REDUCED — simplified pose, fewer bones    (lod_distance_1 <= dist < lod_distance_2)
    LOD_MINIMAL — cheapest root-only rig          (dist >= lod_distance_2)

    Parameters
    ----------
    lod_distance_1, lod_distance_2 : float
        Distance thresholds in simulation units.
    """

    def __init__(
        self,
        lod_distance_1: float = 40.0,
        lod_distance_2: float = 120.0,
    ) -> None:
        self._d1 = lod_distance_1
        self._d2 = lod_distance_2

    def lod_for_distance(self, distance: float) -> LODLevel:
        """Return the appropriate :class:`LODLevel` for *distance*."""
        if distance < self._d1:
            return LODLevel.FULL
        if distance < self._d2:
            return LODLevel.REDUCED
        return LODLevel.MINIMAL


# ---------------------------------------------------------------------------
# DiegeticMultiplayerPresence — top-level orchestrator
# ---------------------------------------------------------------------------

class DiegeticMultiplayerPresence:
    """Stage 25 diegetic presence layer — no UI, no markers, no names.

    Integrates :class:`TrailDecalSystem`, :class:`HeadlampSystem`, and
    :class:`RemotePlayerLOD` into one coherent update cycle.

    Parameters
    ----------
    config : dict | None
        Presence sub-dict from CONFIG_DEFAULTS (``config["presence"]``).
        Falls back to built-in defaults when *None*.

    Usage
    -----
    ::

        presence = DiegeticMultiplayerPresence(config=cfg.get("presence"))

        # Each simulation tick
        presence.update_trails(dt, storm_intensity=climate.storm_intensity)

        # When a TRAIL_EVENT arrives from the network
        presence.add_trail_event(trail_event_dict)

        # Before rendering remote players
        lod   = presence.lod_for_distance(dist_to_remote)
        lamp  = presence.compute_headlamp(day_fraction, visibility)
        segs  = presence.get_active_segments_near(camera_pos, radius=80)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = {**_DEFAULTS, **(config or {})}

        self._trails = TrailDecalSystem(
            max_segments_near    = int(cfg["trails_max_segments_near"]),
            sector_cache_size    = int(cfg["trails_sector_cache_size"]),
            base_ttl_sec         = float(cfg["trails_ttl_sec"]),
            ttl_storm_multiplier = float(cfg["trails_ttl_storm_multiplier"]),
        )
        self._headlamp = HeadlampSystem(
            intensity_day       = float(cfg["headlamp_intensity_day"]),
            intensity_night     = float(cfg["headlamp_intensity_night"]),
            scatter_in_whiteout = float(cfg["headlamp_scatter_in_whiteout"]),
        )
        self._lod = RemotePlayerLOD(
            lod_distance_1 = float(cfg["remote_lod_distance_1"]),
            lod_distance_2 = float(cfg["remote_lod_distance_2"]),
        )

        self._audio_remote_foot_gain    = float(cfg["audio_remote_foot_gain"])
        self._audio_remote_cutoff_storm = float(cfg["audio_remote_cutoff_storm"])

        self._debug_trail_rate: int = 0  # events ingested this second

    # ------------------------------------------------------------------
    # Trail management
    # ------------------------------------------------------------------

    def update_trails(
        self,
        dt:              float,
        storm_intensity: float = 0.0,
    ) -> None:
        """Advance trail TTL timers; remove dead segments.

        Call once per simulation tick.
        """
        self._trails.update(dt, storm_intensity)

    def add_trail_event(self, event: Dict[str, Any]) -> None:
        """Ingest a decoded TRAIL_EVENT dict and add it as a :class:`TrailSegment`.

        Expected keys (from :mod:`src.net.TrailEventProtocol`):
          ``type``, ``pos``, ``dir``, ``strength``, ``material``.
        Optional: ``width``, ``ttl_override``.

        Unknown trail types or missing required keys are silently ignored.
        """
        try:
            trail_type  = TrailType(event.get("type", "footprint"))
            pos         = list(event["pos"])
            direction   = list(event.get("dir",      [0.0, 0.0, 1.0]))
            strength    = float(event.get("strength", 1.0))
            mat_str     = event.get("material", MaterialClass.DUST.value)
            material    = MaterialClass(mat_str)
            width       = float(event.get("width",   0.3))
            ttl_base    = self._trails._base_ttl
            mat_scale   = _MATERIAL_TTL_SCALE.get(material, 1.0)
            ttl         = float(event.get("ttl_override", ttl_base * mat_scale))
        except (KeyError, ValueError, TypeError) as exc:
            _log.debug("DiegeticMultiplayerPresence: bad trail event %s: %s", event, exc)
            return

        seg = TrailSegment(
            pos        = pos,
            direction  = direction,
            width      = width,
            strength   = max(0.0, min(strength, 1.0)),
            trail_type = trail_type,
            material   = material,
            ttl        = ttl,
        )
        self._trails.add_segment(seg)
        self._debug_trail_rate += 1

    def get_active_segments_near(
        self,
        pos:    List[float],
        radius: float = 80.0,
    ) -> List[TrailSegment]:
        """Return trail segments near *pos* for the decal render pass."""
        return self._trails.segments_near(pos, radius)

    # ------------------------------------------------------------------
    # Headlamp
    # ------------------------------------------------------------------

    def compute_headlamp(
        self,
        day_fraction: float,
        visibility:   float = 1.0,
    ) -> HeadlampProfile:
        """Return :class:`HeadlampProfile` for current environment state."""
        return self._headlamp.compute(day_fraction, visibility)

    # ------------------------------------------------------------------
    # LOD
    # ------------------------------------------------------------------

    def lod_for_distance(self, distance: float) -> LODLevel:
        """Return animation :class:`LODLevel` for a remote player at *distance*."""
        return self._lod.lod_for_distance(distance)

    # ------------------------------------------------------------------
    # Audio parameters
    # ------------------------------------------------------------------

    def remote_footstep_gain(self, storm_intensity: float = 0.0) -> float:
        """Gain applied to remote footstep audio.

        In storm conditions the gain is further attenuated to reflect
        masking by wind and dust noise.
        """
        base_gain = self._audio_remote_foot_gain
        storm_atten = 1.0 - 0.5 * min(storm_intensity, 1.0)
        return base_gain * storm_atten

    def remote_footstep_cutoff_hz(self, storm_intensity: float = 0.0) -> float:
        """LP-filter cutoff for remote footstep audio.

        Reduces high-frequency content in storm to give a muffled, distant
        quality — the 'deaf presence' effect.
        """
        clear_cutoff  = 8000.0
        storm_cutoff  = self._audio_remote_cutoff_storm
        t = min(storm_intensity, 1.0)
        return clear_cutoff * (1.0 - t) + storm_cutoff * t

    # ------------------------------------------------------------------
    # Debug info
    # ------------------------------------------------------------------

    def debug_info(self) -> Dict[str, Any]:
        """Return a dict suitable for dev-mode gizmo visualisation."""
        info = {
            "active_trail_segments":       self._trails.active_count(),
            "trail_events_since_last_call": self._debug_trail_rate,
            "lod_d1":                       self._lod._d1,
            "lod_d2":                       self._lod._d2,
            "headlamp_int_day":             self._headlamp._int_day,
            "headlamp_int_night":           self._headlamp._int_night,
            "headlamp_scatter_max":         self._headlamp._scatter_max,
        }
        self._debug_trail_rate = 0
        return info
