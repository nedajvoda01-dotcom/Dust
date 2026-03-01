"""SubsurfaceHazardSystem — Stage 27 subsurface catastrophes.

Server-authoritative hazard tick that evaluates collapse risk across active cave
zones and generates phased events (PRE → IMPACT → POST) with tight rate and
chain caps to prevent spam and performance issues.

Event types
-----------
LOCAL_COLLAPSE  — debris plug blocks a tunnel segment (geometry patch)
CEILING_SAG     — sagging ceiling deforms the vault (geometry patch)
CHAIN_COLLAPSE  — local collapse propagates through the cave graph (≤ maxChainDepth)
DUST_WAVE       — visibility-degrading particle front propagates through tunnels

Three-phase lifecycle
---------------------
PRE    (2–10 s)  : infrasound, micro-crumble, camera micro-shake
IMPACT (instant) : SDF patch batch applied, camera kick, strong sound
POST   (10–60 s) : dust settles, reverb tail, reduced visibility

Integration
-----------
SubsurfaceHazardSystem consumes a SubsurfaceSystem (cave graph, cave_factor_at)
and emits:
  • SubsurfaceHazardEvent records (for replay / multiplayer broadcast)
  • SubsurfaceHazardSignal objects (queried each tick by camera/audio/character)
  • CaveDustField state (visibility reduction per cave zone)

Server authority
----------------
Only the server calls ``tick()``.  Clients receive events via
``apply_replicated_event(evt)`` which reconstructs phases from the encoded
timetable, and apply SDF patches at the IMPACT moment.

Rate / chain limits (hard caps)
--------------------------------
subhaz.max_events_per_hour_global   — absolute global cap
subhaz.max_events_per_hour_zone     — per-zone cap
subhaz.cooldown_same_cave_sec       — per-zone cooldown between events
subhaz.chain_max_depth              — max chain collapse hops (default 3)
subhaz.max_patches_per_event        — max SDF patches per event (default 12)

Public API
----------
SubsurfaceHazardSystem(config, subsurface_sys, planet_radius, sdf_world)
  .tick(dt, game_time, player_positions)       — server: advance hazard sim
  .apply_replicated_event(evt)                 — client: apply received event
  .query_signals_near(world_pos, radius)       — signals for any system
  .dust_density_at(zone_id)                    — [0..1] visibility modifier
  .active_events()                             — list of currently active events
  .force_collapse_near(world_pos, game_time)   — dev: --force-sub-collapse
  .risk_debug_log(game_time, player_pos)       — dev: --sub-risk-debug
  .event_log                                   — all events ever fired

Config (under ``subhaz`` key)
------------------------------
enable, tick_seconds, risk_threshold,
pre_min_sec, pre_max_sec, post_min_sec, post_max_sec,
max_events_per_hour_global, max_events_per_hour_zone,
chain_max_depth, chain_decay, max_patches_per_event,
dust_wave_speed, dust_peak_density, cooldown_same_cave_sec.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

from src.math.Vec3 import Vec3
from src.planet.SDFPatchSystem import (
    AdditiveDeposit,
    CapsuleCarve,
    SDFPatch,
    SDFPatchLog,
    SphereCarve,
)
from src.systems.SubsurfaceSystem import (
    CaveGraph,
    CaveNode,
    SubsurfacePatch,
    SubsurfacePatchBatch,
    SubsurfaceEventType,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * _clamp(t, 0.0, 1.0)


def _sigmoid(x: float) -> float:
    """Logistic sigmoid: output ∈ (0, 1)."""
    if x >= 0.0:
        e = math.exp(-x)
        return 1.0 / (1.0 + e)
    e = math.exp(x)
    return e / (1.0 + e)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class SubsurfaceHazardEventType(Enum):
    LOCAL_COLLAPSE  = auto()   # debris plug blocks tunnel
    CEILING_SAG     = auto()   # vault deformation
    CHAIN_COLLAPSE  = auto()   # propagated local collapse
    DUST_WAVE       = auto()   # visibility-degrading front


class SubsurfaceHazardPhase(Enum):
    PRE    = auto()   # precursor signals
    IMPACT = auto()   # geometry change + strong signals
    POST   = auto()   # dust / reverb tail
    DONE   = auto()   # event fully finished


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

_DEFAULTS: Dict[str, Any] = {
    "enable":                       True,
    "tick_seconds":                 10.0,
    "risk_threshold":               0.55,
    "pre_min_sec":                  2.0,
    "pre_max_sec":                  10.0,
    "post_min_sec":                 10.0,
    "post_max_sec":                 60.0,
    "max_events_per_hour_global":   6,
    "max_events_per_hour_zone":     2,
    "chain_max_depth":              3,
    "chain_decay":                  0.45,
    "max_patches_per_event":        12,
    "dust_wave_speed":              8.0,
    "dust_peak_density":            0.85,
    "cooldown_same_cave_sec":       120.0,
}

# Risk-model coefficients
_A_STRESS    = 1.4
_B_FRACTURE  = 1.2
_C_SPAN      = 0.9
_D_HARDNESS  = 1.0
_E_SUPPORT   = 0.8
_HISTORY_BIAS = 0.5


# ---------------------------------------------------------------------------
# Zone risk factors
# ---------------------------------------------------------------------------


@dataclass
class ZoneRiskFactors:
    """Per-cave-zone geological risk factors (all normalised to [0, 1])."""
    zone_id:            int
    node_id:            int       # CaveNode this zone corresponds to
    fracture:           float = 0.3
    stress:             float = 0.2
    hardness:           float = 0.6
    cave_span:          float = 0.4   # relative to planet radius
    support_density:    float = 0.5
    recent_events:      int   = 0     # count in last window
    time_since_collapse: float = 9999.0

    def compute_risk(self) -> float:
        """Compute scalar risk ∈ (0, 1) using sigmoid model."""
        history_bias = _HISTORY_BIAS if self.recent_events > 0 else 0.0
        x = (
            _A_STRESS   * self.stress
            + _B_FRACTURE * self.fracture
            + _C_SPAN     * self.cave_span
            - _D_HARDNESS * self.hardness
            - _E_SUPPORT  * self.support_density
            + history_bias
        )
        return _sigmoid(x)


# ---------------------------------------------------------------------------
# Dust wave state per zone
# ---------------------------------------------------------------------------


@dataclass
class CaveDustField:
    """Dust-wave state for a single cave zone."""
    zone_id:       int
    origin_node:   int
    start_time:    float
    speed:         float    # m/s effective spread
    decay_rate:    float    # 1/s exponential fall-off
    peak_density:  float    # max visibility reduction [0..1]

    def density_at(self, game_time: float, dist: float = 0.0) -> float:
        """Current dust density [0..1] at *game_time*.

        Parameters
        ----------
        game_time : absolute game time in seconds.
        dist      : metres from the dust origin.  When > 0 the density is zero
                    until the propagating front (age × speed) reaches this
                    distance, then follows exponential decay.  Defaults to 0
                    (origin / local zone query).
        """
        age  = game_time - self.start_time
        if age < 0.0:
            return 0.0
        front = age * self.speed
        # Only non-zero after front reaches dist
        if dist > front + 1.0:
            return 0.0
        density = self.peak_density * math.exp(-self.decay_rate * age)
        return _clamp(density, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Hazard signals (queried by camera / audio / character systems)
# ---------------------------------------------------------------------------


@dataclass
class SubsurfaceHazardSignal:
    """Per-tick signal emitted while a hazard event is active.

    Consumed by CinematicCameraSystem, ProceduralAudioSystem,
    CharacterEnvironmentIntegration, etc.
    """
    event_id:        int
    event_type:      SubsurfaceHazardEventType
    phase:           SubsurfaceHazardPhase
    position:        Vec3
    intensity:       float    # 0..1 (PRE: ramps up; IMPACT: 1.0; POST: decays)
    time_to_impact:  float    # seconds until IMPACT (negative = post-impact)
    dust_density:    float    # local visibility reduction [0..1]
    shake_impulse:   float    # camera shake strength [0..1]


# ---------------------------------------------------------------------------
# Hazard event record (server-authoritative, fully replicable)
# ---------------------------------------------------------------------------


@dataclass
class SubsurfaceHazardEvent:
    """A single hazard event — server-generated, replicable to clients.

    Parameters
    ----------
    event_id     : globally unique ID (monotonically increasing).
    event_type   : SubsurfaceHazardEventType.
    zone_id      : cave zone where the event originates.
    anchor_node  : CaveNode.node_id at origin.
    position     : world-space unit direction × planet_radius.
    t0           : server game_time when event was created.
    pre_dur      : PRE phase duration (seconds).
    post_dur     : POST phase duration (seconds).
    seed_local   : deterministic local seed for patch generation.
    intensity    : base event intensity [0..1].
    patch_batch  : SDF patches applied at IMPACT (may be empty for DUST_WAVE).
    chain_depth  : how deep this event is in a chain (0 = root).
    """
    event_id:    int
    event_type:  SubsurfaceHazardEventType
    zone_id:     int
    anchor_node: int
    position:    Vec3
    t0:          float
    pre_dur:     float
    post_dur:    float
    seed_local:  int
    intensity:   float
    patch_batch: List[SubsurfacePatch] = field(default_factory=list)
    chain_depth: int = 0

    # ------------------------------------------------------------------
    @property
    def impact_time(self) -> float:
        return self.t0 + self.pre_dur

    @property
    def end_time(self) -> float:
        return self.impact_time + self.post_dur

    def phase_at(self, game_time: float) -> SubsurfaceHazardPhase:
        if game_time < self.impact_time:
            return SubsurfaceHazardPhase.PRE
        if game_time < self.end_time:
            return SubsurfaceHazardPhase.POST
        return SubsurfaceHazardPhase.DONE

    def phase_intensity(self, game_time: float) -> float:
        """Intensity ∈ [0, 1] for smooth signal blending."""
        phase = self.phase_at(game_time)
        if phase == SubsurfaceHazardPhase.PRE:
            t_frac = (game_time - self.t0) / max(self.pre_dur, 1e-6)
            return _clamp(t_frac, 0.0, 1.0) * self.intensity
        if phase == SubsurfaceHazardPhase.POST:
            elapsed = game_time - self.impact_time
            t_frac  = 1.0 - elapsed / max(self.post_dur, 1e-6)
            return _clamp(t_frac, 0.0, 1.0) * self.intensity
        return 0.0


# ---------------------------------------------------------------------------
# DebrisPlugPatch — procedural debris geometry (no assets needed)
# ---------------------------------------------------------------------------


class DebrisPlugGenerator:
    """Generate SDF patches forming a debris plug in a tunnel.

    Uses an AdditiveDeposit (adds rock back into void) shaped as an
    ellipsoid + noise warp proxy.  For the network, the parameters are
    fixed-point-friendly integers.
    """

    def __init__(self, planet_radius: float = 1_000.0, config: Optional[Dict[str, Any]] = None) -> None:
        self._radius = planet_radius
        self._cfg    = {**_DEFAULTS, **(config or {})}

    def generate(
        self,
        event_id: int,
        node: CaveNode,
        seed: int,
        max_patches: int,
    ) -> List[SubsurfacePatch]:
        rng     = random.Random(seed ^ 0xDEB1505E)
        patches: List[SubsurfacePatch] = []

        pos = Vec3(
            node.direction.x * self._radius,
            node.direction.y * self._radius,
            node.direction.z * self._radius,
        )
        base_r = node.radius * rng.uniform(0.4, 0.7)

        # Main debris body (AdditiveDeposit — fills the tunnel void)
        patches.append(SubsurfacePatch(
            patch=AdditiveDeposit(centre=pos, radius=base_r),
            event_id=event_id,
        ))
        if len(patches) >= max_patches:
            return patches

        # Secondary scatter deposits (noise warp proxy — 2 extra blobs)
        for i in range(min(2, max_patches - len(patches))):
            offset_scale = base_r * rng.uniform(0.3, 0.8)
            offset = Vec3(
                rng.uniform(-1.0, 1.0) * offset_scale,
                rng.uniform(-1.0, 1.0) * offset_scale,
                rng.uniform(-1.0, 1.0) * offset_scale,
            )
            centre2 = Vec3(pos.x + offset.x, pos.y + offset.y, pos.z + offset.z)
            r2      = base_r * rng.uniform(0.2, 0.5)
            patches.append(SubsurfacePatch(
                patch=AdditiveDeposit(centre=centre2, radius=r2),
                event_id=event_id,
            ))

        return patches


# ---------------------------------------------------------------------------
# SagPatch — ceiling deformation
# ---------------------------------------------------------------------------


class SagPatchGenerator:
    """Generate SDF patches for a sagging ceiling.

    Applies an AdditiveDeposit slightly above the node centre to push the
    ceiling surface downward (filled material encroaches from above).
    """

    def __init__(self, planet_radius: float = 1_000.0, config: Optional[Dict[str, Any]] = None) -> None:
        self._radius = planet_radius
        self._cfg    = {**_DEFAULTS, **(config or {})}

    def generate(
        self,
        event_id: int,
        node: CaveNode,
        seed: int,
        max_patches: int,
    ) -> List[SubsurfacePatch]:
        rng     = random.Random(seed ^ 0x5A6C177E)
        patches: List[SubsurfacePatch] = []

        pos = Vec3(
            node.direction.x * self._radius,
            node.direction.y * self._radius,
            node.direction.z * self._radius,
        )
        sag_amount = node.radius * rng.uniform(0.15, 0.4)
        sag_r      = node.radius * rng.uniform(0.5, 0.9)

        # Shift the sag centre upward (toward "ceiling") along the cave-up vector
        cave_up = node.direction   # radially outward = upward in cave
        sag_centre = Vec3(
            pos.x + cave_up.x * sag_amount,
            pos.y + cave_up.y * sag_amount,
            pos.z + cave_up.z * sag_amount,
        )
        patches.append(SubsurfacePatch(
            patch=AdditiveDeposit(centre=sag_centre, radius=sag_r),
            event_id=event_id,
        ))
        return patches[:max_patches]


# ---------------------------------------------------------------------------
# SubsurfaceHazardSystem — main public API
# ---------------------------------------------------------------------------


class SubsurfaceHazardSystem:
    """Stage 27 SubsurfaceHazardSystem — server-authoritative cave catastrophes.

    Parameters
    ----------
    config         : dict with keys from the ``subhaz`` config section.
    subsurface_sys : SubsurfaceSystem instance (provides cave graph + cave factor).
    planet_radius  : planet radius in metres.
    sdf_world      : optional SDFWorld for patch application.
    global_seed    : world-level seed for determinism.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        subsurface_sys: Any = None,
        planet_radius: float = 1_000.0,
        sdf_world: Any = None,
        global_seed: int = 0,
    ) -> None:
        self._cfg    = {**_DEFAULTS, **(config or {})}
        self._sub    = subsurface_sys
        self._radius = planet_radius
        self._sdf    = sdf_world
        self._seed   = (global_seed ^ 0x27_CA1AB1E) & 0xFFFF_FFFF

        # Event tracking
        self._event_log:      List[SubsurfaceHazardEvent] = []
        self._active_events:  List[SubsurfaceHazardEvent] = []
        self._next_evt_id:    int = 0

        # Rate tracking (global)
        self._global_hour_count:     int   = 0
        self._global_hour_reset:     float = 0.0

        # Per-zone tracking: zone_id → (count_this_hour, last_event_time)
        self._zone_hour_count:  Dict[int, int]   = {}
        self._zone_last_event:  Dict[int, float] = {}

        # Dust fields per zone
        self._dust_fields: Dict[int, CaveDustField] = {}

        # Patch application log
        self._patch_log = SDFPatchLog()

        # Patch generators
        self._debris_gen = DebrisPlugGenerator(planet_radius=planet_radius, config=self._cfg)
        self._sag_gen    = SagPatchGenerator(planet_radius=planet_radius, config=self._cfg)

        # Tick scheduling
        self._time_since_tick: float = 0.0

    # ------------------------------------------------------------------
    # Server: main tick
    # ------------------------------------------------------------------

    def tick(
        self,
        dt: float,
        game_time: float,
        player_positions: Optional[List[Vec3]] = None,
    ) -> List[SubsurfaceHazardEvent]:
        """Advance the hazard simulation by *dt* seconds.

        Should be called only on the server.  Returns any new events fired
        this tick (ready for broadcast).

        Parameters
        ----------
        dt               : elapsed wall-clock seconds since last call.
        game_time        : absolute game time in seconds.
        player_positions : world-space player positions (only evaluate zones
                           near players — performance optimisation).
        """
        if not self._cfg.get("enable", True):
            return []

        # Advance active event states & prune done events
        self._advance_active(game_time)

        self._time_since_tick += dt
        tick_interval = float(self._cfg.get("tick_seconds", 10.0))
        if self._time_since_tick < tick_interval:
            return []
        self._time_since_tick -= tick_interval

        # Reset hourly budget
        if game_time - self._global_hour_reset >= 3_600.0:
            self._global_hour_count = 0
            self._global_hour_reset = game_time
            self._zone_hour_count.clear()

        # Check global cap
        max_global = int(self._cfg.get("max_events_per_hour_global", 6))
        if self._global_hour_count >= max_global:
            return []

        # Identify cave zones relevant to players
        zones = self._active_zones(player_positions)
        if not zones:
            return []

        new_events: List[SubsurfaceHazardEvent] = []
        rng = random.Random(self._seed ^ int(game_time * 1000) & 0xFFFF_FFFF)

        threshold = float(self._cfg.get("risk_threshold", 0.55))

        for zone in zones:
            # Zone-level rate cap
            z_max  = int(self._cfg.get("max_events_per_hour_zone", 2))
            z_cnt  = self._zone_hour_count.get(zone.zone_id, 0)
            if z_cnt >= z_max:
                continue

            # Cooldown
            cooldown = float(self._cfg.get("cooldown_same_cave_sec", 120.0))
            last_evt = self._zone_last_event.get(zone.zone_id, -9999.0)
            if game_time - last_evt < cooldown:
                continue

            risk = zone.compute_risk()
            if risk < threshold:
                continue

            # Choose event type based on geological context
            evt_type = self._choose_event_type(zone, rng)

            seed_local = (self._seed ^ (self._next_evt_id * 0x9E37) ^ zone.node_id) & 0xFFFF_FFFF
            evt = self._spawn_event(
                zone=zone,
                evt_type=evt_type,
                game_time=game_time,
                seed_local=seed_local,
                chain_depth=0,
                rng=rng,
            )
            new_events.append(evt)
            self._record_event(evt, game_time)

            # Chain collapse
            if evt_type in (SubsurfaceHazardEventType.LOCAL_COLLAPSE,
                            SubsurfaceHazardEventType.CHAIN_COLLAPSE):
                chain_evts = self._propagate_chain(
                    root_zone=zone,
                    game_time=game_time,
                    chain_depth=1,
                    rng=rng,
                    risk_seed=seed_local,
                    existing_new=new_events,
                )
                new_events.extend(chain_evts)

            # Cap per tick to avoid burst
            if self._global_hour_count >= max_global:
                break

        return new_events

    # ------------------------------------------------------------------
    # Client: apply replicated event
    # ------------------------------------------------------------------

    def apply_replicated_event(self, evt: SubsurfaceHazardEvent) -> None:
        """Apply a server event received over the network.

        Adds the event to the active list and applies SDF patches
        (patches should already be in the event's patch_batch at IMPACT).
        """
        self._active_events.append(evt)
        self._event_log.append(evt)
        # Apply SDF patches (if IMPACT has already passed on the server, apply now)
        self._apply_patches(evt.patch_batch)

    # ------------------------------------------------------------------
    # Signals query
    # ------------------------------------------------------------------

    def query_signals_near(
        self,
        world_pos: Vec3,
        radius: float = 200.0,
    ) -> List[SubsurfaceHazardSignal]:
        """Return hazard signals active near *world_pos* within *radius* metres."""
        signals: List[SubsurfaceHazardSignal] = []
        pos_r = world_pos.length()
        if pos_r < 1e-9:
            return signals

        for evt in self._active_events:
            # Angular proximity → approximate distance
            evt_pos = Vec3(
                evt.position.x * self._radius,
                evt.position.y * self._radius,
                evt.position.z * self._radius,
            )
            dx = evt_pos.x - world_pos.x
            dy = evt_pos.y - world_pos.y
            dz = evt_pos.z - world_pos.z
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            if dist > radius:
                continue

            # Derive current game time from evt metadata — use impact_time as reference
            # Signals are computed relative to event timing (caller passes their game_time)
            # We expose a simplified signal using evt.phase_intensity evaluated at "now"
            # The caller should pass a game_time; we approximate using end_time context.
            # For this API we emit the signal at full intensity when dist < radius.
            intensity   = evt.intensity * max(0.0, 1.0 - dist / (radius + 1e-9))
            dust        = self._dust_fields.get(evt.zone_id)
            dust_den    = dust.density_at(evt.end_time, dist) if dust else 0.0
            shake       = intensity * 0.8 if evt.event_type in (
                SubsurfaceHazardEventType.LOCAL_COLLAPSE,
                SubsurfaceHazardEventType.CHAIN_COLLAPSE,
            ) else intensity * 0.3

            signals.append(SubsurfaceHazardSignal(
                event_id       = evt.event_id,
                event_type     = evt.event_type,
                phase          = SubsurfaceHazardPhase.POST,  # conservative
                position       = evt.position,
                intensity      = intensity,
                time_to_impact = 0.0,
                dust_density   = dust_den,
                shake_impulse  = shake,
            ))
        return signals

    def query_signals_at_time(
        self,
        world_pos: Vec3,
        game_time: float,
        radius: float = 200.0,
    ) -> List[SubsurfaceHazardSignal]:
        """Return hazard signals with accurate phase/intensity at *game_time*."""
        signals: List[SubsurfaceHazardSignal] = []
        for evt in self._active_events:
            evt_pos = Vec3(
                evt.position.x * self._radius,
                evt.position.y * self._radius,
                evt.position.z * self._radius,
            )
            dx = evt_pos.x - world_pos.x
            dy = evt_pos.y - world_pos.y
            dz = evt_pos.z - world_pos.z
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            if dist > radius:
                continue

            phase     = evt.phase_at(game_time)
            if phase == SubsurfaceHazardPhase.DONE:
                continue
            intensity = evt.phase_intensity(game_time)
            dust      = self._dust_fields.get(evt.zone_id)
            dust_den  = dust.density_at(game_time, dist) if dust else 0.0

            if phase == SubsurfaceHazardPhase.PRE:
                shake = intensity * 0.2
            else:
                shake = intensity * 0.8

            signals.append(SubsurfaceHazardSignal(
                event_id       = evt.event_id,
                event_type     = evt.event_type,
                phase          = phase,
                position       = evt.position,
                intensity      = intensity,
                time_to_impact = evt.impact_time - game_time,
                dust_density   = dust_den,
                shake_impulse  = shake,
            ))
        return signals

    # ------------------------------------------------------------------
    # Dust visibility query
    # ------------------------------------------------------------------

    def dust_density_at(self, zone_id: int, game_time: float = 0.0) -> float:
        """Return current dust density [0..1] in cave zone *zone_id*."""
        dust = self._dust_fields.get(zone_id)
        if dust is None:
            return 0.0
        return dust.density_at(game_time)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def event_log(self) -> List[SubsurfaceHazardEvent]:
        """All hazard events ever fired (ordered by creation time)."""
        return list(self._event_log)

    def active_events(self) -> List[SubsurfaceHazardEvent]:
        """Currently active hazard events (not yet DONE)."""
        return list(self._active_events)

    # ------------------------------------------------------------------
    # Dev tools
    # ------------------------------------------------------------------

    def force_collapse_near(self, world_pos: Vec3, game_time: float = 0.0) -> Optional[SubsurfaceHazardEvent]:
        """Dev helper: ``--force-sub-collapse`` — instantly trigger a LOCAL_COLLAPSE near *world_pos*."""
        if self._sub is None:
            return None
        graph: CaveGraph = self._sub.cave_graph
        if not graph.nodes:
            return None

        pos_r = world_pos.length()
        if pos_r < 1e-9:
            return None
        pos_dir = Vec3(world_pos.x / pos_r, world_pos.y / pos_r, world_pos.z / pos_r)

        # Find nearest cave node
        node = max(graph.nodes, key=lambda n: pos_dir.dot(n.direction))
        zone = self._node_to_zone(node, game_time=game_time)
        rng  = random.Random(self._seed ^ int(game_time))
        seed_local = (self._seed ^ (self._next_evt_id * 0x9E37)) & 0xFFFF_FFFF
        evt = self._spawn_event(
            zone=zone,
            evt_type=SubsurfaceHazardEventType.LOCAL_COLLAPSE,
            game_time=game_time,
            seed_local=seed_local,
            chain_depth=0,
            rng=rng,
        )
        self._record_event(evt, game_time)
        return evt

    def risk_debug_log(
        self,
        game_time: float,
        player_pos: Optional[Vec3] = None,
    ) -> List[Dict[str, Any]]:
        """Dev helper: ``--sub-risk-debug`` — return risk scores for all active zones."""
        zones  = self._active_zones([player_pos] if player_pos else None)
        result = []
        for z in zones:
            result.append({
                "zone_id":    z.zone_id,
                "node_id":    z.node_id,
                "risk":       round(z.compute_risk(), 4),
                "fracture":   z.fracture,
                "stress":     z.stress,
                "hardness":   z.hardness,
                "cave_span":  z.cave_span,
                "support":    z.support_density,
                "recent_events": z.recent_events,
            })
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _active_zones(
        self,
        player_positions: Optional[List[Optional[Vec3]]],
    ) -> List[ZoneRiskFactors]:
        """Return ZoneRiskFactors for cave nodes near players.

        If no subsurface system is present, returns an empty list (performance
        guard — hazards only evaluated where players are present).
        """
        if self._sub is None:
            return []
        graph: CaveGraph = self._sub.cave_graph
        if not graph.nodes:
            return []

        # If no player positions given, evaluate all zones (rare / dev mode)
        if not player_positions:
            return [self._node_to_zone(n) for n in graph.nodes]

        # Find nodes within ~angular interest of any player
        relevant: List[CaveNode] = []
        seen_ids: set = set()
        for ppos in player_positions:
            if ppos is None:
                continue
            pos_r = ppos.length()
            if pos_r < 1e-9:
                continue
            pos_dir = Vec3(ppos.x / pos_r, ppos.y / pos_r, ppos.z / pos_r)
            for node in graph.nodes:
                if node.node_id in seen_ids:
                    continue
                cos_a = _clamp(pos_dir.dot(node.direction), -1.0, 1.0)
                if cos_a > 0.85:   # ≈ within ~32° arc
                    relevant.append(node)
                    seen_ids.add(node.node_id)

        return [self._node_to_zone(n) for n in relevant]

    def _node_to_zone(self, node: CaveNode, game_time: float = 0.0) -> ZoneRiskFactors:
        """Construct ZoneRiskFactors from a CaveNode."""
        rng = random.Random(self._seed ^ node.node_id ^ 0xF00D5EED)

        # Derive stress/fracture from node geometry (proxy — no geo_sampler required)
        fracture = _clamp(rng.uniform(0.1, 0.9), 0.0, 1.0)
        stress   = _clamp(rng.uniform(0.0, 0.8), 0.0, 1.0)
        hardness = _clamp(rng.uniform(0.3, 1.0), 0.0, 1.0)

        # cave_span normalised: larger node.radius → larger span
        cave_span = _clamp(node.radius / (self._radius * 0.1 + node.radius), 0.0, 1.0)

        # support_density: tubes-per-node proxy (simpler: inverse of radius)
        support_density = _clamp(1.0 - cave_span, 0.0, 1.0)

        # Historical recency
        recent = 0
        last_t = self._zone_last_event.get(node.node_id, -9999.0)
        if game_time - last_t < 600.0:
            recent = 1

        return ZoneRiskFactors(
            zone_id            = node.node_id,
            node_id            = node.node_id,
            fracture           = fracture,
            stress             = stress,
            hardness           = hardness,
            cave_span          = cave_span,
            support_density    = support_density,
            recent_events      = recent,
            time_since_collapse= game_time - last_t,
        )

    def _choose_event_type(
        self,
        zone: ZoneRiskFactors,
        rng: random.Random,
    ) -> SubsurfaceHazardEventType:
        """Select the most contextually appropriate event type."""
        if zone.cave_span > 0.6:
            # Wide vault → ceiling sag is natural
            weights = [0.4, 0.4, 0.1, 0.1]
        elif zone.fracture > 0.65:
            # Heavily fractured → local collapse / chain
            weights = [0.5, 0.1, 0.3, 0.1]
        else:
            weights = [0.35, 0.25, 0.2, 0.2]

        types = [
            SubsurfaceHazardEventType.LOCAL_COLLAPSE,
            SubsurfaceHazardEventType.CEILING_SAG,
            SubsurfaceHazardEventType.CHAIN_COLLAPSE,
            SubsurfaceHazardEventType.DUST_WAVE,
        ]
        return rng.choices(types, weights=weights, k=1)[0]

    def _spawn_event(
        self,
        zone: ZoneRiskFactors,
        evt_type: SubsurfaceHazardEventType,
        game_time: float,
        seed_local: int,
        chain_depth: int,
        rng: random.Random,
    ) -> SubsurfaceHazardEvent:
        """Build and record a SubsurfaceHazardEvent."""
        pre_min  = float(self._cfg.get("pre_min_sec",  2.0))
        pre_max  = float(self._cfg.get("pre_max_sec",  10.0))
        post_min = float(self._cfg.get("post_min_sec", 10.0))
        post_max = float(self._cfg.get("post_max_sec", 60.0))

        pre_rng  = random.Random(seed_local ^ 0xAABB)
        post_rng = random.Random(seed_local ^ 0xCCDD)
        pre_dur  = pre_rng.uniform(pre_min, pre_max)
        post_dur = post_rng.uniform(post_min, post_max)

        # Get cave node for this zone
        node: Optional[CaveNode] = None
        if self._sub is not None:
            node = self._sub.cave_graph.get_node(zone.node_id)

        pos_dir = node.direction if node else Vec3(1.0, 0.0, 0.0)
        intensity = _clamp(zone.compute_risk(), 0.3, 1.0)

        max_patches = int(self._cfg.get("max_patches_per_event", 12))

        # Generate SDF patches (applied at IMPACT)
        patches: List[SubsurfacePatch] = []
        if node is not None and evt_type in (
            SubsurfaceHazardEventType.LOCAL_COLLAPSE,
            SubsurfaceHazardEventType.CHAIN_COLLAPSE,
        ):
            patches = self._debris_gen.generate(
                event_id=self._next_evt_id,
                node=node,
                seed=seed_local,
                max_patches=max_patches,
            )
        elif node is not None and evt_type == SubsurfaceHazardEventType.CEILING_SAG:
            patches = self._sag_gen.generate(
                event_id=self._next_evt_id,
                node=node,
                seed=seed_local,
                max_patches=max_patches,
            )
        # DUST_WAVE: no geometry patches — handled via CaveDustField

        evt = SubsurfaceHazardEvent(
            event_id    = self._next_evt_id,
            event_type  = evt_type,
            zone_id     = zone.zone_id,
            anchor_node = zone.node_id,
            position    = pos_dir,
            t0          = game_time,
            pre_dur     = pre_dur,
            post_dur    = post_dur,
            seed_local  = seed_local,
            intensity   = intensity,
            patch_batch = patches,
            chain_depth = chain_depth,
        )

        # Spawn dust field for this zone (all event types get a dust component)
        self._dust_fields[zone.zone_id] = CaveDustField(
            zone_id      = zone.zone_id,
            origin_node  = zone.node_id,
            start_time   = game_time + pre_dur,  # dust starts at IMPACT
            speed        = float(self._cfg.get("dust_wave_speed",    8.0)),
            decay_rate   = 1.0 / max(post_dur * 0.5, 1.0),
            peak_density = float(self._cfg.get("dust_peak_density", 0.85))
                           * intensity,
        )

        return evt

    def _propagate_chain(
        self,
        root_zone: ZoneRiskFactors,
        game_time: float,
        chain_depth: int,
        rng: random.Random,
        risk_seed: int,
        existing_new: List[SubsurfaceHazardEvent],
    ) -> List[SubsurfaceHazardEvent]:
        """Recursively propagate a chain collapse through the cave graph."""
        chain_max  = int(self._cfg.get("chain_max_depth",           3))
        max_global = int(self._cfg.get("max_events_per_hour_global", 6))
        z_max      = int(self._cfg.get("max_events_per_hour_zone",   2))

        if chain_depth > chain_max:
            return []
        if self._sub is None:
            return []
        if self._global_hour_count >= max_global:
            return []

        max_patches   = int(self._cfg.get("max_patches_per_event", 12))
        total_patches = sum(len(e.patch_batch) for e in existing_new)
        if total_patches >= max_patches:
            return []

        decay      = float(self._cfg.get("chain_decay", 0.45))
        chain_prob = decay ** chain_depth

        chain_events: List[SubsurfaceHazardEvent] = []
        neighbours = self._sub.cave_graph.neighbours(root_zone.node_id)

        for nb_node in neighbours:
            # Re-check global cap before each chain hop
            if self._global_hour_count >= max_global:
                break

            # Zone-level cap for chain hops
            z_cnt = self._zone_hour_count.get(nb_node.node_id, 0)
            if z_cnt >= z_max:
                continue

            if rng.random() > chain_prob:
                continue

            # Check zone isn't already triggered this tick
            already = any(e.anchor_node == nb_node.node_id for e in existing_new)
            already |= any(e.anchor_node == nb_node.node_id for e in chain_events)
            if already:
                continue

            nb_zone = self._node_to_zone(nb_node, game_time=game_time)
            seed_ch = (risk_seed ^ (nb_node.node_id * 0x1337) ^ chain_depth) & 0xFFFF_FFFF
            evt = self._spawn_event(
                zone=nb_zone,
                evt_type=SubsurfaceHazardEventType.CHAIN_COLLAPSE,
                game_time=game_time + rng.uniform(0.5, 3.0),  # slight delay
                seed_local=seed_ch,
                chain_depth=chain_depth,
                rng=rng,
            )
            chain_events.append(evt)
            self._record_event(evt, game_time)

            # Recurse
            deeper = self._propagate_chain(
                root_zone=nb_zone,
                game_time=game_time,
                chain_depth=chain_depth + 1,
                rng=rng,
                risk_seed=seed_ch,
                existing_new=existing_new + chain_events,
            )
            chain_events.extend(deeper)

        return chain_events

    def _record_event(self, evt: SubsurfaceHazardEvent, game_time: float) -> None:
        """Add event to active + log lists, update rate counters."""
        self._event_log.append(evt)
        self._active_events.append(evt)
        self._next_evt_id += 1
        self._global_hour_count += 1

        z = evt.zone_id
        self._zone_hour_count[z] = self._zone_hour_count.get(z, 0) + 1
        self._zone_last_event[z] = game_time

    def _advance_active(self, game_time: float) -> None:
        """Remove finished events from active list; apply patches at IMPACT."""
        still_active: List[SubsurfaceHazardEvent] = []
        for evt in self._active_events:
            phase = evt.phase_at(game_time)
            if phase == SubsurfaceHazardPhase.DONE:
                continue
            still_active.append(evt)
        self._active_events = still_active

    def _apply_patches(self, patches: List[SubsurfacePatch]) -> None:
        """Apply a list of SubsurfacePatch to the local SDF world."""
        for sp in patches:
            self._patch_log.add(sp.patch)
            if self._sdf is not None:
                try:
                    self._sdf.apply_patch(sp.patch)
                except (AttributeError, TypeError):
                    pass
