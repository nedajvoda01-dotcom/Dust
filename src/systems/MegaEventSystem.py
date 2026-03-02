"""MegaEventSystem — Stage 33 rare mega-events.

Very infrequent but impactful planetary events that change the mood and
sometimes the geometry of large regions without turning the game into a
constant apocalypse.

Event types
-----------
SUPERCELL_DUST_STORM  — regional super-storm covering 100–800 km
GLOBAL_DUST_VEIL      — planet-wide dust veil that dims the sun for hours
GREAT_RIFT            — tectonic rift with SDF geometry patches
RING_SHADOW_ANOMALY   — anomalous ring-shadow configuration (optional)

Four-phase lifecycle
--------------------
PRE → ONSET → PEAK → AFTERMATH → DONE

Server-authoritative
--------------------
Only the server calls ``update()``.  Clients receive
``MEGA_EVENT_ANNOUNCE`` and reproduce phases from simTime using the
phase duration table embedded in the message.

Caps / fallback
---------------
- Global cooldown (configurable, default 6 h simTime)
- Poisson-style rare gate (seeded, reproducible)
- Max rift segments cap
- Max macro-phenomena spawned by storm cap
- Fallback to ``veil-only`` or ``reduced`` mode under load

Integration
-----------
* AstroClimateCoupler  (stage 29) — applies globalDustVeilFactor to
  insolation / temperature.
* LongHorizonEvolutionSystem (stage 30) — aftermath updates DustThickness,
  IceFilm fields.
* MacroAtmospherePhenomenaSystem (stage 32) — super-storm spawns
  DUST_WALL phenomena (up to cap).
* SubsurfaceHazardSystem (stage 27) — rift ONSET triggers collapse risk.

Public API
----------
MegaEventSystem(config, world_seed)
  .update(dt, sim_time, inputs)        — server: advance scheduler + phases
  .apply_replicated_event(announce)    — client: ingest announce message
  .active_event() → MegaEvent | None   — current event (None if between events)
  .get_announce_message() → dict | None — MEGA_EVENT_ANNOUNCE for broadcast
  .get_global_coeffs() → dict          — globalDustVeilFactor etc. for clients
  .get_rift_patches() → list[RiftPatch] — SDF patches for GREAT_RIFT
  .get_character_modifiers(world_pos, sim_time) → CharacterMegaMod
  .to_state_dict() → dict             — serialisable state for persistence
  .from_state_dict(d) → None          — restore persisted state
  .force_event(event_type, sim_time)  — dev: --force-mega-* flags

Config keys (under ``mega.*``)
------------------------------
enable, cooldown_hours_sim, poisson_lambda,
pre_min_min, pre_max_min, onset_min_min, onset_max_min,
peak_min_min, peak_max_min, aftermath_min_hours, aftermath_max_hours,
storm.radius_km_min, storm.radius_km_max,
storm.max_macro_spawned,
veil.max_factor,
rift.segment_count_max, rift.length_km_max,
rift.width_m_min, rift.width_m_max,
rift.depth_m_min, rift.depth_m_max,
fallback_mode  (``veil-only`` | ``reduced`` | ``none``)
"""
from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

from src.core.Config import Config

_TWO_PI = 2.0 * math.pi


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * _clamp(t)


def _smooth(t: float) -> float:
    """Smoothstep [0..1] → [0..1] with eased edges."""
    t = _clamp(t)
    return t * t * (3.0 - 2.0 * t)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class MegaEventType(Enum):
    SUPERCELL_DUST_STORM = auto()   # regional super-storm
    GLOBAL_DUST_VEIL     = auto()   # planet-wide dust veil
    GREAT_RIFT           = auto()   # tectonic rift (geometry patches)
    RING_SHADOW_ANOMALY  = auto()   # anomalous ring shadow


class MegaEventPhase(Enum):
    PRE       = auto()   # precursor signs
    ONSET     = auto()   # event begins
    PEAK      = auto()   # maximum intensity
    AFTERMATH = auto()   # slow decay
    DONE      = auto()   # fully finished


# ---------------------------------------------------------------------------
# RiftPatch — one SDF segment of a GREAT_RIFT event
# ---------------------------------------------------------------------------

@dataclass
class RiftPatch:
    """A single segment of a tectonic rift.

    Represents a capsule-shaped carve + uplift applied to the planet SDF.
    Patches are numbered sequentially; later patches are applied during PEAK.
    """
    patch_id:      int
    start_lat:     float   # radians
    start_lon:     float   # radians
    end_lat:       float   # radians
    end_lon:       float   # radians
    width_m:       float   # carve half-width metres
    depth_m:       float   # carve depth metres
    uplift_m:      float   # rim uplift metres (positive = raised edge)
    phase_gate:    MegaEventPhase  # earliest phase when this patch activates

    def to_dict(self) -> Dict[str, Any]:
        return {
            "patch_id":   self.patch_id,
            "start_lat":  self.start_lat,
            "start_lon":  self.start_lon,
            "end_lat":    self.end_lat,
            "end_lon":    self.end_lon,
            "width_m":    self.width_m,
            "depth_m":    self.depth_m,
            "uplift_m":   self.uplift_m,
            "phase_gate": self.phase_gate.name,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "RiftPatch":
        return RiftPatch(
            patch_id   = int(d["patch_id"]),
            start_lat  = float(d["start_lat"]),
            start_lon  = float(d["start_lon"]),
            end_lat    = float(d["end_lat"]),
            end_lon    = float(d["end_lon"]),
            width_m    = float(d["width_m"]),
            depth_m    = float(d["depth_m"]),
            uplift_m   = float(d["uplift_m"]),
            phase_gate = MegaEventPhase[d["phase_gate"]],
        )


# ---------------------------------------------------------------------------
# MegaEvent — one complete event record
# ---------------------------------------------------------------------------

@dataclass
class MegaEvent:
    """All data describing a single mega-event lifecycle.

    ``phase_durations`` maps each phase name to its length in seconds.
    ``anchor_lat`` / ``anchor_lon`` is the event epicentre in radians.
    ``anchor_radius_m`` is the affected radius in metres.
    """
    event_id:       int
    event_type:     MegaEventType
    anchor_lat:     float            # radians
    anchor_lon:     float            # radians
    anchor_radius_m: float           # metres
    start_time:     float            # simTime when PRE begins
    phase_durations: Dict[str, float] = field(default_factory=dict)
    seed:           int = 0
    rift_patches:   List[RiftPatch] = field(default_factory=list)
    # Runtime (not serialised to announce, recomputed from start_time)
    _current_phase: MegaEventPhase = field(
        default=MegaEventPhase.PRE, init=False, repr=False
    )

    # ------------------------------------------------------------------
    def phase_start(self, phase: MegaEventPhase) -> float:
        """SimTime when *phase* begins."""
        order = [
            MegaEventPhase.PRE,
            MegaEventPhase.ONSET,
            MegaEventPhase.PEAK,
            MegaEventPhase.AFTERMATH,
        ]
        t = self.start_time
        for p in order:
            if p == phase:
                return t
            t += self.phase_durations.get(p.name, 0.0)
        return t  # DONE

    def end_time(self) -> float:
        return self.phase_start(MegaEventPhase.DONE)

    def current_phase(self, sim_time: float) -> MegaEventPhase:
        """Return the phase at *sim_time*."""
        order = [
            MegaEventPhase.PRE,
            MegaEventPhase.ONSET,
            MegaEventPhase.PEAK,
            MegaEventPhase.AFTERMATH,
        ]
        t = self.start_time
        for p in order:
            t += self.phase_durations.get(p.name, 0.0)
            if sim_time < t:
                return p
        return MegaEventPhase.DONE

    def intensity(self, sim_time: float) -> float:
        """Smooth intensity curve [0..1] over the full event lifecycle."""
        phase = self.current_phase(sim_time)
        if phase == MegaEventPhase.DONE:
            return 0.0
        t_phase = sim_time - self.phase_start(phase)
        dur = self.phase_durations.get(phase.name, 1.0)
        frac = _clamp(t_phase / max(dur, 1e-9))
        if phase == MegaEventPhase.PRE:
            return _smooth(frac) * 0.3
        if phase == MegaEventPhase.ONSET:
            return _lerp(0.3, 1.0, _smooth(frac))
        if phase == MegaEventPhase.PEAK:
            # Slight pulsing around full intensity; clamped to [0, 1].
            return _clamp(0.85 + 0.15 * _smooth(math.sin(frac * math.pi)))
        # AFTERMATH — decay from 1 to 0
        return _lerp(0.9, 0.0, _smooth(frac))

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id":        self.event_id,
            "event_type":      self.event_type.name,
            "anchor_lat":      self.anchor_lat,
            "anchor_lon":      self.anchor_lon,
            "anchor_radius_m": self.anchor_radius_m,
            "start_time":      self.start_time,
            "phase_durations": self.phase_durations,
            "seed":            self.seed,
            "rift_patches":    [p.to_dict() for p in self.rift_patches],
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "MegaEvent":
        evt = MegaEvent(
            event_id        = int(d["event_id"]),
            event_type      = MegaEventType[d["event_type"]],
            anchor_lat      = float(d["anchor_lat"]),
            anchor_lon      = float(d["anchor_lon"]),
            anchor_radius_m = float(d["anchor_radius_m"]),
            start_time      = float(d["start_time"]),
            phase_durations = {k: float(v) for k, v in d.get("phase_durations", {}).items()},
            seed            = int(d.get("seed", 0)),
            rift_patches    = [RiftPatch.from_dict(p) for p in d.get("rift_patches", [])],
        )
        return evt


# ---------------------------------------------------------------------------
# CharacterMegaMod — modifiers applied to the character near an event
# ---------------------------------------------------------------------------

@dataclass
class CharacterMegaMod:
    """Per-frame character movement modifiers due to active mega-event."""
    wind_resistance_add: float = 0.0   # extra drag multiplier [0..1]
    brace_probability:   float = 0.0   # per-second stumble/brace chance
    collapse_risk_add:   float = 0.0   # bonus cave collapse risk [0..1]


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

_SIM_HOUR = 3600.0  # 1 simulated hour in seconds

_DEFAULTS: Dict[str, Any] = {
    "enable":                  True,
    "cooldown_hours_sim":      6.0,
    "poisson_lambda":          0.03,      # expected events per sim-hour
    "pre_min_min":             5.0,       # PRE phase minimum minutes
    "pre_max_min":             30.0,
    "onset_min_min":           1.0,
    "onset_max_min":           10.0,
    "peak_min_min":            10.0,
    "peak_max_min":            60.0,
    "aftermath_min_hours":     1.0,
    "aftermath_max_hours":     12.0,
    "storm": {
        "radius_km_min":       100.0,
        "radius_km_max":       800.0,
        "max_macro_spawned":   8,
    },
    "veil": {
        "max_factor":          0.6,       # max insolation reduction (0=none, 1=full)
    },
    "rift": {
        "segment_count_max":   20,
        "length_km_max":       500.0,
        "width_m_min":         50.0,
        "width_m_max":         500.0,
        "depth_m_min":         50.0,
        "depth_m_max":         500.0,
    },
    "fallback_mode":           "none",    # "veil-only" | "reduced" | "none"
}


def _cfg_get(cfg: Config, *keys: str, default: Any) -> Any:
    try:
        return cfg.get("mega", *keys, default=default)
    except Exception:
        return default


# ---------------------------------------------------------------------------
# MegaEventScheduler — when to fire the next event
# ---------------------------------------------------------------------------

class MegaEventScheduler:
    """Decides whether and which mega-event to trigger.

    Uses a seeded Poisson-style gate:  at each evaluation the scheduler
    checks whether enough simTime has passed (cooldown) and rolls against
    a probabilistic threshold derived from ``hash(worldId, epochIndex)``.

    Parameters
    ----------
    world_seed : int
        Stable world seed used to derive the hash gate.
    cooldown_sec : float
        Minimum simTime between events (default 6 h).
    poisson_lambda : float
        Expected events per sim-hour (governs check probability).
    """

    def __init__(
        self,
        world_seed: int,
        cooldown_sec: float,
        poisson_lambda: float,
    ) -> None:
        self._world_seed     = world_seed
        self._cooldown_sec   = max(60.0, cooldown_sec)
        self._lambda         = max(1e-6, poisson_lambda)
        self._epoch_index    = 0      # increments after each event fires
        self._last_event_time: float = -1e18

    # ------------------------------------------------------------------
    def should_trigger(
        self,
        sim_time: float,
        dt: float,
        scores: Dict[str, float],
    ) -> Optional[MegaEventType]:
        """Return the event type to trigger, or *None*.

        Parameters
        ----------
        sim_time : float  Current server simTime (seconds).
        dt       : float  Tick duration (seconds).
        scores   : dict   ``{type_name: float}`` plausibility scores [0..1].
        """
        # Cooldown gate
        if sim_time - self._last_event_time < self._cooldown_sec:
            return None

        # Probability check: Poisson process approximated by per-tick Bernoulli
        # P(event in dt) = 1 - exp(-lambda * dt / 3600)
        p_tick = 1.0 - math.exp(-self._lambda * dt / _SIM_HOUR)
        gate = self._rare_gate(self._epoch_index)
        if gate > p_tick:
            return None

        # Choose type with highest score (must exceed threshold)
        best_type: Optional[MegaEventType] = None
        best_score = 0.3  # minimum score threshold
        for event_type in MegaEventType:
            s = scores.get(event_type.name, 0.0)
            if s > best_score:
                best_score = s
                best_type = event_type

        return best_type

    def record_triggered(self, sim_time: float) -> None:
        """Call after a trigger fires to reset the cooldown."""
        self._last_event_time = sim_time
        self._epoch_index += 1

    # ------------------------------------------------------------------
    def _rare_gate(self, epoch: int) -> float:
        """Deterministic value in [0, 1] for this world + epoch.

        Lower means "easier to trigger"; higher means "harder".
        """
        raw = hashlib.md5(
            struct.pack(">QQ", self._world_seed & 0xFFFFFFFFFFFFFFFF, epoch)
        ).digest()
        return int.from_bytes(raw[:4], "big") / 0x100000000

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "epoch_index":       self._epoch_index,
            "last_event_time":   self._last_event_time,
        }

    def from_dict(self, d: Dict[str, Any]) -> None:
        self._epoch_index      = int(d.get("epoch_index", 0))
        self._last_event_time  = float(d.get("last_event_time", -1e18))


# ---------------------------------------------------------------------------
# MegaEventSystem — main class
# ---------------------------------------------------------------------------

class MegaEventSystem:
    """Server-authoritative rare mega-event system (Stage 33).

    Parameters
    ----------
    config :
        Global Config (reads ``mega.*`` namespace).
    world_seed :
        Stable integer seed for this world.
    """

    _PLANET_R: float = 1_000_000.0    # planet radius metres (1000 km)

    def __init__(
        self,
        config: Optional[Config] = None,
        world_seed: int = 0,
    ) -> None:
        if config is None:
            config = Config()

        self._world_seed = world_seed
        self._rng_state  = world_seed & 0xFFFFFFFF

        # Read config
        self._enabled: bool = bool(
            _cfg_get(config, "enable", default=_DEFAULTS["enable"])
        )
        cooldown_hours = float(
            _cfg_get(config, "cooldown_hours_sim",
                     default=_DEFAULTS["cooldown_hours_sim"])
        )
        poisson_lambda = float(
            _cfg_get(config, "poisson_lambda",
                     default=_DEFAULTS["poisson_lambda"])
        )

        # Phase durations (minutes → seconds)
        self._pre_min   = float(_cfg_get(config, "pre_min_min",   default=_DEFAULTS["pre_min_min"]))   * 60.0
        self._pre_max   = float(_cfg_get(config, "pre_max_min",   default=_DEFAULTS["pre_max_min"]))   * 60.0
        self._onset_min = float(_cfg_get(config, "onset_min_min", default=_DEFAULTS["onset_min_min"])) * 60.0
        self._onset_max = float(_cfg_get(config, "onset_max_min", default=_DEFAULTS["onset_max_min"])) * 60.0
        self._peak_min  = float(_cfg_get(config, "peak_min_min",  default=_DEFAULTS["peak_min_min"]))  * 60.0
        self._peak_max  = float(_cfg_get(config, "peak_max_min",  default=_DEFAULTS["peak_max_min"]))  * 60.0
        self._aftm_min  = float(_cfg_get(config, "aftermath_min_hours",
                                         default=_DEFAULTS["aftermath_min_hours"])) * _SIM_HOUR
        self._aftm_max  = float(_cfg_get(config, "aftermath_max_hours",
                                         default=_DEFAULTS["aftermath_max_hours"])) * _SIM_HOUR

        # Storm params
        self._storm_r_min    = float(_cfg_get(config, "storm", "radius_km_min",
                                              default=_DEFAULTS["storm"]["radius_km_min"])) * 1000.0
        self._storm_r_max    = float(_cfg_get(config, "storm", "radius_km_max",
                                              default=_DEFAULTS["storm"]["radius_km_max"])) * 1000.0
        self._storm_max_macro = int(_cfg_get(config, "storm", "max_macro_spawned",
                                             default=_DEFAULTS["storm"]["max_macro_spawned"]))

        # Veil params
        self._veil_max_factor = _clamp(float(
            _cfg_get(config, "veil", "max_factor",
                     default=_DEFAULTS["veil"]["max_factor"])
        ))

        # Rift params
        self._rift_seg_max   = max(1, int(_cfg_get(config, "rift", "segment_count_max",
                                                    default=_DEFAULTS["rift"]["segment_count_max"])))
        self._rift_len_max   = float(_cfg_get(config, "rift", "length_km_max",
                                              default=_DEFAULTS["rift"]["length_km_max"])) * 1000.0
        self._rift_w_min     = float(_cfg_get(config, "rift", "width_m_min",
                                              default=_DEFAULTS["rift"]["width_m_min"]))
        self._rift_w_max     = float(_cfg_get(config, "rift", "width_m_max",
                                              default=_DEFAULTS["rift"]["width_m_max"]))
        self._rift_d_min     = float(_cfg_get(config, "rift", "depth_m_min",
                                              default=_DEFAULTS["rift"]["depth_m_min"]))
        self._rift_d_max     = float(_cfg_get(config, "rift", "depth_m_max",
                                              default=_DEFAULTS["rift"]["depth_m_max"]))

        self._fallback_mode: str = str(
            _cfg_get(config, "fallback_mode", default=_DEFAULTS["fallback_mode"])
        )

        # Scheduler
        self._scheduler = MegaEventScheduler(
            world_seed     = world_seed,
            cooldown_sec   = cooldown_hours * _SIM_HOUR,
            poisson_lambda = poisson_lambda,
        )

        # Runtime state
        self._active_event: Optional[MegaEvent] = None
        self._event_counter: int = 0
        self._announce_pending: Optional[Dict[str, Any]] = None
        self._event_log: List[Dict[str, Any]] = []   # abridged log entries

    # ------------------------------------------------------------------
    # Main update (server only)
    # ------------------------------------------------------------------

    def update(
        self,
        dt: float,
        sim_time: float,
        *,
        dust_lift_potential: float = 0.0,
        dust_thickness_mean: float = 0.0,
        ring_shadow_intensity: float = 0.0,
        fracture_fatigue_mean: float = 0.0,
        subsurface_collapse_rate: float = 0.0,
    ) -> None:
        """Advance the mega-event simulation one tick.

        Parameters
        ----------
        dt : float
            Tick duration in simulated seconds.
        sim_time : float
            Current server simulated time (seconds since epoch).
        dust_lift_potential : float
            Mean global DustLiftPotential from AstroClimateCoupler [0..1].
        dust_thickness_mean : float
            Mean global DustThickness from LongHorizonEvolutionSystem [0..1].
        ring_shadow_intensity : float
            Current ring shadow intensity from AstroSystem [0..1].
        fracture_fatigue_mean : float
            Mean FractureFatigue from LongHorizonEvolutionSystem [0..1].
        subsurface_collapse_rate : float
            Recent subsurface collapse rate from SubsurfaceHazardSystem [0..1].
        """
        if not self._enabled:
            return

        # Advance or expire the active event
        if self._active_event is not None:
            if self._active_event.current_phase(sim_time) == MegaEventPhase.DONE:
                self._log_event_ended(self._active_event, sim_time)
                self._active_event = None
            else:
                return  # event in progress; do not schedule another

        # Compute plausibility scores for each event type
        scores = self._compute_scores(
            dust_lift_potential      = dust_lift_potential,
            dust_thickness_mean      = dust_thickness_mean,
            ring_shadow_intensity    = ring_shadow_intensity,
            fracture_fatigue_mean    = fracture_fatigue_mean,
            subsurface_collapse_rate = subsurface_collapse_rate,
        )

        event_type = self._scheduler.should_trigger(sim_time, dt, scores)
        if event_type is None:
            return

        # Apply fallback mode
        event_type = self._apply_fallback(event_type)

        # Create the event
        event = self._create_event(event_type, sim_time)
        self._active_event = event
        self._scheduler.record_triggered(sim_time)
        self._announce_pending = self._build_announce(event)
        self._log_event_started(event)

    # ------------------------------------------------------------------
    # Client-side replication
    # ------------------------------------------------------------------

    def apply_replicated_event(self, announce: Dict[str, Any]) -> None:
        """Ingest a MEGA_EVENT_ANNOUNCE from the server (client-side).

        Rebuilds the full MegaEvent state from the announce message so the
        client can compute phases and intensity deterministically.
        """
        evt = MegaEvent.from_dict(announce)
        self._active_event = evt
        self._announce_pending = None

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def active_event(self) -> Optional[MegaEvent]:
        """Return the current active MegaEvent, or *None*."""
        return self._active_event

    def get_announce_message(self) -> Optional[Dict[str, Any]]:
        """Pop and return the pending MEGA_EVENT_ANNOUNCE dict, or *None*."""
        msg = self._announce_pending
        self._announce_pending = None
        return msg

    def get_global_coeffs(self, sim_time: float) -> Dict[str, float]:
        """Return global atmospheric coefficients driven by any active event.

        Keys
        ----
        ``globalDustVeilFactor``   : insolation reduction [0..1]
        ``scatterBoost``           : atmosphere scatter multiplier [0..∞)
        ``ringEdgeBoost``          : ring-shadow edge amplification
        """
        coeffs: Dict[str, float] = {
            "globalDustVeilFactor": 0.0,
            "scatterBoost":         0.0,
            "ringEdgeBoost":        0.0,
        }
        evt = self._active_event
        if evt is None:
            return coeffs
        inten = evt.intensity(sim_time)
        if evt.event_type == MegaEventType.GLOBAL_DUST_VEIL:
            coeffs["globalDustVeilFactor"] = inten * self._veil_max_factor
            coeffs["scatterBoost"]         = inten * 0.4
        elif evt.event_type == MegaEventType.SUPERCELL_DUST_STORM:
            coeffs["globalDustVeilFactor"] = inten * self._veil_max_factor * 0.25
        elif evt.event_type == MegaEventType.RING_SHADOW_ANOMALY:
            coeffs["ringEdgeBoost"] = inten * 1.5
        return coeffs

    def get_rift_patches(self, sim_time: float) -> List[RiftPatch]:
        """Return active RiftPatch objects for the current GREAT_RIFT event.

        Only patches whose ``phase_gate`` is at or before the current phase
        are returned.
        """
        evt = self._active_event
        if evt is None or evt.event_type != MegaEventType.GREAT_RIFT:
            return []
        phase = evt.current_phase(sim_time)
        phase_order = [
            MegaEventPhase.PRE,
            MegaEventPhase.ONSET,
            MegaEventPhase.PEAK,
            MegaEventPhase.AFTERMATH,
            MegaEventPhase.DONE,
        ]
        phase_idx = phase_order.index(phase)
        return [
            p for p in evt.rift_patches
            if phase_order.index(p.phase_gate) <= phase_idx
        ]

    def get_character_modifiers(
        self, lat: float, lon: float, sim_time: float
    ) -> CharacterMegaMod:
        """Return per-frame character modifiers near *lat/lon*."""
        mod = CharacterMegaMod()
        evt = self._active_event
        if evt is None:
            return mod
        dist = self._great_circle_m(lat, lon, evt.anchor_lat, evt.anchor_lon)
        if dist > evt.anchor_radius_m * 2.0:
            return mod
        proximity = _clamp(1.0 - dist / evt.anchor_radius_m)
        inten = evt.intensity(sim_time)
        if evt.event_type in (
            MegaEventType.SUPERCELL_DUST_STORM,
            MegaEventType.GREAT_RIFT,
        ):
            mod.wind_resistance_add = proximity * inten * 0.6
            mod.brace_probability   = proximity * inten * 0.05
        if evt.event_type == MegaEventType.GREAT_RIFT:
            mod.collapse_risk_add = proximity * inten * 0.4
        return mod

    def get_debug_state(self) -> Dict[str, Any]:
        """Return a dict useful for server-side diagnostics."""
        evt = self._active_event
        return {
            "active":       evt is not None,
            "event_type":   evt.event_type.name if evt else None,
            "event_id":     evt.event_id if evt else None,
            "epoch_index":  self._scheduler._epoch_index,
            "log_length":   len(self._event_log),
        }

    # ------------------------------------------------------------------
    # Dev helpers (--force-mega-* flags)
    # ------------------------------------------------------------------

    def force_event(
        self,
        event_type: MegaEventType,
        sim_time: float,
    ) -> MegaEvent:
        """Immediately create and activate a mega-event of *event_type*.

        Overrides any cooldown.  Intended for dev/debug only.
        """
        evt = self._create_event(event_type, sim_time)
        self._active_event = evt
        self._announce_pending = self._build_announce(evt)
        self._log_event_started(evt)
        return evt

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_state_dict(self) -> Dict[str, Any]:
        """Serialise full runtime state for disk persistence."""
        return {
            "enabled":        self._enabled,
            "event_counter":  self._event_counter,
            "active_event":   self._active_event.to_dict() if self._active_event else None,
            "scheduler":      self._scheduler.to_dict(),
            "event_log":      self._event_log[-256:],   # keep last 256 entries
        }

    def from_state_dict(self, d: Dict[str, Any]) -> None:
        """Restore state from a previously serialised dict."""
        self._event_counter = int(d.get("event_counter", 0))
        ae = d.get("active_event")
        self._active_event = MegaEvent.from_dict(ae) if ae else None
        sched_d = d.get("scheduler", {})
        if sched_d:
            self._scheduler.from_dict(sched_d)
        self._event_log = list(d.get("event_log", []))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_scores(
        self,
        dust_lift_potential: float,
        dust_thickness_mean: float,
        ring_shadow_intensity: float,
        fracture_fatigue_mean: float,
        subsurface_collapse_rate: float,
    ) -> Dict[str, float]:
        storm_score = (
            0.5 * dust_lift_potential
            + 0.3 * dust_thickness_mean
            + 0.2 * ring_shadow_intensity
        )
        veil_score = (
            0.6 * dust_thickness_mean
            + 0.4 * dust_lift_potential
        )
        rift_score = (
            0.5 * fracture_fatigue_mean
            + 0.3 * subsurface_collapse_rate
            + 0.2 * ring_shadow_intensity
        )
        ring_anom_score = (
            0.7 * ring_shadow_intensity
            + 0.3 * dust_lift_potential
        )
        return {
            MegaEventType.SUPERCELL_DUST_STORM.name: _clamp(storm_score),
            MegaEventType.GLOBAL_DUST_VEIL.name:     _clamp(veil_score),
            MegaEventType.GREAT_RIFT.name:           _clamp(rift_score),
            MegaEventType.RING_SHADOW_ANOMALY.name:  _clamp(ring_anom_score),
        }

    def _apply_fallback(self, event_type: MegaEventType) -> MegaEventType:
        if self._fallback_mode == "veil-only":
            return MegaEventType.GLOBAL_DUST_VEIL
        if self._fallback_mode == "reduced":
            if event_type == MegaEventType.GREAT_RIFT:
                return MegaEventType.GLOBAL_DUST_VEIL
        return event_type

    def _rand(self, lo: float = 0.0, hi: float = 1.0) -> float:
        """Simple seeded LCG random in [lo, hi)."""
        self._rng_state = (self._rng_state * 1664525 + 1013904223) & 0xFFFFFFFF
        t = self._rng_state / 0x100000000
        return lo + t * (hi - lo)

    def _create_event(
        self, event_type: MegaEventType, sim_time: float
    ) -> MegaEvent:
        """Construct a new MegaEvent with deterministic parameters."""
        self._event_counter += 1
        event_seed = (self._world_seed ^ (self._event_counter * 2654435761)) & 0xFFFFFFFF

        # Phase durations
        pre_dur    = self._rand(self._pre_min,   self._pre_max)
        onset_dur  = self._rand(self._onset_min, self._onset_max)
        peak_dur   = self._rand(self._peak_min,  self._peak_max)
        aftm_dur   = self._rand(self._aftm_min,  self._aftm_max)

        # Anchor position (deterministic from seed)
        anchor_lat = self._rand(-math.pi * 0.4, math.pi * 0.4)
        anchor_lon = self._rand(-math.pi, math.pi)

        # Radius
        if event_type == MegaEventType.SUPERCELL_DUST_STORM:
            radius_m = self._rand(self._storm_r_min, self._storm_r_max)
        elif event_type == MegaEventType.GLOBAL_DUST_VEIL:
            radius_m = self._PLANET_R * math.pi  # global → huge radius
        elif event_type == MegaEventType.GREAT_RIFT:
            radius_m = self._rand(50_000.0, 300_000.0)
        else:  # RING_SHADOW_ANOMALY
            radius_m = self._PLANET_R * math.pi

        evt = MegaEvent(
            event_id        = self._event_counter,
            event_type      = event_type,
            anchor_lat      = anchor_lat,
            anchor_lon      = anchor_lon,
            anchor_radius_m = radius_m,
            start_time      = sim_time,
            phase_durations = {
                MegaEventPhase.PRE.name:       pre_dur,
                MegaEventPhase.ONSET.name:     onset_dur,
                MegaEventPhase.PEAK.name:      peak_dur,
                MegaEventPhase.AFTERMATH.name: aftm_dur,
            },
            seed = event_seed,
        )

        if event_type == MegaEventType.GREAT_RIFT:
            evt.rift_patches = self._generate_rift_patches(
                evt, event_seed
            )

        return evt

    def _generate_rift_patches(
        self, evt: MegaEvent, seed: int
    ) -> List[RiftPatch]:
        """Build rift segment patches within the cap."""
        # Use a local LCG for reproducible generation from event seed
        state = seed & 0xFFFFFFFF

        def _r(lo: float, hi: float) -> float:
            nonlocal state
            state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
            return lo + (state / 0x100000000) * (hi - lo)

        n_segs = min(
            self._rift_seg_max,
            max(3, int(_r(3, self._rift_seg_max + 1))),
        )

        # Build a chain of points along a great-circle arc
        bearing = _r(0.0, _TWO_PI)
        total_len_m = _r(50_000.0, self._rift_len_max)
        seg_len_m = total_len_m / n_segs

        # Phase distribution: segments assigned to ONSET (first ~20%), PEAK (~60%),
        # AFTERMATH (~20%). At least 1 ONSET segment is always guaranteed so that
        # phase-gating logic is exercisable even for small segment counts.
        onset_boundary = max(1, int(n_segs * 0.2))
        peak_boundary  = max(onset_boundary + 1, int(n_segs * 0.8))

        patches: List[RiftPatch] = []
        lat0, lon0 = evt.anchor_lat, evt.anchor_lon

        for i in range(n_segs):
            lat1, lon1 = self._geodesic_step(lat0, lon0, bearing, seg_len_m)
            if i < onset_boundary:
                gate = MegaEventPhase.ONSET
            elif i < peak_boundary:
                gate = MegaEventPhase.PEAK
            else:
                gate = MegaEventPhase.AFTERMATH

            patches.append(RiftPatch(
                patch_id   = i,
                start_lat  = lat0,
                start_lon  = lon0,
                end_lat    = lat1,
                end_lon    = lon1,
                width_m    = _r(self._rift_w_min, self._rift_w_max),
                depth_m    = _r(self._rift_d_min, self._rift_d_max),
                uplift_m   = _r(10.0, 80.0),
                phase_gate = gate,
            ))
            lat0, lon0 = lat1, lon1

        return patches

    def _build_announce(self, evt: MegaEvent) -> Dict[str, Any]:
        """Build the MEGA_EVENT_ANNOUNCE network message."""
        return {
            "msg_type":    "MEGA_EVENT_ANNOUNCE",
            **evt.to_dict(),
        }

    def _log_event_started(self, evt: MegaEvent) -> None:
        self._event_log.append({
            "action":     "started",
            "event_id":   evt.event_id,
            "event_type": evt.event_type.name,
            "start_time": evt.start_time,
        })

    def _log_event_ended(self, evt: MegaEvent, sim_time: float) -> None:
        self._event_log.append({
            "action":     "ended",
            "event_id":   evt.event_id,
            "event_type": evt.event_type.name,
            "end_time":   sim_time,
        })

    @staticmethod
    def _great_circle_m(
        lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """Approximate great-circle distance in metres."""
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat * 0.5) ** 2 + (
            math.cos(lat1) * math.cos(lat2) * math.sin(dlon * 0.5) ** 2
        )
        c = 2.0 * math.atan2(math.sqrt(max(a, 0.0)), math.sqrt(max(1.0 - a, 0.0)))
        return MegaEventSystem._PLANET_R * c

    @staticmethod
    def _geodesic_step(
        lat: float, lon: float, bearing: float, dist_m: float
    ) -> Tuple[float, float]:
        """Move along a geodesic from (lat, lon) by *dist_m* in *bearing* radians."""
        R = MegaEventSystem._PLANET_R
        d = dist_m / R
        lat2 = math.asin(
            _clamp(
                math.sin(lat) * math.cos(d)
                + math.cos(lat) * math.sin(d) * math.cos(bearing),
                -1.0, 1.0,
            )
        )
        lon2 = lon + math.atan2(
            math.sin(bearing) * math.sin(d) * math.cos(lat),
            math.cos(d) - math.sin(lat) * math.sin(lat2),
        )
        return lat2, lon2
