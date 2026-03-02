"""MacroAtmospherePhenomenaSystem — Stage 32 large-scale atmospheric phenomena.

Generates and manages macro-scale atmospheric events visible from far away:

* **DUST_WALL** — haboob-style dust-wall on the horizon, approaching
* **DRY_LIGHTNING_CLUSTER** — dry lightning inside dense dust (no rain)
* **RING_SHADOW_FRONT** — visible band of changed illumination from ring-shadow edge
* **MEGA_VORTEX** — large slow-moving dust vortex as a moving dark patch

Design goals
------------
* **Server-authoritative** — positions, timings, trajectories decided on server.
* **No per-frame randomness** — all stochastic decisions made once into
  deterministic schedules (``flashTimes`` for lightning, etc.).
* **Parametric / volumetric** — no particle systems on the horizon; phenomena
  are represented as analytical volumes (capsules, slabs, edge masks).
* **Budget-limited** — ``maxMacroPhenomenaActive`` / ``maxDustWallsActive`` etc.
  are hard caps; cheapest LOD applied when budget is tight.
* **Multiplayer-safe** — two clients in the same sector with the same seed + tick
  receive identical ``MacroPhenomenon`` parameters.

Public API
----------
``MacroAtmospherePhenomenaSystem(config=None, world_seed=0)``

``update(dt, sim_time, *, storm_cells, dust_density_fn, wind_speed_fn,
         dust_lift_potential_fn, ring_shadow_edge_fn, sun_dirs)``
    Advance the system. All callbacks are ``(lat_rad, lon_rad) -> float``.

``get_active_phenomena() -> list[MacroPhenomenon]``
    Ordered list of currently active phenomena (server or client side).

``apply_local_coupling(player_pos_lat, player_pos_lon) -> LocalCouplingResult``
    Wind / dust / visibility modifiers for a specific player position.

``get_render_params(player_pos_lat, player_pos_lon, view_dir_xz_angle)
    -> list[RenderParam]``
    Analytical render parameters for the volumetric sky pass (pre-pixelisation).

``get_audio_triggers(player_pos_lat, player_pos_lon, sim_time) -> list[AudioTrigger]``
    Due audio events (rumble, lightning) for this tick.

``get_debug_state() -> dict``
    Dev-console dump of active phenomena counts and nearest distances.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, List, Optional, Tuple

from src.core.Config import Config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


def _smoothstep(edge0: float, edge1: float, x: float) -> float:
    if edge1 <= edge0:
        return 1.0 if x >= edge1 else 0.0
    t = _clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * _clamp(t, 0.0, 1.0)


def _det_hash_float(seed: int, *args: int) -> float:
    """Deterministic float in [0, 1) from integer arguments."""
    raw = hashlib.sha256(
        ":".join(str(x) for x in (seed, *args)).encode()
    ).digest()
    val = (raw[0] << 24 | raw[1] << 16 | raw[2] << 8 | raw[3]) & 0xFFFF_FFFF
    return val / 0xFFFF_FFFF


def _angular_dist(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in radians between two lat/lon pairs."""
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (math.sin(dlat * 0.5) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin(dlon * 0.5) ** 2)
    return 2.0 * math.asin(math.sqrt(max(0.0, a)))


# ---------------------------------------------------------------------------
# Phenomenon type
# ---------------------------------------------------------------------------

class PhenType(IntEnum):
    DUST_WALL               = 0
    DRY_LIGHTNING_CLUSTER   = 1
    RING_SHADOW_FRONT       = 2
    MEGA_VORTEX             = 3


# ---------------------------------------------------------------------------
# MacroPhenomenon — compact network-friendly representation
# ---------------------------------------------------------------------------

@dataclass
class MacroPhenomenon:
    """A single large-scale atmospheric phenomenon (server-authoritative).

    All spatial quantities in radians (lat/lon) and m/s (velocity).
    Times in simulation seconds.
    """
    phen_id:     int       # unique id for this session
    phen_type:   PhenType

    # Spatial anchor
    anchor_lat:  float     # radians
    anchor_lon:  float     # radians
    height:      float     # metres above surface (representative)

    # Shape (approximate capsule/slab dimensions)
    radius:      float     # angular radius or half-length (radians)
    thickness:   float     # metres (depth perpendicular to front)

    # Dynamics
    vel_u:       float     # eastward m/s
    vel_v:       float     # northward m/s

    intensity:   float     # [0, 1]

    start_time:  float     # sim seconds
    end_time:    float     # sim seconds

    seed:        int       # per-phenomenon deterministic seed

    # For RING_SHADOW_FRONT
    front_normal_lon: float = 0.0   # normal direction of front (lon component)
    front_normal_lat: float = 0.0   # normal direction of front (lat component)
    edge_width_rad:   float = 0.0   # angular width of the edge band

    # For DRY_LIGHTNING_CLUSTER — pre-computed flash schedule
    flash_times: Tuple[float, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# LocalCouplingResult — influence of nearby phenomena on player
# ---------------------------------------------------------------------------

@dataclass
class LocalCouplingResult:
    """Additive modifiers for climate coupling at a player position."""
    wind_boost:         float = 0.0   # extra wind speed m/s
    dust_density_add:   float = 0.0   # [0, 1] additive dust
    visibility_mul:     float = 1.0   # multiplier on base visibility
    insolation_mul:     float = 1.0   # multiplier on sunlight (ring front)


# ---------------------------------------------------------------------------
# RenderParam — volumetric render descriptor for one phenomenon
# ---------------------------------------------------------------------------

@dataclass
class RenderParam:
    """Analytical volume descriptor passed to the volumetric sky pass."""
    phen_id:       int
    phen_type:     PhenType

    # Capsule / slab anchor (lat/lon in radians from player POV)
    rel_lat:       float      # relative lat offset to player (radians)
    rel_lon:       float      # relative lon offset to player (radians)

    angular_radius: float     # angular size from horizon (radians)
    horizon_angle:  float     # bearing from player (radians, 0=north, E=pi/2)

    intensity:     float
    base_density:  float      # 0-1 for volumetric integration

    # Lightning flash additive light intensity (0 normally, spike on flash)
    lightning_flash: float    = 0.0

    # Edge width for ring front (radians)
    edge_width_rad: float     = 0.0

    # Forward scattering boost (low-sun dust wall)
    fwd_scatter:   float      = 0.0


# ---------------------------------------------------------------------------
# AudioTrigger
# ---------------------------------------------------------------------------

@dataclass
class AudioTrigger:
    """An audio event to fire this tick."""
    event:      str       # 'dust_wall_rumble' | 'lightning_crack'
    gain:       float     # 0-1
    low_pass:   float     # 0-1 (1 = fully low-pass filtered)
    delay_sec:  float     # seconds to delay playback (distance/c_sound)


# ---------------------------------------------------------------------------
# MacroAtmospherePhenomenaSystem
# ---------------------------------------------------------------------------

class MacroAtmospherePhenomenaSystem:
    """Stage 32 server-authoritative macro-atmosphere phenomena.

    Parameters
    ----------
    config:
        Global Config (reads ``macro.*`` namespace).
    world_seed:
        Stable integer seed for this world / server session.
    """

    _SOUND_SPEED: float = 340.0           # m/s approximate speed of sound
    _PLANET_R:    float = 1_000_000.0     # planet radius metres (1000 km)
    _HALF_PI:     float = math.pi * 0.5

    def __init__(
        self,
        config: Optional[Config] = None,
        world_seed: int = 0,
    ) -> None:
        if config is None:
            config = Config()

        self._enabled: bool = bool(
            config.get("macro", "enable", default=True)
        )
        self._tick_seconds: float = max(
            0.1, float(config.get("macro", "tick_seconds", default=2.0))
        )
        self._max_active: int = max(
            1, int(config.get("macro", "max_active", default=16))
        )

        # --- Dust wall ---
        self._dw_frequency: float = _clamp(
            float(config.get("macro", "dustwall", "frequency", default=0.3)), 0.0, 1.0
        )
        self._dw_len_min: float = float(
            config.get("macro", "dustwall", "length_km_min", default=2.0)
        ) * 1000.0
        self._dw_len_max: float = float(
            config.get("macro", "dustwall", "length_km_max", default=20.0)
        ) * 1000.0
        self._dw_thick_min: float = float(
            config.get("macro", "dustwall", "thickness_m_min", default=200.0)
        )
        self._dw_thick_max: float = float(
            config.get("macro", "dustwall", "thickness_m_max", default=1000.0)
        )
        self._dw_intensity_min: float = float(
            config.get("macro", "dustwall", "intensity_min", default=0.3)
        )
        self._dw_intensity_max: float = float(
            config.get("macro", "dustwall", "intensity_max", default=0.9)
        )
        self._max_dust_walls: int = max(
            1, int(config.get("macro", "dustwall", "max_active", default=4))
        )

        # --- Lightning ---
        self._lightning_enable: bool = bool(
            config.get("macro", "lightning", "enable", default=True)
        )
        self._lightning_rarity: float = _clamp(
            float(config.get("macro", "lightning", "rarity", default=0.15)), 0.0, 1.0
        )
        self._lightning_flash_min: int = max(
            1, int(config.get("macro", "lightning", "flash_count_min", default=2))
        )
        self._lightning_flash_max: int = max(
            self._lightning_flash_min,
            int(config.get("macro", "lightning", "flash_count_max", default=8)),
        )
        self._max_lightning: int = max(
            1, int(config.get("macro", "lightning", "max_active", default=2))
        )

        # --- Ring shadow front ---
        self._rsf_enable: bool = bool(
            config.get("macro", "ringfront", "enable", default=True)
        )
        self._rsf_edge_width: float = float(
            config.get("macro", "ringfront", "edge_width_km", default=5.0)
        ) * 1000.0  # metres -> used as angular width below
        self._max_ring_fronts: int = max(
            1, int(config.get("macro", "ringfront", "max_active", default=2))
        )

        # --- Mega vortex ---
        self._vortex_enable: bool = bool(
            config.get("macro", "vortex", "enable", default=True)
        )
        self._max_vortices: int = max(
            1, int(config.get("macro", "vortex", "max_active", default=2))
        )

        # --- Render ---
        self._vol_steps_near: int = max(
            4, int(config.get("macro", "render", "volumetric_steps_near", default=16))
        )
        self._vol_steps_far: int = max(
            2, int(config.get("macro", "render", "volumetric_steps_far", default=8))
        )

        # --- Audio ---
        self._rumble_gain: float = _clamp(
            float(config.get("macro", "audio", "rumble_gain", default=0.6)), 0.0, 1.0
        )
        self._lightning_gain: float = _clamp(
            float(config.get("macro", "audio", "lightning_gain", default=0.8)), 0.0, 1.0
        )

        self._world_seed: int = int(world_seed)

        # Internal state
        self._phenomena: list[MacroPhenomenon] = []
        self._next_phen_id: int = 1
        self._tick_accum: float = 0.0
        self._tick_index: int = 0

        # Track which lightning cluster flashes have been emitted
        self._emitted_flashes: dict[int, set] = {}   # phen_id -> set of flash indices

    # ------------------------------------------------------------------
    # Public update
    # ------------------------------------------------------------------

    def update(
        self,
        dt: float,
        sim_time: float,
        *,
        storm_cells=None,
        dust_density_fn: Optional[Callable[[float, float], float]] = None,
        wind_speed_fn:   Optional[Callable[[float, float], float]] = None,
        dust_lift_potential_fn: Optional[Callable[[float, float], float]] = None,
        ring_shadow_edge_fn:    Optional[Callable[[float, float], float]] = None,
        sun_dirs: Optional[list] = None,
    ) -> None:
        """Advance the system by *dt* seconds.

        Parameters
        ----------
        dt:
            Frame / tick time in seconds.
        sim_time:
            Server-authoritative simulation time in seconds.
        storm_cells:
            List of ``StormCell``-like objects with ``.center_lat``,
            ``.center_lon``, ``.radius``, ``.intensity``, ``.vel_u``, ``.vel_v``.
        dust_density_fn:
            ``(lat_rad, lon_rad) -> float [0,1]`` dust density field.
        wind_speed_fn:
            ``(lat_rad, lon_rad) -> float`` wind speed in m/s.
        dust_lift_potential_fn:
            ``(lat_rad, lon_rad) -> float [0,1]`` DustLiftPotential.
        ring_shadow_edge_fn:
            ``(lat_rad, lon_rad) -> float [0,1]`` ring shadow edge proximity.
        sun_dirs:
            List of sun direction 3-tuples (optional, for forward-scatter hint).
        """
        if not self._enabled:
            return

        # Advance existing phenomena
        self._advance_phenomena(dt)

        # Server tick: evaluate conditions and spawn new phenomena
        self._tick_accum += dt
        if self._tick_accum >= self._tick_seconds:
            self._tick_accum -= self._tick_seconds
            self._tick_index += 1
            self._run_server_tick(
                sim_time=sim_time,
                storm_cells=storm_cells or [],
                dust_density_fn=dust_density_fn or (lambda la, lo: 0.0),
                wind_speed_fn=wind_speed_fn or (lambda la, lo: 0.0),
                dust_lift_potential_fn=dust_lift_potential_fn or (lambda la, lo: 0.0),
                ring_shadow_edge_fn=ring_shadow_edge_fn or (lambda la, lo: 0.0),
            )

    # ------------------------------------------------------------------
    # Query: active phenomena
    # ------------------------------------------------------------------

    def get_active_phenomena(self) -> list:
        """Return a snapshot of currently active phenomena."""
        return list(self._phenomena)

    # ------------------------------------------------------------------
    # Local coupling
    # ------------------------------------------------------------------

    def apply_local_coupling(
        self,
        player_lat: float,
        player_lon: float,
    ) -> LocalCouplingResult:
        """Compute additive local climate modifiers at *player_lat/lon*.

        Returns
        -------
        LocalCouplingResult
            wind_boost, dust_density_add, visibility_mul, insolation_mul.
        """
        result = LocalCouplingResult()
        for phen in self._phenomena:
            dist_rad = _angular_dist(player_lat, player_lon,
                                     phen.anchor_lat, phen.anchor_lon)
            dist_m = dist_rad * self._PLANET_R

            if phen.phen_type == PhenType.DUST_WALL:
                influence_m = self._dw_len_max * 3.0
                if dist_m < influence_m:
                    falloff = _smoothstep(influence_m, 0.0, dist_m)
                    strength = falloff * phen.intensity
                    result.wind_boost       += strength * 12.0
                    result.dust_density_add += strength * 0.5
                    result.visibility_mul   *= _lerp(1.0, 0.2, strength * 0.8)

            elif phen.phen_type == PhenType.RING_SHADOW_FRONT:
                influence_m = self._rsf_edge_width * 5.0
                if dist_m < influence_m:
                    falloff = _smoothstep(influence_m, 0.0, dist_m)
                    result.insolation_mul *= _lerp(1.0, 0.75, falloff * phen.intensity)
                    result.wind_boost     += falloff * phen.intensity * 4.0

            elif phen.phen_type == PhenType.MEGA_VORTEX:
                influence_m = phen.radius * self._PLANET_R * 2.0
                if dist_m < influence_m:
                    falloff = _smoothstep(influence_m, 0.0, dist_m)
                    result.wind_boost       += falloff * phen.intensity * 8.0
                    result.dust_density_add += falloff * phen.intensity * 0.3

            elif phen.phen_type == PhenType.DRY_LIGHTNING_CLUSTER:
                pass  # lightning doesn't affect local climate directly

        result.wind_boost       = _clamp(result.wind_boost,       0.0, 30.0)
        result.dust_density_add = _clamp(result.dust_density_add, 0.0, 1.0)
        result.visibility_mul   = _clamp(result.visibility_mul,   0.0, 1.0)
        result.insolation_mul   = _clamp(result.insolation_mul,   0.0, 1.0)
        return result

    # ------------------------------------------------------------------
    # Render parameters
    # ------------------------------------------------------------------

    def get_render_params(
        self,
        player_lat: float,
        player_lon: float,
        sim_time: float,
    ) -> list:
        """Return render descriptors for all active phenomena visible from player.

        Designed for the volumetric sky / atmosphere pass **before**
        pixelisation, so no per-pixel shimmer occurs.
        """
        params: list[RenderParam] = []
        for phen in self._phenomena:
            dist_rad = _angular_dist(player_lat, player_lon,
                                     phen.anchor_lat, phen.anchor_lon)
            # Bearing from player to phenomenon
            dlat = phen.anchor_lat - player_lat
            dlon = phen.anchor_lon - player_lon
            bearing = math.atan2(dlon, dlat)  # simplified flat-world bearing

            # Angular radius on horizon (phenomenon height / distance)
            if dist_rad < 1e-9:
                continue
            dist_m = dist_rad * self._PLANET_R
            ang_radius = math.atan2(phen.height, max(1.0, dist_m))

            # Base density falloff with distance
            max_vis_rad = phen.radius * 4.0
            if dist_rad > max_vis_rad:
                continue
            density = phen.intensity * _smoothstep(max_vis_rad, phen.radius * 0.5, dist_rad)

            # Forward scatter boost (for dust wall at low sun)
            fwd_scatter = 0.0
            if phen.phen_type == PhenType.DUST_WALL:
                fwd_scatter = density * 0.6

            # Lightning flash
            flash_val = 0.0
            if phen.phen_type == PhenType.DRY_LIGHTNING_CLUSTER:
                flash_val = self._sample_lightning_flash(phen, sim_time)

            params.append(RenderParam(
                phen_id=phen.phen_id,
                phen_type=phen.phen_type,
                rel_lat=dlat,
                rel_lon=dlon,
                angular_radius=ang_radius,
                horizon_angle=bearing,
                intensity=phen.intensity,
                base_density=density,
                lightning_flash=flash_val,
                edge_width_rad=phen.edge_width_rad,
                fwd_scatter=fwd_scatter,
            ))

        return params

    # ------------------------------------------------------------------
    # Audio triggers
    # ------------------------------------------------------------------

    def get_audio_triggers(
        self,
        player_lat: float,
        player_lon: float,
        sim_time: float,
    ) -> list:
        """Return audio events due this tick for the given player position."""
        triggers: list[AudioTrigger] = []

        for phen in self._phenomena:
            dist_rad = _angular_dist(player_lat, player_lon,
                                     phen.anchor_lat, phen.anchor_lon)
            dist_m = dist_rad * self._PLANET_R

            if phen.phen_type == PhenType.DUST_WALL:
                # Low-frequency rumble; louder when close
                max_dist_m = self._dw_len_max * 4.0
                if dist_m < max_dist_m:
                    t = 1.0 - dist_m / max_dist_m
                    gain = t * t * phen.intensity * self._rumble_gain
                    lp   = 1.0 - t * 0.5    # more low-pass when far
                    triggers.append(AudioTrigger(
                        event="dust_wall_rumble",
                        gain=_clamp(gain, 0.0, 1.0),
                        low_pass=_clamp(lp, 0.0, 1.0),
                        delay_sec=0.0,
                    ))

            elif phen.phen_type == PhenType.DRY_LIGHTNING_CLUSTER:
                # Check if any flash is happening this tick (flash visible = instant,
                # thunder delayed by distance / c_sound)
                flash_val = self._sample_lightning_flash(phen, sim_time)
                if flash_val > 0.1:
                    delay = dist_m / self._SOUND_SPEED
                    gain  = flash_val * phen.intensity * self._lightning_gain
                    # Strong low-pass when far (dust absorbs high freq)
                    max_thunder_m = 30_000.0
                    lp = _clamp(dist_m / max_thunder_m, 0.0, 1.0)
                    triggers.append(AudioTrigger(
                        event="lightning_crack",
                        gain=_clamp(gain, 0.0, 1.0),
                        low_pass=_clamp(lp, 0.0, 1.0),
                        delay_sec=delay,
                    ))

        return triggers

    # ------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------

    def get_debug_state(self) -> dict:
        """Developer console / log dump."""
        by_type: dict[str, int] = {}
        for phen in self._phenomena:
            key = phen.phen_type.name
            by_type[key] = by_type.get(key, 0) + 1
        return {
            "active_count": len(self._phenomena),
            "by_type": by_type,
            "tick_index": self._tick_index,
        }

    # ------------------------------------------------------------------
    # Internal: advance positions and cull expired phenomena
    # ------------------------------------------------------------------

    def _advance_phenomena(self, dt: float) -> None:
        R = self._PLANET_R
        to_remove = []
        for phen in self._phenomena:
            # Advect anchor by velocity
            cos_lat = max(0.01, math.cos(phen.anchor_lat))
            phen.anchor_lat += (phen.vel_v / R) * dt
            phen.anchor_lon += (phen.vel_u / (R * cos_lat)) * dt
            # Wrap longitude
            phen.anchor_lon = (
                (phen.anchor_lon + math.pi) % (2.0 * math.pi) - math.pi
            )
            phen.anchor_lat = _clamp(
                phen.anchor_lat, -self._HALF_PI + 0.01, self._HALF_PI - 0.01
            )
        # Remove expired
        self._phenomena = [
            p for p in self._phenomena if p.end_time > self._tick_index * self._tick_seconds
        ]

    # ------------------------------------------------------------------
    # Internal: server tick — evaluate conditions, spawn phenomena
    # ------------------------------------------------------------------

    def _run_server_tick(
        self,
        sim_time: float,
        storm_cells,
        dust_density_fn:       Callable,
        wind_speed_fn:         Callable,
        dust_lift_potential_fn: Callable,
        ring_shadow_edge_fn:   Callable,
    ) -> None:
        tick = self._tick_index

        # Count current phenomena by type
        dw_count  = sum(1 for p in self._phenomena if p.phen_type == PhenType.DUST_WALL)
        lg_count  = sum(1 for p in self._phenomena
                        if p.phen_type == PhenType.DRY_LIGHTNING_CLUSTER)
        rsf_count = sum(1 for p in self._phenomena
                        if p.phen_type == PhenType.RING_SHADOW_FRONT)
        vx_count  = sum(1 for p in self._phenomena if p.phen_type == PhenType.MEGA_VORTEX)

        # --- Try spawn DUST_WALL from high-dust-lift + storm cells ---
        if (dw_count < self._max_dust_walls
                and len(self._phenomena) < self._max_active):
            self._try_spawn_dust_wall(
                tick, sim_time, storm_cells,
                dust_lift_potential_fn, wind_speed_fn,
            )

        # --- Try spawn DRY_LIGHTNING inside existing dust walls ---
        if (self._lightning_enable
                and lg_count < self._max_lightning
                and len(self._phenomena) < self._max_active):
            self._try_spawn_lightning(tick, sim_time, dust_density_fn, wind_speed_fn)

        # --- Try spawn RING_SHADOW_FRONT ---
        if (self._rsf_enable
                and rsf_count < self._max_ring_fronts
                and len(self._phenomena) < self._max_active):
            self._try_spawn_ring_front(tick, sim_time, ring_shadow_edge_fn, dust_density_fn)

        # --- Try spawn MEGA_VORTEX ---
        if (self._vortex_enable
                and vx_count < self._max_vortices
                and len(self._phenomena) < self._max_active):
            self._try_spawn_vortex(tick, sim_time, wind_speed_fn, dust_lift_potential_fn)

    # ------------------------------------------------------------------
    # Internal: spawn helpers
    # ------------------------------------------------------------------

    def _try_spawn_dust_wall(
        self,
        tick: int,
        sim_time: float,
        storm_cells,
        dust_lift_fn: Callable,
        wind_fn:      Callable,
    ) -> None:
        """Consider spawning a DUST_WALL near a high-lift region or storm cell."""
        # Use existing storm cells as candidate anchor points
        candidates = []
        for sc in storm_cells:
            dlp = _clamp(dust_lift_fn(sc.center_lat, sc.center_lon), 0.0, 1.0)
            if dlp > 0.4 and sc.intensity > 0.5:
                candidates.append((sc.center_lat, sc.center_lon,
                                   sc.vel_u, sc.vel_v, dlp * sc.intensity))

        # Also probe a small set of deterministic grid points
        _PROBE_LATS = (-0.8, -0.4, 0.0, 0.4, 0.8)
        _PROBE_LONS = (-2.5, -1.5, -0.5, 0.5, 1.5, 2.5)
        for plat in _PROBE_LATS:
            for plon in _PROBE_LONS:
                dlp = _clamp(dust_lift_fn(plat, plon), 0.0, 1.0)
                wspd = wind_fn(plat, plon)
                if dlp > 0.55:
                    candidates.append((plat, plon, wspd * 0.5, 0.0, dlp))

        if not candidates:
            return

        # Gate: use deterministic hash so all clients agree
        gate_val = _det_hash_float(self._world_seed, tick, 100)
        if gate_val > self._dw_frequency:
            return

        # Pick best candidate (highest score)
        candidates.sort(key=lambda c: -c[4])
        lat, lon, vel_u, vel_v, score = candidates[0]

        # Jitter anchor slightly (deterministic)
        jlat = (_det_hash_float(self._world_seed, tick, 101) - 0.5) * 0.2
        jlon = (_det_hash_float(self._world_seed, tick, 102) - 0.5) * 0.3
        lat = _clamp(lat + jlat, -self._HALF_PI + 0.05, self._HALF_PI - 0.05)
        lon = ((lon + jlon + math.pi) % (2.0 * math.pi)) - math.pi

        # Shape parameters (deterministic)
        length_m = _lerp(
            self._dw_len_min, self._dw_len_max,
            _det_hash_float(self._world_seed, tick, 103),
        )
        thick_m = _lerp(
            self._dw_thick_min, self._dw_thick_max,
            _det_hash_float(self._world_seed, tick, 104),
        )
        intensity = _lerp(
            self._dw_intensity_min, self._dw_intensity_max,
            score,
        )
        lifetime = _lerp(120.0, 600.0, _det_hash_float(self._world_seed, tick, 105))

        radius_rad = length_m / self._PLANET_R  # angular half-length
        phen_seed = (self._world_seed ^ (tick * 1013) ^ 1) & 0xFFFF_FFFF

        self._phenomena.append(MacroPhenomenon(
            phen_id=self._next_phen_id,
            phen_type=PhenType.DUST_WALL,
            anchor_lat=lat,
            anchor_lon=lon,
            height=thick_m,
            radius=radius_rad,
            thickness=thick_m,
            vel_u=vel_u,
            vel_v=vel_v,
            intensity=intensity,
            start_time=sim_time,
            end_time=sim_time + lifetime,
            seed=phen_seed,
        ))
        self._next_phen_id += 1

    def _try_spawn_lightning(
        self,
        tick: int,
        sim_time: float,
        dust_fn:  Callable,
        wind_fn:  Callable,
    ) -> None:
        """Spawn a DRY_LIGHTNING_CLUSTER inside an existing dense dust wall."""
        # Only spawn inside an existing DUST_WALL
        dust_walls = [p for p in self._phenomena if p.phen_type == PhenType.DUST_WALL]
        if not dust_walls:
            return

        gate_val = _det_hash_float(self._world_seed, tick, 200)
        if gate_val > self._lightning_rarity:
            return

        # Pick the most intense dust wall
        wall = max(dust_walls, key=lambda p: p.intensity)

        # Check dust density (need dense dust)
        dust = dust_fn(wall.anchor_lat, wall.anchor_lon)
        if dust < 0.5:
            return

        # Wind turbulence check
        wspd = wind_fn(wall.anchor_lat, wall.anchor_lon)
        if wspd < 10.0:
            return

        # Build flash schedule deterministically
        n_flashes = int(_lerp(
            self._lightning_flash_min,
            self._lightning_flash_max,
            _det_hash_float(self._world_seed, tick, 201),
        ))
        window = _lerp(10.0, 60.0, _det_hash_float(self._world_seed, tick, 202))
        flash_times = tuple(
            sim_time + window * _det_hash_float(self._world_seed, tick, 210 + i)
            for i in range(n_flashes)
        )

        phen_seed = (self._world_seed ^ (tick * 1013) ^ 2) & 0xFFFF_FFFF
        lifetime = max(flash_times) - sim_time + 2.0

        self._phenomena.append(MacroPhenomenon(
            phen_id=self._next_phen_id,
            phen_type=PhenType.DRY_LIGHTNING_CLUSTER,
            anchor_lat=wall.anchor_lat,
            anchor_lon=wall.anchor_lon,
            height=wall.height * 0.5,
            radius=wall.radius,
            thickness=wall.thickness,
            vel_u=wall.vel_u,
            vel_v=wall.vel_v,
            intensity=wall.intensity,
            start_time=sim_time,
            end_time=sim_time + lifetime,
            seed=phen_seed,
            flash_times=flash_times,
        ))
        self._next_phen_id += 1

    def _try_spawn_ring_front(
        self,
        tick: int,
        sim_time: float,
        ring_edge_fn:   Callable,
        dust_fn:        Callable,
    ) -> None:
        """Spawn a RING_SHADOW_FRONT where ring-shadow edge meets dust/haze."""
        gate_val = _det_hash_float(self._world_seed, tick, 300)
        # Ring fronts are rare
        if gate_val > 0.25:
            return

        # Probe for high ring-edge proximity with visible dust
        _PROBE_LATS = (-0.6, -0.2, 0.2, 0.6)
        _PROBE_LONS = (-2.0, 0.0, 2.0)
        best_score = 0.0
        best_lat, best_lon = 0.0, 0.0
        for plat in _PROBE_LATS:
            for plon in _PROBE_LONS:
                edge  = ring_edge_fn(plat, plon)
                dust  = dust_fn(plat, plon)
                score = edge * (0.3 + dust * 0.7)
                if score > best_score:
                    best_score = score
                    best_lat, best_lon = plat, plon

        if best_score < 0.2:
            return

        edge_width_rad = self._rsf_edge_width / self._PLANET_R
        # The front moves with the ring shadow (slow angular velocity)
        # approximate: 1 degree per 60 s
        vel_u = math.radians(1.0) * self._PLANET_R / 60.0

        lifetime = _lerp(60.0, 300.0, _det_hash_float(self._world_seed, tick, 301))
        phen_seed = (self._world_seed ^ (tick * 1013) ^ 3) & 0xFFFF_FFFF

        self._phenomena.append(MacroPhenomenon(
            phen_id=self._next_phen_id,
            phen_type=PhenType.RING_SHADOW_FRONT,
            anchor_lat=best_lat,
            anchor_lon=best_lon,
            height=10_000.0,
            radius=edge_width_rad * 2.0,
            thickness=self._rsf_edge_width,
            vel_u=vel_u,
            vel_v=0.0,
            intensity=_clamp(best_score, 0.2, 1.0),
            start_time=sim_time,
            end_time=sim_time + lifetime,
            seed=phen_seed,
            front_normal_lon=1.0,
            front_normal_lat=0.0,
            edge_width_rad=edge_width_rad,
        ))
        self._next_phen_id += 1

    def _try_spawn_vortex(
        self,
        tick: int,
        sim_time: float,
        wind_fn:  Callable,
        lift_fn:  Callable,
    ) -> None:
        """Spawn a MEGA_VORTEX in a region combining high wind and dust lift."""
        gate_val = _det_hash_float(self._world_seed, tick, 400)
        if gate_val > 0.10:   # vortices are rare
            return

        # Pick anchor deterministically
        lat = (_det_hash_float(self._world_seed, tick, 401) - 0.5) * math.pi * 0.8
        lon = (_det_hash_float(self._world_seed, tick, 402) - 0.5) * 2.0 * math.pi

        wspd = wind_fn(lat, lon)
        dlp  = lift_fn(lat, lon)
        if wspd < 8.0 or dlp < 0.3:
            return

        radius_rad = _lerp(
            0.02, 0.06,
            _det_hash_float(self._world_seed, tick, 403),
        )
        intensity = _clamp(dlp * 0.7 + wspd / 40.0 * 0.3, 0.2, 0.9)
        lifetime  = _lerp(180.0, 600.0, _det_hash_float(self._world_seed, tick, 404))
        phen_seed = (self._world_seed ^ (tick * 1013) ^ 4) & 0xFFFF_FFFF

        self._phenomena.append(MacroPhenomenon(
            phen_id=self._next_phen_id,
            phen_type=PhenType.MEGA_VORTEX,
            anchor_lat=lat,
            anchor_lon=lon,
            height=3_000.0,
            radius=radius_rad,
            thickness=500.0,
            vel_u=wspd * 0.3,
            vel_v=0.0,
            intensity=intensity,
            start_time=sim_time,
            end_time=sim_time + lifetime,
            seed=phen_seed,
        ))
        self._next_phen_id += 1

    # ------------------------------------------------------------------
    # Internal: lightning flash sampling (analytical, no per-frame RNG)
    # ------------------------------------------------------------------

    def _sample_lightning_flash(
        self,
        phen: MacroPhenomenon,
        sim_time: float,
        flash_duration: float = 0.08,
    ) -> float:
        """Return flash intensity [0,1] for a DRY_LIGHTNING_CLUSTER at *sim_time*.

        The schedule was pre-computed in ``flash_times``; each flash has a
        hard ``flash_duration`` window so it is deterministic and tick-rate
        independent.
        """
        for t_flash in phen.flash_times:
            if abs(sim_time - t_flash) <= flash_duration:
                # Triangular flash shape (peak at t_flash)
                return 1.0 - abs(sim_time - t_flash) / flash_duration
        return 0.0
