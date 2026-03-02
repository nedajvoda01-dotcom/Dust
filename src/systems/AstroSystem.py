"""AstroSystem — Stage 6 / Stage 29 astronomical simulation.

Provides:
- Binary sun circular orbit (barycenter model)
- Planet spin / day-night cycle
- Eclipse detection and eclipseFactor (sun-sun occultation)
- Moon eclipse factor (moon blocking each sun)
- Ring shadow computation (RingShadowFactor) with segment noise
- Moon orbit in ring plane
- InsolationSample API for climate / renderer integration
- Spectral mix (color temperature blend) for renderer
- Astro state hash / keyframe for multiplayer synchronisation

All motion is deterministic for a given seed and config.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field

from src.core.Config import Config
from src.math.Vec3 import Vec3
from src.systems.AstroSystemStub import IAstroSystem

_TWO_PI = 2.0 * math.pi
_BARY_DIST_FACTOR = 3.0   # barycenter distance = orbit_sep * _BARY_DIST_FACTOR

# Blackbody helper constants (approximation for spectral tint)
_BB_TABLE: list[tuple[float, tuple[float, float, float]]] = [
    (2000.0, (1.00, 0.34, 0.02)),
    (3000.0, (1.00, 0.57, 0.22)),
    (4000.0, (1.00, 0.75, 0.49)),
    (5000.0, (1.00, 0.88, 0.72)),
    (5800.0, (1.00, 0.95, 0.88)),
    (6500.0, (0.97, 0.95, 1.00)),
    (8000.0, (0.85, 0.90, 1.00)),
    (10000.0, (0.75, 0.85, 1.00)),
]


def _blackbody_rgb(temp_k: float) -> tuple[float, float, float]:
    """Return an approximate perceptual RGB tint for a blackbody at *temp_k* K."""
    if temp_k <= _BB_TABLE[0][0]:
        return _BB_TABLE[0][1]
    if temp_k >= _BB_TABLE[-1][0]:
        return _BB_TABLE[-1][1]
    for i in range(len(_BB_TABLE) - 1):
        t0, c0 = _BB_TABLE[i]
        t1, c1 = _BB_TABLE[i + 1]
        if t0 <= temp_k <= t1:
            frac = (temp_k - t0) / (t1 - t0)
            return (
                c0[0] + frac * (c1[0] - c0[0]),
                c0[1] + frac * (c1[1] - c0[1]),
                c0[2] + frac * (c1[2] - c0[2]),
            )
    return (1.0, 1.0, 1.0)


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


def _rotate_around_axis(v: Vec3, axis: Vec3, angle: float) -> Vec3:
    """Rodrigues' rotation formula: rotate v around (unit) axis by angle (rad)."""
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    d = axis.dot(v)
    return v * cos_a + axis.cross(v) * sin_a + axis * (d * (1.0 - cos_a))


def _perpendicular_basis(n: Vec3) -> tuple[Vec3, Vec3]:
    """Return two unit vectors (e1, e2) that form an orthonormal basis with n.
    e1 and e2 lie in the plane perpendicular to n."""
    n = n.normalized()
    if abs(n.y) < 0.9:
        ref = Vec3(0.0, 1.0, 0.0)
    else:
        ref = Vec3(1.0, 0.0, 0.0)
    e1 = ref.cross(n).normalized()
    e2 = n.cross(e1).normalized()
    return e1, e2


# ---------------------------------------------------------------------------
# InsolationSample
# ---------------------------------------------------------------------------

@dataclass
class InsolationSample:
    """Insolation data at a surface point.  Consumed by climate / renderer."""
    direct1: float = 0.0        # NdotL contribution from primary sun
    direct2: float = 0.0        # NdotL contribution from secondary sun
    total_direct: float = 0.0   # combined (ring shadow + eclipse applied)
    ring_shadow: float = 0.0    # 0=clear  1=full ring shadow
    eclipse_factor: float = 0.0 # 0=no eclipse  1=maximum overlap (sun-sun occultation)
    moon_eclipse: float = 0.0   # 0=no eclipse  1=full moon eclipse (moon blocks sun1)
    spectral_r: float = 1.0     # spectral mix RGB (blended blackbody tint)
    spectral_g: float = 1.0
    spectral_b: float = 1.0
    dir1: Vec3 = field(default_factory=Vec3.zero)
    dir2: Vec3 = field(default_factory=Vec3.zero)


# ---------------------------------------------------------------------------
# AstroSystem
# ---------------------------------------------------------------------------

class AstroSystem(IAstroSystem):
    """Full astronomical system for Dust Stage 6."""

    def __init__(self, config: Config | None = None, seed: int = 42) -> None:
        if config is None:
            config = Config()

        planet_r: float = config.get("planet", "radius_units", default=1000.0)

        # --- time ---
        self._day_len_s: float = config.get("day", "length_minutes", default=90) * 60.0
        self._binary_period_s: float = config.get("binary", "period_minutes", default=18) * 60.0
        self._time_scale: float = config.get("astro", "time_scale", default=1.0)

        # --- binary orbit ---
        sep_mul: float = config.get("astro", "orbit_separation_mul", default=5.0)
        self._orbit_sep: float = sep_mul * planet_r  # a — separation between the two suns
        mass_ratio: float = config.get("binary", "mass_ratio", default=0.35)  # M2/M1
        total = 1.0 + mass_ratio
        self._r1: float = self._orbit_sep * (mass_ratio / total)   # sun1 orbit radius from barycentre
        self._r2: float = self._orbit_sep * (1.0 / total)          # sun2 orbit radius from barycentre

        # --- intensities ---
        self._sun1_intensity: float = config.get("sun1", "intensity", default=1.0)
        self._sun2_intensity: float = config.get("sun2", "intensity", default=0.35)

        # --- color temperatures (K) for spectral mix ---
        self._sun1_temp_k: float = config.get("astro", "sun1", "temp_k", default=5000.0)
        self._sun2_temp_k: float = config.get("astro", "sun2", "temp_k", default=8500.0)

        # --- angular radii (for eclipse disc overlap) ---
        self._sun1_ang_r: float = math.radians(
            config.get("sun1", "angular_radius_deg", default=1.0))
        self._sun2_ang_r: float = math.radians(
            config.get("sun2", "angular_radius_deg", default=0.7))

        # --- sun-sun occultation softness ---
        self._occult_softness: float = _clamp(
            config.get("astro", "occult", "softness", default=0.5), 0.0, 1.0)

        # --- ring geometry ---
        ring_tilt_rad: float = math.radians(config.get("ring", "tilt_deg", default=14.0))
        self._ring_normal: Vec3 = Vec3(0.0, math.cos(ring_tilt_rad), math.sin(ring_tilt_rad)).normalized()
        self._ring_inner: float = config.get("ring", "inner_radius_mul", default=1.4) * planet_r
        self._ring_outer: float = config.get("ring", "outer_radius_mul", default=2.1) * planet_r
        self._ring_eps: float = (self._ring_outer - self._ring_inner) * 0.05
        self._ring_optical_depth: float = _clamp(
            config.get("astro", "ring", "optical_depth", default=0.8), 0.0, 1.0)
        self._ring_segment_strength: float = _clamp(
            config.get("astro", "ring", "segment_strength", default=0.3), 0.0, 1.0)

        # --- moon ---
        self._moon_orbit_r: float = config.get("moon", "orbit_radius_mul", default=3.5) * planet_r
        self._moon_period_s: float = config.get("moon", "period_minutes", default=45) * 60.0
        moon_radius_mul: float = config.get("moon", "radius_mul", default=0.08)
        self._moon_radius: float = moon_radius_mul * planet_r
        # Moon angular radius as seen from planet centre (approximate)
        self._moon_ang_r: float = math.atan2(self._moon_radius, self._moon_orbit_r)

        # --- fixed axes ---
        self._spin_axis: Vec3 = Vec3(0.0, 1.0, 0.0)
        # Binary orbit: barycenter placed at large distance along +Z.
        # The orbit plane contains the planet→barycenter axis (edge-on geometry)
        # so that the suns can eclipse each other as seen from the planet.
        bary_distance = self._orbit_sep * _BARY_DIST_FACTOR
        self._bary_pos: Vec3 = Vec3(0.0, 0.0, bary_distance)
        self._oe1: Vec3 = Vec3(0.0, 0.0, 1.0)   # toward barycenter
        self._oe2: Vec3 = Vec3(1.0, 0.0, 0.0)   # perpendicular (X)
        # Ring / moon orbit plane
        self._re1, self._re2 = _perpendicular_basis(self._ring_normal)

        # Planet centre fixed at world origin
        self._planet_center: Vec3 = Vec3.zero()

        # --- mutable state ---
        self._t: float = 0.0
        self._spin_angle: float = 0.0
        self._sun1_world_pos: Vec3 = Vec3.zero()
        self._sun2_world_pos: Vec3 = Vec3.zero()
        self._moon_world_pos: Vec3 = Vec3.zero()
        self._sun1_dir: Vec3 = Vec3.zero()   # unit vec from planet centre → sun1
        self._sun2_dir: Vec3 = Vec3.zero()
        self._moon_dir: Vec3 = Vec3.zero()   # unit vec from planet centre → moon

        self._update_state(0.0)

    # ------------------------------------------------------------------
    # IAstroSystem interface
    # ------------------------------------------------------------------

    def update(self, game_time: float) -> None:  # noqa: D102
        """Advance the simulation to game_time (seconds, unscaled)."""
        self._update_state(game_time * self._time_scale)

    def get_sun_directions(self) -> tuple[Vec3, Vec3]:  # noqa: D102
        return self._sun1_dir, self._sun2_dir

    def get_ring_shadow_factor(self, pos: Vec3) -> float:  # noqa: D102
        """Combined ring shadow at pos using the primary (sun1) direction."""
        return self._ring_shadow_for_sun(pos, self._sun1_dir)

    # ------------------------------------------------------------------
    # Extended public API
    # ------------------------------------------------------------------

    @property
    def sun1_world_pos(self) -> Vec3:
        return self._sun1_world_pos

    @property
    def sun2_world_pos(self) -> Vec3:
        return self._sun2_world_pos

    @property
    def moon_world_pos(self) -> Vec3:
        return self._moon_world_pos

    @property
    def spin_angle(self) -> float:
        """Current planet spin angle (radians, monotonically increasing)."""
        return self._spin_angle

    @property
    def planet_spin_axis(self) -> Vec3:
        return self._spin_axis

    @property
    def ring_normal(self) -> Vec3:
        return self._ring_normal

    def get_eclipse_factor(self) -> float:
        """Sun-sun occultation factor (0=no overlap, 1=maximum overlap)."""
        return self._occultation_factor(self._sun1_dir, self._sun2_dir)

    def get_moon_eclipse_factor(self) -> float:
        """Moon eclipse factor for sun1 (0=no eclipse, 1=full moon eclipse)."""
        return self._moon_eclipse_factor(self._sun1_dir)

    def get_ring_shadow_for_sun(self, pos: Vec3, sun_dir: Vec3) -> float:
        """Ring shadow at pos for a specific sun direction (0=clear, 1=full shadow)."""
        return self._ring_shadow_for_sun(pos, sun_dir)

    def get_spectral_mix(self) -> tuple[float, float, float]:
        """Return the current spectral RGB tint (blend of both sun blackbodies).

        The mix weight is proportional to each sun's energy contribution.
        """
        e1 = self._sun1_intensity
        e2 = self._sun2_intensity
        total = e1 + e2
        if total < 1e-9:
            return (1.0, 1.0, 1.0)
        w1 = e1 / total
        w2 = e2 / total
        c1 = _blackbody_rgb(self._sun1_temp_k)
        c2 = _blackbody_rgb(self._sun2_temp_k)
        return (
            c1[0] * w1 + c2[0] * w2,
            c1[1] * w1 + c2[1] * w2,
            c1[2] * w1 + c2[2] * w2,
        )

    def get_astro_state_hash(self) -> str:
        """Return a short hex hash of the current astro state for multiplayer sync."""
        d1, d2 = self._sun1_dir, self._sun2_dir
        m = self._moon_dir
        payload = f"{d1.x:.6f},{d1.y:.6f},{d1.z:.6f}|{d2.x:.6f},{d2.y:.6f},{d2.z:.6f}|{m.x:.6f},{m.y:.6f},{m.z:.6f}|{self._spin_angle:.6f}"
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    def get_astro_keyframe(self) -> dict:
        """Return a serialisable keyframe dict for server→client sync.

        Clients compare the received keyframe against their locally computed
        state (via :meth:`get_astro_state_hash`) and correct any drift.
        """
        d1, d2 = self._sun1_dir, self._sun2_dir
        m = self._moon_dir
        rn = self._ring_normal
        return {
            "type":       "ASTRO_KEYFRAME",
            "simTime":    self._t,
            "sun1Dir":    [d1.x, d1.y, d1.z],
            "sun2Dir":    [d2.x, d2.y, d2.z],
            "moonDir":    [m.x, m.y, m.z],
            "ringNormal": [rn.x, rn.y, rn.z],
            "spinAngle":  self._spin_angle,
            "stateHash":  self.get_astro_state_hash(),
        }

    def sample_insolation(self, world_pos: Vec3, world_normal: Vec3) -> InsolationSample:
        """Compute InsolationSample at a surface point.

        Parameters
        ----------
        world_pos:
            Position in world space (accounting for planet spin).
        world_normal:
            Outward surface normal in world space (accounting for planet spin).
        """
        dir1 = self._sun1_dir
        dir2 = self._sun2_dir

        ndotl1 = max(0.0, world_normal.dot(dir1))
        ndotl2 = max(0.0, world_normal.dot(dir2))

        rs1 = self._ring_shadow_for_sun(world_pos, dir1)
        rs2 = self._ring_shadow_for_sun(world_pos, dir2)
        ring_shadow = max(rs1, rs2)  # dominant ring shadow for the sample

        # Sun-sun occultation (softened)
        ef = self._occultation_factor(dir1, dir2)
        # Moon eclipse (moon blocks sun1)
        me = self._moon_eclipse_factor(dir1)

        d1 = ndotl1 * self._sun1_intensity * (1.0 - rs1) * (1.0 - me)
        d2 = ndotl2 * self._sun2_intensity * (1.0 - rs2) * (1.0 - ef)

        # Spectral mix weighted by actual energy contributions
        e1 = d1 + 1e-12
        e2 = d2 + 1e-12
        w1 = e1 / (e1 + e2)
        w2 = e2 / (e1 + e2)
        c1 = _blackbody_rgb(self._sun1_temp_k)
        c2 = _blackbody_rgb(self._sun2_temp_k)
        sr = c1[0] * w1 + c2[0] * w2
        sg = c1[1] * w1 + c2[1] * w2
        sb = c1[2] * w1 + c2[2] * w2

        return InsolationSample(
            direct1=d1,
            direct2=d2,
            total_direct=d1 + d2,
            ring_shadow=ring_shadow,
            eclipse_factor=ef,
            moon_eclipse=me,
            spectral_r=sr,
            spectral_g=sg,
            spectral_b=sb,
            dir1=dir1,
            dir2=dir2,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_state(self, t: float) -> None:
        self._t = t

        # Planet spin angle
        self._spin_angle = _TWO_PI * (t / self._day_len_s)

        # Binary orbit angle
        binary_angle = _TWO_PI * (t / self._binary_period_s)
        cos_b = math.cos(binary_angle)
        sin_b = math.sin(binary_angle)

        # Suns orbit the barycenter.  The orbit plane contains the
        # planet→barycenter direction (oe1 = +Z) so the two suns align along
        # the line of sight twice per period, producing eclipses.
        sun1_offset = self._oe1 * (cos_b * self._r1) + self._oe2 * (sin_b * self._r1)
        sun2_offset = self._oe1 * (-cos_b * self._r2) + self._oe2 * (-sin_b * self._r2)
        self._sun1_world_pos = self._bary_pos + sun1_offset
        self._sun2_world_pos = self._bary_pos + sun2_offset

        # Sun directions from planet centre (unit vectors toward the suns)
        self._sun1_dir = (self._sun1_world_pos - self._planet_center).normalized()
        self._sun2_dir = (self._sun2_world_pos - self._planet_center).normalized()

        # Moon orbit in ring plane
        moon_angle = _TWO_PI * (t / self._moon_period_s)
        cos_m = math.cos(moon_angle)
        sin_m = math.sin(moon_angle)
        self._moon_world_pos = (
            self._re1 * (cos_m * self._moon_orbit_r)
            + self._re2 * (sin_m * self._moon_orbit_r)
        )
        moon_dist = self._moon_world_pos.length()
        if moon_dist > 1e-9:
            self._moon_dir = self._moon_world_pos * (1.0 / moon_dist)
        else:
            self._moon_dir = Vec3(0.0, 0.0, 1.0)

    def _occultation_factor(self, dir1: Vec3, dir2: Vec3) -> float:
        """Sun-sun occultation overlap factor (0=no overlap, 1=maximum overlap).

        Applies ``_occult_softness`` to soften the transition artistically.
        """
        cos_theta = _clamp(dir1.dot(dir2), -1.0, 1.0)
        theta = math.acos(cos_theta)
        sum_r = self._sun1_ang_r + self._sun2_ang_r
        if sum_r < 1e-12:
            return 0.0
        raw = _clamp((sum_r - theta) / sum_r, 0.0, 1.0)
        # Apply softness: blend between hard step and smoothstep
        soft = _smoothstep(0.0, 1.0, raw)
        return raw * (1.0 - self._occult_softness) + soft * self._occult_softness

    def _moon_eclipse_factor(self, sun_dir: Vec3) -> float:
        """Moon eclipse factor: fraction of sun1 disc blocked by the moon.

        Uses a circle-circle overlap approximation keyed on angular radii.
        Returns 0 when the moon is not in front of the sun and 1 at maximum
        overlap.
        """
        # Angular separation between moon and sun as seen from planet centre
        cos_theta = _clamp(self._moon_dir.dot(sun_dir), -1.0, 1.0)
        # Only eclipse when moon is between planet and sun (same hemisphere)
        moon_dist = self._moon_world_pos.length()
        sun_dist = self._sun1_world_pos.length()
        if moon_dist >= sun_dist:
            return 0.0  # moon is behind the sun
        theta = math.acos(cos_theta)
        sum_r = self._moon_ang_r + self._sun1_ang_r
        if sum_r < 1e-12:
            return 0.0
        return _clamp((sum_r - theta) / sum_r, 0.0, 1.0)

    def _eclipse_factor(self, dir1: Vec3, dir2: Vec3) -> float:
        """Backward-compatible alias for :meth:`_occultation_factor`."""
        return self._occultation_factor(dir1, dir2)

    def _ring_shadow_for_sun(self, world_pos: Vec3, sun_dir: Vec3) -> float:
        """Ring shadow at world_pos for the given sun direction.

        Returns 0.0 (clear) when the ray to the sun does not pass through the
        ring, 1.0 (fully shadowed) when it does.  Soft edges via smoothstep.

        The result is scaled by ``_ring_optical_depth`` and modulated by a
        stable procedural segment mask (``_ring_segment_strength``).
        """
        denom = sun_dir.dot(self._ring_normal)
        if abs(denom) < 1e-9:
            return 0.0  # ray parallel to ring plane
        t_hit = (self._planet_center - world_pos).dot(self._ring_normal) / denom
        if t_hit <= 0.0:
            return 0.0  # intersection behind the point (away from sun)
        p_hit = world_pos + sun_dir * t_hit
        r = (p_hit - self._planet_center).length()
        base_shadow = (
            _smoothstep(self._ring_inner, self._ring_inner + self._ring_eps, r)
            * (1.0 - _smoothstep(self._ring_outer - self._ring_eps, self._ring_outer, r))
        )
        if base_shadow < 1e-6:
            return 0.0
        # Stable segment mask: deterministic noise on ring coordinates
        angle = math.atan2(p_hit.z, p_hit.x) if (abs(p_hit.x) + abs(p_hit.z)) > 1e-9 else 0.0
        radial_frac = (r - self._ring_inner) / max(1e-9, self._ring_outer - self._ring_inner)
        # Coarse sinusoidal bands — no per-frame RNG, so no flicker
        band = 0.5 + 0.5 * math.sin(angle * 7.0 + radial_frac * 19.0)
        segment_mask = 1.0 - self._ring_segment_strength * (1.0 - band)
        return base_shadow * self._ring_optical_depth * segment_mask
