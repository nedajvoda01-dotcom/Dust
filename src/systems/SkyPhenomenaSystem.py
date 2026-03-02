"""SkyPhenomenaSystem — Stage 31 atmospheric-optics renderer.

Generates and evaluates procedural sky phenomena:
  - 22° ice-crystal halo (one ring per sun)
  - Diffraction corona (dust / crystal near-sun glow)
  - Parhelia / sun-dogs (two mock suns at ±22° horizontally)
  - Ring-edge glints / arcs (when ring shadow geometry aligns)
  - Dual-sun bridge glow (soft overlap when the two suns are close)

Design goals
------------
* **No per-frame randomness** — all randomness gated through *timeBuckets* (integer-second
  slots), never per-frame RNG.
* **Temporal smoothing** — exponential lerp (tau configurable) so phenomena
  fade in/out smoothly even at low frame rates.
* **Multiplayer determinism** — gate hash depends only on
  ``worldSeed + timeBucket + regionId``, which is the same for all clients
  given server-authoritative climate & astro state.
* No UI overlays, no texture assets, no lens-flare screen-space noise.

Public API
----------
``update(dt, sim_time, dust_density, ice_crystal_proxy, visibility,
         sun1_dir, sun2_dir, eclipse_fraction1, eclipse_fraction2,
         ring_shadow_factor, camera_view_dir, camera_up)``
    Advance the system by *dt* seconds.

``compute_sky_color_add(view_dir) -> (r, g, b)``
    Return the additive RGB contribution of all phenomena for a given view
    direction (pre-pixel pass, world-space direction).

``get_debug_state() -> dict``
    Developer console / log dump of current scalar strengths.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

from src.core.Config import Config
from src.math.Vec3 import Vec3

# ---------------------------------------------------------------------------
# Module-level helpers (no per-frame RNG allowed here)
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


def _dot3(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross3(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _normalize3(v: tuple[float, float, float]) -> tuple[float, float, float]:
    mag = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if mag < 1e-12:
        return (0.0, 0.0, 1.0)
    inv = 1.0 / mag
    return (v[0] * inv, v[1] * inv, v[2] * inv)


def _add3(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scale3(
    a: tuple[float, float, float],
    s: float,
) -> tuple[float, float, float]:
    return (a[0] * s, a[1] * s, a[2] * s)


def _vec3_to_t(v: Vec3) -> tuple[float, float, float]:
    return (v.x, v.y, v.z)


# ---------------------------------------------------------------------------
# Deterministic gate (no per-frame RNG)
# ---------------------------------------------------------------------------

def _rare_gate(world_seed: int, time_bucket: int, region_id: int, threshold: float) -> bool:
    """Return True when hash(seed, bucket, region) / MAX > (1 - threshold).

    All inputs are integers → fully deterministic for all clients.
    """
    raw = hashlib.sha256(
        f"{world_seed}:{time_bucket}:{region_id}".encode()
    ).digest()
    # Use first 4 bytes as uint32
    val = (raw[0] << 24 | raw[1] << 16 | raw[2] << 8 | raw[3]) / 0xFFFF_FFFF
    return val < threshold


# ---------------------------------------------------------------------------
# PhenomenaConditions (computed each update, drives all renders)
# ---------------------------------------------------------------------------

@dataclass
class PhenomenaConditions:
    """Scalar driver values for all sky phenomena this frame."""
    h_dust: float = 0.0          # [0-1] dust scattering proxy
    h_ice: float = 0.0           # [0-1] ice crystal proxy
    sun1_alt: float = 0.0        # radians — sun1 elevation above horizon
    sun2_alt: float = 0.0        # radians — sun2 elevation above horizon
    low_sun_boost1: float = 0.0  # smoothstep-boosted for low sun1
    low_sun_boost2: float = 0.0  # smoothstep-boosted for low sun2
    ring_edge_boost: float = 0.0 # proximity to ring-shadow edge
    halo_strength1: float = 0.0  # combined halo driver for sun1
    halo_strength2: float = 0.0  # combined halo driver for sun2
    corona_strength: float = 0.0 # corona driver (both suns, averaged)
    parhelia_gate: bool = False   # discrete on/off this bucket
    ring_glint_gate: bool = False # discrete on/off this bucket


# ---------------------------------------------------------------------------
# Smoothed strengths (temporal, no shimmer)
# ---------------------------------------------------------------------------

@dataclass
class _SmoothedStrengths:
    halo1: float = 0.0
    halo2: float = 0.0
    corona: float = 0.0
    parhelia: float = 0.0
    ring_glint: float = 0.0
    bridge_glow: float = 0.0


# ---------------------------------------------------------------------------
# SkyPhenomenaSystem
# ---------------------------------------------------------------------------

class SkyPhenomenaSystem:
    """Stage 31 procedural atmospheric-optics system.

    Parameters
    ----------
    config:
        Global Config (reads ``skyphen.*`` namespace).
    world_seed:
        Stable integer seed for this world / server session.
    """

    # Canonical "up" direction in world space (matches AstroSystem spin axis)
    _UP: tuple[float, float, float] = (0.0, 1.0, 0.0)

    def __init__(
        self,
        config: Config | None = None,
        world_seed: int = 0,
    ) -> None:
        if config is None:
            config = Config()

        self._enabled: bool = bool(config.get("skyphen", "enable", default=True))
        self._halo_theta: float = math.radians(
            config.get("skyphen", "halo_theta_deg", default=22.0)
        )
        self._halo_sigma: float = math.radians(
            config.get("skyphen", "halo_sigma_deg", default=1.5)
        )
        self._halo_k: float = float(config.get("skyphen", "halo_strength_k", default=0.55))
        self._corona_k: float = float(config.get("skyphen", "corona_strength_k", default=0.40))
        self._parhelia_enable: bool = bool(
            config.get("skyphen", "parhelia_enable", default=True)
        )
        self._parhelia_rarity: float = _clamp(
            float(config.get("skyphen", "parhelia_rarity", default=0.25)), 0.0, 1.0
        )
        self._parhelia_tb_sec: float = max(
            1.0, float(config.get("skyphen", "parhelia_timebucket_sec", default=60.0))
        )
        self._ring_glint_k: float = float(
            config.get("skyphen", "ring_glint_strength_k", default=0.35)
        )
        self._ring_glint_rarity: float = _clamp(
            float(config.get("skyphen", "ring_glint_rarity", default=0.30)), 0.0, 1.0
        )
        self._tau: float = max(
            0.1, float(config.get("skyphen", "temporal_tau_sec", default=2.0))
        )
        self._render_scale: float = _clamp(
            float(config.get("skyphen", "render_scale", default=1.0)), 0.01, 1.0
        )

        self._world_seed: int = int(world_seed)

        # Mutable state
        self._cond: PhenomenaConditions = PhenomenaConditions()
        self._smooth: _SmoothedStrengths = _SmoothedStrengths()

        # Current sun directions (unit tuples)
        self._sun1_dir: tuple[float, float, float] = (0.0, 1.0, 0.0)
        self._sun2_dir: tuple[float, float, float] = (0.0, 1.0, 0.0)

        # Eclipse fractions (for attenuating / boosting corona)
        self._eclipse1: float = 0.0
        self._eclipse2: float = 0.0

    # ------------------------------------------------------------------
    # Public update
    # ------------------------------------------------------------------

    def update(
        self,
        dt: float,
        sim_time: float,
        *,
        dust_density: float = 0.0,
        ice_crystal_proxy: float = 0.0,
        visibility: float = 1.0,
        sun1_dir: Vec3 | None = None,
        sun2_dir: Vec3 | None = None,
        eclipse_fraction1: float = 0.0,
        eclipse_fraction2: float = 0.0,
        ring_shadow_factor: float = 0.0,
        camera_up: Vec3 | None = None,
    ) -> None:
        """Advance the system by *dt* seconds.

        Parameters
        ----------
        dt:
            Frame time in seconds.
        sim_time:
            Server-authoritative simulation time in seconds.
        dust_density:
            Normalised dust suspension [0-1].
        ice_crystal_proxy:
            Proxy for ice-crystal density [0-1].
        visibility:
            Atmospheric visibility [0-1] (1 = clear).
        sun1_dir / sun2_dir:
            Unit Vec3 from camera toward each sun.  If None, last value used.
        eclipse_fraction1 / eclipse_fraction2:
            Sun-sun occultation fraction for each sun [0-1].
        ring_shadow_factor:
            Combined ring shadow at camera position [0-1].
        camera_up:
            World-space up vector at camera location (defaults to +Y).
        """
        if not self._enabled:
            return

        # Store sun directions
        if sun1_dir is not None:
            self._sun1_dir = _normalize3(_vec3_to_t(sun1_dir))
        if sun2_dir is not None:
            self._sun2_dir = _normalize3(_vec3_to_t(sun2_dir))
        self._eclipse1 = _clamp(eclipse_fraction1, 0.0, 1.0)
        self._eclipse2 = _clamp(eclipse_fraction2, 0.0, 1.0)

        up = _normalize3(_vec3_to_t(camera_up)) if camera_up is not None else self._UP

        # --- Conditions ---
        cond = self._compute_conditions(
            dust_density=dust_density,
            ice_crystal_proxy=ice_crystal_proxy,
            visibility=visibility,
            ring_shadow_factor=ring_shadow_factor,
            sim_time=sim_time,
            up=up,
        )
        self._cond = cond

        # --- Temporal smoothing ---
        alpha = 1.0 - math.exp(-dt / self._tau)

        # Targets
        t_halo1 = cond.halo_strength1 * (1.0 - self._eclipse1)
        t_halo2 = cond.halo_strength2 * (1.0 - self._eclipse2)
        t_corona = cond.corona_strength
        t_parhelia = (
            cond.halo_strength1 * float(cond.parhelia_gate)
            if self._parhelia_enable else 0.0
        )
        t_ring_glint = (
            cond.ring_edge_boost * float(cond.ring_glint_gate) * float(visibility > 0.15)
        )
        # Bridge glow: soft glow between the two suns when they are close
        cos_sep = _dot3(self._sun1_dir, self._sun2_dir)
        ang_sep = math.acos(_clamp(cos_sep, -1.0, 1.0))
        bridge_raw = _smoothstep(math.radians(30.0), math.radians(5.0), ang_sep)
        t_bridge = bridge_raw * _clamp((dust_density + ice_crystal_proxy) * 0.5, 0.0, 1.0)

        s = self._smooth
        s.halo1 = _lerp(s.halo1, t_halo1, alpha)
        s.halo2 = _lerp(s.halo2, t_halo2, alpha)
        s.corona = _lerp(s.corona, t_corona, alpha)
        s.parhelia = _lerp(s.parhelia, t_parhelia, alpha)
        s.ring_glint = _lerp(s.ring_glint, t_ring_glint, alpha)
        s.bridge_glow = _lerp(s.bridge_glow, t_bridge, alpha)

    # ------------------------------------------------------------------
    # Sky-colour contribution (additive, pre-pixel pass)
    # ------------------------------------------------------------------

    def compute_sky_color_add(
        self,
        view_dir: Vec3,
    ) -> tuple[float, float, float]:
        """Return additive RGB contribution for *view_dir* (world-space unit vector).

        This should be called in the sky / atmosphere pass **before**
        pixelisation so that the angular profiles survive downsampling without
        per-pixel shimmer.
        """
        if not self._enabled:
            return (0.0, 0.0, 0.0)

        vd = _normalize3(_vec3_to_t(view_dir))
        r, g, b = 0.0, 0.0, 0.0

        s = self._smooth

        # --- Halo (sun1) ---
        if s.halo1 > 1e-5:
            h = self._halo_profile(vd, self._sun1_dir) * s.halo1 * self._halo_k
            # Slight warm spectral tint
            r += h * 1.00
            g += h * 0.97
            b += h * 0.92

        # --- Halo (sun2) ---
        if s.halo2 > 1e-5:
            h = self._halo_profile(vd, self._sun2_dir) * s.halo2 * self._halo_k
            # Slight cool spectral tint for the hotter star
            r += h * 0.90
            g += h * 0.95
            b += h * 1.00

        # --- Corona (both suns) ---
        if s.corona > 1e-5:
            c1 = self._corona_profile(vd, self._sun1_dir)
            c2 = self._corona_profile(vd, self._sun2_dir)
            c = (c1 + c2) * 0.5 * s.corona * self._corona_k
            r += c * 1.0
            g += c * 0.98
            b += c * 0.95

        # --- Parhelia ---
        if s.parhelia > 1e-5:
            p = self._parhelia_profile(vd, self._sun1_dir)
            w = p * s.parhelia * self._halo_k * 0.6
            r += w * 1.00
            g += w * 0.93
            b += w * 0.80

        # --- Ring glints ---
        if s.ring_glint > 1e-5:
            rg = self._ring_glint_profile(vd) * s.ring_glint * self._ring_glint_k
            r += rg * 0.85
            g += rg * 0.78
            b += rg * 0.60

        # --- Bridge glow ---
        if s.bridge_glow > 1e-5:
            bg = self._bridge_glow_profile(vd) * s.bridge_glow * 0.20
            r += bg
            g += bg * 0.92
            b += bg * 0.80

        # Clamp to avoid blowing out the scene
        return (
            _clamp(r, 0.0, 0.5),
            _clamp(g, 0.0, 0.5),
            _clamp(b, 0.0, 0.5),
        )

    # ------------------------------------------------------------------
    # Developer debug state
    # ------------------------------------------------------------------

    def get_debug_state(self) -> dict:
        """Return current strength values for dev console / log."""
        s = self._smooth
        c = self._cond
        return {
            "h_dust": c.h_dust,
            "h_ice": c.h_ice,
            "sun1_alt_deg": math.degrees(c.sun1_alt),
            "sun2_alt_deg": math.degrees(c.sun2_alt),
            "ring_edge_boost": c.ring_edge_boost,
            "parhelia_gate": c.parhelia_gate,
            "ring_glint_gate": c.ring_glint_gate,
            "smooth_halo1": s.halo1,
            "smooth_halo2": s.halo2,
            "smooth_corona": s.corona,
            "smooth_parhelia": s.parhelia,
            "smooth_ring_glint": s.ring_glint,
            "smooth_bridge_glow": s.bridge_glow,
        }

    # ------------------------------------------------------------------
    # Internal: condition computation
    # ------------------------------------------------------------------

    def _compute_conditions(
        self,
        dust_density: float,
        ice_crystal_proxy: float,
        visibility: float,
        ring_shadow_factor: float,
        sim_time: float,
        up: tuple[float, float, float],
    ) -> PhenomenaConditions:
        cond = PhenomenaConditions()

        # Saturated environment inputs
        cond.h_dust = _clamp(dust_density, 0.0, 1.0)
        cond.h_ice = _clamp(ice_crystal_proxy, 0.0, 1.0)

        # Sun altitudes
        cond.sun1_alt = math.asin(_clamp(_dot3(self._sun1_dir, up), -1.0, 1.0))
        cond.sun2_alt = math.asin(_clamp(_dot3(self._sun2_dir, up), -1.0, 1.0))

        # Low-sun boost: parhelia and strong halos amplified for low sun
        # (strongest below ~20°, fades above ~45°)
        cond.low_sun_boost1 = _smoothstep(math.radians(45.0), math.radians(5.0), cond.sun1_alt)
        cond.low_sun_boost2 = _smoothstep(math.radians(45.0), math.radians(5.0), cond.sun2_alt)

        # Ring-edge boost: ring glints strongest near shadow boundary
        # Proxy: ring_shadow_factor is highest in-shadow; edge ≈ 0.3–0.7 range
        ring_mid = abs(ring_shadow_factor - 0.5) * 2.0  # 0=edge, 1=centre/clear
        cond.ring_edge_boost = (1.0 - ring_mid) * _clamp(visibility, 0.0, 1.0)

        # Halo: weighted sum of ice + dust (different profiles)
        # Ice crystals dominate classical 22° halo; dust produces softer corona
        cond.halo_strength1 = _clamp(
            0.7 * cond.h_ice + 0.3 * cond.h_dust, 0.0, 1.0
        )
        cond.halo_strength2 = cond.halo_strength1  # both suns same medium

        # Corona: mostly thin dust at intermediate density
        # Very dense dust washes it out (whiteout), very clear = no corona
        corona_raw = cond.h_dust * (1.0 - cond.h_dust * 0.8) + cond.h_ice * 0.3
        cond.corona_strength = _clamp(corona_raw, 0.0, 1.0)

        # Parhelia gate — decisions made per timeBucket (no per-frame random)
        tb_par = int(sim_time / self._parhelia_tb_sec)
        cond.parhelia_gate = (
            _rare_gate(self._world_seed, tb_par, 1, self._parhelia_rarity)
            and cond.h_ice > 0.1
            and cond.low_sun_boost1 > 0.05
        )

        # Ring glint gate — slightly shorter bucket (share parhelia bucket for simplicity)
        tb_rg = int(sim_time / self._parhelia_tb_sec)
        cond.ring_glint_gate = (
            _rare_gate(self._world_seed, tb_rg, 2, self._ring_glint_rarity)
            and cond.ring_edge_boost > 0.05
        )

        return cond

    # ------------------------------------------------------------------
    # Internal: radial profile functions
    # ------------------------------------------------------------------

    @staticmethod
    def _angle_to_sun(
        view_dir: tuple[float, float, float],
        sun_dir: tuple[float, float, float],
    ) -> float:
        """Angular separation in radians between view_dir and sun_dir."""
        cos_t = _clamp(_dot3(view_dir, sun_dir), -1.0, 1.0)
        return math.acos(cos_t)

    def _halo_profile(
        self,
        view_dir: tuple[float, float, float],
        sun_dir: tuple[float, float, float],
    ) -> float:
        """Gaussian ring profile peaking at theta0 (22°)."""
        theta = self._angle_to_sun(view_dir, sun_dir)
        d = theta - self._halo_theta
        sigma = self._halo_sigma
        return math.exp(-(d * d) / (2.0 * sigma * sigma))

    @staticmethod
    def _corona_profile(
        view_dir: tuple[float, float, float],
        sun_dir: tuple[float, float, float],
    ) -> float:
        """Soft diffraction corona: 1/r^2 falloff within ~5° of sun."""
        theta = math.acos(_clamp(_dot3(view_dir, sun_dir), -1.0, 1.0))
        # Two soft rings for corona (innermost intense, second faint)
        sigma1 = math.radians(1.0)
        sigma2 = math.radians(3.0)
        ring1 = math.exp(-(theta * theta) / (2.0 * sigma1 * sigma1))
        ring2 = 0.35 * math.exp(-((theta - math.radians(2.5)) ** 2) / (2.0 * sigma2 * sigma2))
        return _clamp(ring1 + ring2, 0.0, 1.0)

    def _parhelia_profile(
        self,
        view_dir: tuple[float, float, float],
        sun_dir: tuple[float, float, float],
    ) -> float:
        """Two mock-sun blobs at ±22° horizontally from the sun."""
        up = self._UP
        # Horizontal right basis perpendicular to up
        fwd = _normalize3(sun_dir)
        right_raw = _cross3(fwd, up)
        r_len = math.sqrt(right_raw[0]**2 + right_raw[1]**2 + right_raw[2]**2)
        if r_len < 1e-9:
            return 0.0
        right = (right_raw[0] / r_len, right_raw[1] / r_len, right_raw[2] / r_len)

        # Rotate sun_dir ±22° around the up axis (approximate: shift along right)
        # Use Rodrigues to rotate fwd by ±halo_theta around up
        cos_a = math.cos(self._halo_theta)
        sin_a = math.sin(self._halo_theta)

        def _rot_around_up(v: tuple, sign: float) -> tuple[float, float, float]:
            sa = sign * sin_a
            # Rodrigues: v*cos + (up×v)*sin + up*(up·v)*(1-cos)
            up_dot_v = _dot3(up, v)
            cross_uv = _cross3(up, v)
            return (
                v[0] * cos_a + cross_uv[0] * sa + up[0] * up_dot_v * (1.0 - cos_a),
                v[1] * cos_a + cross_uv[1] * sa + up[1] * up_dot_v * (1.0 - cos_a),
                v[2] * cos_a + cross_uv[2] * sa + up[2] * up_dot_v * (1.0 - cos_a),
            )

        par_dir_pos = _normalize3(_rot_around_up(fwd, +1.0))
        par_dir_neg = _normalize3(_rot_around_up(fwd, -1.0))

        sigma = math.radians(2.5)  # parhelia blob width
        val = 0.0
        for pd in (par_dir_pos, par_dir_neg):
            t = math.acos(_clamp(_dot3(view_dir, pd), -1.0, 1.0))
            val += math.exp(-(t * t) / (2.0 * sigma * sigma))
        return _clamp(val, 0.0, 1.0)

    def _ring_glint_profile(
        self,
        view_dir: tuple[float, float, float],
    ) -> float:
        """Soft arc along the ring plane as seen from the camera."""
        # The ring plane is approximated as containing the sun directions.
        # We compute how close view_dir is to the great circle connecting
        # sun1 and sun2 (a proxy for the ring plane arc).
        sun1 = self._sun1_dir
        sun2 = self._sun2_dir

        # Normal to the plane spanned by sun1 and sun2
        plane_n_raw = _cross3(sun1, sun2)
        n_len = math.sqrt(
            plane_n_raw[0] ** 2 + plane_n_raw[1] ** 2 + plane_n_raw[2] ** 2
        )
        if n_len < 1e-9:
            return 0.0
        plane_n = (
            plane_n_raw[0] / n_len,
            plane_n_raw[1] / n_len,
            plane_n_raw[2] / n_len,
        )

        # Distance from view_dir to this great circle = |dot(view_dir, plane_n)|
        dist_to_arc = abs(_dot3(view_dir, plane_n))
        sigma_arc = math.radians(3.0)
        return math.exp(-(dist_to_arc * dist_to_arc) / (2.0 * sigma_arc * sigma_arc))

    def _bridge_glow_profile(
        self,
        view_dir: tuple[float, float, float],
    ) -> float:
        """Soft glow along the great-circle arc between the two suns."""
        sun1 = self._sun1_dir
        sun2 = self._sun2_dir

        # Midpoint direction
        mid_raw = (
            sun1[0] + sun2[0],
            sun1[1] + sun2[1],
            sun1[2] + sun2[2],
        )
        mid = _normalize3(mid_raw)

        # View angle from midpoint
        theta = math.acos(_clamp(_dot3(view_dir, mid), -1.0, 1.0))
        # Angular half-separation
        cos_sep = _clamp(_dot3(sun1, sun2), -1.0, 1.0)
        half_sep = math.acos(cos_sep) * 0.5

        # Profile: Gaussian centred on midpoint, width = half_sep
        sigma = max(math.radians(5.0), half_sep)
        return math.exp(-(theta * theta) / (2.0 * sigma * sigma))
