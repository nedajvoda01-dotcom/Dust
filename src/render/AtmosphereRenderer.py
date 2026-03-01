"""AtmosphereRenderer — Stage 11 atmospheric rendering.

Provides CPU-side atmosphere compositing for Dust:
  - Sky colour via simplified Rayleigh + Mie scatter (two-sun support)
  - Aerial perspective / haze (distance-based fog, height-dependent density)
  - Volumetric dust (ray-march, 8–24 steps, Mie phase per sun)
  - Sun disc + corona (procedural, no textures)
  - Cinematic colour-grade pass (filmic curve, warm/cool tint, far desaturation)
  - Debug render modes (haze-only, vol-dust-only, transmittance, sky-only)

All computations are pure Python / floating-point maths — no texture assets,
no OpenGL calls.  The output colours are intended for consumption by the
render pipeline (draw to framebuffer, pixel stage, etc.).

Debug mode constants (pass as *debug_mode* to :meth:`composite`):
  ``AtmosphereRenderer.DEBUG_HAZE_ONLY``
  ``AtmosphereRenderer.DEBUG_VOL_DUST_ONLY``
  ``AtmosphereRenderer.DEBUG_TRANSMITTANCE``
  ``AtmosphereRenderer.DEBUG_SKY_ONLY``
"""
from __future__ import annotations

import math

from src.core.Config import Config
from src.math.Vec3 import Vec3

# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------

def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * _clamp(t, 0.0, 1.0)


def _lerp3(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    t: float,
) -> tuple[float, float, float]:
    t = _clamp(t, 0.0, 1.0)
    return (a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t,
            a[2] + (b[2] - a[2]) * t)


def _add3(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _mul3(
    c: tuple[float, float, float],
    s: float,
) -> tuple[float, float, float]:
    return (c[0] * s, c[1] * s, c[2] * s)


def _dot3(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _normalize3(v: tuple[float, float, float]) -> tuple[float, float, float]:
    mag = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if mag < 1e-12:
        return (0.0, 0.0, 1.0)
    inv = 1.0 / mag
    return (v[0] * inv, v[1] * inv, v[2] * inv)


def _vec3_to_tuple(v: Vec3) -> tuple[float, float, float]:
    return (v.x, v.y, v.z)


# ---------------------------------------------------------------------------
# Scatter functions
# ---------------------------------------------------------------------------

def _rayleigh_phase(cos_theta: float) -> float:
    """Standard Rayleigh phase function (normalised)."""
    return 0.75 * (1.0 + cos_theta * cos_theta)


def _mie_phase(cos_theta: float, g: float) -> float:
    """Henyey-Greenstein Mie phase function."""
    g2 = g * g
    denom = 1.0 + g2 - 2.0 * g * cos_theta
    if denom < 1e-9:
        return 1.0
    return (1.0 - g2) / (denom * math.sqrt(denom))


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

# Default camera height above the surface used when no explicit altitude is
# provided (e.g., for climate dust sampling when only the altitude offset is
# known).  Matches the default planet_radius_units from CONFIG_DEFAULTS.
_CAMERA_BASE_HEIGHT: float = 1000.0

# ---------------------------------------------------------------------------
# AtmosphereRenderer
# ---------------------------------------------------------------------------

class AtmosphereRenderer:
    """CPU-side atmosphere compositor for Dust Stage 11.

    Parameters
    ----------
    config:
        Game :class:`~src.core.Config.Config` instance (reads ``atmo.*`` and
        ``grade.*`` keys).
    astro:
        :class:`~src.systems.AstroSystem.AstroSystem` or compatible stub.
        Used for sun directions / intensities / eclipse factor.
    climate:
        :class:`~src.systems.ClimateSystem.ClimateSystem` or compatible stub.
        Used for dust density / visibility along sample rays.
    """

    # Debug mode string constants.
    DEBUG_HAZE_ONLY        = "DEBUG_HAZE_ONLY"
    DEBUG_VOL_DUST_ONLY    = "DEBUG_VOL_DUST_ONLY"
    DEBUG_TRANSMITTANCE    = "DEBUG_TRANSMITTANCE"
    DEBUG_SKY_ONLY         = "DEBUG_SKY_ONLY"

    # Sun colours (same warm/cool palette as SkyPrimitivesRenderer)
    _SUN1_COLOR: tuple[float, float, float] = (1.00, 0.88, 0.60)   # warm key
    _SUN2_COLOR: tuple[float, float, float] = (0.70, 0.85, 1.00)   # cool rim/fill

    # ---------------------------------------------------------------------------
    # Construction
    # ---------------------------------------------------------------------------

    def __init__(
        self,
        config: Config | None = None,
        astro=None,
        climate=None,
    ) -> None:
        if config is None:
            config = Config()

        self._astro   = astro
        self._climate = climate

        # --- Atmosphere parameters ---
        self._haze_density_base:   float = config.get("atmo", "haze_density_base",   default=0.003)
        self._haze_height_falloff: float = config.get("atmo", "haze_height_falloff", default=0.001)
        self._rayleigh_strength:   float = config.get("atmo", "rayleigh_strength",   default=1.0)
        self._mie_strength:        float = config.get("atmo", "mie_strength",        default=0.8)
        self._mie_g:               float = config.get("atmo", "mie_g",               default=0.76)
        self._vol_steps:           int   = int(config.get("atmo", "vol_steps",       default=16))
        self._vol_max_dist:        float = config.get("atmo", "vol_max_distance",    default=5000.0)
        self._vol_density_scale:   float = config.get("atmo", "vol_density_scale",   default=1.0)
        self._whiteout_threshold:  float = config.get("atmo", "whiteout_threshold",  default=0.7)

        # Sun2 contribution limit (prevents it from washing out key sun)
        self._sun2_sky_limit: float = config.get("atmo", "sun2_sky_limit", default=0.5)

        # --- Grade parameters ---
        self._filmic_toe:       float = config.get("grade", "filmic_toe",       default=0.04)
        self._filmic_shoulder:  float = config.get("grade", "filmic_shoulder",  default=0.85)
        self._shadow_tint:  tuple = tuple(config.get("grade", "shadow_tint",   default=[0.80, 0.85, 1.00]))
        self._highlight_tint: tuple = tuple(config.get("grade", "highlight_tint", default=[1.00, 0.95, 0.80]))
        self._sat_far_scale:    float = config.get("grade", "sat_far_scale",    default=0.30)

    # ---------------------------------------------------------------------------
    # 1. Sky colour — simplified Rayleigh + Mie, two suns
    # ---------------------------------------------------------------------------

    def compute_sky_color(
        self,
        view_dir: tuple[float, float, float],
        camera_altitude: float = 0.0,
        eclipse_factor: float = 0.0,
    ) -> tuple[float, float, float]:
        """Return (r, g, b) sky colour for *view_dir*.

        Combines Rayleigh (blue/purple) and Mie (dusty warm) scatter from
        both suns.  Eclipse reduces the contribution from Sun2.

        Parameters
        ----------
        view_dir:
            Unit direction vector toward which the camera is looking.
        camera_altitude:
            Camera height above the surface (simulation units).  Higher
            altitude → less Rayleigh scatter.
        eclipse_factor:
            Eclipse overlap factor [0, 1].  Reduces Sun2 sky contribution.
        """
        vd = _normalize3(view_dir)

        dir1: tuple[float, float, float] = (0.0, 1.0, 0.0)
        dir2: tuple[float, float, float] = (0.0, 1.0, 0.0)
        i1 = 1.0
        i2 = 0.35
        dust = 0.12

        if self._astro is not None:
            d1, d2 = self._astro.get_sun_directions()
            dir1 = _vec3_to_tuple(d1)
            dir2 = _vec3_to_tuple(d2)
            i1 = getattr(self._astro, "_sun1_intensity", 1.0)
            i2 = getattr(self._astro, "_sun2_intensity", 0.35)

        if self._climate is not None:
            camera_pos = Vec3(0.0, _CAMERA_BASE_HEIGHT + camera_altitude, 0.0)
            dust = self._climate.sample_dust(camera_pos)

        # Height-dependent Rayleigh attenuation
        height_scale = math.exp(-camera_altitude * self._haze_height_falloff)

        sky = (0.0, 0.0, 0.0)

        for sun_dir, sun_intensity, sun_color, is_sun2 in (
            (dir1, i1, self._SUN1_COLOR, False),
            (dir2, i2, self._SUN2_COLOR, True),
        ):
            cos_theta = _clamp(_dot3(vd, _normalize3(sun_dir)), -1.0, 1.0)

            # Rayleigh: scattered (blue) component
            r_phase = _rayleigh_phase(cos_theta)
            # Mie: dusty forward-scattering component boosted by dust
            m_phase = _mie_phase(cos_theta, self._mie_g)

            r_contrib = self._rayleigh_strength * r_phase * height_scale
            m_contrib = self._mie_strength      * m_phase * (1.0 + dust * 2.0)

            scatter = r_contrib + m_contrib
            eff_intensity = sun_intensity * scatter

            if is_sun2:
                eff_intensity *= (1.0 - eclipse_factor)
                eff_intensity = min(eff_intensity, self._sun2_sky_limit * sun_intensity)

            sky = _add3(sky, _mul3(sun_color, eff_intensity))

        # Atmosphere scattering colour tint (warm Mars-like)
        atmo_tint = (0.72, 0.58, 0.45)
        sky = (sky[0] * atmo_tint[0], sky[1] * atmo_tint[1], sky[2] * atmo_tint[2])

        # Clamp
        return (_clamp(sky[0], 0.0, 1.0), _clamp(sky[1], 0.0, 1.0), _clamp(sky[2], 0.0, 1.0))

    # ---------------------------------------------------------------------------
    # 2. Aerial perspective / haze
    # ---------------------------------------------------------------------------

    def compute_haze_factor(
        self,
        distance: float,
        camera_altitude: float = 0.0,
        dust: float | None = None,
    ) -> float:
        """Return fog factor in **[0, 1]**, monotonically increasing with distance.

        Uses Beer–Lambert exponential model:
        ``fog = 1 − exp(−distance × density)``

        Parameters
        ----------
        distance:
            World-space distance to the surface point.
        camera_altitude:
            Height above the surface; higher altitude = thinner haze.
        dust:
            Dust concentration [0, 1] at camera position.  Falls back to
            ClimateSystem query if not supplied.
        """
        if dust is None:
            if self._climate is not None:
                camera_pos = Vec3(0.0, _CAMERA_BASE_HEIGHT + camera_altitude, 0.0)
                dust = self._climate.sample_dust(camera_pos)
            else:
                dust = 0.12

        # Effective density: base + dust contribution + height falloff
        height_factor = math.exp(-camera_altitude * self._haze_height_falloff)
        density = self._haze_density_base * (1.0 + dust * 3.0) * height_factor
        fog = 1.0 - math.exp(-distance * density)
        return _clamp(fog, 0.0, 1.0)

    def apply_haze(
        self,
        surface_color: tuple[float, float, float],
        distance: float,
        view_dir: tuple[float, float, float],
        camera_altitude: float = 0.0,
        dust: float | None = None,
        eclipse_factor: float = 0.0,
    ) -> tuple[float, float, float]:
        """Blend *surface_color* toward the haze colour by fog factor.

        The haze colour is the sky colour for the given *view_dir*, giving
        an "aerial perspective" effect — distant objects take on the sky hue.
        """
        fog = self.compute_haze_factor(distance, camera_altitude, dust)
        haze_color = self.compute_sky_color(view_dir, camera_altitude, eclipse_factor)
        return _lerp3(surface_color, haze_color, fog)

    # ---------------------------------------------------------------------------
    # 3. Volumetric dust — ray march
    # ---------------------------------------------------------------------------

    def compute_volumetric(
        self,
        ray_origin: tuple[float, float, float],
        ray_dir: tuple[float, float, float],
        max_dist: float | None = None,
        camera_altitude: float = 0.0,
    ) -> tuple[tuple[float, float, float], float]:
        """Ray-march volumetric dust from *ray_origin* along *ray_dir*.

        Returns
        -------
        vol_color : tuple[float, float, float]
            In-scattered light accumulated along the ray.
        transmittance : float
            Fraction of background light that passes through [0, 1].
            Lower values indicate more dust (potential whiteout).
        """
        if max_dist is None:
            max_dist = self._vol_max_dist

        rd = _normalize3(ray_dir)
        steps = max(1, self._vol_steps)
        step_size = max_dist / steps

        transmittance = 1.0
        vol_r = vol_g = vol_b = 0.0

        # Sun directions and intensities (defaults if no astro)
        dir1: tuple[float, float, float] = (0.0, 1.0, 0.0)
        dir2: tuple[float, float, float] = (0.0, 1.0, 0.0)
        i1, i2 = 1.0, 0.35

        if self._astro is not None:
            d1, d2 = self._astro.get_sun_directions()
            dir1 = _vec3_to_tuple(d1)
            dir2 = _vec3_to_tuple(d2)
            i1 = getattr(self._astro, "_sun1_intensity", 1.0)
            i2 = getattr(self._astro, "_sun2_intensity", 0.35)

        for s in range(steps):
            t = (s + 0.5) * step_size
            sample_pos_t = (
                ray_origin[0] + rd[0] * t,
                ray_origin[1] + rd[1] * t,
                ray_origin[2] + rd[2] * t,
            )

            # Sample dust density at this world position
            if self._climate is not None:
                sv = Vec3(sample_pos_t[0], sample_pos_t[1], sample_pos_t[2])
                dust_d = self._climate.sample_dust(sv)
            else:
                dust_d = 0.12

            density = dust_d * self._vol_density_scale

            # Beer–Lambert transmittance step
            d_transmit = math.exp(-density * step_size * 0.001)
            transmittance *= d_transmit

            # In-scatter: Mie phase from each sun
            for sun_dir, sun_intensity, sun_color in (
                (dir1, i1, self._SUN1_COLOR),
                (dir2, i2, self._SUN2_COLOR),
            ):
                cos_theta = _clamp(_dot3(rd, _normalize3(sun_dir)), -1.0, 1.0)
                phase = _mie_phase(cos_theta, self._mie_g)
                scatter = dust_d * phase * sun_intensity * step_size * 0.0005
                # Whiteout: blend toward white when dust is very high
                if dust_d > self._whiteout_threshold:
                    white_blend = (dust_d - self._whiteout_threshold) / (1.0 - self._whiteout_threshold + 1e-9)
                    sc = _lerp3(sun_color, (1.0, 1.0, 1.0), white_blend * 0.7)
                else:
                    sc = sun_color
                vol_r += sc[0] * scatter * transmittance
                vol_g += sc[1] * scatter * transmittance
                vol_b += sc[2] * scatter * transmittance

            if transmittance < 0.01:
                break  # fully opaque

        vol_color = (
            _clamp(vol_r, 0.0, 1.0),
            _clamp(vol_g, 0.0, 1.0),
            _clamp(vol_b, 0.0, 1.0),
        )
        return vol_color, _clamp(transmittance, 0.0, 1.0)

    # ---------------------------------------------------------------------------
    # 4. Sun disc + corona
    # ---------------------------------------------------------------------------

    def compute_sun_contribution(
        self,
        view_dir: tuple[float, float, float],
        eclipse_factor: float = 0.0,
    ) -> tuple[float, float, float]:
        """Return (r, g, b) contribution of sun discs and coronas.

        Disc: smoothstep within angular radius.
        Corona: exponential falloff from disc edge.
        """
        vd = _normalize3(view_dir)
        result = (0.0, 0.0, 0.0)

        if self._astro is None:
            return result

        d1, d2 = self._astro.get_sun_directions()
        ang1 = getattr(self._astro, "_sun1_ang_r", math.radians(1.0))
        ang2 = getattr(self._astro, "_sun2_ang_r", math.radians(0.7))
        i1 = getattr(self._astro, "_sun1_intensity", 1.0)
        i2 = getattr(self._astro, "_sun2_intensity", 0.35)

        for sun_dir_v, ang_r, sun_color, sun_intensity, is_sun2 in (
            (d1, ang1, self._SUN1_COLOR, i1, False),
            (d2, ang2, self._SUN2_COLOR, i2, True),
        ):
            sd = _normalize3(_vec3_to_tuple(sun_dir_v))
            cos_theta = _clamp(_dot3(vd, sd), -1.0, 1.0)
            theta = math.acos(cos_theta)

            # Disc: bright filled circle
            disc = _smoothstep(ang_r * 1.1, ang_r * 0.9, theta)

            # Corona / halo: exponential falloff outside disc
            corona_width = ang_r * 4.0
            corona = math.exp(-max(0.0, theta - ang_r) / max(1e-9, corona_width)) * 0.3

            contrib = (disc + corona) * sun_intensity
            if is_sun2:
                contrib *= (1.0 - eclipse_factor * 0.8)

            result = _add3(result, _mul3(sun_color, contrib))

        return (
            _clamp(result[0], 0.0, 1.0),
            _clamp(result[1], 0.0, 1.0),
            _clamp(result[2], 0.0, 1.0),
        )

    # ---------------------------------------------------------------------------
    # 5. Cinematic colour grade
    # ---------------------------------------------------------------------------

    def color_grade(
        self,
        color: tuple[float, float, float],
        fog_factor: float = 0.0,
    ) -> tuple[float, float, float]:
        """Apply cinematic colour grading.

        1. Filmic compression of highlights (toe + shoulder).
        2. Warm highlights / cool shadows split-toning.
        3. Atmospheric desaturation increasing with *fog_factor*.

        Parameters
        ----------
        color:
            Linear (r, g, b) surface colour after haze compositing.
        fog_factor:
            Haze factor [0, 1] for the corresponding pixel — used to
            scale the far-desaturation amount.
        """
        r, g, b = color

        # 1. Filmic curve (soft roll-off for highlights)
        r = self._filmic(r)
        g = self._filmic(g)
        b = self._filmic(b)

        # 2. Split-toning: warm highlights, cool shadows
        luma = 0.299 * r + 0.587 * g + 0.114 * b
        tint = _lerp3(self._shadow_tint, self._highlight_tint, luma)
        r *= tint[0]
        g *= tint[1]
        b *= tint[2]

        # 3. Atmospheric desaturation increases with distance (fog_factor)
        luma2 = 0.299 * r + 0.587 * g + 0.114 * b
        desat = fog_factor * self._sat_far_scale
        r = _lerp(r, luma2, desat)
        g = _lerp(g, luma2, desat)
        b = _lerp(b, luma2, desat)

        return (_clamp(r, 0.0, 1.0), _clamp(g, 0.0, 1.0), _clamp(b, 0.0, 1.0))

    # ---------------------------------------------------------------------------
    # 6. Full composite pass
    # ---------------------------------------------------------------------------

    def composite(
        self,
        surface_color: tuple[float, float, float],
        distance: float,
        view_dir: tuple[float, float, float],
        ray_origin: tuple[float, float, float] | None = None,
        camera_altitude: float = 0.0,
        dust: float | None = None,
        debug_mode: str | None = None,
    ) -> tuple[float, float, float]:
        """Full atmosphere composite for a single pixel.

        Applies (in order): volumetric dust → aerial haze → colour grade.
        Returns final (r, g, b) ready for framebuffer output.

        Parameters
        ----------
        surface_color:
            Opaque surface colour from the geometry pass.
        distance:
            World-space depth to the surface point.
        view_dir:
            Unit viewing direction for the pixel.
        ray_origin:
            Camera world position (for volumetric sampling).  Defaults to
            origin if *None*.
        camera_altitude:
            Camera height above the surface.
        dust:
            Explicit dust value; falls back to ClimateSystem if *None*.
        debug_mode:
            One of the ``DEBUG_*`` class constants to override the output.
        """
        if ray_origin is None:
            ray_origin = (0.0, _CAMERA_BASE_HEIGHT + camera_altitude, 0.0)

        eclipse_factor = 0.0
        if self._astro is not None and hasattr(self._astro, "get_eclipse_factor"):
            eclipse_factor = self._astro.get_eclipse_factor()

        # --- Debug modes ---
        if debug_mode == self.DEBUG_SKY_ONLY:
            return self.compute_sky_color(view_dir, camera_altitude, eclipse_factor)

        if debug_mode == self.DEBUG_HAZE_ONLY:
            fog = self.compute_haze_factor(distance, camera_altitude, dust)
            return (fog, fog, fog)

        if debug_mode == self.DEBUG_TRANSMITTANCE:
            _, transmit = self.compute_volumetric(ray_origin, view_dir, distance, camera_altitude)
            return (transmit, transmit, transmit)

        if debug_mode == self.DEBUG_VOL_DUST_ONLY:
            vol_color, _ = self.compute_volumetric(ray_origin, view_dir, distance, camera_altitude)
            return vol_color

        # --- Full pipeline ---

        # Step 1: volumetric dust compositing
        vol_color, transmittance = self.compute_volumetric(
            ray_origin, view_dir, distance, camera_altitude
        )
        lit_surface = _mul3(surface_color, transmittance)
        after_vol = _add3(lit_surface, vol_color)
        after_vol = (
            _clamp(after_vol[0], 0.0, 1.0),
            _clamp(after_vol[1], 0.0, 1.0),
            _clamp(after_vol[2], 0.0, 1.0),
        )

        # Step 2: aerial perspective / haze
        fog = self.compute_haze_factor(distance, camera_altitude, dust)
        after_haze = self.apply_haze(after_vol, distance, view_dir, camera_altitude, dust, eclipse_factor)

        # Step 3: colour grade
        return self.color_grade(after_haze, fog)

    # ---------------------------------------------------------------------------
    # Private helpers
    # ---------------------------------------------------------------------------

    def _filmic(self, x: float) -> float:
        """Piecewise filmic tone-mapping curve (toe + shoulder).

        * Below *filmic_toe*: linear pass-through.
        * Above *filmic_shoulder*: compressed toward 1.
        * Mid-range: smooth cubic transition.
        """
        if x <= self._filmic_toe:
            return x
        if x >= self._filmic_shoulder:
            # Soft shoulder: asymptotic to 1
            excess = x - self._filmic_shoulder
            return 1.0 - math.exp(-excess * 3.0) * (1.0 - self._filmic_shoulder)
        # Mid-range cubic ease
        t = (x - self._filmic_toe) / (self._filmic_shoulder - self._filmic_toe)
        t = t * t * (3.0 - 2.0 * t)   # smoothstep
        return _lerp(self._filmic_toe, self._filmic_shoulder, t)


# ---------------------------------------------------------------------------
# Module-level helper (smoothstep — mirrors AstroSystem version)
# ---------------------------------------------------------------------------

def _smoothstep(edge0: float, edge1: float, x: float) -> float:
    if edge1 <= edge0:
        return 1.0 if x >= edge1 else 0.0
    t = _clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)
