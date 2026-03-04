"""LightingModelFinal — Stage 62 cinematic dual-sun + ring lighting.

Implements §3.2 (Lighting Stage) and §5:

* Two directional lights (sun1, sun2) with distinct colour temperatures.
* Atmospheric scattering contribution baked into the sun colours (simplified
  but stable — no per-frame recompute of full Rayleigh/Mie).
* Ring shadow attenuation applied on top of the lighting result.
* Soft AO term (very gentle, no noise).
* Soft PCF-style shadow factor input (0 = fully shadowed, 1 = fully lit).

All maths is pure Python — no OpenGL dependency.  Consumers pass the result
to :class:`~src.render.ToneMapperLocked.ToneMapperLocked` before display.

Public API
----------
LightSample (dataclass)
LightingModelFinal(config=None)
  .evaluate(normal, view_dir, ao, shadow, ring_shadow) → LightSample
  .sun1_dir : tuple[float,float,float]
  .sun2_dir : tuple[float,float,float]
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

_Vec3 = Tuple[float, float, float]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dot(a: _Vec3, b: _Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _normalize(v: _Vec3) -> _Vec3:
    mag = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if mag < 1e-12:
        return (0.0, 1.0, 0.0)
    inv = 1.0 / mag
    return (v[0] * inv, v[1] * inv, v[2] * inv)


def _add(a: _Vec3, b: _Vec3) -> _Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _mul(v: _Vec3, s: float) -> _Vec3:
    return (v[0] * s, v[1] * s, v[2] * s)


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


def _kelvin_to_rgb(kelvin: float) -> _Vec3:
    """Convert a colour temperature (K) to an approximate linear RGB tint.

    Uses a simple polynomial fit that is accurate enough for art direction;
    no look-up tables or texture assets required.
    """
    t = _clamp(kelvin, 1000.0, 40000.0) / 100.0

    # Red channel
    if t <= 66.0:
        r = 1.0
    else:
        r = 329.698727446 * ((t - 60.0) ** -0.1332047592) / 255.0
    r = _clamp(r, 0.0, 1.0)

    # Green channel
    if t <= 66.0:
        g = (99.4708025861 * math.log(t) - 161.1195681661) / 255.0
    else:
        g = 288.1221695283 * ((t - 60.0) ** -0.0755148492) / 255.0
    g = _clamp(g, 0.0, 1.0)

    # Blue channel
    if t >= 66.0:
        b = 1.0
    elif t <= 19.0:
        b = 0.0
    else:
        b = (138.5177312231 * math.log(t - 10.0) - 305.0447927307) / 255.0
    b = _clamp(b, 0.0, 1.0)

    return (r, g, b)


# ---------------------------------------------------------------------------
# LightSample
# ---------------------------------------------------------------------------

@dataclass
class LightSample:
    """Output of :meth:`LightingModelFinal.evaluate`.

    Attributes
    ----------
    diffuse : (r, g, b)
        Combined diffuse irradiance from both suns.
    ambient : (r, g, b)
        Ambient / sky irradiance (AO-weighted).
    specular : (r, g, b)
        Specular highlight (very soft, capped at low intensity).
    total : (r, g, b)
        ``diffuse + ambient + specular`` — ready for tone-mapping.
    shadow_factor : float
        Effective shadow in [0, 1] (0 = black, 1 = fully lit).
    """
    diffuse: _Vec3 = (0.0, 0.0, 0.0)
    ambient: _Vec3 = (0.0, 0.0, 0.0)
    specular: _Vec3 = (0.0, 0.0, 0.0)
    total: _Vec3 = (0.0, 0.0, 0.0)
    shadow_factor: float = 1.0


# ---------------------------------------------------------------------------
# LightingModelFinal
# ---------------------------------------------------------------------------

class LightingModelFinal:
    """Cinematic dual-sun + ring lighting model (§3.2, §5).

    Parameters
    ----------
    config :
        Optional dict; reads ``render.sun1_color``, ``render.sun2_color``,
        ``render.ring_shadow_strength`` and ``render.shadow_quality``.
        Falls back to Kelvin-derived colours (5 000 K / 8 500 K) when not
        provided.
    """

    # Default sun directions (normalised) — artistically chosen
    _DEFAULT_SUN1_DIR: _Vec3 = _normalize((0.6, 1.0, 0.4))
    _DEFAULT_SUN2_DIR: _Vec3 = _normalize((-0.3, 0.8, 0.7))

    # Intensities
    _SUN1_INTENSITY = 1.0
    _SUN2_INTENSITY = 0.45   # secondary is dimmer
    _AMBIENT_INTENSITY = 0.12
    _SPECULAR_MAX = 0.18     # hard cap — no harsh highlights

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = (config or {}).get("render", {}) or {}

        # Colour temperatures → linear RGB tints
        sun1_k = float(cfg.get("sun1_temp_k", 5000.0))
        sun2_k = float(cfg.get("sun2_temp_k", 8500.0))
        sun1_c = cfg.get("sun1_color")
        self._sun1_color: _Vec3 = tuple(sun1_c) if sun1_c is not None else _kelvin_to_rgb(sun1_k)  # type: ignore[assignment]
        sun2_c = cfg.get("sun2_color")
        self._sun2_color: _Vec3 = tuple(sun2_c) if sun2_c is not None else _kelvin_to_rgb(sun2_k)  # type: ignore[assignment]

        # Ring shadow strength (§5.2)
        self._ring_shadow_strength: float = float(
            cfg.get("ring_shadow_strength", 0.35)
        )

        # Soft AO strength
        self._ao_strength: float = 0.5

        # Shadow quality: "soft" uses PCF weight, "hard" uses binary
        shadow_q = cfg.get("shadow_quality", "soft")
        self._shadow_soft: bool = (shadow_q != "hard")

        # Sun directions (can be overridden at runtime by AstroSystem)
        self.sun1_dir: _Vec3 = self._DEFAULT_SUN1_DIR
        self.sun2_dir: _Vec3 = self._DEFAULT_SUN2_DIR

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def update_sun_directions(
        self,
        sun1_dir: _Vec3,
        sun2_dir: _Vec3,
    ) -> None:
        """Update both sun directions (call once per frame from AstroSystem)."""
        self.sun1_dir = _normalize(sun1_dir)
        self.sun2_dir = _normalize(sun2_dir)

    def evaluate(
        self,
        normal: _Vec3,
        view_dir: _Vec3,
        ao: float = 1.0,
        shadow: float = 1.0,
        ring_shadow: float = 0.0,
    ) -> LightSample:
        """Compute cinematic lighting for a single surface point.

        Parameters
        ----------
        normal :
            Surface normal (normalised, world space).
        view_dir :
            View direction toward camera (normalised, world space).
        ao :
            Ambient occlusion factor in [0, 1]; 1 = fully unoccluded.
        shadow :
            PCF shadow factor in [0, 1]; 1 = fully lit, 0 = fully in shadow.
        ring_shadow :
            Ring shadow attenuation in [0, 1]; 1 = maximum ring shadow.

        Returns
        -------
        LightSample
        """
        n = _normalize(normal)
        v = _normalize(view_dir)

        # --- Shadow factor ---
        eff_shadow = self._compute_shadow(shadow, ring_shadow)

        # --- Diffuse (Lambert) from both suns ---
        diff1 = _clamp(_dot(n, self.sun1_dir), 0.0, 1.0) * self._SUN1_INTENSITY * eff_shadow
        diff2 = _clamp(_dot(n, self.sun2_dir), 0.0, 1.0) * self._SUN2_INTENSITY

        diffuse = _add(
            _mul(self._sun1_color, diff1),
            _mul(self._sun2_color, diff2),
        )

        # --- Ambient (soft, AO-weighted) ---
        ao_f = _clamp(ao, 0.0, 1.0) * self._ao_strength + (1.0 - self._ao_strength)
        # Ambient tint is a blend of both sun colours toward cool sky
        ambient_color: _Vec3 = (0.18, 0.22, 0.32)
        ambient = _mul(ambient_color, self._AMBIENT_INTENSITY * ao_f)

        # --- Specular (very soft Phong, capped) ---
        specular = self._compute_specular(n, v, eff_shadow)

        # --- Assemble total ---
        total: _Vec3 = (
            _clamp(diffuse[0] + ambient[0] + specular[0], 0.0, 4.0),
            _clamp(diffuse[1] + ambient[1] + specular[1], 0.0, 4.0),
            _clamp(diffuse[2] + ambient[2] + specular[2], 0.0, 4.0),
        )

        return LightSample(
            diffuse=diffuse,
            ambient=ambient,
            specular=specular,
            total=total,
            shadow_factor=eff_shadow,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute_shadow(self, shadow: float, ring_shadow: float) -> float:
        """Combine PCF shadow and ring shadow into a single factor."""
        pcf = _clamp(shadow, 0.0, 1.0)
        ring = _clamp(ring_shadow, 0.0, 1.0) * self._ring_shadow_strength
        # Ring attenuates light multiplicatively; PCF attenuates further
        return _clamp(pcf * (1.0 - ring), 0.0, 1.0)

    def _compute_specular(
        self,
        n: _Vec3,
        v: _Vec3,
        shadow: float,
    ) -> _Vec3:
        """Soft Phong specular from sun1 (primary).  Very gentle cap."""
        # Reflect sun1_dir about normal
        dot_nl = _dot(n, self.sun1_dir)
        refl: _Vec3 = (
            2.0 * dot_nl * n[0] - self.sun1_dir[0],
            2.0 * dot_nl * n[1] - self.sun1_dir[1],
            2.0 * dot_nl * n[2] - self.sun1_dir[2],
        )
        spec_dot = _clamp(_dot(refl, v), 0.0, 1.0)
        # Low shininess (8) for very soft highlight
        spec_intensity = (spec_dot ** 8) * self._SPECULAR_MAX * shadow
        return _mul(self._sun1_color, spec_intensity)
