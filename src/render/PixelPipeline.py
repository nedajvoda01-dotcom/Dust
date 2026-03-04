"""PixelPipeline — Stage 62 master render pipeline compositor (§3, §11).

Orchestrates the full Stage 62 rendering sequence:

    1. Lighting  (LightingModelFinal)
    2. Atmosphere fog + scattering  (AtmosphereModel)
    3. Tone-mapping  (ToneMapperLocked)
    4. Colour grading  (ColorGradingProfile)
    5. Pixel quantization  (PixelQuantizer)

This module is the single entry point for render-loop consumers.  It exposes
a high-level ``process_buffer`` method that accepts a list of
:class:`SurfaceSample` objects and returns a fully processed flat pixel
buffer.

It also acts as the authoritative keeper of the Stage 62 config (§12):
parameters are read once at construction and locked until
``apply_tuning_epoch`` is called explicitly (anti-auto-tuning, §12 note).

Pure Python — no OpenGL.

Public API
----------
SurfaceSample (dataclass)
PixelPipeline(config=None)
  .process_buffer(samples, width, height) → list[tuple[float,float,float]]
  .set_weather(dust_density, storm_active)
  .set_contrast_from_weather(dust_density, storm_active)
  .apply_tuning_epoch(new_config)
  .config_snapshot() → dict
"""
from __future__ import annotations

import math
import hashlib
import json
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from src.render.LightingModelFinal import LightingModelFinal
from src.render.ToneMapperLocked import ToneMapperLocked
from src.render.AtmosphereModel import AtmosphereModel
from src.render.RingShadowProjector import RingShadowProjector
from src.render.ColorGradingProfile import ColorGradingProfile
from src.render.PixelQuantizer import PixelQuantizer, PixelQuantizerConfig

_Vec3 = Tuple[float, float, float]


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


def _lerp3(a: _Vec3, b: _Vec3, t: float) -> _Vec3:
    t = _clamp(t, 0.0, 1.0)
    return (a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t,
            a[2] + (b[2] - a[2]) * t)


# ---------------------------------------------------------------------------
# SurfaceSample
# ---------------------------------------------------------------------------

@dataclass
class SurfaceSample:
    """Per-fragment input to the PixelPipeline.

    Attributes
    ----------
    position : (x, y, z)
        World-space surface position (planet-local coords).
    normal : (x, y, z)
        World-space surface normal (normalised).
    view_dir : (x, y, z)
        Direction from surface toward camera (normalised).
    base_color : (r, g, b)
        Pre-lit material colour in linear space.
    ao : float
        Ambient occlusion [0, 1]; 1 = fully unoccluded.
    shadow : float
        PCF shadow factor [0, 1]; 1 = fully lit.
    distance : float
        Distance from camera to this surface point (sim units).
    """
    position: _Vec3 = (0.0, 0.0, 0.0)
    normal: _Vec3 = (0.0, 1.0, 0.0)
    view_dir: _Vec3 = (0.0, 0.0, 1.0)
    base_color: _Vec3 = (0.72, 0.58, 0.45)
    ao: float = 1.0
    shadow: float = 1.0
    distance: float = 100.0


# ---------------------------------------------------------------------------
# PixelPipeline
# ---------------------------------------------------------------------------

# Locked config keys (§12) — these are never auto-tuned.
_LOCKED_KEYS = (
    "internal_resolution",
    "pixel_resolution",
    "pixel_scale_mode",
    "tone_mapper",
    "shadow_quality",
    "fog_density_base",
    "dust_color_shift",
    "ring_shadow_strength",
    "sun1_color",
    "sun2_color",
    "camera_inertia",
    "fov_base",
)


class PixelPipeline:
    """Master Stage 62 render pipeline (§3, §11).

    Parameters
    ----------
    config :
        Optional dict matching CONFIG_DEFAULTS.json structure.  Reads
        ``render.*`` and ``camera.*`` keys.  Locked keys (§12) are captured
        at construction; only :meth:`apply_tuning_epoch` may update them.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        self._raw_config: dict = {}
        self._locked_snapshot: dict = {}
        self._lighting: LightingModelFinal
        self._tonemapper: ToneMapperLocked
        self._atmosphere: AtmosphereModel
        self._ring: RingShadowProjector
        self._grading: ColorGradingProfile
        self._quantizer: PixelQuantizer
        self._setup(config)

    # ------------------------------------------------------------------
    # Internal — initialisation
    # ------------------------------------------------------------------

    def _setup(self, config: Optional[dict]) -> None:
        """Build (or rebuild) all sub-systems from *config*."""
        self._raw_config = dict(config or {})
        render_cfg = self._raw_config.get("render", {}) or {}

        # Snapshot locked keys
        self._locked_snapshot = {k: render_cfg.get(k) for k in _LOCKED_KEYS}

        # Build sub-systems
        self._lighting = LightingModelFinal(self._raw_config)
        self._tonemapper = ToneMapperLocked(self._raw_config)
        self._atmosphere = AtmosphereModel(self._raw_config)
        self._ring = RingShadowProjector(self._raw_config)
        self._grading = ColorGradingProfile(self._raw_config)

        # PixelQuantizer uses PixelQuantizerConfig
        int_res = render_cfg.get("internal_resolution", [1920, 1080])
        pix_res = render_cfg.get("pixel_resolution", [853, 480])
        scale_mode = render_cfg.get("pixel_scale_mode", "nearest")
        pq_cfg = PixelQuantizerConfig(
            internal_resolution=(int(int_res[0]), int(int_res[1])),
            pixel_resolution=(int(pix_res[0]), int(pix_res[1])),
            pixel_scale_mode=scale_mode,
        )
        self._quantizer = PixelQuantizer(pq_cfg)

    # ------------------------------------------------------------------
    # Public — weather / per-frame state
    # ------------------------------------------------------------------

    def set_weather(
        self,
        dust_density: float,
        storm_active: bool = False,
        altitude: float = 0.0,
    ) -> None:
        """Update atmosphere state once per weather tick."""
        self._atmosphere.update(dust_density, storm_active, altitude)
        self.set_contrast_from_weather(dust_density, storm_active)

    def set_contrast_from_weather(
        self,
        dust_density: float,
        storm_active: bool = False,
    ) -> None:
        """Drive ToneMapper contrast from weather (§4.3)."""
        # Storm → lower contrast; clear dual-sun → higher
        if storm_active:
            contrast = 0.75
        else:
            contrast = 1.0 + (1.0 - dust_density) * 0.25
        self._tonemapper.set_contrast(contrast)

    def update_sun_directions(
        self,
        sun1_dir: _Vec3,
        sun2_dir: _Vec3,
    ) -> None:
        """Pass per-frame sun directions from AstroSystem."""
        self._lighting.update_sun_directions(sun1_dir, sun2_dir)

    # ------------------------------------------------------------------
    # Public — main processing
    # ------------------------------------------------------------------

    def process_buffer(
        self,
        samples: List[SurfaceSample],
        width: int,
        height: int,
    ) -> List[_Vec3]:
        """Run the full Stage 62 pipeline on a flat list of surface samples.

        Parameters
        ----------
        samples :
            List of :class:`SurfaceSample`; must have length ``width*height``.
        width, height :
            Dimensions of the sample buffer.

        Returns
        -------
        list of (r, g, b)
            Final display buffer in sRGB [0, 1], dimensions ``width×height``.
        """
        lit: List[_Vec3] = []
        for s in samples:
            color = self._shade_sample(s)
            lit.append(color)

        # Tone-map + grade
        toned = self._tonemapper.apply_buffer(lit)
        graded = self._grading.grade_buffer(toned)

        # Pixel quantize
        output = self._quantizer.quantize(graded, width, height)
        return output

    # ------------------------------------------------------------------
    # Public — tuning epoch (§12)
    # ------------------------------------------------------------------

    def apply_tuning_epoch(self, new_config: dict) -> None:
        """Replace the entire config and rebuild all sub-systems.

        This is the **only** sanctioned way to change locked parameters.
        """
        self._setup(dict(new_config))

    def config_snapshot(self) -> dict:
        """Return a serialisable dict of current locked config keys."""
        return dict(self._locked_snapshot)

    def visual_hash(self) -> str:
        """Return a stable hex digest of locked config keys.

        Useful for ``test_visual_hash_stable_across_restarts`` (§13.6).
        """
        payload = json.dumps(self._locked_snapshot, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _shade_sample(self, s: SurfaceSample) -> _Vec3:
        """Compute lit + fog colour for one surface sample."""
        # Ring shadow
        ring_sh = self._ring.shadow_at(s.position, self._lighting.sun1_dir)

        # Lighting
        light = self._lighting.evaluate(
            s.normal, s.view_dir,
            ao=s.ao,
            shadow=s.shadow,
            ring_shadow=ring_sh,
        )

        # Modulate base colour by diffuse + ambient
        r = s.base_color[0] * (light.diffuse[0] + light.ambient[0]) + light.specular[0]
        g = s.base_color[1] * (light.diffuse[1] + light.ambient[1]) + light.specular[1]
        b = s.base_color[2] * (light.diffuse[2] + light.ambient[2]) + light.specular[2]

        # Cold bias from ring shadow
        cold = self._ring.cold_bias_at(ring_sh)
        b = _clamp(b + cold * 0.08, 0.0, 4.0)

        # Atmosphere: fog blend + scatter
        fog = self._atmosphere.fog_factor(s.distance)
        scatter = self._atmosphere.scatter_hint(s.distance)
        fog_color = self._atmosphere.color_shift

        fogged = _lerp3((r, g, b), fog_color, fog)
        r = _clamp(fogged[0] + scatter[0], 0.0, 4.0)
        g = _clamp(fogged[1] + scatter[1], 0.0, 4.0)
        b_out = _clamp(fogged[2] + scatter[2], 0.0, 4.0)

        return (r, g, b_out)
