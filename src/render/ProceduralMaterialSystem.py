"""ProceduralMaterialSystem — Stage 10/28 texture-free surface material (v2).

No bitmap textures, texture samplers, normal-map files, or UV-based
patterns are used anywhere in this module.  All colour, roughness, and
micro-normal information is derived from:

  * world-space position / normal
  * geological fields  (hardness, fracture, boundaryType)
  * climate fields     (dust, wetness, ice, temperature)
  * procedural 3D noise (see ProceduralNoise)
  * BiomeAtlas biome classification (Stage 28)

Public API
----------
MaterialInput     — dataclass carrying all per-fragment inputs
MaterialOutput    — dataclass carrying colour, roughness, micro_normal, emissive
DebugMode         — enum selecting a debug visualisation channel
ProceduralMaterialSystem.evaluate(inp, debug_mode) → MaterialOutput

Rock types (virtual, no files)
-------------------------------
0  DustLayer       — light, matte; dominates on flat dusty surfaces
1  BasaltRock      — dark; common in convergent / mountain zones
2  LayeredSediment — horizontal banding by height; prominent in rifts
3  FracturedRock   — high contrast, fine noise fractures
4  IceFilm         — near-specular, blue-tinted; present when ice > threshold

Stage 28 additions
------------------
* evaluate() now accepts an optional ``biome_id`` (BiomeId) parameter.
  When provided, the BiomeParams base albedo and roughness range are used
  instead of the legacy rock-type blend.  If None, the legacy path runs.
* New DebugMode values: DEBUG_BIOME, DEBUG_ICE_OVERLAY, DEBUG_DUST_OVERLAY.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from src.math.Vec3 import Vec3
from src.planet.TectonicPlatesSystem import BoundaryType
from src.render.ProceduralNoise import fbm3, gradient3
from src.systems.BiomeAtlasSystem import BIOME_PARAMS, BiomeId

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_UP = Vec3(0.0, 1.0, 0.0)

# Rock base colours (linear, no gamma)
_COLOR_DUST       = Vec3(0.82, 0.74, 0.60)   # warm tan
_COLOR_BASALT     = Vec3(0.22, 0.20, 0.19)   # dark grey-brown
_COLOR_SEDIMENT_A = Vec3(0.68, 0.55, 0.40)   # lighter layer
_COLOR_SEDIMENT_B = Vec3(0.44, 0.36, 0.28)   # darker layer
_COLOR_FRACTURE   = Vec3(0.28, 0.24, 0.22)   # fractured rock
_COLOR_ICE        = Vec3(0.72, 0.82, 0.95)   # pale blue ice

# Roughness per rock type
_ROUGH_DUST      = 0.92
_ROUGH_BASALT    = 0.65
_ROUGH_SEDIMENT  = 0.75
_ROUGH_FRACTURE  = 0.80
_ROUGH_ICE       = 0.15

# Micro-normal strength per rock type (amplitude ≤ 0.25 per spec)
_MICRO_DUST      = 0.05
_MICRO_BASALT    = 0.15
_MICRO_SEDIMENT  = 0.10
_MICRO_FRACTURE  = 0.22
_MICRO_ICE       = 0.04

# Noise scale used for micro-detail (small = large features; spec §8)
_MICRO_SCALE     = 0.18

# Layer spacing (world units) for sediment terracing
_LAYER_STEP      = 80.0

# Ice presence threshold
_ICE_THRESHOLD   = 0.25

# Wet-darkening factors
_WET_DARK_K      = 0.35   # max albedo reduction
_WET_ROUGH_K     = 0.30   # max roughness reduction


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class DebugMode(Enum):
    """Development visualisation channels."""
    NONE              = auto()
    DEBUG_SLOPE       = auto()
    DEBUG_DUST        = auto()
    DEBUG_ROCKTYPE    = auto()
    DEBUG_FRACTURE    = auto()
    DEBUG_BOUNDARY    = auto()
    DEBUG_BIOME       = auto()   # Stage 28: false-colour by BiomeId
    DEBUG_ICE_OVERLAY = auto()   # Stage 28: ice overlay intensity
    DEBUG_DUST_OVERLAY = auto()  # Stage 28: dust overlay intensity


# ---------------------------------------------------------------------------
# Data transfer objects
# ---------------------------------------------------------------------------

@dataclass
class MaterialInput:
    """All per-fragment data the material system needs.

    ``slope`` is expected to be in [0, 1] where 0 = flat (normal = Up)
    and 1 = vertical.  ``curvature`` is negative for concave regions and
    positive for convex regions; typical range −1 .. +1.

    ``biome_id`` (Stage 28) — when provided, the biome's base albedo and
    roughness range take priority over the legacy rock-type blend.
    """
    world_pos:     Vec3
    world_normal:  Vec3
    height:        float          # radial height (world units)
    slope:         float          # 1 - dot(worldNormal, Up); [0, 1]
    curvature:     float          # concave < 0, convex > 0
    # Geological fields
    hardness:      float          # [0, 1]
    fracture:      float          # [0, 1]
    boundary_type: BoundaryType
    # Climate fields
    dust:          float          # [0, 1]
    wetness:       float          # [0, 1]
    ice:           float          # [0, 1]
    temperature:   float          # Kelvin
    # Stage 28: optional biome classification
    biome_id:      Optional[BiomeId] = None


@dataclass
class MaterialOutput:
    """Result of one material evaluation."""
    color:        Vec3    # linear RGB, [0, 1] per component
    roughness:    float   # [0, 1]
    micro_normal: Vec3    # perturbed world-space normal (normalised)
    emissive:     Vec3    # additive emission (near-zero unless lava etc.)
    rock_type_id: int     # dominant rock type index for debug


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


def _lerp(a: float, b: float, t: float) -> float:
    return a + t * (b - a)


def _lerp_v(a: Vec3, b: Vec3, t: float) -> Vec3:
    return a + (b - a) * t


def _saturate(v: float) -> float:
    return _clamp(v, 0.0, 1.0)


def _smoothstep(edge0: float, edge1: float, x: float) -> float:
    t = _saturate((x - edge0) / (edge1 - edge0) if edge1 != edge0 else 0.0)
    return t * t * (3.0 - 2.0 * t)


# ---------------------------------------------------------------------------
# Heatmap helper (blue → cyan → green → yellow → red)
# ---------------------------------------------------------------------------

def _heatmap(t: float) -> Vec3:
    """Convert scalar [0, 1] to a visually distinct heat colour."""
    t = _saturate(t)
    if t < 0.25:
        s = t / 0.25
        return Vec3(0.0, s, 1.0)
    elif t < 0.5:
        s = (t - 0.25) / 0.25
        return Vec3(0.0, 1.0, 1.0 - s)
    elif t < 0.75:
        s = (t - 0.5) / 0.25
        return Vec3(s, 1.0, 0.0)
    else:
        s = (t - 0.75) / 0.25
        return Vec3(1.0, 1.0 - s, 0.0)


# ---------------------------------------------------------------------------
# Rock-type colours for DEBUG_ROCKTYPE
# ---------------------------------------------------------------------------

_ROCKTYPE_DEBUG_COLORS = [
    Vec3(0.9, 0.8, 0.5),   # 0 DustLayer     — tan
    Vec3(0.2, 0.2, 0.7),   # 1 BasaltRock    — blue-grey
    Vec3(0.3, 0.7, 0.3),   # 2 LayeredSediment — green
    Vec3(0.8, 0.3, 0.2),   # 3 FracturedRock — red
    Vec3(0.6, 0.9, 1.0),   # 4 IceFilm       — light blue
]

# Boundary accent colours for DEBUG_BOUNDARY
_BOUNDARY_DEBUG_COLORS = {
    BoundaryType.NONE:       Vec3(0.3, 0.3, 0.3),
    BoundaryType.CONVERGENT: Vec3(0.9, 0.2, 0.1),
    BoundaryType.DIVERGENT:  Vec3(0.1, 0.5, 0.9),
    BoundaryType.TRANSFORM:  Vec3(0.9, 0.7, 0.1),
}

# Stage 28: false-colour per BiomeId for DEBUG_BIOME
_BIOME_DEBUG_COLORS: dict[Optional[BiomeId], Vec3] = {
    BiomeId.IRON_SAND:          Vec3(0.80, 0.35, 0.20),
    BiomeId.BASALT_PLATEAU:     Vec3(0.20, 0.20, 0.20),
    BiomeId.ASH_FIELDS:         Vec3(0.55, 0.52, 0.50),
    BiomeId.SALT_FLATS:         Vec3(0.95, 0.95, 0.90),
    BiomeId.GLASSY_IMPACT:      Vec3(0.20, 0.75, 0.60),
    BiomeId.ICE_CRUST:          Vec3(0.60, 0.80, 1.00),
    BiomeId.SCREE_SLOPES:       Vec3(0.65, 0.50, 0.35),
    BiomeId.FRACTURE_BANDS:     Vec3(0.70, 0.10, 0.10),
    BiomeId.SUBSURFACE_BEDROCK: Vec3(0.15, 0.10, 0.20),
    None:                       Vec3(0.50, 0.50, 0.50),
}


# ---------------------------------------------------------------------------
# ProceduralMaterialSystem
# ---------------------------------------------------------------------------

class ProceduralMaterialSystem:
    """Evaluates a texture-free procedural surface material.

    The system is stateless: ``evaluate`` takes a ``MaterialInput`` and
    returns a ``MaterialOutput`` with no side effects and no mutable
    state, making it deterministic and thread-safe.
    """

    # No __init__ needed — the system has no configuration state.
    # If per-scene tuning is required, subclass or add constructor params.

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def evaluate(
        self,
        inp: MaterialInput,
        debug_mode: DebugMode = DebugMode.NONE,
    ) -> MaterialOutput:
        """Compute colour, roughness and micro-normal for *inp*.

        Parameters
        ----------
        inp:
            All per-fragment data.  If ``inp.biome_id`` is set (Stage 28)
            the biome's base albedo and roughness guide the output.
        debug_mode:
            When not ``NONE`` the colour is replaced by a diagnostic
            heatmap; roughness and micro_normal are still computed
            normally so downstream passes remain valid.
        """
        # 1. Compute masks
        slope_mask     = self._slope_mask(inp.slope)
        concavity      = self._concavity(inp.curvature)
        height_band_t  = self._height_band(inp.height)

        # 2. Derive rock-type weights
        (w_dust, w_basalt, w_sediment, w_fracture,
         dominant_id) = self._rock_weights(inp, slope_mask, concavity, height_band_t)

        # 3. Base rock colour & roughness (legacy path)
        rock_color, rock_rough = self._blend_rock(
            w_dust, w_basalt, w_sediment, w_fracture)

        # 3b. Stage 28: biome override — blend toward BiomeParams base albedo
        if inp.biome_id is not None:
            bp = BIOME_PARAMS[inp.biome_id]
            # Use fracture/slope as blending factor toward biome colour
            biome_t = _saturate(0.6 + (1.0 - inp.fracture) * 0.3)
            rock_color = _lerp_v(rock_color, bp.base_albedo, biome_t)
            # Roughness from biome range, modulated by fracture
            biome_rough = _lerp(bp.roughness_min, bp.roughness_max, inp.fracture)
            rock_rough  = _lerp(rock_rough, biome_rough, biome_t)
            # Micro-normal strength from biome
            micro_strength = _lerp(
                w_dust * _MICRO_DUST + w_basalt * _MICRO_BASALT +
                w_sediment * _MICRO_SEDIMENT + w_fracture * _MICRO_FRACTURE,
                bp.micro_normal_str,
                biome_t,
            )
        else:
            micro_strength = (
                w_dust     * _MICRO_DUST    +
                w_basalt   * _MICRO_BASALT  +
                w_sediment * _MICRO_SEDIMENT +
                w_fracture * _MICRO_FRACTURE
            )

        # 4. Micro-normal perturbation (noise gradient, world-space, max 3 oct.)
        micro_normal = self._micro_normal(inp.world_pos, inp.world_normal, micro_strength)

        # 5. Dust overlay
        dust_mask = _saturate(inp.dust * (1.0 - slope_mask * 0.5) * (1.0 + concavity))
        color     = _lerp_v(rock_color, _COLOR_DUST, dust_mask)
        roughness = _lerp(rock_rough, _ROUGH_DUST, dust_mask)

        # 6. Boundary accent (linear dark line)
        if inp.boundary_type != BoundaryType.NONE:
            accent = inp.fracture * 0.3
            color  = color * (1.0 - accent)

        # 7. Ice overlay
        if inp.ice > _ICE_THRESHOLD:
            ice_mask = _smoothstep(_ICE_THRESHOLD, _ICE_THRESHOLD + 0.3, inp.ice)
            color     = _lerp_v(color, _COLOR_ICE, ice_mask)
            roughness = _lerp(roughness, _ROUGH_ICE, ice_mask)
            dominant_id = 4

        # 8. Wet darkening (modifier, not a separate rock type)
        if inp.wetness > 0.0:
            dark_k = inp.wetness * _WET_DARK_K
            color  = color * (1.0 - dark_k)
            roughness *= 1.0 - inp.wetness * _WET_ROUGH_K

        # 9. Clamp
        color     = Vec3(
            _saturate(color.x),
            _saturate(color.y),
            _saturate(color.z),
        )
        roughness = _saturate(roughness)

        # 10. Debug override (replaces colour only)
        if debug_mode != DebugMode.NONE:
            color = self._debug_color(debug_mode, inp, dominant_id)

        return MaterialOutput(
            color        = color,
            roughness    = roughness,
            micro_normal = micro_normal,
            emissive     = Vec3.zero(),
            rock_type_id = dominant_id,
        )

    # ------------------------------------------------------------------
    # Masks
    # ------------------------------------------------------------------

    @staticmethod
    def _slope_mask(slope: float) -> float:
        """1 = steep, 0 = flat.  Dust prefers low slope_mask values."""
        return _smoothstep(0.10, 0.55, slope)

    @staticmethod
    def _concavity(curvature: float) -> float:
        """Map curvature to a [0, 1] concavity factor.

        Concave regions (negative curvature) → 1 (attract dust).
        Convex regions (positive curvature) → 0.
        """
        return _saturate(-curvature * 2.0)

    @staticmethod
    def _height_band(height: float) -> float:
        """Smooth band parameter in [0, 1] cycling by height layer."""
        band_raw = math.fmod(abs(height), _LAYER_STEP) / _LAYER_STEP
        # Smooth within each band
        return _smoothstep(0.0, 0.5, band_raw) * (1.0 - _smoothstep(0.5, 1.0, band_raw))

    # ------------------------------------------------------------------
    # Rock type weights
    # ------------------------------------------------------------------

    @staticmethod
    def _rock_weights(
        inp: MaterialInput,
        slope_mask: float,
        concavity: float,
        height_band_t: float,
    ) -> tuple[float, float, float, float, int]:
        """Return (w_dust, w_basalt, w_sediment, w_fracture, dominant_id).

        Weights sum to approximately 1.0 before dust overlay is applied
        in the blending stage.  No branching — all paths use smoothstep/lerp.
        """
        # --- BasaltRock: convergent zones + high hardness + steep ---
        w_basalt = _saturate(
            _smoothstep(0.6, 1.0, inp.hardness) * slope_mask +
            (1.0 if inp.boundary_type == BoundaryType.CONVERGENT else 0.0) * 0.5
        )

        # --- FracturedRock: high fracture field ---
        w_fracture = _smoothstep(0.35, 0.85, inp.fracture)

        # --- LayeredSediment: divergent/rifts + band by height ---
        w_sediment_base = (
            1.0 if inp.boundary_type in (BoundaryType.DIVERGENT, BoundaryType.NONE)
            else 0.3
        )
        w_sediment = _saturate(
            w_sediment_base * height_band_t * (1.0 - inp.hardness * 0.5)
        )

        # --- DustLayer: flat + dusty + concave ---
        flat_factor  = 1.0 - slope_mask
        w_dust_base  = _saturate(flat_factor * (1.0 + concavity * 0.5))

        # --- Normalise so primary rock weights sum to 1 ---
        rock_sum = w_basalt + w_fracture + w_sediment + 1e-6
        if rock_sum > 1.0:
            scale = 1.0 / rock_sum
            w_basalt   *= scale
            w_fracture *= scale
            w_sediment *= scale
        # Dust does not compete at normalisation — it is applied as an overlay later
        w_dust = w_dust_base   # kept separate

        # Dominant rock type for debug channel
        weights_rock = [w_dust, w_basalt, w_sediment, w_fracture]
        dominant_id  = weights_rock.index(max(weights_rock))

        return w_dust, w_basalt, w_sediment, w_fracture, dominant_id

    # ------------------------------------------------------------------
    # Colour / roughness blend
    # ------------------------------------------------------------------

    @staticmethod
    def _blend_rock(
        w_dust: float,
        w_basalt: float,
        w_sediment: float,
        w_fracture: float,
    ) -> tuple[Vec3, float]:
        """Linearly blend rock colours and roughness values."""
        # Sediment uses height_band_t to alternate between two tones
        # (here we mix the two sediment tones in a fixed 0.5 ratio for the
        # base blend; the height_band_t variation is baked into w_sediment)
        sediment_color = _lerp_v(_COLOR_SEDIMENT_A, _COLOR_SEDIMENT_B, 0.5)

        total_w = w_dust + w_basalt + w_sediment + w_fracture
        if total_w < 1e-6:
            return _COLOR_BASALT, _ROUGH_BASALT

        inv = 1.0 / total_w
        wd = w_dust     * inv
        wb = w_basalt   * inv
        ws = w_sediment * inv
        wf = w_fracture * inv

        color = (
            _COLOR_DUST      * wd +
            _COLOR_BASALT    * wb +
            sediment_color   * ws +
            _COLOR_FRACTURE  * wf
        )
        roughness = (
            _ROUGH_DUST     * wd +
            _ROUGH_BASALT   * wb +
            _ROUGH_SEDIMENT * ws +
            _ROUGH_FRACTURE * wf
        )
        return color, roughness

    # ------------------------------------------------------------------
    # Micro-normal
    # ------------------------------------------------------------------

    @staticmethod
    def _micro_normal(
        world_pos: Vec3,
        world_normal: Vec3,
        strength: float,
    ) -> Vec3:
        """Perturb *world_normal* using 3D noise gradient.

        Maximum amplitude is capped to 0.25 (spec §8).
        Uses only 2 octaves to avoid high-frequency aliasing.
        """
        strength = _clamp(strength, 0.0, 0.25)
        if strength < 1e-4:
            n = world_normal.length()
            return world_normal / n if n > 1e-9 else Vec3(0.0, 1.0, 0.0)

        p = world_pos * _MICRO_SCALE
        grad = gradient3(p)
        perturbed = world_normal + grad * strength
        n = perturbed.length()
        if n < 1e-9:
            return Vec3(0.0, 1.0, 0.0)
        return perturbed / n

    # ------------------------------------------------------------------
    # Debug colour
    # ------------------------------------------------------------------

    @staticmethod
    def _debug_color(
        mode: DebugMode,
        inp: MaterialInput,
        dominant_id: int,
    ) -> Vec3:
        if mode == DebugMode.DEBUG_SLOPE:
            return _heatmap(inp.slope)
        if mode == DebugMode.DEBUG_DUST:
            return _heatmap(inp.dust)
        if mode == DebugMode.DEBUG_ROCKTYPE:
            idx = _clamp(dominant_id, 0, len(_ROCKTYPE_DEBUG_COLORS) - 1)
            return _ROCKTYPE_DEBUG_COLORS[int(idx)]
        if mode == DebugMode.DEBUG_FRACTURE:
            return _heatmap(inp.fracture)
        if mode == DebugMode.DEBUG_BOUNDARY:
            return _BOUNDARY_DEBUG_COLORS.get(inp.boundary_type, Vec3(0.5, 0.5, 0.5))
        if mode == DebugMode.DEBUG_BIOME:
            return _BIOME_DEBUG_COLORS.get(inp.biome_id, Vec3(0.5, 0.5, 0.5))
        if mode == DebugMode.DEBUG_ICE_OVERLAY:
            return _heatmap(inp.ice)
        if mode == DebugMode.DEBUG_DUST_OVERLAY:
            return _heatmap(inp.dust)
        # Fallback (should not be reached)
        return Vec3(1.0, 0.0, 1.0)
