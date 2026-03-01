"""BiomeAtlasSystem — Stage 28 procedural biome/surface classification.

No texture files are used.  All biome selection is based on deterministic
noise functions evaluated from world-space inputs: latitude, longitude,
height, geological fields, climate, and planet seed.

Public API
----------
BiomeId          — enum of surface/cave biome types
BiomeParams      — per-biome material parameters
OverlayState     — transient overlay layers (dust, ice, debris, wetness)
BiomeAtlas       — main system; get_biome() and get_overlays()
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from src.math.Vec3 import Vec3
from src.render.ProceduralNoise import fbm3, noise3

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


def _saturate(v: float) -> float:
    return _clamp(v, 0.0, 1.0)


def _smoothstep(edge0: float, edge1: float, x: float) -> float:
    t = _saturate((x - edge0) / (edge1 - edge0) if edge1 != edge0 else 0.0)
    return t * t * (3.0 - 2.0 * t)


def _lerp(a: float, b: float, t: float) -> float:
    return a + t * (b - a)


def _lerp_v(a: Vec3, b: Vec3, t: float) -> Vec3:
    return a + (b - a) * t


# ---------------------------------------------------------------------------
# BiomeId
# ---------------------------------------------------------------------------

class BiomeId(Enum):
    """Surface and cave biome identifiers (extensible)."""
    IRON_SAND         = auto()   # red-iron sand plains
    BASALT_PLATEAU    = auto()   # dark basalt plateaus
    ASH_FIELDS        = auto()   # volcanic ash fields
    SALT_FLATS        = auto()   # bright salt/mineral flats
    GLASSY_IMPACT     = auto()   # glassy impact-melt zones
    ICE_CRUST         = auto()   # frost / ice crust
    SCREE_SLOPES      = auto()   # rocky debris slopes
    FRACTURE_BANDS    = auto()   # tectonic fracture belts
    SUBSURFACE_BEDROCK = auto()  # cave walls / deep rock


# ---------------------------------------------------------------------------
# BiomeParams
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BiomeParams:
    """Material parameters for a biome.

    base_albedo      — linear-space RGB base colour
    roughness_min    — minimum PBR roughness
    roughness_max    — maximum PBR roughness
    micro_normal_str — world-space noise perturbation amplitude [0, 0.25]
    dust_retention   — how readily dust accumulates [0, 1]
    ice_affinity     — how readily ice/frost forms [0, 1]
    debris_propensity — likelihood of scree/debris overlay [0, 1]
    """
    base_albedo:       Vec3
    roughness_min:     float
    roughness_max:     float
    micro_normal_str:  float
    dust_retention:    float
    ice_affinity:      float
    debris_propensity: float


# Default per-biome parameters
BIOME_PARAMS: dict[BiomeId, BiomeParams] = {
    BiomeId.IRON_SAND: BiomeParams(
        base_albedo       = Vec3(0.62, 0.32, 0.22),
        roughness_min     = 0.75, roughness_max = 0.90,
        micro_normal_str  = 0.08,
        dust_retention    = 0.80, ice_affinity = 0.10, debris_propensity = 0.20,
    ),
    BiomeId.BASALT_PLATEAU: BiomeParams(
        base_albedo       = Vec3(0.18, 0.17, 0.16),
        roughness_min     = 0.60, roughness_max = 0.72,
        micro_normal_str  = 0.15,
        dust_retention    = 0.30, ice_affinity = 0.15, debris_propensity = 0.35,
    ),
    BiomeId.ASH_FIELDS: BiomeParams(
        base_albedo       = Vec3(0.30, 0.28, 0.27),
        roughness_min     = 0.78, roughness_max = 0.92,
        micro_normal_str  = 0.06,
        dust_retention    = 0.90, ice_affinity = 0.10, debris_propensity = 0.15,
    ),
    BiomeId.SALT_FLATS: BiomeParams(
        base_albedo       = Vec3(0.88, 0.85, 0.80),
        roughness_min     = 0.50, roughness_max = 0.65,
        micro_normal_str  = 0.04,
        dust_retention    = 0.55, ice_affinity = 0.30, debris_propensity = 0.05,
    ),
    BiomeId.GLASSY_IMPACT: BiomeParams(
        base_albedo       = Vec3(0.25, 0.28, 0.26),
        roughness_min     = 0.15, roughness_max = 0.35,
        micro_normal_str  = 0.10,
        dust_retention    = 0.20, ice_affinity = 0.20, debris_propensity = 0.45,
    ),
    BiomeId.ICE_CRUST: BiomeParams(
        base_albedo       = Vec3(0.75, 0.84, 0.95),
        roughness_min     = 0.12, roughness_max = 0.28,
        micro_normal_str  = 0.04,
        dust_retention    = 0.15, ice_affinity = 0.95, debris_propensity = 0.05,
    ),
    BiomeId.SCREE_SLOPES: BiomeParams(
        base_albedo       = Vec3(0.45, 0.40, 0.36),
        roughness_min     = 0.80, roughness_max = 0.95,
        micro_normal_str  = 0.20,
        dust_retention    = 0.20, ice_affinity = 0.20, debris_propensity = 0.85,
    ),
    BiomeId.FRACTURE_BANDS: BiomeParams(
        base_albedo       = Vec3(0.20, 0.18, 0.17),
        roughness_min     = 0.70, roughness_max = 0.85,
        micro_normal_str  = 0.22,
        dust_retention    = 0.25, ice_affinity = 0.12, debris_propensity = 0.60,
    ),
    BiomeId.SUBSURFACE_BEDROCK: BiomeParams(
        base_albedo       = Vec3(0.22, 0.20, 0.19),
        roughness_min     = 0.65, roughness_max = 0.80,
        micro_normal_str  = 0.18,
        dust_retention    = 0.10, ice_affinity = 0.05, debris_propensity = 0.40,
    ),
}


# ---------------------------------------------------------------------------
# OverlayState
# ---------------------------------------------------------------------------

@dataclass
class OverlayState:
    """Transient environmental overlay thicknesses.

    All values in [0, 1].  They are derived from climate state and are NOT
    part of the deterministic biome map — they change with climate events.
    """
    dust_thickness:  float = 0.0   # dust deposition / storm blow
    ice_film:        float = 0.0   # frost / ice crust
    debris_thickness: float = 0.0  # rockfall / scree after events
    wetness:         float = 0.0   # surface moisture


# ---------------------------------------------------------------------------
# GeoInput / ClimateInput — lightweight containers for callers
# ---------------------------------------------------------------------------

@dataclass
class GeoInput:
    """Geological properties at a surface point."""
    fracture:       float = 0.0    # [0, 1]
    hardness:       float = 0.5    # [0, 1]
    slope:          float = 0.0    # [0, 1]  1 = vertical
    height:         float = 0.0    # world-space radial height (m)
    is_convergent:  bool  = False
    is_divergent:   bool  = False
    is_transform:   bool  = False
    is_cave:        bool  = False


@dataclass
class ClimateInput:
    """Climate snapshot used for overlay computation."""
    temperature:      float = 290.0   # K
    storm_intensity:  float = 0.0     # [0, 1]
    dust_suspension:  float = 0.0     # [0, 1]  from ClimateSystem
    ring_shadow:      float = 0.0     # [0, 1]  0=none 1=full shadow
    wetness:          float = 0.0     # [0, 1]


# ---------------------------------------------------------------------------
# BiomeAtlas
# ---------------------------------------------------------------------------

# Noise scales (world-space / rad, unitless for lat-lon)
_MACRO_SCALE = 0.08    # low-frequency  (~50-500 km equivalent)
_MESO_SCALE  = 0.80    # medium-frequency (~1-20 km)
_DOMAIN_WARP = 0.15    # domain warp to prevent grid artefacts

# Thresholds / weights
_FRACTURE_BAND_THRESHOLD = 0.55   # fracture above this → FRACTURE_BANDS
_SCREE_SLOPE_THRESHOLD   = 0.65   # slope above this  → SCREE_SLOPES
_ICE_TEMP_THRESHOLD      = 220.0  # K — below this ice crust is common
_SALT_HEIGHT_MAX         = -50.0  # depth below baseline → salt flats likely
_IMPACT_NOISE_THRESHOLD  = 0.88   # rare Worley-like speckle for impact zones

# Overlay constants
_DUST_MAX               = 1.0
_ICE_MAX                = 1.0
_DEBRIS_MAX             = 1.0
_DUST_DEPOSITION_RATE   = 0.6    # fraction retained in depressions
_DUST_EROSION_RATE_WIND = 0.8    # fraction removed on steep slopes in wind


class BiomeAtlas:
    """Deterministic biome classifier.

    Parameters
    ----------
    seed : int
        Planet generation seed — same seed → same biome map.
    """

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        # Seed-derived offsets to break axis symmetry
        s = seed & 0xFFFF
        self._offset_x = (((s * 1664525 + 1013904223) & 0x7FFFFFFF) / 2147483648.0) * 100.0
        self._offset_y = (((s * 22695477 + 1) & 0x7FFFFFFF) / 2147483648.0) * 100.0
        self._offset_z = (((s * 6364136223846793005 + 1442695040888963407) & 0x7FFFFFFF) / 2147483648.0) * 100.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_biome(
        self,
        lat: float,
        lon: float,
        geo: GeoInput,
        climate: Optional[ClimateInput] = None,
    ) -> BiomeId:
        """Return the dominant BiomeId for a surface or cave point.

        Parameters
        ----------
        lat : float
            Latitude in radians [-π/2, π/2].
        lon : float
            Longitude in radians [-π, π].
        geo : GeoInput
            Geological properties.
        climate : ClimateInput | None
            Current climate state (used only for ICE_CRUST).
        """
        if geo.is_cave:
            return self._cave_biome(lat, lon, geo)

        # --- Domain warp to avoid grid-aligned boundaries ---
        p_macro = self._macro_point(lat, lon)
        warp = Vec3(
            fbm3(p_macro + Vec3(13.7, 0.0, 0.0), octaves=2) - 0.5,
            fbm3(p_macro + Vec3(0.0, 17.3, 0.0), octaves=2) - 0.5,
            fbm3(p_macro + Vec3(0.0, 0.0, 5.1),  octaves=2) - 0.5,
        )
        p_warped = p_macro + warp * _DOMAIN_WARP

        macro_n = fbm3(p_warped, octaves=2)            # [0, 1]
        meso_n  = fbm3(self._meso_point(lat, lon), octaves=2)

        scores: dict[BiomeId, float] = {}

        # CAVE interior → excluded above; surface scores follow
        temp = climate.temperature if climate else 290.0
        ring_shadow = climate.ring_shadow if climate else 0.0

        # --- FRACTURE_BANDS ---
        scores[BiomeId.FRACTURE_BANDS] = (
            _smoothstep(_FRACTURE_BAND_THRESHOLD - 0.1, 1.0, geo.fracture) * 1.5 +
            (0.5 if (geo.is_convergent or geo.is_transform) else 0.0) +
            _smoothstep(0.3, 0.8, meso_n) * 0.3
        )

        # --- BASALT_PLATEAU ---
        scores[BiomeId.BASALT_PLATEAU] = (
            _smoothstep(0.55, 1.0, geo.hardness) * 0.8 +
            (0.6 if geo.is_convergent else 0.0) +
            _smoothstep(0.25, 0.65, geo.slope) * 0.5 +
            (1.0 - macro_n) * 0.3
        )

        # --- ASH_FIELDS ---
        scores[BiomeId.ASH_FIELDS] = (
            (0.6 if geo.is_divergent else 0.0) +
            _smoothstep(0.4, 0.7, macro_n) * 0.5 +
            _smoothstep(0.0, 0.3, geo.fracture) * 0.3
        )

        # --- SALT_FLATS ---
        depression = _smoothstep(0.0, -200.0, geo.height) if geo.height < 0.0 else 0.0
        scores[BiomeId.SALT_FLATS] = (
            depression * 0.8 +
            (1.0 - geo.slope) * _smoothstep(0.45, 0.75, macro_n) * 0.6 +
            (1.0 - _smoothstep(0.2, 0.8, geo.fracture)) * 0.3
        )

        # --- GLASSY_IMPACT ---
        impact_n = noise3(p_macro * 5.3 + Vec3(self._offset_z, 0.0, 0.0))
        scores[BiomeId.GLASSY_IMPACT] = (
            _smoothstep(_IMPACT_NOISE_THRESHOLD, 1.0, impact_n) * 2.0 +
            _smoothstep(0.5, 0.9, 1.0 - meso_n) * 0.3
        )

        # --- SCREE_SLOPES ---
        scores[BiomeId.SCREE_SLOPES] = (
            _smoothstep(_SCREE_SLOPE_THRESHOLD - 0.1, 1.0, geo.slope) * 1.4 +
            _smoothstep(0.4, 0.8, geo.fracture) * 0.4
        )

        # --- IRON_SAND (default open plains) ---
        scores[BiomeId.IRON_SAND] = (
            (1.0 - geo.slope) * 0.6 +
            _smoothstep(0.35, 0.65, macro_n) * 0.5 +
            0.3   # baseline: this is the most common biome
        )

        # --- ICE_CRUST ---
        polar_factor = _smoothstep(math.pi * 0.3, math.pi * 0.5, abs(lat))
        cold_factor  = _smoothstep(_ICE_TEMP_THRESHOLD + 30.0, _ICE_TEMP_THRESHOLD - 20.0, temp)
        scores[BiomeId.ICE_CRUST] = (
            (polar_factor + cold_factor + ring_shadow * 0.5) * 0.7
        )

        # --- Pick winner ---
        best_biome = max(scores, key=lambda b: scores[b])
        return best_biome

    def get_overlays(
        self,
        biome: BiomeId,
        geo: GeoInput,
        climate: ClimateInput,
    ) -> OverlayState:
        """Compute transient overlay thicknesses from climate state.

        The returned values depend on (biome, geo, climate) and change
        as the climate evolves — they are NOT baked into the biome map.
        """
        params = BIOME_PARAMS[biome]

        # --- Dust ---
        # Storms increase dust; wind on slopes erodes it; low areas deposit more
        base_dust = climate.dust_suspension
        slope_erosion = _smoothstep(0.4, 0.9, geo.slope) * _DUST_EROSION_RATE_WIND
        concavity_bonus = _saturate(-geo.slope * 2.0 + 0.5)   # flatter = more deposition
        dust = _saturate(
            base_dust * params.dust_retention * (1.0 + concavity_bonus * _DUST_DEPOSITION_RATE)
            - slope_erosion * climate.storm_intensity * 0.3
        )

        # --- Ice ---
        temp_factor = _smoothstep(
            _ICE_TEMP_THRESHOLD + 20.0,
            _ICE_TEMP_THRESHOLD - 30.0,
            climate.temperature,
        )
        ice = _saturate(
            temp_factor * params.ice_affinity +
            climate.ring_shadow * params.ice_affinity * 0.4
        ) * _ICE_MAX

        # Debris is driven by external events (collapses, rockfalls).
        # Base value is 0; callers update it after event-driven triggers.
        debris = 0.0

        # --- Wetness ---
        wetness = climate.wetness * (1.0 - _smoothstep(0.3, 1.0, geo.slope))

        return OverlayState(
            dust_thickness   = _saturate(dust * _DUST_MAX),
            ice_film         = _saturate(ice),
            debris_thickness = _saturate(debris * _DEBRIS_MAX),
            wetness          = _saturate(wetness),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _macro_point(self, lat: float, lon: float) -> Vec3:
        """Convert lat/lon to a 3D noise point at macro scale."""
        return Vec3(
            lat  * _MACRO_SCALE + self._offset_x,
            lon  * _MACRO_SCALE + self._offset_y,
            0.5 * _MACRO_SCALE + self._offset_z,
        )

    def _meso_point(self, lat: float, lon: float) -> Vec3:
        """Convert lat/lon to a 3D noise point at meso scale."""
        return Vec3(
            lat  * _MESO_SCALE + self._offset_y,
            lon  * _MESO_SCALE + self._offset_z,
            1.3 * _MESO_SCALE + self._offset_x,
        )

    def _cave_biome(self, lat: float, lon: float, geo: GeoInput) -> BiomeId:
        """Select biome for subsurface/cave regions."""
        p = self._macro_point(lat, lon) * 3.0
        cave_n = fbm3(p + Vec3(7.7, 3.3, 5.5), octaves=2)
        if cave_n > 0.72:
            return BiomeId.GLASSY_IMPACT      # glassy pocket
        if geo.fracture > 0.5 or cave_n > 0.55:
            return BiomeId.ASH_FIELDS         # ashy cave zone
        return BiomeId.SUBSURFACE_BEDROCK     # default bedrock
