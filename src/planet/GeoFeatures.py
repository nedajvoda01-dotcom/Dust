"""GeoFeatures — geological feature seeds for SDF and event systems.

Generates deterministic structural features from tectonic data:
    * FaultLine   — arcs along transform boundaries
    * WeaknessZone — volumetric zones under rifts
    * VoidPocket  — potential void / cave seeds under rifts + high porosity

These structures are used by SDFGenerator to modulate material channel
values (hardness, fracture, porosity) without triggering actual
geo-events (those are Stage 9).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional

from src.math.Vec3 import Vec3
from src.planet.TectonicPlatesSystem import (
    BoundaryType,
    TectonicPlatesSystem,
)


# ---------------------------------------------------------------------------
# GeoFeatureSeed types
# ---------------------------------------------------------------------------

class GeoFeatureKind(Enum):
    FAULT_LINE    = auto()   # transform boundary arc
    WEAKNESS_ZONE = auto()   # rift / divergent weakness volume
    VOID_POCKET   = auto()   # potential cave pocket


@dataclass
class GeoFeatureSeed:
    """One geological feature seed anchored to the sphere surface."""
    kind:       GeoFeatureKind
    anchor_dir: Vec3           # unit direction of the feature's centre
    axis_dir:   Vec3           # unit tangent — feature extends along this arc
    arc_length: float          # approximate angular span (radians)
    width:      float          # angular half-width (radians)
    intensity:  float          # 0..1 — fracture / porosity multiplier


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

# Number of probe directions used to locate boundary feature anchors
_PROBE_COUNT = 1024


def generate_geo_features(
    system: TectonicPlatesSystem,
    seed: int,
) -> List[GeoFeatureSeed]:
    """
    Sample the tectonic field and emit a deterministic list of
    GeoFeatureSeed objects.

    Parameters
    ----------
    system : built TectonicPlatesSystem
    seed   : additional seed offset (typically same as world seed)
    """
    rng = random.Random(seed ^ 0xFEED_FA11)
    features: List[GeoFeatureSeed] = []

    if system.field is None:
        return features

    # --- collect boundary-zone anchors from the equirectangular grid ---
    transform_anchors: List[Vec3] = []
    divergent_anchors: List[Vec3] = []

    for direction, col, row in system.field._iter_directions():
        cell = system.field._cells[system.field._idx(col, row)]
        if cell.boundary_type == BoundaryType.TRANSFORM and cell.boundary_strength > 0.3:
            transform_anchors.append(direction)
        elif cell.boundary_type == BoundaryType.DIVERGENT and cell.boundary_strength > 0.25:
            divergent_anchors.append(direction)

    # Thin to at most ~64 anchors per type to avoid thousands of features
    def _thin(dirs: List[Vec3], max_n: int, rng: random.Random) -> List[Vec3]:
        if len(dirs) <= max_n:
            return dirs
        rng.shuffle(dirs)
        return dirs[:max_n]

    transform_anchors = _thin(transform_anchors, 64, rng)
    divergent_anchors = _thin(divergent_anchors, 64, rng)

    # --- FaultLine features from transform boundaries ---
    for anchor in transform_anchors:
        # Axis: tangent perpendicular to the radial direction
        ref = Vec3(0.0, 1.0, 0.0) if abs(anchor.x) < 0.9 else Vec3(0.0, 0.0, 1.0)
        axis = ref.cross(anchor).normalized()
        features.append(GeoFeatureSeed(
            kind       = GeoFeatureKind.FAULT_LINE,
            anchor_dir = anchor,
            axis_dir   = axis,
            arc_length = rng.uniform(0.1, 0.4),   # radians
            width      = rng.uniform(0.005, 0.015),
            intensity  = rng.uniform(0.4, 1.0),
        ))

    # --- WeaknessZone and VoidPocket from divergent boundaries ---
    for anchor in divergent_anchors:
        ref = Vec3(0.0, 1.0, 0.0) if abs(anchor.x) < 0.9 else Vec3(0.0, 0.0, 1.0)
        axis = ref.cross(anchor).normalized()
        features.append(GeoFeatureSeed(
            kind       = GeoFeatureKind.WEAKNESS_ZONE,
            anchor_dir = anchor,
            axis_dir   = axis,
            arc_length = rng.uniform(0.15, 0.5),
            width      = rng.uniform(0.02, 0.06),
            intensity  = rng.uniform(0.3, 0.9),
        ))
        # Occasional void pocket under rifts
        if rng.random() < 0.3:
            offset_angle = rng.uniform(-0.05, 0.05)
            # Slightly offset anchor along axis
            import math as _m
            oa = anchor + axis * _m.sin(offset_angle)
            oa_len = oa.length()
            if oa_len > 1e-9:
                oa = oa * (1.0 / oa_len)
            features.append(GeoFeatureSeed(
                kind       = GeoFeatureKind.VOID_POCKET,
                anchor_dir = oa,
                axis_dir   = axis,
                arc_length = rng.uniform(0.01, 0.05),
                width      = rng.uniform(0.01, 0.03),
                intensity  = rng.uniform(0.5, 1.0),
            ))

    return features


# ---------------------------------------------------------------------------
# GeoFeatureInfluence — fast query: how much does a direction feel each feature
# ---------------------------------------------------------------------------

def feature_influence(feature: GeoFeatureSeed, direction: Vec3) -> float:
    """
    Return influence ∈ [0, 1] of *feature* at *direction*.

    Uses angular distance from the feature arc; falls off with a
    Gaussian in the cross-axis and arc-axis directions.
    """
    # Distance from arc axis plane (chord distance approximation)
    cos_a = max(-1.0, min(1.0, direction.dot(feature.anchor_dir)))
    arc_dist = math.acos(cos_a)

    if arc_dist > feature.arc_length + feature.width * 4.0:
        return 0.0

    # Distance perpendicular to the axis direction in tangent plane
    # Project direction onto the (anchor, axis) plane
    along_axis = direction.dot(feature.axis_dir)
    perp_sq    = max(0.0, 1.0 - along_axis * along_axis)
    perp_dist  = math.sqrt(perp_sq)   # approx angular distance from arc spine

    t = perp_dist / (feature.width + 1e-9)
    influence = math.exp(-t * t * 2.0) * feature.intensity

    # Also attenuate if beyond the arc length
    if arc_dist > feature.arc_length:
        excess = (arc_dist - feature.arc_length) / (feature.width * 2.0 + 1e-9)
        influence *= math.exp(-excess * excess)

    return min(1.0, influence)
