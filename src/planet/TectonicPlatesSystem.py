"""TectonicPlatesSystem — deterministic tectonic-plate model for Stage 5.

Overview
--------
* Generates N spherical Voronoi plates from a seed.
* Assigns each surface direction to a plate (SphericalVoronoi).
* Classifies plate-boundary type: convergent / divergent / transform
  (PlateBoundaryClassifier).
* Accumulates stress and fracture fields over simulation time.
* Produces stable, readable landforms used by PlanetHeightProvider:
    convergent  → mountain ridges
    divergent   → rift valleys
    transform   → canyons / fault scars

All generation is deterministic for a given (seed, config) pair.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

from src.math.Vec3 import Vec3

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class CrustType(Enum):
    CONTINENTAL = auto()   # thick, buoyant
    OCEANIC     = auto()   # thin, dense


class BoundaryType(Enum):
    NONE        = auto()   # well inside a plate
    CONVERGENT  = auto()   # plates collide → ridges / mountains
    DIVERGENT   = auto()   # plates pull apart → rifts
    TRANSFORM   = auto()   # plates slide past each other → canyons


# ---------------------------------------------------------------------------
# Plate
# ---------------------------------------------------------------------------

@dataclass
class Plate:
    """One tectonic plate."""
    id:              int
    center_dir:      Vec3         # unit vector to plate centre on sphere
    velocity_tangent: Vec3        # tangent velocity vector at center_dir
    crust_type:      CrustType
    strength:        float        # 0..1; higher = harder to deform


# ---------------------------------------------------------------------------
# SphericalVoronoi — lightweight utility
# ---------------------------------------------------------------------------

class SphericalVoronoi:
    """
    Generates and queries spherical Voronoi cells from a set of seed
    directions.

    Voronoi cells are defined by *nearest-centre angular distance*
    — no polygon construction is needed.
    """

    def __init__(self, centers: List[Vec3]) -> None:
        self._centers: List[Vec3] = centers

    # ------------------------------------------------------------------
    @staticmethod
    def generate_centers(seed: int, count: int) -> List[Vec3]:
        """
        Produce *count* well-distributed unit-sphere directions from *seed*.

        Uses Mitchell's best-candidate: tries K candidates and keeps the
        one furthest from all already-placed centres.
        """
        rng = random.Random(seed ^ 0x7EC8_3A1B)
        K = max(10, count * 4)

        def _rand_dir() -> Vec3:
            while True:
                x = rng.uniform(-1.0, 1.0)
                y = rng.uniform(-1.0, 1.0)
                z = rng.uniform(-1.0, 1.0)
                sq = x * x + y * y + z * z
                if 0.001 < sq <= 1.0:
                    s = math.sqrt(sq)
                    return Vec3(x / s, y / s, z / s)

        centers: List[Vec3] = []
        for _ in range(count):
            best: Optional[Vec3] = None
            best_min_dist = -1.0
            for _ in range(K):
                cand = _rand_dir()
                if not centers:
                    best = cand
                    break
                # Angular distance as acos of dot product
                min_dist = min(
                    math.acos(max(-1.0, min(1.0, cand.dot(c))))
                    for c in centers
                )
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best = cand
            centers.append(best)  # type: ignore[arg-type]
        return centers

    # ------------------------------------------------------------------
    def nearest(self, direction: Vec3) -> int:
        """Return index of the nearest centre to *direction*."""
        best_idx  = 0
        best_dot  = -2.0
        for i, c in enumerate(self._centers):
            d = direction.dot(c)
            if d > best_dot:
                best_dot = d
                best_idx = i
        return best_idx

    # ------------------------------------------------------------------
    def two_nearest(self, direction: Vec3) -> Tuple[int, int]:
        """Return indices (first, second) nearest to *direction*."""
        first_idx  = 0
        second_idx = 1
        first_dot  = -2.0
        second_dot = -2.0
        for i, c in enumerate(self._centers):
            d = direction.dot(c)
            if d > first_dot:
                second_dot = first_dot
                second_idx = first_idx
                first_dot  = d
                first_idx  = i
            elif d > second_dot:
                second_dot = d
                second_idx = i
        return first_idx, second_idx


# ---------------------------------------------------------------------------
# PlateBoundaryClassifier
# ---------------------------------------------------------------------------

class PlateBoundaryClassifier:
    """
    Classifies the tectonic boundary type for an arbitrary surface point.

    The classification uses the relative tangential velocity of the two
    nearest plates projected onto a local coordinate frame at the query
    point.
    """

    # Angular distance from the Voronoi edge at which "boundary zone" starts.
    BOUNDARY_THRESHOLD: float = 0.08   # radians ≈ 4.6 °

    def __init__(
        self,
        plates: List[Plate],
        voronoi: SphericalVoronoi,
    ) -> None:
        self._plates  = plates
        self._voronoi = voronoi

    # ------------------------------------------------------------------
    def classify(
        self,
        direction: Vec3,
    ) -> Tuple[BoundaryType, float, int, int]:
        """
        Return ``(boundary_type, boundary_strength, plate_a_id, plate_b_id)``.

        *boundary_strength* is 0 for interior points; in boundary zones it
        reflects the magnitude of relative plate motion (0..1 normalised).
        """
        idx_a, idx_b = self._voronoi.two_nearest(direction)
        plate_a = self._plates[idx_a]
        plate_b = self._plates[idx_b]

        # Angular distances to both centres
        dot_a = max(-1.0, min(1.0, direction.dot(plate_a.center_dir)))
        dot_b = max(-1.0, min(1.0, direction.dot(plate_b.center_dir)))
        dist_a = math.acos(dot_a)
        dist_b = math.acos(dot_b)

        # "Voronoi proximity" — difference in distances to the two nearest
        # centres.  Near zero → near the edge.
        d_diff = dist_b - dist_a   # ≥ 0 by construction of two_nearest

        if d_diff > self.BOUNDARY_THRESHOLD:
            return BoundaryType.NONE, 0.0, idx_a, idx_b

        # --- project each plate's velocity tangent onto the query point ---
        def _project_velocity(plate: Plate, pt: Vec3) -> Vec3:
            """Project plate velocity (defined at centre) onto tangent plane at pt."""
            # Radial component at pt:
            radial = pt * pt.dot(plate.velocity_tangent)
            return plate.velocity_tangent - radial

        va = _project_velocity(plate_a, direction)
        vb = _project_velocity(plate_b, direction)

        rel = vb - va   # relative velocity of B w.r.t. A

        rel_len = rel.length()
        if rel_len < 1e-9:
            return BoundaryType.NONE, 0.0, idx_a, idx_b

        # Approximate boundary normal: points from centre_a to centre_b,
        # projected onto the tangent plane at direction.
        diff = plate_b.center_dir - plate_a.center_dir
        radial_diff = direction * direction.dot(diff)
        boundary_normal = (diff - radial_diff)
        bn_len = boundary_normal.length()
        if bn_len < 1e-9:
            # degenerate — centres nearly antipodal; treat as none
            return BoundaryType.NONE, 0.0, idx_a, idx_b
        boundary_normal = boundary_normal * (1.0 / bn_len)

        # Component of relative velocity along boundary normal
        normal_component = rel.dot(boundary_normal)
        # Component along boundary tangent
        boundary_tangent = direction.cross(boundary_normal).normalized()
        if boundary_tangent.is_near_zero():
            boundary_tangent = Vec3(0.0, 1.0, 0.0)
        tangent_component = rel.dot(boundary_tangent)

        # Blend factor: at d_diff=0 → full boundary; at threshold → 0
        blend = 1.0 - d_diff / self.BOUNDARY_THRESHOLD
        strength = blend * rel_len

        # Classification: dominant component determines type
        if abs(normal_component) >= abs(tangent_component):
            btype = (
                BoundaryType.CONVERGENT if normal_component < 0.0
                else BoundaryType.DIVERGENT
            )
        else:
            btype = BoundaryType.TRANSFORM

        return btype, min(1.0, strength), idx_a, idx_b


# ---------------------------------------------------------------------------
# PlateField — low-resolution spherical field
# ---------------------------------------------------------------------------

@dataclass
class PlateFieldCell:
    """One cell of the PlateField grid (equirectangular 512×256)."""
    plate_id:         int   = 0
    boundary_type:    BoundaryType = BoundaryType.NONE
    boundary_strength: float = 0.0
    stress:           float = 0.0
    fracture:         float = 0.0
    hardness:         float = 1.0


class PlateField:
    """
    Equirectangular grid of PlateFieldCells covering the whole sphere.

    Grid coordinates:
        longitude θ  ∈ [-π, π]     — cols 0..W-1
        latitude  φ  ∈ [-π/2, π/2] — rows 0..H-1

    Resolution is intentionally coarse (128×64 default) so that
    the tectonic influence manifests as large-scale geology, not
    high-frequency noise.
    """

    def __init__(self, width: int = 128, height: int = 64) -> None:
        self.W = width
        self.H = height
        self._cells: List[PlateFieldCell] = [
            PlateFieldCell() for _ in range(width * height)
        ]

    # ------------------------------------------------------------------
    def _idx(self, col: int, row: int) -> int:
        col = col % self.W
        row = max(0, min(self.H - 1, row))
        return row * self.W + col

    def _dir_to_cr(self, direction: Vec3) -> Tuple[int, int]:
        """Map unit direction → (col, row) in the equirectangular grid."""
        lon = math.atan2(direction.y, direction.x)           # [-π, π]
        lat = math.asin(max(-1.0, min(1.0, direction.z)))    # [-π/2, π/2]
        col = int((lon + math.pi) / (2.0 * math.pi) * self.W) % self.W
        row = int((lat + math.pi * 0.5) / math.pi * self.H)
        row = max(0, min(self.H - 1, row))
        return col, row

    # ------------------------------------------------------------------
    def get_cell(self, direction: Vec3) -> PlateFieldCell:
        col, row = self._dir_to_cr(direction)
        return self._cells[self._idx(col, row)]

    def set_cell(self, direction: Vec3, cell: PlateFieldCell) -> None:
        col, row = self._dir_to_cr(direction)
        self._cells[self._idx(col, row)] = cell

    # ------------------------------------------------------------------
    def _iter_directions(self):
        """Yield (Vec3, col, row) for every cell centre."""
        for row in range(self.H):
            lat = ((row + 0.5) / self.H) * math.pi - math.pi * 0.5
            for col in range(self.W):
                lon = ((col + 0.5) / self.W) * 2.0 * math.pi - math.pi
                x = math.cos(lat) * math.cos(lon)
                y = math.cos(lat) * math.sin(lon)
                z = math.sin(lat)
                yield Vec3(x, y, z), col, row

    # ------------------------------------------------------------------
    def build(self, plates: List[Plate], classifier: PlateBoundaryClassifier) -> None:
        """Populate every cell from the voronoi/classifier."""
        for direction, col, row in self._iter_directions():
            btype, bstrength, idx_a, _ = classifier.classify(direction)
            plate = plates[idx_a]
            cell = PlateFieldCell(
                plate_id          = idx_a,
                boundary_type     = btype,
                boundary_strength = bstrength,
                stress            = 0.0,
                fracture          = 0.0,
                hardness          = plate.strength,
            )
            self._cells[self._idx(col, row)] = cell

    # ------------------------------------------------------------------
    def update(self, dt: float, stress_rate: float, fracture_rate: float) -> None:
        """Accumulate stress and fracture on boundary cells; relax interior cells."""
        for cell in self._cells:
            if cell.boundary_type != BoundaryType.NONE:
                cell.stress = min(1.0, cell.stress + cell.boundary_strength * dt * stress_rate)
                cell.fracture = min(1.0, cell.fracture + cell.stress * dt * fracture_rate)
            else:
                cell.stress = max(0.0, cell.stress - dt * stress_rate * 0.3)


# ---------------------------------------------------------------------------
# TectonicPlatesSystem
# ---------------------------------------------------------------------------

class TectonicPlatesSystem:
    """
    Top-level geological simulation system.

    Usage::

        system = TectonicPlatesSystem(seed=42, plate_count=18)
        system.build()
        system.update(dt=0.016)   # call each game tick
        sampler = system.get_field_sampler()  # → GeoFieldSampler
    """

    # Config defaults (overridable via constructor kwargs)
    DEFAULT_PLATE_COUNT:   int   = 18
    DEFAULT_STRESS_RATE:   float = 0.01
    DEFAULT_FRACTURE_RATE: float = 0.02
    DEFAULT_FIELD_W:       int   = 128
    DEFAULT_FIELD_H:       int   = 64

    def __init__(
        self,
        seed: int,
        plate_count: int      = DEFAULT_PLATE_COUNT,
        stress_rate: float    = DEFAULT_STRESS_RATE,
        fracture_rate: float  = DEFAULT_FRACTURE_RATE,
        field_width: int      = DEFAULT_FIELD_W,
        field_height: int     = DEFAULT_FIELD_H,
    ) -> None:
        self.seed          = seed
        self.plate_count   = plate_count
        self.stress_rate   = stress_rate
        self.fracture_rate = fracture_rate
        self._field_w      = field_width
        self._field_h      = field_height

        self.plates:     List[Plate]             = []
        self.voronoi:    Optional[SphericalVoronoi]          = None
        self.classifier: Optional[PlateBoundaryClassifier]  = None
        self.field:      Optional[PlateField]                = None

    # ------------------------------------------------------------------
    def build(self) -> None:
        """Generate plates, build Voronoi, classify field cells."""
        rng = random.Random(self.seed ^ 0xC0FFEE_1234)

        centers = SphericalVoronoi.generate_centers(self.seed, self.plate_count)
        self.voronoi = SphericalVoronoi(centers)

        self.plates = []
        for i, c in enumerate(centers):
            # Tangent velocity: random vector in tangent plane at c
            ref = Vec3(0.0, 1.0, 0.0) if abs(c.x) < 0.9 else Vec3(0.0, 0.0, 1.0)
            t1 = ref.cross(c).normalized()
            speed  = rng.uniform(0.1, 1.0)
            angle  = rng.uniform(0.0, 2.0 * math.pi)
            t2 = c.cross(t1).normalized()
            vel = t1 * (speed * math.cos(angle)) + t2 * (speed * math.sin(angle))
            crust  = CrustType.CONTINENTAL if rng.random() < 0.55 else CrustType.OCEANIC
            strength = rng.uniform(0.4, 1.0)
            self.plates.append(Plate(
                id              = i,
                center_dir      = c,
                velocity_tangent = vel,
                crust_type      = crust,
                strength        = strength,
            ))

        self.classifier = PlateBoundaryClassifier(self.plates, self.voronoi)
        self.field = PlateField(self._field_w, self._field_h)
        self.field.build(self.plates, self.classifier)

    # ------------------------------------------------------------------
    def update(self, dt: float) -> None:
        """Advance stress/fracture accumulation by *dt* seconds."""
        if self.field is not None:
            self.field.update(dt, self.stress_rate, self.fracture_rate)

    # ------------------------------------------------------------------
    def sample_plate_id(self, direction: Vec3) -> int:
        """Plate id for *direction*."""
        if self.voronoi is None:
            return 0
        return self.voronoi.nearest(direction)

    # ------------------------------------------------------------------
    def sample_boundary(self, direction: Vec3) -> Tuple[BoundaryType, float]:
        """(BoundaryType, boundary_strength) for *direction*."""
        if self.classifier is None:
            return BoundaryType.NONE, 0.0
        btype, strength, _, _ = self.classifier.classify(direction)
        return btype, strength

    # ------------------------------------------------------------------
    def sample_field_cell(self, direction: Vec3) -> Optional[PlateFieldCell]:
        """Return the low-resolution field cell for *direction*."""
        if self.field is None:
            return None
        return self.field.get_cell(direction)

    # ------------------------------------------------------------------
    def get_field_sampler(self):
        """Return a GeoFieldSampler wrapping this system (import is deferred)."""
        from src.planet.GeoFieldSampler import GeoFieldSampler
        return GeoFieldSampler(self)
