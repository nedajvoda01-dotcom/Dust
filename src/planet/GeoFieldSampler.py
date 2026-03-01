"""GeoFieldSampler — high-level query interface for geological fields.

Provides a single sampling point for any surface direction:
    * plateId
    * boundaryType
    * stress
    * fracture
    * stability
    * hardness

This is the primary interface used by PlanetHeightProvider,
SDFGenerator, and GeoEventSystem.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.math.Vec3 import Vec3
from src.planet.TectonicPlatesSystem import (
    BoundaryType,
    TectonicPlatesSystem,
)


# ---------------------------------------------------------------------------
# GeoSample — returned by GeoFieldSampler.sample()
# ---------------------------------------------------------------------------

@dataclass
class GeoSample:
    plate_id:          int
    boundary_type:     BoundaryType
    boundary_strength: float
    stress:            float   # 0..1
    fracture:          float   # 0..1
    stability:         float   # 0..1  (1 = fully stable)
    hardness:          float   # 0..1


# ---------------------------------------------------------------------------
# GeoFieldSampler
# ---------------------------------------------------------------------------

class GeoFieldSampler:
    """
    Thin façade over TectonicPlatesSystem that returns a rich GeoSample
    for any surface direction.

    Stability is derived as:
        stability = 1 - clamp(fracture * 0.7 + stress * 0.3, 0, 1)

    Hardness comes from the PlateField cell (plate's intrinsic strength,
    decreased by fracture).
    """

    def __init__(self, system: TectonicPlatesSystem) -> None:
        self._sys = system

    # ------------------------------------------------------------------
    def sample(self, direction: Vec3) -> GeoSample:
        """Return full geological fields for *direction* (unit vector)."""
        cell = self._sys.sample_field_cell(direction)
        if cell is None:
            return GeoSample(
                plate_id          = 0,
                boundary_type     = BoundaryType.NONE,
                boundary_strength = 0.0,
                stress            = 0.0,
                fracture          = 0.0,
                stability         = 1.0,
                hardness          = 1.0,
            )

        stress   = cell.stress
        fracture = cell.fracture
        hardness = max(0.0, cell.hardness - fracture * 0.4)
        stability = max(0.0, 1.0 - (fracture * 0.7 + stress * 0.3))

        return GeoSample(
            plate_id          = cell.plate_id,
            boundary_type     = cell.boundary_type,
            boundary_strength = cell.boundary_strength,
            stress            = stress,
            fracture          = fracture,
            stability         = stability,
            hardness          = hardness,
        )

    # ------------------------------------------------------------------
    # Convenience attribute-style accessors
    # ------------------------------------------------------------------

    def plate_id(self, direction: Vec3) -> int:
        return self._sys.sample_plate_id(direction)

    def boundary_type(self, direction: Vec3) -> BoundaryType:
        cell = self._sys.sample_field_cell(direction)
        return cell.boundary_type if cell else BoundaryType.NONE

    def stress(self, direction: Vec3) -> float:
        cell = self._sys.sample_field_cell(direction)
        return cell.stress if cell else 0.0

    def fracture(self, direction: Vec3) -> float:
        cell = self._sys.sample_field_cell(direction)
        return cell.fracture if cell else 0.0

    def stability(self, direction: Vec3) -> float:
        return self.sample(direction).stability

    def hardness(self, direction: Vec3) -> float:
        return self.sample(direction).hardness
