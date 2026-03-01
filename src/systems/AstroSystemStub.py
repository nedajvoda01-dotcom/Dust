"""IAstroSystem stub — interface for future astronomical simulation."""
from __future__ import annotations

from abc import ABC, abstractmethod

from src.math.Vec3 import Vec3


class IAstroSystem(ABC):
    @abstractmethod
    def update(self, game_time: float) -> None: ...

    @abstractmethod
    def get_sun_directions(self) -> tuple[Vec3, Vec3]:
        """Returns (primary_sun_dir, secondary_sun_dir) as unit vectors."""
        ...

    @abstractmethod
    def get_ring_shadow_factor(self, pos: Vec3) -> float:
        """Shadow intensity from ring at surface position pos (0=no shadow, 1=full)."""
        ...


class AstroSystemStub(IAstroSystem):
    """Stub: returns constant values. Replace with real implementation in Stage 3+."""

    def update(self, game_time: float) -> None:
        pass

    def get_sun_directions(self) -> tuple[Vec3, Vec3]:
        return Vec3(0.7, 0.7, 0.0).normalized(), Vec3(-0.5, 0.5, 0.5).normalized()

    def get_ring_shadow_factor(self, pos: Vec3) -> float:
        return 0.0
