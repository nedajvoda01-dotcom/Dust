"""IClimateSystem stub — interface for future climate simulation."""
from __future__ import annotations

from abc import ABC, abstractmethod

from src.math.Vec3 import Vec3


class IClimateSystem(ABC):
    @abstractmethod
    def update(self, game_time: float) -> None: ...

    @abstractmethod
    def sample_wind(self, pos: Vec3) -> Vec3:
        """Wind vector at surface position."""
        ...

    @abstractmethod
    def sample_dust(self, pos: Vec3) -> float:
        """Dust suspension (0–1) at surface position."""
        ...


class ClimateSystemStub(IClimateSystem):
    def update(self, game_time: float) -> None:
        pass

    def sample_wind(self, pos: Vec3) -> Vec3:
        return Vec3(4.5, 0.0, 0.0)

    def sample_dust(self, pos: Vec3) -> float:
        return 0.12
