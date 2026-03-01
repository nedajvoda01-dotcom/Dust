"""IGeoEventSystem stub — interface for future geological event simulation."""
from __future__ import annotations

from abc import ABC, abstractmethod

from src.math.Vec3 import Vec3


class IGeoEventSystem(ABC):
    @abstractmethod
    def update(self, game_time: float) -> None: ...

    @abstractmethod
    def query_ground_stability(self, pos: Vec3) -> float:
        """Ground stability at pos (0=unstable, 1=stable)."""
        ...


class GeoEventSystemStub(IGeoEventSystem):
    def update(self, game_time: float) -> None:
        pass

    def query_ground_stability(self, pos: Vec3) -> float:
        return 1.0
