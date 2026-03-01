"""ISubsurfaceSystem stub — interface for the Stage 26 subsurface world."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from src.math.Vec3 import Vec3


class ISubsurfaceSystem(ABC):
    """Minimal interface implemented by SubsurfaceSystem (and this stub)."""

    @abstractmethod
    def update(self, dt: float, game_time: float, player_pos: "Vec3 | None") -> None:
        """Advance simulation by *dt* seconds."""
        ...

    @abstractmethod
    def cave_factor_at(self, world_pos: Vec3) -> float:
        """Return cave-atmosphere factor ∈ [0, 1] at *world_pos*.

        0 = fully on the open surface;  1 = deep underground.
        """
        ...

    @abstractmethod
    def portals(self) -> List[Vec3]:
        """Return world-space unit directions of all known surface entry points."""
        ...


class SubsurfaceSystemStub(ISubsurfaceSystem):
    def update(self, dt: float, game_time: float, player_pos: "Vec3 | None") -> None:
        pass

    def cave_factor_at(self, world_pos: Vec3) -> float:
        return 0.0

    def portals(self) -> List[Vec3]:
        return []
