"""Components — data bags attached to entities."""
from __future__ import annotations

from src.math.Vec3 import Vec3


class MeshComponent:
    """Reference to a mesh (by name/id). Actual mesh data lives in Renderer."""
    def __init__(self, mesh_id: str = "sphere") -> None:
        self.mesh_id: str = mesh_id
        self.visible: bool = True


class VelocityComponent:
    def __init__(self) -> None:
        self.linear: Vec3 = Vec3.zero()


class PlayerComponent:
    """Marker + movement state for the player entity."""
    def __init__(self) -> None:
        self.unit_dir: Vec3 = Vec3(0.0, 1.0, 0.0)  # current position on unit sphere
        self.speed: float = 0.0
        self.intent: Vec3 = Vec3.zero()              # WASD intent direction (world-space)
