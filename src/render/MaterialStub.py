"""MaterialStub — placeholder material descriptor (no textures)."""
from __future__ import annotations

from src.math.Vec3 import Vec3


class MaterialStub:
    """Simple colour + lighting stub. No textures."""
    def __init__(
        self,
        diffuse: Vec3 | None = None,
        emissive: Vec3 | None = None,
    ) -> None:
        self.diffuse: Vec3 = diffuse if diffuse is not None else Vec3(0.7, 0.6, 0.5)
        self.emissive: Vec3 = emissive if emissive is not None else Vec3.zero()

    # Preset materials
    @staticmethod
    def planet_surface() -> "MaterialStub":
        return MaterialStub(diffuse=Vec3(0.72, 0.58, 0.45))

    @staticmethod
    def debug_up() -> "MaterialStub":
        return MaterialStub(emissive=Vec3(0.0, 1.0, 0.0))

    @staticmethod
    def debug_forward() -> "MaterialStub":
        return MaterialStub(emissive=Vec3(0.0, 0.5, 1.0))

    @staticmethod
    def space_background() -> "MaterialStub":
        return MaterialStub(diffuse=Vec3(0.02, 0.02, 0.05))
