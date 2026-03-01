"""Transform — position + rotation + scale for a scene entity."""
from __future__ import annotations

from src.math.Vec3 import Vec3
from src.math.Quat import Quat
from src.math.Mat4 import Mat4


class Transform:
    __slots__ = ("position", "rotation", "scale")

    def __init__(
        self,
        position: Vec3 | None = None,
        rotation: Quat | None = None,
        scale: Vec3 | None = None,
    ) -> None:
        self.position: Vec3 = position if position is not None else Vec3.zero()
        self.rotation: Quat = rotation if rotation is not None else Quat.identity()
        self.scale: Vec3 = scale if scale is not None else Vec3.one()

    def to_matrix(self) -> Mat4:
        t = Mat4.translation(self.position)
        r = Mat4(self.rotation.to_matrix4())
        s = Mat4.scale(self.scale)
        return t * r * s

    def forward(self) -> Vec3:
        return self.rotation.rotate_vec(Vec3(0.0, 0.0, -1.0))

    def up(self) -> Vec3:
        return self.rotation.rotate_vec(Vec3(0.0, 1.0, 0.0))

    def right(self) -> Vec3:
        return self.rotation.rotate_vec(Vec3(1.0, 0.0, 0.0))
