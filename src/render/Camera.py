"""Camera — perspective camera with floating-origin-aware view matrix."""
from __future__ import annotations

import math

from src.math.Vec3 import Vec3
from src.math.Quat import Quat
from src.math.Mat4 import Mat4
from src.math.PlanetMath import PlanetMath


class Camera:
    def __init__(
        self,
        fov_deg: float = 70.0,
        near: float = 0.1,
        far: float = 10000.0,
    ) -> None:
        self.fov_rad: float = math.radians(fov_deg)
        self.near: float = near
        self.far: float = far
        self.aspect: float = 16.0 / 9.0

        # Local position (render-space, after floating origin rebase)
        self.local_position: Vec3 = Vec3(0.0, 1001.8, 0.0)
        self.rotation: Quat = Quat.identity()

        # Orbital "look around" yaw angle in local tangent frame
        self.yaw: float = 0.0

    def set_aspect(self, width: int, height: int) -> None:
        self.aspect = width / max(height, 1)

    def projection_matrix(self) -> Mat4:
        return Mat4.perspective(self.fov_rad, self.aspect, self.near, self.far)

    def view_matrix(self) -> Mat4:
        eye = self.local_position
        up = PlanetMath.up_at_position(eye) if not eye.is_near_zero() else Vec3(0.0, 1.0, 0.0)
        fwd = self.rotation.rotate_vec(Vec3(0.0, 0.0, -1.0))
        target = eye + fwd
        return Mat4.look_at(eye, target, up)

    def align_to_surface(self, dt: float, stiffness: float = 8.0) -> None:
        """Smoothly align camera up with planet up at current position."""
        target_up = PlanetMath.up_at_position(self.local_position)
        if target_up.is_near_zero():
            return
        self.rotation = PlanetMath.align_up(self.rotation, target_up, dt, stiffness)

    def forward_tangent(self) -> Vec3:
        """Camera forward projected onto the tangent plane."""
        fwd = self.rotation.rotate_vec(Vec3(0.0, 0.0, -1.0))
        return PlanetMath.tangent_forward(
            PlanetMath.up_at_position(self.local_position), fwd
        )
