"""Quat — unit quaternion for 3D rotations."""
from __future__ import annotations

import math

from src.math.Vec3 import Vec3


class Quat:
    """Unit quaternion: w + xi + yj + zk."""
    __slots__ = ("w", "x", "y", "z")

    def __init__(self, w: float = 1.0, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
        self.w = float(w)
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    @staticmethod
    def identity() -> "Quat":
        return Quat(1.0, 0.0, 0.0, 0.0)

    @staticmethod
    def from_axis_angle(axis: Vec3, angle_rad: float) -> "Quat":
        a = axis.normalized()
        s = math.sin(angle_rad * 0.5)
        c = math.cos(angle_rad * 0.5)
        return Quat(c, a.x * s, a.y * s, a.z * s)

    @staticmethod
    def from_to_rotation(from_vec: Vec3, to_vec: Vec3) -> "Quat":
        """Shortest rotation that takes from_vec onto to_vec (both need not be unit)."""
        a = from_vec.normalized()
        b = to_vec.normalized()
        d = a.dot(b)
        if d > 0.9999:
            return Quat.identity()
        if d < -0.9999:
            perp = Vec3(1.0, 0.0, 0.0)
            if abs(a.dot(perp)) > 0.99:
                perp = Vec3(0.0, 1.0, 0.0)
            axis = a.cross(perp).normalized()
            return Quat.from_axis_angle(axis, math.pi)
        axis = a.cross(b)
        q = Quat(1.0 + d, axis.x, axis.y, axis.z)
        return q.normalized()

    def normalized(self) -> "Quat":
        n = math.sqrt(self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z)
        if n < 1e-30:
            return Quat.identity()
        return Quat(self.w / n, self.x / n, self.y / n, self.z / n)

    def conjugate(self) -> "Quat":
        return Quat(self.w, -self.x, -self.y, -self.z)

    def __mul__(self, other: "Quat") -> "Quat":
        w1, x1, y1, z1 = self.w, self.x, self.y, self.z
        w2, x2, y2, z2 = other.w, other.x, other.y, other.z
        return Quat(
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        )

    def rotate_vec(self, v: Vec3) -> Vec3:
        """Rotate vector v by this quaternion."""
        q = Quat(0.0, v.x, v.y, v.z)
        r = self * q * self.conjugate()
        return Vec3(r.x, r.y, r.z)

    def to_matrix4(self) -> list:
        """Return column-major 4x4 matrix (16 floats) for OpenGL."""
        w, x, y, z = self.w, self.x, self.y, self.z
        return [
            1 - 2*y*y - 2*z*z,  2*x*y + 2*w*z,      2*x*z - 2*w*y,      0.0,
            2*x*y - 2*w*z,      1 - 2*x*x - 2*z*z,  2*y*z + 2*w*x,      0.0,
            2*x*z + 2*w*y,      2*y*z - 2*w*x,      1 - 2*x*x - 2*y*y,  0.0,
            0.0,                0.0,                0.0,                  1.0,
        ]

    @staticmethod
    def slerp(a: "Quat", b: "Quat", t: float) -> "Quat":
        """Spherical linear interpolation between two unit quaternions."""
        dot = a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z
        if dot < 0.0:
            b = Quat(-b.w, -b.x, -b.y, -b.z)
            dot = -dot
        if dot > 0.9995:
            r = Quat(
                a.w + t * (b.w - a.w),
                a.x + t * (b.x - a.x),
                a.y + t * (b.y - a.y),
                a.z + t * (b.z - a.z),
            )
            return r.normalized()
        theta_0 = math.acos(dot)
        theta = theta_0 * t
        sin_theta = math.sin(theta)
        sin_theta_0 = math.sin(theta_0)
        s1 = sin_theta / sin_theta_0
        s0 = math.cos(theta) - dot * s1
        return Quat(
            s0 * a.w + s1 * b.w,
            s0 * a.x + s1 * b.x,
            s0 * a.y + s1 * b.y,
            s0 * a.z + s1 * b.z,
        ).normalized()

    def __repr__(self) -> str:
        return f"Quat(w={self.w:.6g}, x={self.x:.6g}, y={self.y:.6g}, z={self.z:.6g})"
