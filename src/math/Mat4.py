"""Mat4 — 4x4 column-major matrix for transforms and projection."""
from __future__ import annotations

import math
from src.math.Vec3 import Vec3


class Mat4:
    """Column-major 4x4 float matrix. m[col*4+row]."""
    __slots__ = ("m",)

    def __init__(self, m: list | None = None) -> None:
        if m is None:
            self.m: list = [0.0] * 16
        else:
            self.m = list(m)

    @staticmethod
    def identity() -> "Mat4":
        r = Mat4()
        r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1.0
        return r

    @staticmethod
    def perspective(fov_rad: float, aspect: float, near: float, far: float) -> "Mat4":
        f = 1.0 / math.tan(fov_rad * 0.5)
        r = Mat4()
        r.m[0] = f / aspect
        r.m[5] = f
        r.m[10] = (far + near) / (near - far)
        r.m[11] = -1.0
        r.m[14] = (2.0 * far * near) / (near - far)
        return r

    @staticmethod
    def look_at(eye: Vec3, center: Vec3, up: Vec3) -> "Mat4":
        f = (center - eye).normalized()
        r_vec = f.cross(up).normalized()
        u_vec = r_vec.cross(f)
        m = Mat4.identity()
        m.m[0] = r_vec.x
        m.m[4] = r_vec.y
        m.m[8] = r_vec.z
        m.m[1] = u_vec.x
        m.m[5] = u_vec.y
        m.m[9] = u_vec.z
        m.m[2] = -f.x
        m.m[6] = -f.y
        m.m[10] = -f.z
        m.m[12] = -r_vec.dot(eye)
        m.m[13] = -u_vec.dot(eye)
        m.m[14] = f.dot(eye)
        return m

    @staticmethod
    def translation(v: Vec3) -> "Mat4":
        r = Mat4.identity()
        r.m[12] = v.x
        r.m[13] = v.y
        r.m[14] = v.z
        return r

    @staticmethod
    def scale(v: Vec3) -> "Mat4":
        r = Mat4.identity()
        r.m[0] = v.x
        r.m[5] = v.y
        r.m[10] = v.z
        return r

    def __mul__(self, other: "Mat4") -> "Mat4":
        a, b = self.m, other.m
        c = [0.0] * 16
        for col in range(4):
            for row in range(4):
                for k in range(4):
                    c[col * 4 + row] += a[k * 4 + row] * b[col * 4 + k]
        return Mat4(c)

    def to_list(self) -> list:
        return list(self.m)
