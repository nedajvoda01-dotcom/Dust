"""Vec3 — 3-component float vector."""
from __future__ import annotations

import math
from typing import Iterator


class Vec3:
    __slots__ = ("x", "y", "z")

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    # --- factory ---
    @staticmethod
    def zero() -> "Vec3":
        return Vec3(0.0, 0.0, 0.0)

    @staticmethod
    def one() -> "Vec3":
        return Vec3(1.0, 1.0, 1.0)

    @staticmethod
    def up() -> "Vec3":
        return Vec3(0.0, 1.0, 0.0)

    @staticmethod
    def forward() -> "Vec3":
        return Vec3(0.0, 0.0, -1.0)

    @staticmethod
    def right() -> "Vec3":
        return Vec3(1.0, 0.0, 0.0)

    # --- arithmetic ---
    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, s: float) -> "Vec3":
        return Vec3(self.x * s, self.y * s, self.z * s)

    def __rmul__(self, s: float) -> "Vec3":
        return self.__mul__(s)

    def __truediv__(self, s: float) -> "Vec3":
        return Vec3(self.x / s, self.y / s, self.z / s)

    def __neg__(self) -> "Vec3":
        return Vec3(-self.x, -self.y, -self.z)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vec3):
            return NotImplemented
        return self.x == other.x and self.y == other.y and self.z == other.z

    def __repr__(self) -> str:
        return f"Vec3({self.x:.6g}, {self.y:.6g}, {self.z:.6g})"

    # --- geometry ---
    def dot(self, other: "Vec3") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Vec3") -> "Vec3":
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def length_sq(self) -> float:
        return self.x * self.x + self.y * self.y + self.z * self.z

    def length(self) -> float:
        return math.sqrt(self.length_sq())

    def normalized(self) -> "Vec3":
        n = self.length()
        if n < 1e-30:
            return Vec3.zero()
        return self / n

    def lerp(self, other: "Vec3", t: float) -> "Vec3":
        return self * (1.0 - t) + other * t

    def is_near_zero(self, eps: float = 1e-9) -> bool:
        return self.length_sq() < eps * eps

    def to_tuple(self) -> tuple:
        return (self.x, self.y, self.z)

    def __iter__(self) -> Iterator[float]:
        yield self.x
        yield self.y
        yield self.z
