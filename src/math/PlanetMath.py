"""PlanetMath — geodesic math, surface movement, orientation."""
from __future__ import annotations

import math
from dataclasses import dataclass

from src.math.Vec3 import Vec3
from src.math.Quat import Quat


@dataclass
class LatLong:
    lat_rad: float  # latitude in radians, -pi/2 .. pi/2 (south to north)
    lon_rad: float  # longitude in radians, -pi .. pi


class PlanetMath:
    """All planet-surface geodesy. Planet center is the world origin (0,0,0)."""

    # --- Basic geometry ---

    @staticmethod
    def planet_center() -> Vec3:
        return Vec3(0.0, 0.0, 0.0)

    @staticmethod
    def to_unit_sphere(p: Vec3) -> Vec3:
        return p.normalized()

    @staticmethod
    def up_at_position(world_pos: Vec3) -> Vec3:
        """Unit vector pointing away from planet center at world_pos."""
        return (world_pos - PlanetMath.planet_center()).normalized()

    @staticmethod
    def down_at_position(world_pos: Vec3) -> Vec3:
        return -PlanetMath.up_at_position(world_pos)

    # --- Lat/Long conversions ---

    @staticmethod
    def from_direction(unit_dir: Vec3) -> LatLong:
        """Convert a unit sphere direction to LatLong (Y-up convention)."""
        d = unit_dir.normalized()
        lat = math.asin(max(-1.0, min(1.0, d.y)))
        lon = math.atan2(d.x, d.z)
        return LatLong(lat_rad=lat, lon_rad=lon)

    @staticmethod
    def direction_from_lat_long(ll: LatLong) -> Vec3:
        """Convert LatLong to unit sphere direction (Y-up convention)."""
        cos_lat = math.cos(ll.lat_rad)
        x = cos_lat * math.sin(ll.lon_rad)
        y = math.sin(ll.lat_rad)
        z = cos_lat * math.cos(ll.lon_rad)
        return Vec3(x, y, z)

    # --- Surface movement ---

    @staticmethod
    def tangent_forward(unit_dir: Vec3, desired_dir_world: Vec3) -> Vec3:
        """Project desired_dir_world onto the tangent plane at unit_dir."""
        up = unit_dir.normalized()
        proj = desired_dir_world - up * up.dot(desired_dir_world)
        if proj.is_near_zero():
            return Vec3.zero()
        return proj.normalized()

    @staticmethod
    def move_along_surface(unit_dir: Vec3, tangent_vel: Vec3, arc_len: float) -> Vec3:
        """
        Integrate movement along the sphere surface.
        unit_dir:   current position on unit sphere.
        tangent_vel: unit tangent direction of movement.
        arc_len:    distance (in radians on unit sphere, or metres/radius).
        Returns new unit-sphere position (always normalized).
        """
        if arc_len == 0.0 or tangent_vel.is_near_zero():
            return unit_dir.normalized()
        t = tangent_vel.normalized()
        # Rotation axis is perpendicular to both up and tangent
        axis = t.cross(unit_dir.normalized()).normalized()
        # Rotate unit_dir around axis by arc_len radians
        q = Quat.from_axis_angle(axis, arc_len)
        new_pos = q.rotate_vec(unit_dir.normalized())
        return new_pos.normalized()

    # --- Orientation ---

    @staticmethod
    def align_up(current_rot: Quat, target_up: Vec3, dt: float, stiffness: float) -> Quat:
        """
        Smoothly rotate current_rot so that its local up axis aligns with target_up.
        Uses slerp for smooth interpolation.
        """
        # Current "up" as seen by the rotation
        local_up = current_rot.rotate_vec(Vec3(0.0, 1.0, 0.0))
        # Compute delta rotation from current up to target up
        correction = Quat.from_to_rotation(local_up, target_up)
        # slerp factor: smaller dt/stiffness = slower alignment
        alpha = min(1.0, dt * stiffness)
        return Quat.slerp(current_rot, correction * current_rot, alpha).normalized()
