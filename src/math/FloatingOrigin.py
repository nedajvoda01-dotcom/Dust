"""FloatingOrigin — prevents float precision loss on large planets."""
from __future__ import annotations

from src.math.Vec3 import Vec3
from src.math.PlanetMath import PlanetMath, LatLong
from src.core.Logger import Logger

_TAG = "FloatingOrigin"


class FloatingOrigin:
    """
    Keeps the render-space origin near the player to preserve float precision.

    world_pos = local_pos + origin_offset

    When local_pos magnitude exceeds rebase_threshold:
      - origin_offset += local_pos
      - all scene local positions are shifted by -local_pos
      - geo coordinates are updated from the new world_pos
    """

    def __init__(self, rebase_threshold: float = 2000.0, planet_radius: float = 1000.0) -> None:
        self.rebase_threshold: float = rebase_threshold
        self.planet_radius: float = planet_radius

        # High-precision geodetic location of the origin
        self.origin_offset: Vec3 = Vec3(0.0, 0.0, 0.0)

        # Low-precision local (render-space) player position
        self.local_position: Vec3 = Vec3(0.0, float(planet_radius), 0.0)

        # Cached geo coords
        self.geo: LatLong = PlanetMath.from_direction(Vec3(0.0, 1.0, 0.0))

    # --- world / local conversion ---

    def world_position(self) -> Vec3:
        return self.local_position + self.origin_offset

    def to_local(self, world_pos: Vec3) -> Vec3:
        return world_pos - self.origin_offset

    def set_local_position(self, local_pos: Vec3) -> None:
        self.local_position = local_pos

    # --- rebase ---

    def try_rebase(self, scene_objects: list) -> bool:
        """
        If local_position exceeds rebase_threshold, shift everything.
        scene_objects: list of objects with a .local_position: Vec3 attribute.
        Returns True if a rebase occurred.
        """
        if self.local_position.length() < self.rebase_threshold:
            return False

        offset = self.local_position
        self.origin_offset = self.origin_offset + offset
        self.local_position = Vec3(0.0, 0.0, 0.0)

        for obj in scene_objects:
            obj.local_position = obj.local_position - offset

        # Update geo coordinates
        world = self.origin_offset
        unit = PlanetMath.to_unit_sphere(world)
        self.geo = PlanetMath.from_direction(unit)
        Logger.info(_TAG, f"Rebase. origin={self.origin_offset}, geo=({self.geo.lat_rad:.4f}, {self.geo.lon_rad:.4f})")
        return True

    def update_geo(self) -> None:
        """Refresh geo coordinates from current world position."""
        world = self.world_position()
        unit = PlanetMath.to_unit_sphere(world)
        self.geo = PlanetMath.from_direction(unit)
