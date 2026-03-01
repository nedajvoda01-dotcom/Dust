"""SpawnAnchor — shared spawn point for multiplayer.

All players spawn near a single anchor point on the planet surface.
Each player's exact offset is deterministically derived from a hash of
their ``player_key`` so spawn positions are stable and collision-free.

Algorithm (Stage 21 spec §10)
------------------------------
1. ``angle = hash(player_key) mod 2π``
2. ``offset = rotate(tangentBasis, angle) * spawn_radius``
3. Normalise the offset direction back onto the sphere surface.

The anchor itself is set by the server when the first player spawns (or
from a saved world state).

Public API
----------
SpawnAnchor(anchor_pos, radius_m, planet_radius)
  .anchor                          → list[float]  (get/set)
  .get_spawn_for_player(player_key) → list[float]  [x, y, z]
"""
from __future__ import annotations

import hashlib
import math
from typing import List, Optional


def _cross(a: List[float], b: List[float]) -> List[float]:
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


class SpawnAnchor:
    """Manages the shared spawn anchor and per-player offsets.

    Parameters
    ----------
    anchor_pos:
        Initial anchor position [x, y, z] in world space.
        Defaults to ``[0.0, planet_radius + 1.8, 0.0]`` (north pole-ish).
    radius_m:
        Radius of the scatter circle around the anchor (metres / sim units).
    planet_radius:
        Used to compute the angular arc for the scatter offset.
    """

    def __init__(
        self,
        anchor_pos:    Optional[List[float]] = None,
        radius_m:      float = 5.0,
        planet_radius: float = 1000.0,
    ) -> None:
        self._planet_r = planet_radius
        self._radius   = radius_m
        if anchor_pos is not None:
            self._anchor = list(anchor_pos)
        else:
            self._anchor = [0.0, planet_radius + 1.8, 0.0]

    # ------------------------------------------------------------------

    @property
    def anchor(self) -> List[float]:
        return list(self._anchor)

    @anchor.setter
    def anchor(self, pos: List[float]) -> None:
        self._anchor = list(pos)

    def get_spawn_for_player(self, player_key: str) -> List[float]:
        """Return a deterministic spawn position near the anchor.

        The position lies on the sphere surface at an angular distance of
        ``radius_m / planet_radius`` radians from the anchor, at an angle
        derived from the SHA-256 hash of *player_key*.
        """
        # Determine angle from player key hash
        h = int(hashlib.sha256(player_key.encode("utf-8")).hexdigest(), 16)
        angle = (h % 100_000) / 100_000.0 * 2.0 * math.pi

        ax, ay, az = self._anchor
        r = math.sqrt(ax * ax + ay * ay + az * az)
        if r < 1e-9:
            return list(self._anchor)

        # Unit direction of the anchor
        ux, uy, uz = ax / r, ay / r, az / r

        # Build two orthogonal tangent vectors in the plane perpendicular to up
        if abs(uy) < 0.9:
            ref = [0.0, 1.0, 0.0]
        else:
            ref = [1.0, 0.0, 0.0]

        tang0 = _cross([ux, uy, uz], ref)
        t0l = math.sqrt(tang0[0] ** 2 + tang0[1] ** 2 + tang0[2] ** 2)
        if t0l < 1e-9:
            return list(self._anchor)
        tang0 = [t / t0l for t in tang0]
        tang1 = _cross([ux, uy, uz], tang0)

        # Offset direction in tangent plane
        ox = math.cos(angle) * tang0[0] + math.sin(angle) * tang1[0]
        oy = math.cos(angle) * tang0[1] + math.sin(angle) * tang1[1]
        oz = math.cos(angle) * tang0[2] + math.sin(angle) * tang1[2]

        # Arc length in radians
        arc = self._radius / max(self._planet_r, 1.0)

        # Move *arc* radians from the anchor direction toward *offset*
        nx = ux + arc * ox
        ny = uy + arc * oy
        nz = uz + arc * oz
        nl = math.sqrt(nx * nx + ny * ny + nz * nz)
        if nl < 1e-9:
            return list(self._anchor)
        nx, ny, nz = nx / nl, ny / nl, nz / nl

        # Project back to sphere at the same radius as the anchor
        return [nx * r, ny * r, nz * r]
