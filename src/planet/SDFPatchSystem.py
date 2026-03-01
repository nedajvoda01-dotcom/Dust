"""SDFPatchSystem — local SDF modifications (carve/deposit) with a replay log.

CSG convention for SDF patches
-------------------------------
We use the **smooth-union / smooth-subtraction** convention consistent with
the rest of the subsystem:

  d > 0  →  air / outside
  d < 0  →  rock / inside

Carve (excavate rock → air):
    new_d = max(old_d, -shape_sdf)

  This pushes the distance towards positive (air) inside the shape.
  Equivalent to CSG subtraction: new_solid = old_solid MINUS shape.

Deposit (add rock → fill air):
    new_d = min(old_d, shape_sdf)

  This pushes the distance towards negative (rock) inside the shape.

Patch application is strictly **additive to the log**: patches are never
removed and are always replayed in insertion order to stay deterministic.

Supported patch types
---------------------
SphereCarve    — excavate a sphere (radius, centre)
CapsuleCarve   — excavate a capsule (two endpoints, radius)  [stub — geometry
                 only; full parametrisation may be extended later]
SplineCarve    — reserved stub for spline-driven tunnels
AdditiveDeposit — add rock inside a sphere
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from src.math.Vec3 import Vec3
from src.planet.SDFChunk import SDFChunk, MATERIAL_AIR, MATERIAL_ROCK


# ---------------------------------------------------------------------------
# Patch base
# ---------------------------------------------------------------------------

@dataclass
class SDFPatch:
    """Base class for all SDF patches.  Subclasses override ``apply_to_chunk``."""

    def apply_to_chunk(self, chunk: SDFChunk) -> bool:
        """
        Modify *chunk* in-place.  Returns True if any voxel was changed.
        Sets ``chunk.dirty = True`` when changes are made.
        """
        raise NotImplementedError  # pragma: no cover

    # ------------------------------------------------------------------
    def _affects_chunk(self, chunk: SDFChunk) -> bool:
        """Quick bounding-box reject: True when the patch might overlap chunk."""
        raise NotImplementedError  # pragma: no cover


# ---------------------------------------------------------------------------
# SphereCarve
# ---------------------------------------------------------------------------

@dataclass
class SphereCarve(SDFPatch):
    """Excavate a sphere of radius *r* centred at world-space *centre*."""

    centre: Vec3
    radius: float

    # ------------------------------------------------------------------
    def _affects_chunk(self, chunk: SDFChunk) -> bool:
        """True when any voxel corner might be within reach of the sphere."""
        cx, cy, cz = self.centre.x, self.centre.y, self.centre.z
        R = chunk.resolution
        # Check all 8 chunk corners for overlap (conservative AABB test)
        xs = [chunk.get_pos(0, 0, 0)[0], chunk.get_pos(R-1, R-1, R-1)[0]]
        ys = [chunk.get_pos(0, 0, 0)[1], chunk.get_pos(R-1, R-1, R-1)[1]]
        zs = [chunk.get_pos(0, 0, 0)[2], chunk.get_pos(R-1, R-1, R-1)[2]]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        min_z, max_z = min(zs), max(zs)
        # Closest point on AABB to sphere centre
        px = max(min_x, min(cx, max_x))
        py = max(min_y, min(cy, max_y))
        pz = max(min_z, min(cz, max_z))
        dist_sq = (px - cx)**2 + (py - cy)**2 + (pz - cz)**2
        return dist_sq <= self.radius * self.radius

    # ------------------------------------------------------------------
    def apply_to_chunk(self, chunk: SDFChunk) -> bool:
        if not self._affects_chunk(chunk):
            return False

        cx, cy, cz = self.centre.x, self.centre.y, self.centre.z
        r          = self.radius
        R          = chunk.resolution
        changed    = False

        for k in range(R):
            for j in range(R):
                for i in range(R):
                    wx, wy, wz = chunk.get_pos(i, j, k)
                    dx = wx - cx; dy = wy - cy; dz = wz - cz
                    # SDF of the sphere: d_sphere < 0 inside the sphere
                    d_sphere = math.sqrt(dx*dx + dy*dy + dz*dz) - r
                    old_d    = chunk.get_d(i, j, k)
                    # Carve: new_d = max(old_d, -d_sphere)
                    new_d    = max(old_d, -d_sphere)
                    if new_d != old_d:
                        chunk.set_d(i, j, k, new_d)
                        # Update material
                        mat = MATERIAL_AIR if new_d >= 0.0 else MATERIAL_ROCK
                        chunk.material_field[chunk.flat_index(i, j, k)] = mat
                        changed = True

        if changed:
            chunk.dirty = True
        return changed


# ---------------------------------------------------------------------------
# CapsuleCarve (stub — correct geometry, extensible)
# ---------------------------------------------------------------------------

@dataclass
class CapsuleCarve(SDFPatch):
    """
    Excavate a capsule (cylinder with hemispherical caps) between *a* and *b*
    with radius *r*.  Acts as a straight tunnel.
    """

    a:      Vec3
    b:      Vec3
    radius: float

    # ------------------------------------------------------------------
    @staticmethod
    def _capsule_sdf(px: float, py: float, pz: float,
                     ax: float, ay: float, az: float,
                     bx: float, by: float, bz: float,
                     r: float) -> float:
        """Signed distance from point P to capsule (A, B, r)."""
        abx = bx - ax; aby = by - ay; abz = bz - az
        apx = px - ax; apy = py - ay; apz = pz - az
        ab2 = abx*abx + aby*aby + abz*abz
        t   = (apx*abx + apy*aby + apz*abz) / ab2 if ab2 > 1e-30 else 0.0
        t   = max(0.0, min(1.0, t))
        qx  = ax + t * abx; qy = ay + t * aby; qz = az + t * abz
        dx  = px - qx; dy = py - qy; dz = pz - qz
        return math.sqrt(dx*dx + dy*dy + dz*dz) - r

    def _affects_chunk(self, chunk: SDFChunk) -> bool:
        # Conservative: test sphere at midpoint with half-length + radius
        R  = chunk.resolution
        mx = (self.a.x + self.b.x) * 0.5
        my = (self.a.y + self.b.y) * 0.5
        mz = (self.a.z + self.b.z) * 0.5
        half_len = math.sqrt(
            (self.b.x - self.a.x)**2 +
            (self.b.y - self.a.y)**2 +
            (self.b.z - self.a.z)**2
        ) * 0.5
        bound_r = half_len + self.radius
        xs = [chunk.get_pos(0, 0, 0)[0], chunk.get_pos(R-1, R-1, R-1)[0]]
        ys = [chunk.get_pos(0, 0, 0)[1], chunk.get_pos(R-1, R-1, R-1)[1]]
        zs = [chunk.get_pos(0, 0, 0)[2], chunk.get_pos(R-1, R-1, R-1)[2]]
        px = max(min(xs), min(mx, max(xs)))
        py = max(min(ys), min(my, max(ys)))
        pz = max(min(zs), min(mz, max(zs)))
        dist_sq = (px-mx)**2 + (py-my)**2 + (pz-mz)**2
        return dist_sq <= bound_r * bound_r

    def apply_to_chunk(self, chunk: SDFChunk) -> bool:
        if not self._affects_chunk(chunk):
            return False

        ax, ay, az = self.a.x, self.a.y, self.a.z
        bx, by, bz = self.b.x, self.b.y, self.b.z
        r          = self.radius
        R          = chunk.resolution
        changed    = False

        for k in range(R):
            for j in range(R):
                for i in range(R):
                    wx, wy, wz = chunk.get_pos(i, j, k)
                    d_cap  = self._capsule_sdf(wx, wy, wz, ax, ay, az, bx, by, bz, r)
                    old_d  = chunk.get_d(i, j, k)
                    new_d  = max(old_d, -d_cap)
                    if new_d != old_d:
                        chunk.set_d(i, j, k, new_d)
                        mat = MATERIAL_AIR if new_d >= 0.0 else MATERIAL_ROCK
                        chunk.material_field[chunk.flat_index(i, j, k)] = mat
                        changed = True

        if changed:
            chunk.dirty = True
        return changed


# ---------------------------------------------------------------------------
# SplineCarve (reserved stub)
# ---------------------------------------------------------------------------

@dataclass
class SplineCarve(SDFPatch):
    """Stub — spline-driven tunnel carve.  Not yet implemented."""

    control_points: List[Vec3]
    radius:         float

    def _affects_chunk(self, chunk: SDFChunk) -> bool:
        return False  # stub

    def apply_to_chunk(self, chunk: SDFChunk) -> bool:
        return False  # stub


# ---------------------------------------------------------------------------
# AdditiveDeposit
# ---------------------------------------------------------------------------

@dataclass
class AdditiveDeposit(SDFPatch):
    """Add rock inside a sphere (inverse of SphereCarve)."""

    centre: Vec3
    radius: float

    def _affects_chunk(self, chunk: SDFChunk) -> bool:
        cx, cy, cz = self.centre.x, self.centre.y, self.centre.z
        R  = chunk.resolution
        xs = [chunk.get_pos(0, 0, 0)[0], chunk.get_pos(R-1, R-1, R-1)[0]]
        ys = [chunk.get_pos(0, 0, 0)[1], chunk.get_pos(R-1, R-1, R-1)[1]]
        zs = [chunk.get_pos(0, 0, 0)[2], chunk.get_pos(R-1, R-1, R-1)[2]]
        px = max(min(xs), min(cx, max(xs)))
        py = max(min(ys), min(cy, max(ys)))
        pz = max(min(zs), min(cz, max(zs)))
        dist_sq = (px-cx)**2 + (py-cy)**2 + (pz-cz)**2
        return dist_sq <= self.radius * self.radius

    def apply_to_chunk(self, chunk: SDFChunk) -> bool:
        if not self._affects_chunk(chunk):
            return False

        cx, cy, cz = self.centre.x, self.centre.y, self.centre.z
        r          = self.radius
        R          = chunk.resolution
        changed    = False

        for k in range(R):
            for j in range(R):
                for i in range(R):
                    wx, wy, wz = chunk.get_pos(i, j, k)
                    dx = wx - cx; dy = wy - cy; dz = wz - cz
                    d_sphere = math.sqrt(dx*dx + dy*dy + dz*dz) - r
                    old_d    = chunk.get_d(i, j, k)
                    # Deposit: new_d = min(old_d, d_sphere)
                    new_d    = min(old_d, d_sphere)
                    if new_d != old_d:
                        chunk.set_d(i, j, k, new_d)
                        mat = MATERIAL_AIR if new_d >= 0.0 else MATERIAL_ROCK
                        chunk.material_field[chunk.flat_index(i, j, k)] = mat
                        changed = True

        if changed:
            chunk.dirty = True
        return changed


# ---------------------------------------------------------------------------
# SDFPatchLog
# ---------------------------------------------------------------------------

class SDFPatchLog:
    """
    Ordered, append-only log of SDF patches.

    Patches are applied deterministically in insertion order.  The full log
    can be replayed to any chunk at any time, which guarantees reproducibility
    from seed + log content.
    """

    def __init__(self) -> None:
        self._patches: List[SDFPatch] = []

    # ------------------------------------------------------------------
    def add(self, patch: SDFPatch) -> None:
        """Append a patch to the log."""
        self._patches.append(patch)

    def patches(self) -> List[SDFPatch]:
        """Read-only view of all patches (in insertion order)."""
        return list(self._patches)

    def __len__(self) -> int:
        return len(self._patches)

    # ------------------------------------------------------------------
    def apply_to_chunk(self, chunk: SDFChunk) -> bool:
        """
        Replay all patches in log order onto *chunk*.
        Returns True if any patch changed the chunk.
        """
        any_changed = False
        for patch in self._patches:
            if patch.apply_to_chunk(chunk):
                any_changed = True
        return any_changed

    # ------------------------------------------------------------------
    def get_affected_chunks_hint(self, chunk_coords: List) -> List:
        """
        Return subset of *chunk_coords* that may be affected by at least one
        patch.  Uses each patch's ``_affects_chunk`` bounding-box test.

        This is a hint only — false positives are acceptable, false negatives
        are not (all affected chunks must be included).

        Parameters
        ----------
        chunk_coords : list of SDFChunkCoord (used to look up chunks externally)
        """
        # Without chunk objects we cannot test; return all as conservative hint.
        return list(chunk_coords)
