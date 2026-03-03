"""SDFPatch — a delta-distance patch applied to the base SDF.

A patch is a localised modification (subtraction or addition) stored as
a compact, serialisable record.  Patches are never removed — they are
replayed in insertion order to maintain determinism.

Supported types
---------------
SphereDent  — subtracts a smooth sphere-shaped dent from the surface
              (footprint / cosmetic displacement from a player or event).
SphereDeposit — adds material inside a sphere (unused in this pass, stub).

Public API
----------
SDFPatch(patch_id, revision, cx, cy, cz, radius, strength, kind)
  .delta(x, y, z)   → float   signed distance delta at point
  .affects(x, y, z, margin) → bool   conservative spatial reject
  .to_dict()        → dict    serialisable record
  SDFPatch.from_dict(d) → SDFPatch   deserialise
"""
from __future__ import annotations

import math
from typing import Any, Dict


# Patch kind constants
KIND_SPHERE_DENT    = "sphere_dent"
KIND_SPHERE_DEPOSIT = "sphere_deposit"


class SDFPatch:
    """One localised SDF delta record.

    Parameters
    ----------
    patch_id :
        Unique integer identifier (server-assigned, monotonically increasing).
    revision :
        ``sdf_revision`` of the volume when this patch was applied.
    cx, cy, cz :
        World-space centre of the patch.
    radius :
        Influence radius in simulation units.
    strength :
        Maximum signed distance change at the centre (positive = dent/carve,
        negative = deposit).  Typical cosmetic footprint: 0.05 – 0.3 units.
    kind :
        ``KIND_SPHERE_DENT`` or ``KIND_SPHERE_DEPOSIT``.
    """

    __slots__ = (
        "patch_id", "revision",
        "cx", "cy", "cz",
        "radius", "strength", "kind",
    )

    def __init__(
        self,
        patch_id: int,
        revision: int,
        cx: float,
        cy: float,
        cz: float,
        radius: float,
        strength: float,
        kind: str = KIND_SPHERE_DENT,
    ) -> None:
        self.patch_id = int(patch_id)
        self.revision = int(revision)
        self.cx       = float(cx)
        self.cy       = float(cy)
        self.cz       = float(cz)
        self.radius   = float(radius)
        self.strength = float(strength)
        self.kind     = str(kind)

    # ------------------------------------------------------------------
    def affects(self, x: float, y: float, z: float, margin: float = 0.0) -> bool:
        """Return True when (x, y, z) is within radius + margin of the centre."""
        dx = x - self.cx
        dy = y - self.cy
        dz = z - self.cz
        r  = self.radius + margin
        return (dx * dx + dy * dy + dz * dz) <= r * r

    def delta(self, x: float, y: float, z: float) -> float:
        """Signed distance delta contributed by this patch at (x, y, z).

        A ``sphere_dent`` pushes the surface outward (adds positive distance)
        inside the sphere, carving a cavity.
        A ``sphere_deposit`` pushes the surface inward (adds negative distance).
        """
        dx   = x - self.cx
        dy   = y - self.cy
        dz   = z - self.cz
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        if dist >= self.radius:
            return 0.0
        # Smooth falloff: 1 at centre → 0 at radius
        t    = 1.0 - dist / self.radius
        bump = self.strength * t * t
        if self.kind == KIND_SPHERE_DENT:
            return bump
        # KIND_SPHERE_DEPOSIT
        return -bump

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "patch_id": self.patch_id,
            "revision": self.revision,
            "cx":       self.cx,
            "cy":       self.cy,
            "cz":       self.cz,
            "radius":   self.radius,
            "strength": self.strength,
            "kind":     self.kind,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SDFPatch":
        return cls(
            patch_id=int(d["patch_id"]),
            revision=int(d["revision"]),
            cx=float(d["cx"]),
            cy=float(d["cy"]),
            cz=float(d["cz"]),
            radius=float(d["radius"]),
            strength=float(d["strength"]),
            kind=str(d.get("kind", KIND_SPHERE_DENT)),
        )
