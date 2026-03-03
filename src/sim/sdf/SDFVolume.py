"""SDFVolume — composite SDF: base sphere + additive patch list.

Usage
-----
vol = SDFVolume(radius=1000.0)
vol.apply_patch(SDFPatch(...))
d = vol.eval(x, y, z)

The volume is the single source of signed-distance truth for physics,
collision detection, and rendering on the server.  Patches are applied
additively in insertion order (deterministic replay).

Public API
----------
SDFVolume(radius, patches=None)
  .eval(x, y, z)                 → float   signed distance
  .apply_patch(patch)            → None    append + bump sdf_revision
  .patches_since(revision)       → list    patches added after *revision*
  .sdf_revision                  → int     current revision counter
  .to_baseline_dict()            → dict    base parameters only
"""
from __future__ import annotations

from typing import List, Optional

from src.sim.sdf.SDFBase import SDFBase
from src.sim.sdf.SDFPatch import SDFPatch


class SDFVolume:
    """Planet SDF = sphere base + ordered patch deltas.

    Parameters
    ----------
    radius :
        Base planet radius.
    patches :
        Optional initial patch list (for reloading from disk).
    """

    def __init__(
        self,
        radius: float = 1000.0,
        patches: Optional[List[SDFPatch]] = None,
    ) -> None:
        self._base:     SDFBase      = SDFBase(radius)
        self._patches:  List[SDFPatch] = list(patches) if patches else []
        self._revision: int          = len(self._patches)

    # ------------------------------------------------------------------
    @property
    def sdf_revision(self) -> int:
        """Number of patches applied so far (monotonically increasing)."""
        return self._revision

    @property
    def base_radius(self) -> float:
        return self._base.radius

    # ------------------------------------------------------------------
    def eval(self, x: float, y: float, z: float) -> float:
        """Return signed distance at (x, y, z).

        d > 0  outside / air
        d < 0  inside / rock
        """
        d = self._base.eval(x, y, z)
        for p in self._patches:
            if p.affects(x, y, z):
                d += p.delta(x, y, z)
        return d

    # ------------------------------------------------------------------
    def apply_patch(self, patch: SDFPatch) -> None:
        """Append *patch* and bump ``sdf_revision``."""
        self._patches.append(patch)
        self._revision += 1

    # ------------------------------------------------------------------
    def patches_since(self, revision: int) -> List[SDFPatch]:
        """Return patches whose ``revision`` field is greater than *revision*.

        Used for incremental replication: a client sends its last known
        ``sdf_revision``; the server replies with only the new patches.
        """
        return [p for p in self._patches if p.revision > revision]

    def all_patches(self) -> List[SDFPatch]:
        return list(self._patches)

    # ------------------------------------------------------------------
    def to_baseline_dict(self) -> dict:
        """Serialise base parameters (not patches) for WORLD_BASELINE."""
        return {
            "planet_radius": self._base.radius,
            "sdf_revision":  self._revision,
        }
