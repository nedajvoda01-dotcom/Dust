"""DeformationIntegrator — Stage 35 per-frame deformation tick driver.

Owns a cache of active DeformationField instances (LRU by chunk_id) and
applies ContactSample streams each frame, subject to budget limits.

DeformationIntegrator
    - apply_samples(samples, dt)  — apply a batch of ContactSamples
    - relax_tick(dt, storm_multiplier)  — advance relaxation on all active chunks
    - get_field(chunk_id)         — retrieve (or create) a field for a chunk
    - effective_mu(chunk_id, ix, iy, material, m_base) -> float
    - debug_info() -> dict

Budget control
    maxChunkDeformUploadsPerFrame limits how many chunks get GPU-upload
    notifications per frame.  Contact samples beyond the upload budget are
    still applied to the field; only the "dirty" flag is budgeted.
"""
from __future__ import annotations

import math
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

from src.physics.MaterialYieldModel import MaterialClass, MaterialYieldModel
from src.surface.DeformationField import ContactSample, DeformationField


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


# ---------------------------------------------------------------------------
# DeformationIntegrator
# ---------------------------------------------------------------------------

class DeformationIntegrator:
    """Frame-level driver for surface deformation.

    Parameters
    ----------
    config:
        Optional Config object for ``deform.*`` keys.
    """

    def __init__(self, config=None) -> None:
        self._cfg   = config
        self._model = MaterialYieldModel(config)

        # Config-driven parameters
        self._grid_res: int   = int(self._get("grid_res", 64))
        self._max_chunks: int = int(self._get("cache.max_active_chunks", 16))
        self._max_uploads: int = int(self._get(
            "render.max_uploads_per_frame", 4))

        # LRU cache: chunk_id → DeformationField
        self._fields: OrderedDict[object, DeformationField] = OrderedDict()

        # Chunks that need a GPU upload this frame
        self._dirty: set = set()

        # Stats
        self._upload_count_this_frame: int = 0
        self._samples_applied: int = 0

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def apply_samples(
        self,
        samples: List[ContactSample],
        dt: float,
    ) -> None:
        """Apply a batch of ContactSamples (one frame's worth).

        Budget: at most ``max_uploads_per_frame`` chunks are flagged dirty.
        All samples are applied regardless of the upload budget.
        """
        self._upload_count_this_frame = 0
        self._samples_applied = 0

        for sample in samples:
            chunk_id = self._chunk_id_for_sample(sample)
            f        = self.get_field(chunk_id)
            params   = self._model.get(sample.material)
            f.apply_contact_sample(sample, params, dt)
            self._samples_applied += 1

            # Mark dirty (upload budget)
            if chunk_id not in self._dirty:
                if self._upload_count_this_frame < self._max_uploads:
                    self._dirty.add(chunk_id)
                    self._upload_count_this_frame += 1

    def relax_tick(
        self,
        dt: float,
        storm_multiplier: float = 1.0,
    ) -> None:
        """Advance relaxation on all active cached chunks."""
        for f in self._fields.values():
            f.relax(dt, storm_multiplier)

    def get_field(self, chunk_id: object) -> DeformationField:
        """Return the DeformationField for *chunk_id*, creating if needed.

        Evicts the least-recently-used entry when the cache is full.
        """
        if chunk_id in self._fields:
            self._fields.move_to_end(chunk_id)
            return self._fields[chunk_id]

        # Evict LRU if at capacity
        if len(self._fields) >= self._max_chunks:
            self._fields.popitem(last=False)

        f = DeformationField(
            chunk_id=chunk_id,
            grid_res=self._grid_res,
            m_base=self._get("m_base_default", 0.5),
        )
        # Wire up relaxation taus from config
        tau_h = self._get("relax_tau_h_sec", 120.0)
        tau_m = self._get("relax_tau_m_sec", 90.0)
        f.set_relaxation_taus(tau_h, tau_m)

        self._fields[chunk_id] = f
        return f

    def effective_mu(
        self,
        chunk_id: object,
        ix: int,
        iy: int,
        material: MaterialClass,
        ice_film: float = 0.0,
    ) -> float:
        """Return effective friction at a grid cell.

        Used by ContactManager to feed back into MotorStack.
        """
        if chunk_id not in self._fields:
            return self._model.get(material).base_friction
        f      = self._fields[chunk_id]
        m_val  = f.m_at(ix, iy)
        m_base = f.m_base
        return self._model.effective_friction(material, m_val, m_base, ice_film)

    def consume_dirty_set(self) -> set:
        """Return and clear the set of chunk IDs that need GPU upload."""
        dirty = set(self._dirty)
        self._dirty.clear()
        return dirty

    def debug_info(self) -> dict:
        """Return diagnostic snapshot (dev-mode only)."""
        return {
            "active_chunks":             len(self._fields),
            "dirty_this_frame":          len(self._dirty),
            "uploads_this_frame":        self._upload_count_this_frame,
            "samples_applied_this_frame": self._samples_applied,
        }

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _get(self, key: str, default):
        """Read a ``deform.*`` config value with a fallback."""
        if self._cfg is None:
            return default
        v = self._cfg.get("deform", key, default=None)
        return v if v is not None else default

    @staticmethod
    def _chunk_id_for_sample(sample: ContactSample) -> object:
        """Map a sample's grid coords to a chunk identifier.

        Simple implementation: chunk_id is a tuple (chunk_x, chunk_y)
        derived by dividing by a large constant.  In practice the caller
        should supply per-chunk IDs directly; this is the fallback for
        standalone testing.
        """
        # Treat every sample as belonging to chunk (0, 0) for now;
        # real integration would use (ix // grid_res, iy // grid_res).
        return (0, 0)
