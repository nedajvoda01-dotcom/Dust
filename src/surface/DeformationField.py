"""DeformationField — Stage 35 per-chunk deformation state.

Stores two quantised scalar fields for one surface chunk:

* H(x, y) — vertical displacement [int16, units: mm].
  Negative = indentation below original surface; positive = raised berm.
* M(x, y) — loose material layer thickness [uint16, arbitrary units].
  Represents the mass that can be bulldozed / displaced.

Fields are stored as flat Python arrays (int) for portability; conversion
to/from bytes is handled by DeformStampCodec.

DeformationField
    Created per active chunk.  The resolution (grid_res × grid_res) is
    configured globally (default 64).

ContactSample
    Input record from MotorStack/ContactManager: normal force, tangential
    force, slip velocity, contact area, material class, world position.

Public API
----------
DeformationField(chunk_id, grid_res, m_base)
DeformationField.apply_contact_sample(sample, params, dt)
DeformationField.relax(dt, storm_multiplier)
DeformationField.h_at(ix, iy) -> float          # metres
DeformationField.m_at(ix, iy) -> float          # normalised [0,1]
DeformationField.field_hash() -> str
"""
from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass, field
from typing import Tuple

from src.physics.MaterialYieldModel import MaterialClass, MaterialParams

# Quantisation constants
_H_SCALE   = 1000.0   # int16 units per metre  (1 unit = 1 mm)
_H_MIN     = -32768
_H_MAX     = 32767
_M_MAX_INT = 65535     # uint16 range


def _clamp_int(v: int, lo: int, hi: int) -> int:
    return lo if v < lo else (hi if v > hi else v)


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


# ---------------------------------------------------------------------------
# ContactSample
# ---------------------------------------------------------------------------

@dataclass
class ContactSample:
    """Single contact input record for one foot/hand contact patch.

    Attributes
    ----------
    world_ix, world_iy:
        Grid cell coordinates within the chunk (0 ≤ ix, iy < grid_res).
    fn:
        Normal (compressive) force magnitude [N].
    ft_x, ft_y:
        Tangential force components in the 2-D grid plane [N].
    v_rel_x, v_rel_y:
        Relative slip velocity in the 2-D grid plane [m/s].
    area:
        Contact patch area [m²].
    material:
        MaterialClass of the surface at this contact.
    tick_index:
        Simulation tick index (for ordering / dedup).
    """
    world_ix:   int
    world_iy:   int
    fn:         float
    ft_x:       float
    ft_y:       float
    v_rel_x:    float
    v_rel_y:    float
    area:       float
    material:   MaterialClass = MaterialClass.DUST
    tick_index: int = 0


# ---------------------------------------------------------------------------
# DeformationField
# ---------------------------------------------------------------------------

class DeformationField:
    """H + M deformation field for one surface chunk.

    Parameters
    ----------
    chunk_id:
        Opaque identifier for the owning chunk (for hashing / logging).
    grid_res:
        Square grid resolution (e.g. 64 → 64×64 cells).
    m_base:
        Baseline M value [0, 1].  M is initialised to this value and
        relaxes back toward it over time.
    """

    def __init__(
        self,
        chunk_id: object,
        grid_res: int = 64,
        m_base: float = 0.5,
    ) -> None:
        self.chunk_id  = chunk_id
        self.grid_res  = grid_res
        self.m_base    = _clamp(m_base, 0.0, 1.0)
        n              = grid_res * grid_res

        # H: int16 values (stored as plain ints for pure-Python portability)
        self._h: list[int] = [0] * n
        # M: uint16 values normalised: stored as ints in [0, _M_MAX_INT]
        m_init_int         = int(round(self.m_base * _M_MAX_INT))
        self._m: list[int] = [m_init_int] * n

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def h_at(self, ix: int, iy: int) -> float:
        """Vertical displacement at (ix, iy) in metres (negative = indent)."""
        return self._h[self._idx(ix, iy)] / _H_SCALE

    def m_at(self, ix: int, iy: int) -> float:
        """Loose material value at (ix, iy) normalised [0, 1]."""
        return self._m[self._idx(ix, iy)] / _M_MAX_INT

    # ------------------------------------------------------------------
    # Physics
    # ------------------------------------------------------------------

    def apply_contact_sample(
        self,
        sample: ContactSample,
        params: MaterialParams,
        dt: float,
    ) -> None:
        """Integrate one contact sample into the H and M fields.

        Implements:
        * Plastic indentation (section 5.1)
        * Bulldozing / berm formation (section 5.2)
        * Slip groove (section 5.3)
        """
        if dt <= 0.0:
            return

        ix = _clamp_int(sample.world_ix, 0, self.grid_res - 1)
        iy = _clamp_int(sample.world_iy, 0, self.grid_res - 1)
        idx = self._idx(ix, iy)

        area = max(sample.area, 1e-6)
        pressure = sample.fn / area

        # -- 5.1 Plastic indentation --
        if pressure > params.yield_strength and params.indent_k > 0.0:
            dh_m = params.indent_k * (pressure - params.yield_strength) * dt
            dh_i = int(round(dh_m * _H_SCALE))
            self._h[idx] = _clamp_int(self._h[idx] - dh_i, _H_MIN, _H_MAX)

            # Compaction: twice the depth ratio (loose material packs against
            # the indenter), capped at 5 % per step to avoid jumps.
            dm_frac = _clamp(dh_m * 2.0, 0.0, 0.05)
            dm_i    = int(round(dm_frac * _M_MAX_INT))
            self._m[idx] = _clamp_int(self._m[idx] - dm_i, 0, _M_MAX_INT)

            # Spread a small berm to the 4 cardinal neighbours
            berm_i = dm_i // 4
            for nix, niy in self._neighbours(ix, iy):
                ni = self._idx(nix, niy)
                self._m[ni] = _clamp_int(self._m[ni] + berm_i, 0, _M_MAX_INT)
                # Neighbours are also raised a little (berm)
                self._h[ni] = _clamp_int(
                    self._h[ni] + berm_i // 2, _H_MIN, _H_MAX)

        # -- 5.2 Bulldozing (mass transport along slip direction) --
        slip_mag = math.sqrt(sample.v_rel_x ** 2 + sample.v_rel_y ** 2)
        ft_mag   = math.sqrt(sample.ft_x ** 2 + sample.ft_y ** 2)

        slip_thresh = 0.05   # m/s
        ft_thresh   = 1.0    # N

        if (slip_mag > slip_thresh and ft_mag > ft_thresh
                and params.mass_push_k > 0.0):
            # Direction of mass transport: opposite to slip
            if slip_mag > 0.0:
                sx = -sample.v_rel_x / slip_mag
                sy = -sample.v_rel_y / slip_mag
            else:
                sx, sy = 0.0, 0.0

            # Target neighbour cell in push direction
            target_ix = ix + (1 if sx > 0.5 else (-1 if sx < -0.5 else 0))
            target_iy = iy + (1 if sy > 0.5 else (-1 if sy < -0.5 else 0))
            target_ix = _clamp_int(target_ix, 0, self.grid_res - 1)
            target_iy = _clamp_int(target_iy, 0, self.grid_res - 1)

            move_frac = _clamp(
                params.mass_push_k * slip_mag * ft_mag * dt * 0.001, 0.0, 0.2)
            move_i    = int(round(move_frac * _M_MAX_INT))
            move_i    = min(move_i, self._m[idx])

            self._m[idx] = _clamp_int(self._m[idx] - move_i, 0, _M_MAX_INT)
            ti = self._idx(target_ix, target_iy)
            self._m[ti] = _clamp_int(self._m[ti] + move_i, 0, _M_MAX_INT)

        # -- 5.3 Slip groove (anisotropic H reduction along slip) --
        if slip_mag > slip_thresh and params.slip_track_k > 0.0:
            dh_slip_m = params.slip_track_k * slip_mag * dt * 0.05
            dh_slip_i = int(round(dh_slip_m * _H_SCALE))
            self._h[idx] = _clamp_int(
                self._h[idx] - dh_slip_i, _H_MIN, _H_MAX)

    def relax(self, dt: float, storm_multiplier: float = 1.0) -> None:
        """Relax H and M fields toward resting state.

        H relaxes toward 0.
        M relaxes toward m_base.
        Storm/wind multiplier accelerates relaxation.
        """
        if dt <= 0.0:
            return

        # We use per-material tau via the params passed at construction;
        # here we use a default tau baked in for the whole field.
        # The integrator passes storm_multiplier to speed up during storms.
        sm = max(storm_multiplier, 1.0)

        # H relaxation: H *= exp(-dt * sm / tau_h)
        # Without material reference here we use a stored tau set from outside.
        # Default: tau_h = 120 s for dust-like behaviour.
        tau_h = getattr(self, "_tau_h", 120.0)
        tau_m = getattr(self, "_tau_m", 90.0)

        decay_h = math.exp(-dt * sm / tau_h)
        decay_m = math.exp(-dt * sm / tau_m)

        m_base_i = int(round(self.m_base * _M_MAX_INT))

        n = self.grid_res * self.grid_res
        for i in range(n):
            # H toward 0
            self._h[i] = int(round(self._h[i] * decay_h))
            # M toward m_base
            m_diff     = m_base_i - self._m[i]
            self._m[i] = _clamp_int(
                self._m[i] + int(round(m_diff * (1.0 - decay_m))),
                0, _M_MAX_INT,
            )

    def set_relaxation_taus(self, tau_h: float, tau_m: float) -> None:
        """Override relaxation time constants [s]."""
        self._tau_h = max(tau_h, 1.0)
        self._tau_m = max(tau_m, 1.0)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_bytes_h(self) -> bytes:
        """Pack H field as little-endian int16 bytes."""
        return struct.pack(f"<{len(self._h)}h", *self._h)

    def to_bytes_m(self) -> bytes:
        """Pack M field as little-endian uint16 bytes."""
        return struct.pack(f"<{len(self._m)}H", *self._m)

    def from_bytes_h(self, data: bytes) -> None:
        """Unpack H field from little-endian int16 bytes."""
        n = self.grid_res * self.grid_res
        self._h = list(struct.unpack(f"<{n}h", data))

    def from_bytes_m(self, data: bytes) -> None:
        """Unpack M field from little-endian uint16 bytes."""
        n = self.grid_res * self.grid_res
        self._m = list(struct.unpack(f"<{n}H", data))

    def field_hash(self) -> str:
        """Stable MD5 hash of both fields (for determinism tests)."""
        h_bytes = self.to_bytes_h()
        m_bytes = self.to_bytes_m()
        return hashlib.md5(h_bytes + m_bytes).hexdigest()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _idx(self, ix: int, iy: int) -> int:
        return iy * self.grid_res + ix

    def _neighbours(self, ix: int, iy: int) -> list[Tuple[int, int]]:
        """Return valid 4-cardinal neighbours."""
        res = self.grid_res
        result = []
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nx, ny = ix + dx, iy + dy
            if 0 <= nx < res and 0 <= ny < res:
                result.append((nx, ny))
        return result
