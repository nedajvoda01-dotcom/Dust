"""InsolationField — Stage 7 spherical insolation grid.

Maintains a W×H equirectangular grid of InsolCell values, keyed by
lat/long in the **planet's rotating frame** (fixed to the surface).

Each cell stores the insolation contributed by both suns, ring shadow,
and eclipse factor, computed by calling AstroSystem.sample_insolation for
the cell's world-space position at the time of the update.

Tiled updates spread cost across multiple ticks; double-buffering
(prev / next) prevents sampling artefacts between updates.

API
---
update(game_time, astro, planet_radius)       — one tick (partial update)
force_full_update(game_time, astro, planet_r) — synchronous full update
sample_at(world_pos) -> InsolCell             — O(1) bilinear sample
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from src.core.Config import Config
from src.math.PlanetMath import PlanetMath, LatLong
from src.math.Vec3 import Vec3
from src.systems.AstroSystem import AstroSystem

_TWO_PI = 2.0 * math.pi
_HALF_PI = math.pi * 0.5


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------

@dataclass
class InsolCell:
    """Per-cell insolation data stored in the field."""
    direct1: float = 0.0          # primary sun contribution
    direct2: float = 0.0          # secondary sun contribution
    direct_total: float = 0.0     # combined (ring shadow + eclipse applied)
    ring_shadow_eff: float = 0.0  # 0 = clear, 1 = full ring shadow
    eclipse_eff: float = 0.0      # 0 = no eclipse, 1 = maximum overlap


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rotate_y(v: Vec3, angle: float) -> Vec3:
    """Rotate Vec3 around the Y axis by *angle* radians."""
    c, s = math.cos(angle), math.sin(angle)
    return Vec3(v.x * c + v.z * s, v.y, -v.x * s + v.z * c)


class _FieldBuffer:
    """Flat parallel arrays for one W×H insolation grid.

    Using separate arrays avoids per-cell object allocation and is
    cache-friendly for the bilinear sampler.
    """
    __slots__ = ("direct1", "direct2", "direct_total",
                 "ring_shadow_eff", "eclipse_eff")

    def __init__(self, size: int) -> None:
        self.direct1: list[float] = [0.0] * size
        self.direct2: list[float] = [0.0] * size
        self.direct_total: list[float] = [0.0] * size
        self.ring_shadow_eff: list[float] = [0.0] * size
        self.eclipse_eff: list[float] = [0.0] * size


# ---------------------------------------------------------------------------
# InsolationField
# ---------------------------------------------------------------------------

class InsolationField:
    """Equirectangular insolation field on the planet sphere.

    Grid layout:
      row 0 → south pole (-90° lat), row H-1 → north pole (+90° lat)
      col 0 → -180° lon, col W-1 → +180° lon (cell centres)

    Cells are indexed in the planet's rotating frame: the world position of
    each cell is ``_rotate_y(cell_dir, spin_angle) * planet_radius``, so the
    field always reflects "what sun exposure does this geographic location
    currently have?".

    Tiled update strategy:
      Each call to ``update()`` processes ``cells_per_tick`` cells in the
      ``_next`` buffer.  When the buffer is complete it is swapped with
      ``_prev`` and a new cycle begins.  Sampling always reads from
      ``_prev`` (the most recently completed full snapshot).
    """

    def __init__(
        self,
        config: Config,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        self._w: int = (
            width if width is not None
            else config.get("insolation", "field_width", default=64)
        )
        self._h: int = (
            height if height is not None
            else config.get("insolation", "field_height", default=32)
        )
        self._n: int = self._w * self._h

        # Cells processed per update() call.
        # Default targets a full pass in 10 ticks.
        raw_cpt = config.get("insolation", "cells_per_tick", default=None)
        self._cells_per_tick: int = (
            int(raw_cpt) if raw_cpt is not None else max(1, self._n // 10)
        )

        # Double buffers — _prev is always the latest completed snapshot
        self._prev = _FieldBuffer(self._n)
        self._next = _FieldBuffer(self._n)

        # Tiled-update state
        self._next_cursor: int = 0
        self._next_complete: bool = True  # triggers first swap on update()

        # Pre-compute per-cell planet-frame unit directions (constant)
        self._cell_dirs: list[Vec3] = self._precompute_dirs()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        game_time: float,
        astro: AstroSystem,
        planet_radius: float,
    ) -> None:
        """Advance insolation computation by one tick (tiled update).

        Call once per game frame (or at desired update rate).
        When the in-progress ``_next`` buffer is complete, it is swapped
        to become the new ``_prev`` and a fresh cycle begins.
        """
        if self._next_complete:
            self._prev, self._next = self._next, self._prev
            self._next_cursor = 0
            self._next_complete = False

        end = min(self._next_cursor + self._cells_per_tick, self._n)
        spin = astro.spin_angle
        eclipse_eff = astro.get_eclipse_factor()

        for idx in range(self._next_cursor, end):
            local_dir = self._cell_dirs[idx]
            world_dir = _rotate_y(local_dir, spin)
            world_pos = world_dir * planet_radius
            s = astro.sample_insolation(world_pos, world_dir)
            self._next.direct1[idx] = s.direct1
            self._next.direct2[idx] = s.direct2
            self._next.direct_total[idx] = s.total_direct
            self._next.ring_shadow_eff[idx] = s.ring_shadow
            self._next.eclipse_eff[idx] = eclipse_eff

        self._next_cursor = end
        if self._next_cursor >= self._n:
            self._next_complete = True

    def force_full_update(
        self,
        game_time: float,
        astro: AstroSystem,
        planet_radius: float,
    ) -> None:
        """Synchronously update all cells and swap to ``_prev``.

        Useful for deterministic tests, initial seeding, or when the
        caller can afford the one-off cost.
        """
        spin = astro.spin_angle
        eclipse_eff = astro.get_eclipse_factor()
        buf = self._next

        for idx in range(self._n):
            local_dir = self._cell_dirs[idx]
            world_dir = _rotate_y(local_dir, spin)
            world_pos = world_dir * planet_radius
            s = astro.sample_insolation(world_pos, world_dir)
            buf.direct1[idx] = s.direct1
            buf.direct2[idx] = s.direct2
            buf.direct_total[idx] = s.total_direct
            buf.ring_shadow_eff[idx] = s.ring_shadow
            buf.eclipse_eff[idx] = eclipse_eff

        # Swap so _prev holds the fresh snapshot
        self._prev, self._next = self._next, self._prev
        self._next_cursor = 0
        self._next_complete = True

    def sample_at(self, world_pos: Vec3) -> InsolCell:
        """Bilinear sample from the insolation field.

        Parameters
        ----------
        world_pos:
            A planet-frame direction (or any world-space vector from the
            planet centre): lat/long is extracted and used to index the
            field.  The caller should pass the **planet-frame** position of
            the surface point (i.e., before applying spin rotation).

        Returns
        -------
        InsolCell
            Bilinearly interpolated cell values.  O(1) regardless of
            field resolution.
        """
        ll = PlanetMath.from_direction(world_pos)
        u = (ll.lon_rad + math.pi) / _TWO_PI        # [0, 1]
        v = (ll.lat_rad + _HALF_PI) / math.pi       # [0, 1]

        u = max(0.0, min(1.0, u))
        v = max(0.0, min(1.0, v))

        fx = u * (self._w - 1)
        fy = v * (self._h - 1)

        x0 = int(fx)
        y0 = int(fy)
        x1 = min(x0 + 1, self._w - 1)
        y1 = min(y0 + 1, self._h - 1)
        tx = fx - x0
        ty = fy - y0

        buf = self._prev
        i00 = y0 * self._w + x0
        i10 = y0 * self._w + x1
        i01 = y1 * self._w + x0
        i11 = y1 * self._w + x1

        def _bilerp(arr: list[float]) -> float:
            v00, v10 = arr[i00], arr[i10]
            v01, v11 = arr[i01], arr[i11]
            return (
                (v00 * (1.0 - tx) + v10 * tx) * (1.0 - ty)
                + (v01 * (1.0 - tx) + v11 * tx) * ty
            )

        return InsolCell(
            direct1=_bilerp(buf.direct1),
            direct2=_bilerp(buf.direct2),
            direct_total=_bilerp(buf.direct_total),
            ring_shadow_eff=_bilerp(buf.ring_shadow_eff),
            eclipse_eff=_bilerp(buf.eclipse_eff),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _precompute_dirs(self) -> list[Vec3]:
        """Pre-compute planet-frame unit direction for each cell centre."""
        dirs: list[Vec3] = []
        for y in range(self._h):
            lat = -_HALF_PI + math.pi * (y + 0.5) / self._h
            for x in range(self._w):
                lon = -math.pi + _TWO_PI * (x + 0.5) / self._w
                dirs.append(PlanetMath.direction_from_lat_long(LatLong(lat, lon)))
        return dirs
