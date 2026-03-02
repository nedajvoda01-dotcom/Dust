"""ContactImpulseCollector — Stage 36 physics-to-sound contact aggregation.

Collects raw contact data from the physics layer and groups it into
:class:`ContactImpulse` packets that are consumed by :class:`ExcitationGenerator`.

Architecture
------------
Physics tick (every frame)
  → ContactImpulseCollector.record(...)     # immediate, O(1)
  → ContactImpulseCollector.flush(dt)       # call once per audio tick (5–10 ms)
  → list[ContactImpulse]                   # passed to ExcitationGenerator

ContactImpulse fields
---------------------
impulse_magnitude : float    total normal impulse [N·s or normalised]
contact_duration  : float    seconds the contact lasted
material_pair     : tuple    (mat_id_A, mat_id_B) — order-normalised
slip_ratio        : float    0 = pure impact, 1 = pure sliding
contact_area      : float    approximate contact area [m² or normalised]
world_pos         : tuple    (x, y, z) world position

Public API
----------
ContactImpulseCollector(config=None)
  .record(fn, ft, v_rel, mat_a, mat_b, area, duration, world_pos) → None
  .flush(dt) → list[ContactImpulse]
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# ContactImpulse
# ---------------------------------------------------------------------------

@dataclass
class ContactImpulse:
    """Aggregated contact event ready for excitation generation."""
    impulse_magnitude: float
    contact_duration:  float
    material_pair:     Tuple[int, int]
    slip_ratio:        float
    contact_area:      float
    world_pos:         Tuple[float, float, float]


# ---------------------------------------------------------------------------
# ContactImpulseCollector
# ---------------------------------------------------------------------------

class ContactImpulseCollector:
    """Receives per-frame physics contact data and emits aggregated impulses.

    Parameters
    ----------
    config :
        Optional dict; uses ``audio.network_impulse_hz`` (default 300) to
        determine the internal aggregation rate.
    max_pending : int
        Hard cap on buffered contacts before forced flush.
    """

    _DEFAULT_IMPULSE_HZ = 300.0

    def __init__(
        self,
        config: Optional[dict] = None,
        max_pending: int = 512,
    ) -> None:
        audio = (config or {}).get("audio", {})
        self._impulse_hz: float = float(
            audio.get("network_impulse_hz", self._DEFAULT_IMPULSE_HZ)
        )
        self._flush_interval: float = 1.0 / max(1.0, self._impulse_hz)
        self._max_pending = max_pending

        self._pending: List[_RawContact] = []
        self._acc_dt:  float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(
        self,
        fn:        float,
        ft:        float,
        v_rel:     float,
        mat_a:     int,
        mat_b:     int,
        area:      float,
        duration:  float,
        world_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        """Record a raw physics contact sample.

        Parameters
        ----------
        fn        : normal force [N or normalised]
        ft        : tangential (friction) force magnitude
        v_rel     : relative contact-point speed [m/s or normalised]
        mat_a/b   : material IDs of the two surfaces
        area      : contact area
        duration  : contact duration this frame [s]
        world_pos : world position of the contact point
        """
        if len(self._pending) >= self._max_pending:
            return   # budget guard: silently drop when saturated
        self._pending.append(
            _RawContact(fn, ft, v_rel, mat_a, mat_b, area, duration, world_pos)
        )

    def flush(self, dt: float) -> List[ContactImpulse]:
        """Advance the internal clock and return any accumulated impulses.

        Should be called every audio tick (typically every 5–10 ms).  Returns
        a list that may be empty if the interval has not elapsed yet.
        """
        self._acc_dt += dt
        if self._acc_dt < self._flush_interval and self._pending:
            # not yet at flush boundary — accumulate
            return []

        self._acc_dt = 0.0
        raw = self._pending
        self._pending = []
        if not raw:
            return []
        return _aggregate(raw)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

@dataclass
class _RawContact:
    fn:        float
    ft:        float
    v_rel:     float
    mat_a:     int
    mat_b:     int
    area:      float
    duration:  float
    world_pos: Tuple[float, float, float]


def _aggregate(raw: List[_RawContact]) -> List[ContactImpulse]:
    """Group raw contacts by normalised material pair and aggregate.

    Contacts with the same material pair are summed; each unique pair
    produces at most one :class:`ContactImpulse` per flush batch.
    """
    buckets: dict = {}
    for c in raw:
        key = (min(c.mat_a, c.mat_b), max(c.mat_a, c.mat_b))
        if key not in buckets:
            buckets[key] = _Accum(key)
        buckets[key].add(c)

    return [acc.to_impulse() for acc in buckets.values()]


class _Accum:
    """Accumulator for one material-pair bucket."""

    def __init__(self, pair: Tuple[int, int]) -> None:
        self.pair      = pair
        self.total_fn  = 0.0
        self.total_ft  = 0.0
        self.max_v_rel = 0.0
        self.total_dur = 0.0
        self.total_area = 0.0
        self.count     = 0
        self._pos_x    = 0.0
        self._pos_y    = 0.0
        self._pos_z    = 0.0

    def add(self, c: _RawContact) -> None:
        self.total_fn   += c.fn * c.duration
        self.total_ft   += c.ft * c.duration
        self.max_v_rel   = max(self.max_v_rel, c.v_rel)
        self.total_dur  += c.duration
        self.total_area += c.area
        self.count      += 1
        self._pos_x     += c.world_pos[0]
        self._pos_y     += c.world_pos[1]
        self._pos_z     += c.world_pos[2]

    def to_impulse(self) -> ContactImpulse:
        dur = max(1e-9, self.total_dur)
        avg_fn = self.total_fn / dur
        avg_ft = self.total_ft / dur
        impulse_mag = math.sqrt(avg_fn * avg_fn + avg_ft * avg_ft) * dur
        slip = avg_ft / max(1e-9, avg_fn + avg_ft)
        n = max(1, self.count)
        return ContactImpulse(
            impulse_magnitude=impulse_mag,
            contact_duration=dur,
            material_pair=self.pair,
            slip_ratio=min(1.0, slip),
            contact_area=self.total_area / n,
            world_pos=(
                self._pos_x / n,
                self._pos_y / n,
                self._pos_z / n,
            ),
        )
