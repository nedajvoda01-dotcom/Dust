"""GraspConstraintBinder — Stage 40 §8.

Server-authoritative constraint factory.  Only the server creates and
destroys :class:`GraspConstraint` objects.  Clients submit
:class:`ContactCandidate` proposals; the server validates them and
either confirms or rejects.

Public API
----------
GraspType (enum)
ContactCandidate (dataclass)
GraspConstraint (dataclass)

GraspConstraintBinder(config=None)
  .propose(candidate)           → bool          # client call
  .tick(dt, sim_time)           → list[ConstraintEvent]
  .get_active(player_id)        → GraspConstraint | None
  .force_break(constraint_id)   → None
"""
from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.math.Vec3 import Vec3


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


# ---------------------------------------------------------------------------
# GraspType
# ---------------------------------------------------------------------------

class GraspType(enum.Enum):
    """Supported contact configurations (§4)."""
    HAND_TO_HAND       = "hand_to_hand"
    HAND_TO_FOREARM    = "hand_to_forearm"
    HAND_TO_SUIT_HARNESS = "hand_to_suit_harness"
    WRIST_TETHER       = "wrist_tether"   # optional tether (§4)


# ---------------------------------------------------------------------------
# ContactCandidate — proposed by client
# ---------------------------------------------------------------------------

@dataclass
class ContactCandidate:
    """A grasp-contact proposal submitted to the server binder (§7).

    Attributes
    ----------
    player_a, player_b :
        Integer player IDs; A is the one who needs help.
    grasp_type :
        Which contact configuration was detected.
    anchor_a, anchor_b :
        Local anchor positions (bone-space) on each character.
    normal :
        Contact normal direction (world space).
    relative_velocity :
        Relative speed at contact point [m/s].
    estimated_force_capacity :
        Max force the contact geometry can bear [N or normalised].
    tick :
        Server tick at which this candidate was generated.
    """
    player_a:                int
    player_b:                int
    grasp_type:              GraspType
    anchor_a:                Vec3
    anchor_b:                Vec3
    normal:                  Vec3
    relative_velocity:       float
    estimated_force_capacity: float
    tick:                    int = 0


# ---------------------------------------------------------------------------
# GraspConstraint — server-confirmed constraint
# ---------------------------------------------------------------------------

@dataclass
class GraspConstraint:
    """A live grasp constraint maintained by the server (§8).

    Attributes
    ----------
    id :
        Unique constraint identifier.
    player_a, player_b :
        Player IDs.
    anchor_a, anchor_b :
        Bone-space anchor positions.
    max_force :
        Force magnitude at which the constraint transmits full tension [N].
    break_force :
        Force magnitude above which the constraint breaks [N].
    damping :
        Velocity damping coefficient at the joint.
    rest_length :
        Natural length of the constraint [m]; 0 = hard grasp.
    created_at_tick :
        Server tick when this constraint was created.
    current_force :
        Running constraint force estimate (updated by solver).
    """
    id:             int
    player_a:       int
    player_b:       int
    anchor_a:       Vec3
    anchor_b:       Vec3
    grasp_type:     GraspType
    max_force:      float
    break_force:    float
    damping:        float
    rest_length:    float
    created_at_tick: int
    current_force:  float = 0.0


# ---------------------------------------------------------------------------
# ConstraintEvent — binder output
# ---------------------------------------------------------------------------

@dataclass
class ConstraintEvent:
    """A create / update / break event for network replication (§12.3)."""
    kind:       str   # "create" | "update" | "break"
    constraint: GraspConstraint


# ---------------------------------------------------------------------------
# GraspConstraintBinder
# ---------------------------------------------------------------------------

class GraspConstraintBinder:
    """Server-authoritative grasp constraint manager (§8).

    Parameters
    ----------
    config :
        Optional dict; reads ``grasp.*`` keys.
    """

    _DEFAULT_MAX_FORCE       = 800.0    # N (normalised units)
    _DEFAULT_BREAK_FORCE     = 1200.0
    _DEFAULT_DAMPING         = 0.85
    _DEFAULT_MAX_PER_PLAYER  = 1        # MVP: one grasp per player (§15)
    _DEFAULT_CONFIRM_TIMEOUT = 200      # ms

    def __init__(self, config: Optional[dict] = None) -> None:
        gcfg = (config or {}).get("grasp", {}) or {}
        self._max_force:    float = float(gcfg.get("max_force",          self._DEFAULT_MAX_FORCE))
        self._break_force:  float = float(gcfg.get("break_force",        self._DEFAULT_BREAK_FORCE))
        self._damping:      float = float(gcfg.get("damping",            self._DEFAULT_DAMPING))
        self._max_per_player: int = int(gcfg.get("max_active_per_player", self._DEFAULT_MAX_PER_PLAYER))
        self._confirm_timeout_ms: float = float(gcfg.get("server_confirm_timeout_ms", self._DEFAULT_CONFIRM_TIMEOUT))

        self._active: Dict[int, GraspConstraint] = {}   # id → constraint
        self._player_constraints: Dict[int, int] = {}   # player_id → constraint_id
        self._next_id: int = 1
        self._current_tick: int = 0
        self._events: List[ConstraintEvent] = []

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    @property
    def active_constraints(self) -> Dict[int, GraspConstraint]:
        """Read-only view of currently active constraints."""
        return self._active

    def propose(self, candidate: ContactCandidate) -> bool:
        """Validate and optionally confirm a client-proposed contact candidate.

        Returns ``True`` if a new :class:`GraspConstraint` was created.
        """
        # Check per-player limit (§15)
        if self._player_has_active_grasp(candidate.player_a):
            return False
        if self._player_has_active_grasp(candidate.player_b):
            return False

        # Sanity: the two players must be distinct
        if candidate.player_a == candidate.player_b:
            return False

        constraint = GraspConstraint(
            id=self._next_id,
            player_a=candidate.player_a,
            player_b=candidate.player_b,
            anchor_a=candidate.anchor_a,
            anchor_b=candidate.anchor_b,
            grasp_type=candidate.grasp_type,
            max_force=min(self._max_force, candidate.estimated_force_capacity),
            break_force=self._break_force,
            damping=self._damping,
            rest_length=0.0 if candidate.grasp_type != GraspType.WRIST_TETHER else 1.5,
            created_at_tick=self._current_tick,
            current_force=0.0,
        )

        self._next_id += 1
        self._active[constraint.id] = constraint
        self._player_constraints[candidate.player_a] = constraint.id
        self._player_constraints[candidate.player_b] = constraint.id
        self._events.append(ConstraintEvent(kind="create", constraint=constraint))
        return True

    def tick(self, dt: float, sim_time: float) -> List[ConstraintEvent]:
        """Advance the binder by one server tick.

        Returns accumulated :class:`ConstraintEvent` objects since last call.
        These should be replicated to all clients (§12.3).
        """
        self._current_tick += 1
        evts = list(self._events)
        self._events.clear()
        return evts

    def get_active(self, player_id: int) -> Optional[GraspConstraint]:
        """Return the active constraint for *player_id*, or ``None``."""
        cid = self._player_constraints.get(player_id)
        if cid is None:
            return None
        return self._active.get(cid)

    def force_break(self, constraint_id: int) -> None:
        """Immediately break a constraint (server forces, e.g. on lag/de-sync §15)."""
        constraint = self._active.pop(constraint_id, None)
        if constraint is None:
            return
        self._player_constraints.pop(constraint.player_a, None)
        self._player_constraints.pop(constraint.player_b, None)
        self._events.append(ConstraintEvent(kind="break", constraint=constraint))

    def update_force(self, constraint_id: int, force: float) -> Optional[ConstraintEvent]:
        """Update a constraint's current force and break it if over threshold.

        Returns a break event if the constraint was broken, otherwise ``None``.
        """
        c = self._active.get(constraint_id)
        if c is None:
            return None
        c.current_force = force
        if force > c.break_force:
            self.force_break(constraint_id)
            # The break event is queued in _events; retrieve it
            if self._events and self._events[-1].kind == "break":
                return self._events[-1]
        return None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _player_has_active_grasp(self, player_id: int) -> bool:
        cid = self._player_constraints.get(player_id)
        if cid is None:
            return False
        return cid in self._active
