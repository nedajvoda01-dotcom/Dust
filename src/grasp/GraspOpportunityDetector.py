"""GraspOpportunityDetector — Stage 40 §5.

Analyses per-tick inputs from Perception (Stage 37) and SocialCoupler
(Stage 39) to produce a scored :class:`GraspCandidate` when a grasp
opportunity exists.

Score formula (§5)::

    score = reachable * (1 - selfRisk) * otherNeedsHelp * timeToInterceptFactor

If ``score >= config.grasp.score_threshold``, ``candidate`` is non-None and
the caller should trigger :class:`~src.grasp.ReachPlanner.ReachPlanner`.

Public API
----------
GraspOpportunityDetector(config=None)
  .detect(self_pos, self_vel, other_pos, other_vel,
          self_risk, assist_opportunity, other_is_slipping,
          other_risk, dt) → GraspCandidate | None
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from src.math.Vec3 import Vec3


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


# ---------------------------------------------------------------------------
# GraspCandidate
# ---------------------------------------------------------------------------

@dataclass
class GraspCandidate:
    """A scored grasp opportunity emitted by the detector.

    Attributes
    ----------
    score :
        Composite opportunity score in [0..1]; higher = more urgent.
    other_pos :
        World position of the player who needs help.
    other_vel :
        Current velocity of the player who needs help.
    distance :
        Euclidean distance between self and other [m].
    """
    score:     float
    other_pos: Vec3
    other_vel: Vec3
    distance:  float


# ---------------------------------------------------------------------------
# GraspOpportunityDetector
# ---------------------------------------------------------------------------

class GraspOpportunityDetector:
    """Detects and scores grasp opportunities (§5).

    Parameters
    ----------
    config :
        Optional dict; reads ``grasp.*`` keys.
    """

    _DEFAULT_REACH_RADIUS    = 2.0   # metres
    _DEFAULT_SCORE_THRESHOLD = 0.35
    _DEFAULT_TTC_MAX         = 3.0   # seconds; time-to-contact horizon

    def __init__(self, config: Optional[dict] = None) -> None:
        gcfg = (config or {}).get("grasp", {}) or {}
        self._reach_radius:    float = float(gcfg.get("reach_radius",    self._DEFAULT_REACH_RADIUS))
        self._score_threshold: float = float(gcfg.get("score_threshold", self._DEFAULT_SCORE_THRESHOLD))
        self._ttc_max:         float = float(gcfg.get("ttc_max_sec",     self._DEFAULT_TTC_MAX))

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def detect(
        self,
        self_pos:           Vec3,
        self_vel:           Vec3,
        other_pos:          Vec3,
        other_vel:          Vec3,
        self_risk:          float,
        assist_opportunity: float,
        other_is_slipping:  bool  = False,
        other_risk:         float = 0.0,
        dt:                 float = 0.0,
    ) -> Optional[GraspCandidate]:
        """Evaluate one potential grasp target.

        Returns a :class:`GraspCandidate` if the opportunity is above the
        configured threshold, otherwise ``None``.
        """
        diff = other_pos - self_pos
        dist = diff.length()

        # 1. Reachable factor — 1 when inside reach radius, falls off outside
        if dist < 1e-6:
            reachable = 0.0
        elif dist <= self._reach_radius:
            reachable = 1.0 - (dist / self._reach_radius) * 0.5   # [0.5..1]
        else:
            reachable = 0.0   # out of reach: hard cutoff

        # 2. Self-safety factor
        self_safe = _clamp(1.0 - self_risk, 0.0, 1.0)

        # 3. Other-needs-help factor
        slip_factor = 1.0 if other_is_slipping else 0.0
        risk_factor = _clamp(other_risk, 0.0, 1.0)
        other_needs = _clamp(
            slip_factor * 0.6 + risk_factor * 0.4 + assist_opportunity * 0.5,
            0.0, 1.0
        )

        # 4. Time-to-intercept factor (§5) — relative closing speed
        rel_vel = other_vel - self_vel
        closing  = -rel_vel.dot(diff) / max(dist, 1e-6)   # positive = approaching
        # If already very close, time-to-intercept is short → factor near 1
        if dist < self._reach_radius:
            ttc = dist / max(abs(closing), 0.1)
            ttc_factor = _clamp(1.0 - ttc / self._ttc_max, 0.0, 1.0)
            # Slipping players require immediate response — raise floor
            floor = 0.7 if other_is_slipping else 0.3
            ttc_factor = max(ttc_factor, floor)
        else:
            ttc_factor = 0.0   # not reachable; already zero from reachable

        score = reachable * self_safe * other_needs * ttc_factor

        if score < self._score_threshold:
            return None

        return GraspCandidate(
            score=_clamp(score, 0.0, 1.0),
            other_pos=other_pos,
            other_vel=other_vel,
            distance=dist,
        )
