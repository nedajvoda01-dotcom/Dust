"""GroupCohesion — Stage 39 speed/distance alignment for parallel walk (§8).

Computes two outputs that bias the social layer when agents move together:

* ``alignment_strength`` (0..1) — how strongly to match the other's speed/dir
* ``preferred_distance`` (m)    — desired separation, blended from personalities

Public API
----------
GroupCohesion(config=None)
  .compute(self_pos, self_vel, others, personality, shared_risk) → (alignment_strength, preferred_distance)
"""
from __future__ import annotations

import math
from typing import List, Optional

from src.math.Vec3 import Vec3
from src.social.SocialNetInputs import SocialAgentState
from src.social.PersonalitySeed import PersonalityParams


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


class GroupCohesion:
    """Computes alignment strength and preferred inter-agent distance (§8).

    Parameters
    ----------
    config :
        Optional dict; reads ``social.alignment_strength_max`` and
        ``social.presence_radius``.
    """

    _DEFAULT_ALIGN_MAX      = 0.8
    _DEFAULT_PRESENCE_RADIUS = 20.0

    def __init__(self, config: Optional[dict] = None) -> None:
        scfg = ((config or {}).get("social", {})) or {}
        self._align_max:       float = float(
            scfg.get("alignment_strength_max", self._DEFAULT_ALIGN_MAX)
        )
        self._presence_radius: float = float(
            scfg.get("presence_radius", self._DEFAULT_PRESENCE_RADIUS)
        )

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def compute(
        self,
        self_pos:    Vec3,
        self_vel:    Vec3,
        others:      List[SocialAgentState],
        personality: PersonalityParams,
        shared_risk: float = 0.0,
    ) -> tuple:
        """Return ``(alignment_strength, preferred_distance)``.

        Parameters
        ----------
        self_pos, self_vel :
            Own position and current velocity.
        others :
            Nearby agent states (all within presence radius are considered).
        personality :
            Own personality scalars.
        shared_risk :
            ``max(self_risk, other_risk)`` — reduces alignment when risky.
        """
        if not others:
            return 0.0, personality.personalSpace

        # Compute weighted average velocity of visible neighbours
        total_w = 0.0
        vel_acc = Vec3.zero()
        nearest_dist = math.inf

        for agent in others:
            diff = agent.position - self_pos
            dist = diff.length()
            if dist > self._presence_radius or dist < 1e-6:
                continue
            w = (1.0 - dist / self._presence_radius) ** 2
            total_w += w
            vel_acc = vel_acc + agent.velocity * w
            if dist < nearest_dist:
                nearest_dist = dist

        if total_w < 1e-6:
            return 0.0, personality.personalSpace

        avg_vel = vel_acc * (1.0 / total_w)

        # Alignment: high when self and others move at similar speed/direction
        self_speed = self_vel.length()
        avg_speed  = avg_vel.length()

        if self_speed > 1e-3 and avg_speed > 1e-3:
            cos_sim = _clamp(
                self_vel.dot(avg_vel) / (self_speed * avg_speed), -1.0, 1.0
            )
        else:
            cos_sim = 0.0

        # Alignment strength: higher when moving parallel, modulated by sociability
        # and suppressed by high shared_risk
        raw_align = _clamp((cos_sim + 1.0) * 0.5, 0.0, 1.0)
        align = raw_align * personality.sociability * self._align_max
        align = align * _clamp(1.0 - shared_risk, 0.0, 1.0)

        # Preferred distance: personality personal space + risk pushes apart slightly
        preferred = personality.personalSpace * (1.0 + shared_risk * 0.5)

        return _clamp(align, 0.0, 1.0), _clamp(preferred, 0.5, 10.0)
