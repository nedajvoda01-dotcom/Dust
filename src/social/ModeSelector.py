"""ModeSelector — Stage 39 social mode selection with hysteresis (§7).

Selects one of seven :class:`SocialMode` values based on perception state,
nearby agent states, and personality.  Each mode is gated by its own
:class:`~src.intent.ModeHysteresis.ModeHysteresis` to prevent rapid
oscillation (§13 — mode_hysteresis_margin).

Social modes (§7)
-----------------
1. Ignore             — too far, or own risk too high
2. AcknowledgePresence — noticed, brief body-pause + orient
3. ParallelWalk       — moving alongside at matched speed
4. Follow             — other leads; low visibility
5. YieldPath          — trajectory collision predicted
6. AssistPrep         — other slipping, self safe and helpful
7. Regroup            — separation after prior proximity

Public API
----------
SocialMode (enum)
SocialModeSelector(config=None)
  .select(perception_state, others, personality, yield_bias,
          alignment_strength, sim_time) → SocialMode
"""
from __future__ import annotations

import enum
import math
from typing import Dict, List, Optional, Tuple

from src.math.Vec3 import Vec3
from src.perception.PerceptionSystem import PerceptionState
from src.intent.ModeHysteresis import ModeHysteresis
from src.social.SocialNetInputs import SocialAgentState
from src.social.PersonalitySeed import PersonalityParams


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


# ---------------------------------------------------------------------------
# Social mode enumeration
# ---------------------------------------------------------------------------

class SocialMode(enum.Enum):
    """Seven social coupling modes (§7)."""
    Ignore              = 0
    AcknowledgePresence = 1
    ParallelWalk        = 2
    Follow              = 3
    YieldPath           = 4
    AssistPrep          = 5
    Regroup             = 6


# ---------------------------------------------------------------------------
# Hysteresis defaults (enter, exit) per mode
# ---------------------------------------------------------------------------

_HYSTERESIS_DEFAULTS: Dict[SocialMode, Tuple[float, float]] = {
    SocialMode.Ignore:              (0.00, 0.00),
    SocialMode.AcknowledgePresence: (0.25, 0.12),
    SocialMode.ParallelWalk:        (0.35, 0.18),
    SocialMode.Follow:              (0.35, 0.20),
    SocialMode.YieldPath:           (0.30, 0.15),
    SocialMode.AssistPrep:          (0.40, 0.25),
    SocialMode.Regroup:             (0.30, 0.15),
}


# ---------------------------------------------------------------------------
# SocialModeSelector
# ---------------------------------------------------------------------------

class SocialModeSelector:
    """Selects the current social mode using scored hysteresis (§7).

    Parameters
    ----------
    config :
        Optional dict; reads ``social.mode_hysteresis_margin``,
        ``social.assist_prep_threshold``,
        ``social.follow_in_low_visibility_enable``,
        ``social.presence_radius``.
    """

    _DEFAULT_ASSIST_THRESHOLD = 0.4
    _DEFAULT_PRESENCE_RADIUS  = 20.0

    def __init__(self, config: Optional[dict] = None) -> None:
        scfg = ((config or {}).get("social", {})) or {}
        margin = float(scfg.get("mode_hysteresis_margin", 0.1))
        self._assist_threshold = float(
            scfg.get("assist_prep_threshold", self._DEFAULT_ASSIST_THRESHOLD)
        )
        self._presence_radius = float(
            scfg.get("presence_radius", self._DEFAULT_PRESENCE_RADIUS)
        )
        self._follow_low_vis = bool(
            scfg.get("follow_in_low_visibility_enable", True)
        )

        # Build per-mode hysteresis
        self._hysteresis: Dict[SocialMode, ModeHysteresis] = {}
        for mode, (enter, exit_) in _HYSTERESIS_DEFAULTS.items():
            e = _clamp(enter - margin, 0.0, 1.0)
            x = _clamp(exit_  - margin, 0.0, e)
            self._hysteresis[mode] = ModeHysteresis(
                enter_threshold=e,
                exit_threshold=x,
            )

        self._prev_mode = SocialMode.Ignore

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def select(
        self,
        perception:         PerceptionState,
        others:             List[SocialAgentState],
        personality:        PersonalityParams,
        yield_bias:         float = 0.0,
        alignment_strength: float = 0.0,
        sim_time:           float = 0.0,
    ) -> SocialMode:
        """Return the best social mode given current inputs.

        Parameters
        ----------
        perception :
            Own PerceptionState from Stage 37.
        others :
            Nearby social agent states.
        personality :
            Deterministic personality scalars.
        yield_bias :
            Output of CollisionYield.compute [0..1].
        alignment_strength :
            Output of GroupCohesion.compute [0..1].
        sim_time :
            Current simulation time (used for deterministic tie-breaking).
        """
        scores = self._score_all(
            perception, others, personality, yield_bias, alignment_strength
        )

        # Ignore is the unconditional fallback
        self._hysteresis[SocialMode.Ignore].update(1.0)
        best_mode  = SocialMode.Ignore
        best_score = -1.0

        for mode, score in scores.items():
            if mode == SocialMode.Ignore:
                continue
            active = self._hysteresis[mode].update(score)
            if active and score > best_score:
                best_score = score
                best_mode  = mode

        self._prev_mode = best_mode
        return best_mode

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _nearest_other(
        self,
        others: List[SocialAgentState],
        self_pos: Vec3,
    ) -> Optional[SocialAgentState]:
        """Return the closest agent within presence radius, or None."""
        best: Optional[SocialAgentState] = None
        best_dist = math.inf
        for a in others:
            d = (a.position - self_pos).length()
            if d < best_dist:
                best_dist = d
                best = a
        if best_dist > self._presence_radius:
            return None
        return best

    def _score_all(
        self,
        perception:         PerceptionState,
        others:             List[SocialAgentState],
        personality:        PersonalityParams,
        yield_bias:         float,
        alignment_strength: float,
    ) -> Dict[SocialMode, float]:
        """Return a raw activation score for every social mode."""
        own_risk = perception.globalRisk
        presence = perception.presenceNear
        visibility = perception.visibility
        assist_opp = perception.assistOpportunity

        # High risk suppresses most social engagement (§4)
        risk_suppression = _clamp(1.0 - own_risk * 1.5, 0.0, 1.0)

        # Check whether any nearby agent is in distress
        any_slipping = any(a.is_slipping or a.is_stumbling for a in others)

        # AssistPrep: other in distress, self safe, personality helpful (§10)
        assist_score = _clamp(
            assist_opp
            * personality.helpfulness
            * _clamp(1.0 - own_risk, 0.0, 1.0),
            0.0, 1.0,
        ) if assist_opp >= self._assist_threshold else 0.0

        # ParallelWalk: both moving, alignment high, moderate presence (§7)
        parallel_score = _clamp(
            alignment_strength * presence * risk_suppression, 0.0, 1.0
        )

        # Follow: other ahead + low visibility + sociable (§7)
        follow_score = 0.0
        if self._follow_low_vis and visibility < 0.4:
            follow_score = _clamp(
                (1.0 - visibility) * presence * personality.sociability * risk_suppression,
                0.0, 1.0,
            )

        # YieldPath: predicted collision (§9)
        yield_score = _clamp(yield_bias * (0.5 + personality.caution * 0.5), 0.0, 1.0)

        # AcknowledgePresence: other is near, but not moving together (§7)
        ack_score = _clamp(
            presence * personality.sociability * risk_suppression * 0.7,
            0.0, 1.0,
        )

        # Regroup: was close, now separating — bring them back (§7)
        regroup_score = 0.0
        if self._prev_mode in (
            SocialMode.ParallelWalk, SocialMode.Follow,
            SocialMode.AssistPrep, SocialMode.Regroup,
        ):
            # Presence dropped — try to stay together
            regroup_score = _clamp(
                (1.0 - presence) * personality.sociability * risk_suppression * 0.5,
                0.0, 1.0,
            )

        return {
            SocialMode.Ignore:              0.0,
            SocialMode.AcknowledgePresence: ack_score,
            SocialMode.ParallelWalk:        parallel_score,
            SocialMode.Follow:              follow_score,
            SocialMode.YieldPath:           yield_score,
            SocialMode.AssistPrep:          assist_score,
            SocialMode.Regroup:             regroup_score,
        }
