"""SocialCoupler — Stage 39 Emergent Social Coupling orchestrator.

Sits between Stage 37 PerceptionSystem and Stage 38 IntentArbitrator and
adds social biases to motor-intent selection based on the presence and
state of other nearby players.

Architecture (§3)::

    PerceptionState + List[SocialAgentState]
            ↓
    SocialCoupler
            ↓
    SocialBiases
            ↓
    IntentArbitrator (38)
            ↓
    MotorIntent → MotorStack

SocialCoupler does **not** move the body directly — it emits
:class:`SocialBiases` that the IntentArbitrator layer can use to
modulate strategy selection and intent construction.

Public API
----------
SocialBiases (dataclass)
  .preferredDistance       float (m)
  .maxApproachSpeedScale   float [0..1]
  .alignmentStrength       float [0..1]
  .attentionToOther        float [0..1]
  .assistBias              float [0..1]
  .yieldBias               float [0..1]
  .yieldDir                Vec3
  .jointHazardBias         float [0..1]

SocialCoupler(config=None, sim_time=0.0, player_id=0)
  .update(dt, sim_time, perception_state, others) → SocialBiases
  .current_mode      → SocialMode
  .current_biases    → SocialBiases
  .debug_info()      → dict
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

from src.math.Vec3 import Vec3
from src.perception.PerceptionSystem import PerceptionState
from src.social.SocialNetInputs import SocialAgentState
from src.social.PersonalitySeed import PersonalitySeed, PersonalityParams
from src.social.ModeSelector import SocialModeSelector, SocialMode
from src.social.CollisionYield import CollisionYield
from src.social.GroupCohesion import GroupCohesion


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


# ---------------------------------------------------------------------------
# SocialBiases — output struct consumed by IntentArbitrator
# ---------------------------------------------------------------------------

@dataclass
class SocialBiases:
    """Social coupling biases produced by SocialCoupler (§4).

    All scalars are in [0..1] unless otherwise noted.
    """
    preferredDistance:     float = 2.0    # metres — desired inter-agent gap
    maxApproachSpeedScale: float = 1.0    # caps approach speed when closing
    alignmentStrength:     float = 0.0    # match other's velocity direction/speed
    attentionToOther:      float = 0.0    # orient gaze/torso toward nearest agent
    assistBias:            float = 0.0    # raise assistWillingness in IntentArbitrator
    yieldBias:             float = 0.0    # raise yieldBias when on collision course
    yieldDir:              Vec3  = field(default_factory=Vec3.zero)  # avoidance dir
    jointHazardBias:       float = 0.0    # move toward safer shared zone


# ---------------------------------------------------------------------------
# SocialCoupler
# ---------------------------------------------------------------------------

class SocialCoupler:
    """Orchestrates all social coupling sub-modules and emits SocialBiases.

    Parameters
    ----------
    config :
        Optional dict; reads ``social.*`` keys.
    sim_time :
        Initial simulation time [s].
    player_id :
        Stable integer player identifier — used for deterministic personality.
    """

    _DEFAULT_TICK_HZ = 5.0

    def __init__(
        self,
        config:    Optional[dict] = None,
        sim_time:  float          = 0.0,
        player_id: int            = 0,
    ) -> None:
        self._cfg       = config or {}
        self._player_id = int(player_id)

        scfg = self._cfg.get("social", {}) or {}
        self._tick_dt: float = 1.0 / max(1.0, float(
            scfg.get("tick_hz", self._DEFAULT_TICK_HZ)
        ))
        self._t_next: float = sim_time

        # Sub-modules
        self._personality_seed = PersonalitySeed(config)
        self._personality: PersonalityParams = self._personality_seed.generate(player_id)

        self._mode_selector  = SocialModeSelector(config)
        self._collision_yield = CollisionYield(config)
        self._group_cohesion  = GroupCohesion(config)

        # State
        self._current_mode   = SocialMode.Ignore
        self._current_biases = SocialBiases()

        # Own position/velocity cache (updated by caller via perception env)
        self._self_pos: Vec3 = Vec3.zero()
        self._self_vel: Vec3 = Vec3.zero()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    @property
    def current_mode(self) -> SocialMode:
        """Most recently selected social mode."""
        return self._current_mode

    @property
    def current_biases(self) -> SocialBiases:
        """Most recently computed SocialBiases."""
        return self._current_biases

    def update(
        self,
        dt:               float,
        sim_time:         float,
        perception:       PerceptionState,
        others:           List[SocialAgentState],
        self_pos:         Optional[Vec3] = None,
        self_vel:         Optional[Vec3] = None,
    ) -> SocialBiases:
        """Advance the social coupler and return current SocialBiases.

        Parameters
        ----------
        dt :
            Elapsed time since last call [s].
        sim_time :
            Current absolute simulation time [s].
        perception :
            Own PerceptionState from Stage 37.
        others :
            Replicated states of other nearby players.
        self_pos :
            Own world position (optional; cached between calls).
        self_vel :
            Own current velocity (optional; cached between calls).
        """
        if self_pos is not None:
            self._self_pos = self_pos
        if self_vel is not None:
            self._self_vel = self_vel

        if sim_time < self._t_next:
            return self._current_biases

        self._t_next = sim_time + self._tick_dt

        if not others:
            self._current_mode   = SocialMode.Ignore
            self._current_biases = SocialBiases()
            return self._current_biases

        # 1. Collision yield (§9)
        nearest = self._nearest_other(others)
        yield_bias = 0.0
        yield_dir  = Vec3.zero()
        if nearest is not None:
            yield_bias, yield_dir = self._collision_yield.compute(
                self_pos=self._self_pos,
                self_vel=self._self_vel,
                other_pos=nearest.position,
                other_vel=nearest.velocity,
                self_risk=perception.globalRisk,
                other_risk=nearest.global_risk,
                caution=self._personality.caution,
            )

        # 2. Group cohesion (§8)
        shared_risk = max(
            perception.globalRisk,
            max((a.global_risk for a in others), default=0.0),
        )
        alignment_strength, preferred_dist = self._group_cohesion.compute(
            self_pos=self._self_pos,
            self_vel=self._self_vel,
            others=others,
            personality=self._personality,
            shared_risk=shared_risk,
        )

        # 3. Social mode selection (§7)
        mode = self._mode_selector.select(
            perception=perception,
            others=others,
            personality=self._personality,
            yield_bias=yield_bias,
            alignment_strength=alignment_strength,
            sim_time=sim_time,
        )

        # 4. Build SocialBiases from mode + sub-module outputs (§4)
        biases = self._build_biases(
            mode=mode,
            perception=perception,
            nearest=nearest,
            yield_bias=yield_bias,
            yield_dir=yield_dir,
            alignment_strength=alignment_strength,
            preferred_dist=preferred_dist,
            shared_risk=shared_risk,
        )

        self._current_mode   = mode
        self._current_biases = biases
        return biases

    def debug_info(self) -> dict:
        """Return a serialisable snapshot for logging / dev overlays (§16)."""
        b = self._current_biases
        return {
            "social_mode":           self._current_mode.name,
            "preferredDistance":     b.preferredDistance,
            "maxApproachSpeedScale": b.maxApproachSpeedScale,
            "alignmentStrength":     b.alignmentStrength,
            "attentionToOther":      b.attentionToOther,
            "assistBias":            b.assistBias,
            "yieldBias":             b.yieldBias,
            "yieldDir":              (b.yieldDir.x, b.yieldDir.y, b.yieldDir.z),
            "jointHazardBias":       b.jointHazardBias,
            "personality_sociability":  self._personality.sociability,
            "personality_caution":      self._personality.caution,
            "personality_helpfulness":  self._personality.helpfulness,
            "personality_personalSpace": self._personality.personalSpace,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _nearest_other(
        self, others: List[SocialAgentState]
    ) -> Optional[SocialAgentState]:
        best: Optional[SocialAgentState] = None
        best_d = math.inf
        for a in others:
            d = (a.position - self._self_pos).length()
            if d < best_d:
                best_d = d
                best = a
        return best

    def _build_biases(
        self,
        mode:               SocialMode,
        perception:         PerceptionState,
        nearest:            Optional[SocialAgentState],
        yield_bias:         float,
        yield_dir:          Vec3,
        alignment_strength: float,
        preferred_dist:     float,
        shared_risk:        float,
    ) -> SocialBiases:
        """Construct SocialBiases from the selected mode and sub-module outputs."""
        own_risk = perception.globalRisk

        # Default biases
        pref_dist         = preferred_dist
        max_approach      = 1.0
        align             = alignment_strength
        attention         = 0.0
        assist            = 0.0
        y_bias            = yield_bias
        y_dir             = yield_dir
        joint_hazard      = 0.0

        # Attention toward nearest other when any social mode active
        if nearest is not None and mode != SocialMode.Ignore:
            diff = nearest.position - self._self_pos
            d = diff.length()
            if d > 1e-6:
                attention = _clamp(perception.presenceNear, 0.0, 1.0)

        # Mode-specific overrides
        if mode == SocialMode.Ignore:
            align    = 0.0
            attention = 0.0
            assist   = 0.0
            y_bias   = 0.0

        elif mode == SocialMode.AcknowledgePresence:
            # Brief body pause — slow approach, orient toward other
            max_approach = 0.5
            align        = 0.0

        elif mode == SocialMode.ParallelWalk:
            # Speed/direction matching; preferred distance from personality
            max_approach = 0.8

        elif mode == SocialMode.Follow:
            # Follow at comfortable distance; don't over-approach
            max_approach = 0.7
            align        = _clamp(alignment_strength * 1.2, 0.0, 1.0)

        elif mode == SocialMode.YieldPath:
            # Yield trajectory; slow down
            max_approach = _clamp(1.0 - yield_bias, 0.2, 1.0)

        elif mode == SocialMode.AssistPrep:
            # Move toward, reduce speed, raise assist willingness (§10)
            assist       = _clamp(perception.assistOpportunity * self._personality.helpfulness, 0.0, 1.0)
            max_approach = 0.6
            align        = 0.4
            # Preferred distance: close but not inside personal space
            pref_dist    = max(pref_dist * 0.6, 1.0)

        elif mode == SocialMode.Regroup:
            # Slow, seek each other out
            max_approach = 0.6
            align        = _clamp(alignment_strength * 0.5, 0.0, 1.0)

        # Joint hazard: if shared risk is high, move to safer common zone (§8)
        if shared_risk > 0.3:
            joint_hazard = _clamp((shared_risk - 0.3) / 0.7, 0.0, 1.0)
            # Under high hazard, social activity is suppressed (§4 — no "hugging in a storm")
            if shared_risk > 0.8:
                align     = 0.0
                assist    = 0.0
                attention = _clamp(attention * 0.3, 0.0, 1.0)

        return SocialBiases(
            preferredDistance=_clamp(pref_dist, 0.3, 15.0),
            maxApproachSpeedScale=_clamp(max_approach, 0.0, 1.0),
            alignmentStrength=_clamp(align, 0.0, 1.0),
            attentionToOther=_clamp(attention, 0.0, 1.0),
            assistBias=_clamp(assist, 0.0, 1.0),
            yieldBias=_clamp(y_bias, 0.0, 1.0),
            yieldDir=y_dir,
            jointHazardBias=_clamp(joint_hazard, 0.0, 1.0),
        )
