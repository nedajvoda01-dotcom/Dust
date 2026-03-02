"""StrategySelector — Stage 38 motor strategy selection (§5).

Maps a :class:`PerceptionState` + :class:`CostBreakdown` to the optimal
:class:`MotorMode` and the corresponding :class:`MotorIntent`.

Ten motor modes are defined (§5).  For each, an *activation score* is
computed from the current perception state.  Per-mode
:class:`~src.intent.ModeHysteresis.ModeHysteresis` gates prevent rapid
oscillation (§11).  The mode with the highest score among those that
pass hysteresis is selected.

MotorIntent (§3)
----------------
All intent fields are [0..1] scalars or Vec3 unit vectors:

* ``desiredVelocity``     — scaled input velocity
* ``stanceWidthBias``     — 0 = normal, 1 = wide stance
* ``stepLengthBias``      — 0 = tiny steps, 1 = full stride
* ``bracePreference``     — 0 = relaxed, 1 = braced
* ``attentionTargetDir``  — where to orient head/gaze
* ``proximityPreference`` — preferred inter-player spacing scale
* ``assistWillingness``   — 0 = ignore others, 1 = move to assist

Public API
----------
MotorMode (enum)
MotorIntent (dataclass)
StrategySelector(config=None)
  .select(state, cost, input_velocity) → (MotorMode, MotorIntent)
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from src.math.Vec3 import Vec3
from src.perception.PerceptionSystem import PerceptionState
from src.intent.CostEvaluator import CostBreakdown
from src.intent.ModeHysteresis import ModeHysteresis


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


# ---------------------------------------------------------------------------
# Motor mode enumeration
# ---------------------------------------------------------------------------

class MotorMode(enum.Enum):
    """The ten motor modes available to the IntentArbitrator (§5)."""
    NormalLocomotion    = 0
    CautiousLocomotion  = 1
    WideStanceStabilize = 2
    SlowApproach        = 3
    BraceMode           = 4
    PauseAndAssess      = 5
    AlignWithOther      = 6
    AssistPreparation   = 7
    PosturalAdjustment  = 8
    MicroReposition     = 9


# ---------------------------------------------------------------------------
# MotorIntent
# ---------------------------------------------------------------------------

@dataclass
class MotorIntent:
    """Body-level motor intent produced by IntentArbitrator (§3).

    Not an animation directive — MotorStack translates this into physics.
    """
    desiredVelocity:     Vec3  = field(default_factory=Vec3.zero)
    stanceWidthBias:     float = 0.0   # 0 = normal, 1 = wide
    stepLengthBias:      float = 1.0   # 0 = tiny steps, 1 = normal
    bracePreference:     float = 0.0   # 0 = relaxed, 1 = braced
    attentionTargetDir:  Vec3  = field(default_factory=Vec3.zero)
    proximityPreference: float = 1.0   # preferred spacing multiplier
    assistWillingness:   float = 0.0   # 0 = ignore others, 1 = assist


# ---------------------------------------------------------------------------
# Per-mode hysteresis defaults (enter, exit) — spec §11
# ---------------------------------------------------------------------------

_HYSTERESIS_DEFAULTS: Dict[MotorMode, Tuple[float, float]] = {
    MotorMode.NormalLocomotion:    (0.00, 0.00),   # always eligible
    MotorMode.CautiousLocomotion:  (0.35, 0.20),
    MotorMode.WideStanceStabilize: (0.40, 0.25),
    MotorMode.SlowApproach:        (0.35, 0.20),
    MotorMode.BraceMode:           (0.60, 0.40),
    MotorMode.PauseAndAssess:      (0.65, 0.45),
    MotorMode.AlignWithOther:      (0.30, 0.15),
    MotorMode.AssistPreparation:   (0.40, 0.25),
    MotorMode.PosturalAdjustment:  (0.20, 0.10),
    MotorMode.MicroReposition:     (0.20, 0.10),
}


# ---------------------------------------------------------------------------
# StrategySelector
# ---------------------------------------------------------------------------

class StrategySelector:
    """Selects the optimal motor mode and computes the corresponding MotorIntent.

    Strategy selection follows §4 and §5: for each mode an *activation
    score* is computed from the risk components; per-mode hysteresis
    prevents oscillation; the highest-scoring eligible mode wins.

    Parameters
    ----------
    config :
        Optional dict; reads ``intent.mode_hysteresis_margin`` and
        ``intent.max_speed_scale_under_risk``, ``intent.max_stance_bias``,
        ``intent.assist_threshold``.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = ((config or {}).get("intent", {})) or {}
        margin = float(cfg.get("mode_hysteresis_margin", 0.0))

        self._max_speed_scale = float(cfg.get("max_speed_scale_under_risk", 0.2))
        self._max_stance_bias = float(cfg.get("max_stance_bias",            0.9))
        self._assist_threshold = float(cfg.get("assist_threshold",          0.4))

        # Build per-mode hysteresis objects
        self._hysteresis: Dict[MotorMode, ModeHysteresis] = {}
        for mode, (enter, exit_) in _HYSTERESIS_DEFAULTS.items():
            # Clamp to avoid invalid thresholds after margin adjustment
            e = _clamp(enter - margin, 0.0, 1.0)
            x = _clamp(exit_  - margin, 0.0, e)
            self._hysteresis[mode] = ModeHysteresis(
                enter_threshold=e,
                exit_threshold=x,
            )

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def select(
        self,
        state:          PerceptionState,
        cost:           CostBreakdown,
        input_velocity: Optional[Vec3] = None,
    ) -> Tuple[MotorMode, MotorIntent]:
        """Select the best motor mode and return a MotorIntent.

        Parameters
        ----------
        state :
            Current aggregated PerceptionState.
        cost :
            Pre-computed CostBreakdown for *state*.
        input_velocity :
            Desired movement velocity provided by the player/AI controller.
            Zero or None is treated as idle.

        Returns
        -------
        (MotorMode, MotorIntent)
        """
        if input_velocity is None:
            input_velocity = Vec3.zero()

        # Compute activation scores for every mode
        scores: Dict[MotorMode, float] = self._score_all(state, cost, input_velocity)

        # NormalLocomotion is the unconditional fallback; all other modes
        # compete for selection by highest activation score via hysteresis.
        self._hysteresis[MotorMode.NormalLocomotion].update(1.0)
        best_mode  = MotorMode.NormalLocomotion
        best_score = -1.0

        for mode, score in scores.items():
            if mode == MotorMode.NormalLocomotion:
                continue
            active = self._hysteresis[mode].update(score)
            if active and score > best_score:
                best_score = score
                best_mode  = mode

        intent = self._build_intent(best_mode, state, cost, input_velocity)
        return best_mode, intent

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _score_all(
        self,
        state: PerceptionState,
        cost:  CostBreakdown,
        input_velocity: Vec3,
    ) -> Dict[MotorMode, float]:
        """Return a raw activation score for every motor mode."""
        idle_factor = 1.0 - _clamp(input_velocity.length(), 0.0, 1.0)

        return {
            # NormalLocomotion: high when total cost is low
            MotorMode.NormalLocomotion:    _clamp(1.0 - cost.total / max(cost.total + 1e-6, 7.0), 0.0, 1.0),

            # CautiousLocomotion: dominant slip, visibility, or balance
            MotorMode.CautiousLocomotion:  _clamp(max(cost.slip, cost.visibility, cost.balance) * 0.7, 0.0, 1.0),

            # WideStanceStabilize: high wind or slip
            MotorMode.WideStanceStabilize: _clamp(max(cost.wind, cost.slip) * 0.9, 0.0, 1.0),

            # SlowApproach: poor visibility
            MotorMode.SlowApproach:        _clamp(cost.visibility * 0.9, 0.0, 1.0),

            # BraceMode: high combined balance + vibration + wind
            MotorMode.BraceMode:           _clamp(
                max(cost.balance, cost.vibration, cost.wind) * 0.85, 0.0, 1.0
            ),

            # PauseAndAssess: high vibration or audio uncertainty
            MotorMode.PauseAndAssess:      _clamp(
                max(cost.vibration, cost.uncertainty) * 0.95, 0.0, 1.0
            ),

            # AlignWithOther: other players nearby
            MotorMode.AlignWithOther:      _clamp(cost.proximity * 0.9, 0.0, 1.0),

            # AssistPreparation: other player slipping + self safe
            MotorMode.AssistPreparation:   _clamp(
                state.assistOpportunity * (1.0 - state.globalRisk), 0.0, 1.0
            ),

            # PosturalAdjustment: idle with some brace bias
            MotorMode.PosturalAdjustment:  _clamp(idle_factor * state.braceBias * 0.8, 0.0, 1.0),

            # MicroReposition: idle with very low overall risk
            MotorMode.MicroReposition:     _clamp(
                idle_factor * (1.0 - state.globalRisk) * 0.4, 0.0, 1.0
            ),
        }

    def _build_intent(
        self,
        mode:           MotorMode,
        state:          PerceptionState,
        cost:           CostBreakdown,
        input_velocity: Vec3,
    ) -> MotorIntent:
        """Construct a MotorIntent appropriate for *mode* and current state."""
        # Base: apply movementConfidence speed scaling (§6.3)
        speed_scale = _clamp(
            state.movementConfidence,
            self._max_speed_scale,
            1.0,
        )
        base_vel = input_velocity * speed_scale

        # Defaults
        stance  = 0.0
        step    = 1.0
        brace   = state.braceBias
        attn    = state.attentionDir
        prox    = 1.0
        assist  = 0.0

        if mode == MotorMode.NormalLocomotion:
            pass  # all defaults

        elif mode == MotorMode.CautiousLocomotion:
            # Slightly slower, shorter steps
            base_vel = input_velocity * _clamp(speed_scale * 0.7, self._max_speed_scale, 1.0)
            step     = 0.6
            stance   = _clamp(cost.slip * 0.5, 0.0, self._max_stance_bias)

        elif mode == MotorMode.WideStanceStabilize:
            # §6.1: wide stance, short steps, braced (spec example: high wind + slip)
            stance   = _clamp(max(state.windLoad, state.slipRisk) * self._max_stance_bias, 0.0, self._max_stance_bias)
            step     = _clamp(1.0 - max(state.windLoad, state.slipRisk) * 0.8, 0.1, 0.8)
            brace    = _clamp(max(state.windLoad, state.slipRisk), 0.0, 1.0)
            base_vel = input_velocity * _clamp(speed_scale * 0.6, self._max_speed_scale, 1.0)

        elif mode == MotorMode.SlowApproach:
            # §6.3: low visibility → slow movement
            base_vel = input_velocity * _clamp(speed_scale * 0.4, self._max_speed_scale, 1.0)
            step     = 0.4

        elif mode == MotorMode.BraceMode:
            brace    = _clamp(state.globalRisk, 0.6, 1.0)
            stance   = _clamp(state.globalRisk * self._max_stance_bias, 0.0, self._max_stance_bias)
            step     = 0.3
            base_vel = input_velocity * _clamp(speed_scale * 0.5, self._max_speed_scale, 1.0)

        elif mode == MotorMode.PauseAndAssess:
            # §6.2: pause + orient attention
            base_vel = Vec3.zero()
            step     = 0.0
            brace    = _clamp(state.globalRisk, 0.0, 1.0)

        elif mode == MotorMode.AlignWithOther:
            # §6.4: match speed, reduce lateral drift
            base_vel = input_velocity * _clamp(speed_scale * 0.8, self._max_speed_scale, 1.0)
            prox     = 0.8

        elif mode == MotorMode.AssistPreparation:
            # §8: move toward other player, braced
            brace    = _clamp(0.6 + state.assistOpportunity * 0.4, 0.0, 1.0)
            assist   = _clamp(state.assistOpportunity, 0.0, 1.0)
            base_vel = input_velocity * _clamp(speed_scale * 0.5, self._max_speed_scale, 1.0)

        elif mode == MotorMode.PosturalAdjustment:
            # §7: idle, weight shift
            base_vel = Vec3.zero()
            stance   = _clamp(state.braceBias * 0.5, 0.0, self._max_stance_bias)

        elif mode == MotorMode.MicroReposition:
            # §7: tiny idle repositioning step
            base_vel = Vec3.zero()

        return MotorIntent(
            desiredVelocity     = base_vel,
            stanceWidthBias     = _clamp(stance, 0.0, 1.0),
            stepLengthBias      = _clamp(step,   0.0, 1.0),
            bracePreference     = _clamp(brace,  0.0, 1.0),
            attentionTargetDir  = attn,
            proximityPreference = _clamp(prox,   0.0, 2.0),
            assistWillingness   = _clamp(assist, 0.0, 1.0),
        )
