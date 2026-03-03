"""InjuryToFootstepBias — Stage 48 injury → footstep planner biases.

Translates per-joint :class:`~src.injury.InjurySystem.InjuryState` into
footstep-planning biases so the character naturally shifts weight away from
injured legs, widens stance, and places feet more cautiously.

Design
------
* ``weightBias_l`` / ``weightBias_r`` — fraction of body weight preferred
  on each side.  An injured left ankle shifts weight to the right.
* ``stanceWidthBias`` — extra stance width (positive = wider).
* ``stepCaution_l`` / ``stepCaution_r`` — per-foot placement caution [0..1]:
  higher → smaller steps, more time in double-support.
* ``avoidLeanLeft`` / ``avoidLeanRight`` — boolean hints for the balance
  controller to avoid lateral lean toward the injured side.

Public API
----------
FootstepBias (dataclass)
InjuryToFootstepBias(config=None)
  .bias(state) → FootstepBias
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.injury.InjurySystem import InjuryState


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * _clamp(t, 0.0, 1.0)


# ---------------------------------------------------------------------------
# FootstepBias — output struct
# ---------------------------------------------------------------------------

@dataclass
class FootstepBias:
    """Injury-derived footstep planner biases.

    Attributes
    ----------
    weightBias_l :
        Preferred weight fraction on the left leg [0..1]; nominal 0.5.
    weightBias_r :
        Preferred weight fraction on the right leg [0..1]; nominal 0.5.
    stanceWidthBias :
        Extra stance width [normalised, 0..1].
    stepCaution_l :
        Left-foot step caution [0..1]; higher = shorter, slower steps.
    stepCaution_r :
        Right-foot step caution [0..1].
    avoidLeanLeft :
        Hint to the balance controller to limit leftward lean.
    avoidLeanRight :
        Hint to the balance controller to limit rightward lean.
    """
    weightBias_l:    float = 0.5
    weightBias_r:    float = 0.5
    stanceWidthBias: float = 0.0
    stepCaution_l:   float = 0.0
    stepCaution_r:   float = 0.0
    avoidLeanLeft:   bool  = False
    avoidLeanRight:  bool  = False


# ---------------------------------------------------------------------------
# InjuryToFootstepBias
# ---------------------------------------------------------------------------

class InjuryToFootstepBias:
    """Derives :class:`FootstepBias` from :class:`InjuryState`.

    Parameters
    ----------
    config :
        Optional dict; reads ``injury.*`` keys.
    """

    _DEFAULT_MAX_WEIGHT_SHIFT  = 0.20   # max shift from 0.5 to 0.7
    _DEFAULT_MAX_WIDTH_BIAS    = 0.15   # normalised stance width gain
    _DEFAULT_MAX_CAUTION       = 0.80
    _DEFAULT_LEAN_THRESHOLD    = 0.30   # pain avoidance above which lean is avoided

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg  = config or {}
        icfg = cfg.get("injury", {}) or {}

        self._max_weight_shift = float(icfg.get("max_weight_shift",  self._DEFAULT_MAX_WEIGHT_SHIFT))
        self._max_width_bias   = float(icfg.get("max_width_bias",    self._DEFAULT_MAX_WIDTH_BIAS))
        self._max_caution      = float(icfg.get("max_step_caution",  self._DEFAULT_MAX_CAUTION))
        self._lean_threshold   = float(icfg.get("lean_threshold",    self._DEFAULT_LEAN_THRESHOLD))

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def bias(self, state: InjuryState) -> FootstepBias:
        """Compute footstep biases from the current injury state.

        Parameters
        ----------
        state :
            Current :class:`InjuryState`.

        Returns
        -------
        FootstepBias
        """
        # Lower-limb pain avoidance signals (worst of ankle/knee/hip per side)
        pain_l = max(
            state.joints["ankle_l"].painAvoidance,
            state.joints["knee_l"].painAvoidance,
            state.joints["hip_l"].painAvoidance,
        )
        pain_r = max(
            state.joints["ankle_r"].painAvoidance,
            state.joints["knee_r"].painAvoidance,
            state.joints["hip_r"].painAvoidance,
        )

        # Weight preference: shift away from the painful side
        # pain_l↑ → prefer right leg → weightBias_r↑, weightBias_l↓
        shift_l = _clamp(pain_l - pain_r, 0.0, 1.0) * self._max_weight_shift
        shift_r = _clamp(pain_r - pain_l, 0.0, 1.0) * self._max_weight_shift
        weight_l = _clamp(0.5 - shift_l + shift_r, 0.0, 1.0)
        weight_r = _clamp(1.0 - weight_l, 0.0, 1.0)

        # Stance width: widen if either side is painful
        total_pain     = _clamp(pain_l + pain_r, 0.0, 1.0)
        stance_bias    = total_pain * self._max_width_bias

        # Step caution: directly driven by pain avoidance
        caution_l = _clamp(pain_l * self._max_caution, 0.0, self._max_caution)
        caution_r = _clamp(pain_r * self._max_caution, 0.0, self._max_caution)

        # Lean avoidance hints
        avoid_lean_l = pain_l >= self._lean_threshold
        avoid_lean_r = pain_r >= self._lean_threshold

        return FootstepBias(
            weightBias_l=weight_l,
            weightBias_r=weight_r,
            stanceWidthBias=stance_bias,
            stepCaution_l=caution_l,
            stepCaution_r=caution_r,
            avoidLeanLeft=avoid_lean_l,
            avoidLeanRight=avoid_lean_r,
        )
