"""InjuryToQPAdapter — Stage 48 injury → QP/motor parameter adjustments.

Translates per-joint :class:`~src.injury.InjurySystem.InjuryState` into
concrete motor parameter reductions that the whole-body QP controller,
balance controller, and footstep planner consume.

Design
------
* Joint torque capacity ``tau_max`` is reduced for injured joints (floored
  at ``tau_floor``).
* Joint stiffness is reduced proportionally (floored at ``stiffness_min``).
* ``braceBias`` is elevated when upper-limb or lower-back injury is high.
* Body-yaw rate and step length are capped when globalInjuryIndex is high.
* All reductions are clamped by ``max_total_influence`` so control is never
  completely lost (§ 43 input contract respected).

Public API
----------
InjuryMotorAdjustment (dataclass)
InjuryToQPAdapter(config=None)
  .adapt(state) → InjuryMotorAdjustment
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from src.injury.InjurySystem import InjuryState, JOINT_NAMES


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * _clamp(t, 0.0, 1.0)


# ---------------------------------------------------------------------------
# InjuryMotorAdjustment — output struct
# ---------------------------------------------------------------------------

@dataclass
class InjuryMotorAdjustment:
    """Injury-derived motor parameter adjustments.

    Attributes
    ----------
    tau_scale :
        Per-joint torque capacity multiplier (keyed by joint name).
        1.0 = no reduction.
    stiffness_scale :
        Per-joint stiffness multiplier. 1.0 = no reduction.
    braceBias :
        Extra arm-brace probability added to the base brace bias [0..1].
    bodyYawRateLimit :
        Maximum allowed body-yaw rate multiplier (1.0 = unrestricted).
    stepLengthScale :
        Step-length reduction multiplier (1.0 = no reduction).
    doubleSupportBias :
        Extra double-support phase fraction [0..1].
    globalInjuryIndex :
        Pass-through of the aggregate injury severity [0..1].
    """
    tau_scale:         Dict[str, float] = field(
        default_factory=lambda: {n: 1.0 for n in JOINT_NAMES}
    )
    stiffness_scale:   Dict[str, float] = field(
        default_factory=lambda: {n: 1.0 for n in JOINT_NAMES}
    )
    braceBias:         float = 0.0
    bodyYawRateLimit:  float = 1.0
    stepLengthScale:   float = 1.0
    doubleSupportBias: float = 0.0
    globalInjuryIndex: float = 0.0


# ---------------------------------------------------------------------------
# InjuryToQPAdapter
# ---------------------------------------------------------------------------

class InjuryToQPAdapter:
    """Maps :class:`InjuryState` to :class:`InjuryMotorAdjustment`.

    Parameters
    ----------
    config :
        Optional dict; reads ``injury.*`` keys.
    """

    _DEFAULT_TAU_FLOOR       = 0.75
    _DEFAULT_STIFFNESS_MIN   = 0.70
    _DEFAULT_MAX_INFLUENCE   = 0.60
    _DEFAULT_MAX_BRACE_BIAS  = 0.50
    _DEFAULT_YAW_RATE_MIN    = 0.40   # at max globalInjuryIndex
    _DEFAULT_STEP_LENGTH_MIN = 0.60   # at max globalInjuryIndex
    _DEFAULT_MAX_DS_BIAS     = 0.25

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg  = config or {}
        icfg = cfg.get("injury", {}) or {}

        self._tau_floor        = float(icfg.get("tau_floor",          self._DEFAULT_TAU_FLOOR))
        self._stiffness_min    = float(icfg.get("stiffness_min",      self._DEFAULT_STIFFNESS_MIN))
        self._max_influence    = float(icfg.get("max_total_influence", self._DEFAULT_MAX_INFLUENCE))
        self._max_brace_bias   = float(icfg.get("max_brace_bias",     self._DEFAULT_MAX_BRACE_BIAS))
        self._yaw_rate_min     = float(icfg.get("yaw_rate_min",       self._DEFAULT_YAW_RATE_MIN))
        self._step_length_min  = float(icfg.get("step_length_min",    self._DEFAULT_STEP_LENGTH_MIN))
        self._max_ds_bias      = float(icfg.get("max_ds_bias",        self._DEFAULT_MAX_DS_BIAS))

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def adapt(self, state: InjuryState) -> InjuryMotorAdjustment:
        """Compute motor adjustments from the current injury state.

        Parameters
        ----------
        state :
            Current :class:`InjuryState`.

        Returns
        -------
        InjuryMotorAdjustment
        """
        tau_scale:       Dict[str, float] = {}
        stiffness_scale: Dict[str, float] = {}

        for name in JOINT_NAMES:
            j = state.joints[name]
            injury_factor = _clamp(j.strain + j.acute * 0.4, 0.0, self._max_influence)
            # Torque capacity: lerp from 1.0 down to tau_floor
            tau_scale[name] = _lerp(1.0, self._tau_floor, injury_factor)
            # Stiffness: lerp from 1.0 down to stiffness_min
            stiffness_scale[name] = _lerp(1.0, self._stiffness_min, injury_factor)

        # Brace bias — driven by upper-limb and lower-back pain avoidance
        upper = ["shoulder_l", "shoulder_r", "elbow_l", "elbow_r",
                 "wrist_l", "wrist_r", "lower_back"]
        pain_upper = max(state.joints[n].painAvoidance for n in upper)
        brace_bias = _clamp(pain_upper * self._max_brace_bias, 0.0, self._max_brace_bias)

        g = state.globalInjuryIndex
        yaw_rate_limit   = _lerp(1.0, self._yaw_rate_min,     g)
        step_length_scale = _lerp(1.0, self._step_length_min, g * 0.7)
        ds_bias          = g * self._max_ds_bias

        return InjuryMotorAdjustment(
            tau_scale=tau_scale,
            stiffness_scale=stiffness_scale,
            braceBias=brace_bias,
            bodyYawRateLimit=yaw_rate_limit,
            stepLengthScale=step_length_scale,
            doubleSupportBias=ds_bias,
            globalInjuryIndex=g,
        )
