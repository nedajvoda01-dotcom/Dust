"""FatigueToMotorAdapter — Stage 44 fatigue → motor parameter degradation.

Translates a :class:`~src.fatigue.FatigueSystem.FatigueState` into concrete
motor parameter scales that MotorCore, the balance controller, the footstep
planner, and the arm-brace system consume.

All output values are [0..1] multipliers or additive biases.  The adapter
does **not** own any state; it is a pure mapping function.

Design
------
* ``maxTorqueScale``  — joint torque capacity (floors at ``torque_floor``).
* ``stiffnessScale``  — joint stiffness (1 = nominal, lower = floppier).
* ``dampingScale``    — joint damping (1 = nominal, higher = heavier).
* ``reactionDelay``   — added latency to balance correction [s].
* ``recoveryThreshold`` — raised threshold → brace earlier.
* ``stepLengthScale`` — shorter steps with fatigue.
* ``stepWidthBias``   — wider stance with fatigue.
* ``doubleSupportBias`` — longer double-support phase.
* ``footPlacementNoise`` — deterministic foot placement error [m].
* ``braceBias``       — extra arm-brace probability.
* ``gripForceScale``  — grip force capacity (floors at ``grip_scale_min``).

Public API
----------
MotorParams (dataclass)
FatigueToMotorAdapter(config=None)
  .adapt(state, tick_bucket, world_seed) → MotorParams
"""
from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass
from typing import Optional

from src.fatigue.FatigueSystem import FatigueState


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * _clamp(t, 0.0, 1.0)


# ---------------------------------------------------------------------------
# MotorParams — output struct
# ---------------------------------------------------------------------------

@dataclass
class MotorParams:
    """Fatigue-derived motor parameter scales and biases.

    All values are ready to multiply against or add to the corresponding
    nominal motor parameters.
    """
    maxTorqueScale:       float = 1.0
    stiffnessScale:       float = 1.0
    dampingScale:         float = 1.0
    reactionDelay:        float = 0.0    # seconds of extra latency
    recoveryThreshold:    float = 0.0    # additive bias to balance threshold
    stepLengthScale:      float = 1.0
    stepWidthBias:        float = 0.0    # additive to nominal width
    doubleSupportBias:    float = 0.0    # fraction of extra double-support
    footPlacementNoise:   float = 0.0    # deterministic placement error [m]
    braceBias:            float = 0.0    # extra arm-brace probability
    gripForceScale:       float = 1.0


# ---------------------------------------------------------------------------
# FatigueToMotorAdapter
# ---------------------------------------------------------------------------

class FatigueToMotorAdapter:
    """Maps :class:`FatigueState` to :class:`MotorParams`.

    Parameters
    ----------
    config :
        Optional dict; reads ``fatigue.*`` keys.
    """

    _DEFAULT_TORQUE_FLOOR         = 0.65
    _DEFAULT_STIFFNESS_SCALE_MIN  = 0.6
    _DEFAULT_GRIP_SCALE_MIN       = 0.5
    _DEFAULT_MAX_REACTION_DELAY   = 0.08   # seconds
    _DEFAULT_MAX_STEP_LENGTH_LOSS = 0.35   # fraction lost at max fatigue
    _DEFAULT_MAX_STEP_WIDTH_GAIN  = 0.25   # extra width normalised
    _DEFAULT_MAX_DOUBLE_SUP_GAIN  = 0.30
    _DEFAULT_MAX_FOOT_NOISE       = 0.06   # metres
    _DEFAULT_MAX_BRACE_BIAS       = 0.7

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg  = config or {}
        fcfg = cfg.get("fatigue", {}) or {}

        self._torque_floor        = float(fcfg.get("torque_floor",        self._DEFAULT_TORQUE_FLOOR))
        self._stiffness_scale_min = float(fcfg.get("stiffness_scale_min", self._DEFAULT_STIFFNESS_SCALE_MIN))
        self._grip_scale_min      = float(fcfg.get("grip_scale_min",      self._DEFAULT_GRIP_SCALE_MIN))
        self._max_reaction_delay  = float(fcfg.get("max_reaction_delay",  self._DEFAULT_MAX_REACTION_DELAY))

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def adapt(
        self,
        state:       FatigueState,
        tick_bucket: int = 0,
        world_seed:  int = 0,
    ) -> MotorParams:
        """Compute motor parameters from the current fatigue state.

        Parameters
        ----------
        state :
            Current :class:`FatigueState`.
        tick_bucket :
            Coarse tick counter (increments each fatigue tick); used as
            the seed for the **deterministic** foot-placement noise so
            the same inputs always yield the same noise value.
        world_seed :
            World-level seed for domain isolation.

        Returns
        -------
        MotorParams
        """
        f = 1.0 - state.energy          # fatigue factor [0..1]
        n = state.neuromuscularNoise     # noise [0..1]
        c = state.coordination           # coordination [0..1]
        g = state.gripReserve            # grip [0..1]

        # Torque capacity degrades with fatigue (floored)
        torque_scale = _lerp(1.0, self._torque_floor, f)

        # Stiffness decreases, damping increases (movements feel heavier)
        stiffness_scale = _lerp(1.0, self._stiffness_scale_min, f * 0.8)
        damping_scale   = _lerp(1.0, 1.4, f * 0.6)

        # Balance reaction delay grows with noise
        reaction_delay = n * self._max_reaction_delay

        # Recovery threshold bias (character braces sooner when tired)
        recovery_threshold = f * 0.25

        # Footstep adaptations
        step_length_scale    = _lerp(1.0, 1.0 - self._DEFAULT_MAX_STEP_LENGTH_LOSS, f)
        step_width_bias      = f * self._DEFAULT_MAX_STEP_WIDTH_GAIN
        double_support_bias  = f * self._DEFAULT_MAX_DOUBLE_SUP_GAIN

        # Deterministic foot-placement noise (§6.4)
        foot_noise = n * self._DEFAULT_MAX_FOOT_NOISE * self._det_noise(tick_bucket, world_seed)

        # Arm brace bias — increases with fatigue and coordination loss
        brace_bias = _clamp(f * 0.6 + (1.0 - c) * 0.4, 0.0, self._DEFAULT_MAX_BRACE_BIAS)

        # Grip force scale (floored at grip_scale_min)
        grip_force_scale = _lerp(1.0, self._grip_scale_min, 1.0 - g)

        return MotorParams(
            maxTorqueScale      = torque_scale,
            stiffnessScale      = stiffness_scale,
            dampingScale        = damping_scale,
            reactionDelay       = reaction_delay,
            recoveryThreshold   = recovery_threshold,
            stepLengthScale     = step_length_scale,
            stepWidthBias       = step_width_bias,
            doubleSupportBias   = double_support_bias,
            footPlacementNoise  = foot_noise,
            braceBias           = brace_bias,
            gripForceScale      = grip_force_scale,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _det_noise(tick_bucket: int, world_seed: int) -> float:
        """Return a deterministic [0..1] noise scalar for foot placement.

        Uses SHA-256 counter mode so the same (tick_bucket, world_seed)
        always yields the same value — no random() is used.
        """
        data   = struct.pack(">QQ", tick_bucket & 0xFFFFFFFFFFFFFFFF,
                             world_seed & 0xFFFFFFFFFFFFFFFF)
        digest = hashlib.sha256(data).digest()
        raw    = int.from_bytes(digest[:8], "big")
        return (raw >> 11) * (1.0 / (1 << 53))
