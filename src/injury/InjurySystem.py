"""InjurySystem — Stage 48 Injury & Recovery Micro-Physics (no UI).

Models micro-traumas and joint overloads as temporary **mechanical
constraints**, not health-points.  There is **no UI**: the player reads
the character's condition through gait changes, micro-pauses, weight
redistribution, and equipment sounds.

Design
------
* A deterministic tick (2–10 Hz, default 5 Hz) integrates joint-load
  inputs into per-joint ``strain`` and ``acute`` values.
* ``painAvoidance`` is a smoothed behavioural response: the character
  protects an injured joint by offloading weight.
* Recovery is automatic under low-load, stable conditions.
* The server is authoritative; clients receive coarse snapshots.

Joints tracked (13 total)
--------------------------
left/right: ankle, knee, hip, shoulder, elbow, wrist  → 12
lower_back: 1

Public API
----------
JointInjury (dataclass)
InjuryState (dataclass)
InjurySystem(config=None)
  .tick(dt, load_input, env_input) → InjuryState
  .state                           → InjuryState
  .force_strain(joint, strain)     — dev / test helper
  .debug_info()                    → dict
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * _clamp(t, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Joint identifiers
# ---------------------------------------------------------------------------

JOINT_NAMES: List[str] = [
    "ankle_l", "ankle_r",
    "knee_l",  "knee_r",
    "hip_l",   "hip_r",
    "shoulder_l", "shoulder_r",
    "elbow_l", "elbow_r",
    "wrist_l", "wrist_r",
    "lower_back",
]


# ---------------------------------------------------------------------------
# JointInjury — per-joint injury record
# ---------------------------------------------------------------------------

@dataclass
class JointInjury:
    """Injury state for a single joint/segment.

    Attributes
    ----------
    strain :
        Accumulated chronic overload (0 = healthy, 1 = severely strained).
    acute :
        Acute spike from a sudden impact or jerk (0 = none, 1 = severe).
    painAvoidance :
        Behavioural protection bias (0 = no avoidance, 1 = maximum caution).
    recoveryRate :
        Current recovery rate [units/s]; computed each tick.
    """
    strain:       float = 0.0
    acute:        float = 0.0
    painAvoidance: float = 0.0
    recoveryRate:  float = 0.0


# ---------------------------------------------------------------------------
# InjuryState — full-body injury snapshot
# ---------------------------------------------------------------------------

@dataclass
class InjuryState:
    """Full-body injury state.

    Attributes
    ----------
    joints :
        Per-joint :class:`JointInjury` keyed by name (see :data:`JOINT_NAMES`).
    globalInjuryIndex :
        Aggregate injury severity [0..1] — max of weighted joint contributions.
    """
    joints: Dict[str, JointInjury] = field(
        default_factory=lambda: {name: JointInjury() for name in JOINT_NAMES}
    )
    globalInjuryIndex: float = 0.0


# ---------------------------------------------------------------------------
# LoadInput — per-tick physics inputs for one joint
# ---------------------------------------------------------------------------

@dataclass
class JointLoad:
    """Physics-derived load for one joint over the last tick.

    Attributes
    ----------
    tau :
        Joint torque magnitude [Nm, normalised 0..1 relative to tau_max].
    tau_max :
        Configured peak torque capacity [normalised, typically 1.0].
    omega :
        Joint angular velocity magnitude [rad/s, normalised 0..1].
    impactImpulse :
        Contact impulse magnitude (0 if no impact this tick) [normalised 0..1].
    graspForcePeak :
        Peak constraint force from grasp system [normalised 0..1].
    """
    tau:            float = 0.0
    tau_max:        float = 1.0
    omega:          float = 0.0
    impactImpulse:  float = 0.0
    graspForcePeak: float = 0.0


@dataclass
class LoadInput:
    """All joint loads for one injury tick.

    Attributes
    ----------
    joints :
        Per-joint :class:`JointLoad` keyed by name.  Missing joints default
        to zero load.
    """
    joints: Dict[str, JointLoad] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# InjuryEnvInput — environmental conditions consumed by recovery model
# ---------------------------------------------------------------------------

@dataclass
class InjuryEnvInput:
    """Environmental and motion conditions for the injury tick.

    Attributes
    ----------
    speed :
        Current movement speed [m/s].
    windLoad :
        Perceived wind load [0..1].
    supportQuality :
        Ground support quality [0..1].
    isGrasping :
        True when the player is sustaining a grasp on another body.
    """
    speed:          float = 0.0
    windLoad:       float = 0.0
    supportQuality: float = 1.0
    isGrasping:     bool  = False


# ---------------------------------------------------------------------------
# InjurySystem
# ---------------------------------------------------------------------------

class InjurySystem:
    """Integrates joint-load and environment inputs into :class:`InjuryState`.

    Parameters
    ----------
    config :
        Optional dict; reads ``injury.*`` keys.
    """

    # ------- defaults -------------------------------------------------------
    _DEFAULT_TICK_HZ          = 5.0
    _DEFAULT_K_STRAIN         = 0.04    # chronic overload accumulation rate
    _DEFAULT_K_ACUTE          = 0.6     # acute spike scaling
    _DEFAULT_IMPACT_THRESHOLD = 0.25    # impulse threshold before acute damage
    _DEFAULT_TAU_FLOOR        = 0.75    # minimum torque capacity under injury
    _DEFAULT_STIFFNESS_MIN    = 0.70    # minimum stiffness under injury
    _DEFAULT_RECOVER_K        = 0.015   # strain recovery rate at full rest
    _DEFAULT_ACUTE_DECAY_K    = 0.05    # acute decay rate
    _DEFAULT_PAIN_AVOIDANCE_K = 0.08    # pain avoidance adaptation rate
    _DEFAULT_MAX_INFLUENCE    = 0.60    # caps total motor influence
    _DEFAULT_STRAIN_POWER     = 2.0     # exponent p in (tau/tau_max)^p
    _DEFAULT_ACUTE_TO_STRAIN  = 0.15    # fraction of acute that converts to strain/s
    _DEFAULT_REST_SPEED_MAX   = 0.5     # m/s threshold for recovery
    _DEFAULT_REST_WIND_MAX    = 0.25    # wind threshold for recovery
    _DEFAULT_REST_SUPPORT_MIN = 0.65    # support threshold for recovery

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg  = config or {}
        icfg = cfg.get("injury", {}) or {}

        self._enabled          = bool(icfg.get("enable", True))
        self._tick_hz          = float(icfg.get("tick_hz",          self._DEFAULT_TICK_HZ))
        self._k_strain         = float(icfg.get("k_strain",         self._DEFAULT_K_STRAIN))
        self._k_acute          = float(icfg.get("k_acute",          self._DEFAULT_K_ACUTE))
        self._impact_threshold = float(icfg.get("impact_threshold", self._DEFAULT_IMPACT_THRESHOLD))
        self._tau_floor        = float(icfg.get("tau_floor",        self._DEFAULT_TAU_FLOOR))
        self._stiffness_min    = float(icfg.get("stiffness_min",    self._DEFAULT_STIFFNESS_MIN))
        self._recover_k        = float(icfg.get("recover_k",        self._DEFAULT_RECOVER_K))
        self._acute_decay_k    = float(icfg.get("acute_decay_k",    self._DEFAULT_ACUTE_DECAY_K))
        self._pain_avoidance_k = float(icfg.get("pain_avoidance_k", self._DEFAULT_PAIN_AVOIDANCE_K))
        self._max_influence    = float(icfg.get("max_total_influence", self._DEFAULT_MAX_INFLUENCE))
        self._strain_power     = float(icfg.get("strain_power",     self._DEFAULT_STRAIN_POWER))
        self._acute_to_strain  = float(icfg.get("acute_to_strain",  self._DEFAULT_ACUTE_TO_STRAIN))

        rest = icfg.get("rest_condition_thresholds", {}) or {}
        self._rest_speed_max   = float(rest.get("speed",   self._DEFAULT_REST_SPEED_MAX))
        self._rest_wind_max    = float(rest.get("wind",    self._DEFAULT_REST_WIND_MAX))
        self._rest_support_min = float(rest.get("support", self._DEFAULT_REST_SUPPORT_MIN))

        self._state = InjuryState()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    @property
    def state(self) -> InjuryState:
        """Current :class:`InjuryState`."""
        return self._state

    def tick(
        self,
        dt:    float,
        load:  LoadInput,
        env:   InjuryEnvInput,
    ) -> InjuryState:
        """Advance the injury integrator by ``dt`` seconds.

        Parameters
        ----------
        dt :
            Elapsed simulation time [s] since last tick.
        load :
            Per-joint physics loads for this tick.
        env :
            Environmental and motion conditions.

        Returns
        -------
        InjuryState
            Updated injury state.
        """
        if not self._enabled or dt <= 0.0:
            return self._state

        resting = self._is_resting(env)
        new_joints: Dict[str, JointInjury] = {}

        for name in JOINT_NAMES:
            j    = self._state.joints[name]
            jl   = load.joints.get(name, JointLoad())
            new_joints[name] = self._tick_joint(j, jl, dt, resting)

        # Global injury index — weighted max across all joints
        leg_joints   = {"ankle_l", "ankle_r", "knee_l", "knee_r", "hip_l", "hip_r"}
        upper_joints = {"shoulder_l", "shoulder_r", "elbow_l", "elbow_r",
                        "wrist_l", "wrist_r", "lower_back"}
        scores = []
        for name, j in new_joints.items():
            injury_factor = _clamp(j.strain + j.acute * 0.5, 0.0, 1.0)
            # Leg joints weighted slightly higher (locomotion critical)
            w = 1.2 if name in leg_joints else 1.0
            scores.append(w * injury_factor)
        global_idx = _clamp(max(scores) if scores else 0.0, 0.0, 1.0)

        self._state = InjuryState(
            joints=new_joints,
            globalInjuryIndex=global_idx,
        )
        return self._state

    def force_strain(self, joint: str, strain: float) -> None:
        """Dev helper: force strain level on a specific joint.

        Parameters
        ----------
        joint :
            Joint name (must be in :data:`JOINT_NAMES`).
        strain :
            Desired strain level [0..1].
        """
        if joint not in self._state.joints:
            return
        j = self._state.joints[joint]
        self._state.joints[joint] = JointInjury(
            strain=_clamp(strain, 0.0, 1.0),
            acute=j.acute,
            painAvoidance=_clamp(strain * 0.8, 0.0, 1.0),
            recoveryRate=j.recoveryRate,
        )
        self._recompute_global()

    def debug_info(self) -> dict:
        """Return current injury scalars for logging / dev output."""
        result = {"globalInjuryIndex": self._state.globalInjuryIndex}
        for name, j in self._state.joints.items():
            result[name] = {
                "strain":       round(j.strain, 4),
                "acute":        round(j.acute, 4),
                "painAvoidance": round(j.painAvoidance, 4),
                "recoveryRate": round(j.recoveryRate, 6),
            }
        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _tick_joint(
        self,
        j:       JointInjury,
        jl:      JointLoad,
        dt:      float,
        resting: bool,
    ) -> JointInjury:
        """Update a single joint's injury state for one tick."""

        # § 5.1 Chronic overload from sustained torque
        tau_ratio = jl.tau / max(jl.tau_max, 1e-6)
        strain_rate = self._k_strain * (tau_ratio ** self._strain_power)

        # § 5.2 Acute spike from impact impulse
        excess_impulse = _clamp(jl.impactImpulse - self._impact_threshold, 0.0, 1.0)
        acute_spike    = self._k_acute * excess_impulse

        # Grasp force jerk contributes to upper-limb acute
        acute_spike += self._k_acute * 0.5 * _clamp(jl.graspForcePeak, 0.0, 1.0)

        new_acute = _clamp(j.acute + acute_spike - self._acute_decay_k * dt, 0.0, 1.0)

        # § 5.3 acute → strain conversion
        acute_to_strain = j.acute * self._acute_to_strain * dt
        new_strain = j.strain + strain_rate * dt + acute_to_strain

        # § 8 Recovery
        recover_rate = 0.0
        if resting:
            recover_rate = self._recover_k
            new_strain  = new_strain - recover_rate * dt
            new_acute   = max(0.0, new_acute - self._acute_decay_k * dt)

        new_strain = _clamp(new_strain, 0.0, 1.0)
        new_acute  = _clamp(new_acute, 0.0, 1.0)

        # § 7 painAvoidance — smoothed behavioural response
        injury_signal  = _clamp(new_strain + new_acute * 0.6, 0.0, 1.0)
        target_pain    = injury_signal
        pain_tau       = 5.0 if injury_signal > j.painAvoidance else 20.0
        alpha_pain     = 1.0 - math.exp(-dt / max(pain_tau, 1e-6))
        new_pain       = _clamp(
            j.painAvoidance + (target_pain - j.painAvoidance) * alpha_pain,
            0.0, 1.0,
        )
        if resting:
            new_pain = _clamp(new_pain - self._pain_avoidance_k * dt, 0.0, 1.0)

        return JointInjury(
            strain=new_strain,
            acute=new_acute,
            painAvoidance=new_pain,
            recoveryRate=recover_rate,
        )

    def _is_resting(self, env: InjuryEnvInput) -> bool:
        """Return True when conditions support recovery."""
        return (
            env.speed          <= self._rest_speed_max
            and env.windLoad   <= self._rest_wind_max
            and env.supportQuality >= self._rest_support_min
            and not env.isGrasping
        )

    def _recompute_global(self) -> None:
        """Recompute globalInjuryIndex from current joint states."""
        leg_joints = {"ankle_l", "ankle_r", "knee_l", "knee_r", "hip_l", "hip_r"}
        scores = []
        for name, j in self._state.joints.items():
            w = 1.2 if name in leg_joints else 1.0
            scores.append(w * _clamp(j.strain + j.acute * 0.5, 0.0, 1.0))
        self._state.globalInjuryIndex = _clamp(
            max(scores) if scores else 0.0, 0.0, 1.0
        )
