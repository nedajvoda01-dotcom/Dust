"""ProceduralAnimationSystem — Stage 17 procedural animation without assets.

Generates character animation entirely from math — no clips, no mocap, no
external files.  All motion is synthesised on-the-fly from physics signals,
environment parameters, and deterministic seeded noise.

Subsystems
----------
GaitGenerator    — continuous phase-based gait, analytic joint angles
FootPlacement    — stance foot-locking + 2-bone IK correction
BodyBalancer     — pelvis spring-damper, spine lean into wind/slope
ActionLayers     — brace / stumble / crouch / slip / grab / fall overlays
StyleController  — cartoon exaggeration, squash/stretch, timing variation
PoseComposer     — additive layer blending with per-bone masks
ProceduralAnimationSystem — main API; reads CharacterPhysicalController +
                            AnimParamFrame + AnimEventStream each tick

Public API
----------
ProceduralAnimationSystem(config=None, global_seed=42, character_id=0)
  .update(ctrl, anim_frame, events, dt, game_time)  — advance one tick
  .pose              → Dict[str, BonePose]           — current pose
  .gait_phase        → float                         — current [0..1) phase
  .foot_world(side)  → Optional[Vec3]                — locked foot world-pos
  .pose_hash()       → str                           — deterministic SHA-256
  .get_debug_info()  → dict                          — gizmo/log data (no UI)

Debug flags (module-level booleans)
------------------------------------
DEBUG_DRAW_SKELETON       — log all bone world positions each tick
DEBUG_DRAW_FOOT_TARGETS   — log foot swing targets each tick
DEBUG_SHOW_GAIT_PHASE     — log gait phase each tick
DEBUG_ACTIVE_LAYERS       — log active action layers each tick
"""
from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

from src.math.Vec3 import Vec3
from src.systems.CharacterPhysicalController import CharacterState
from src.systems.CharacterEnvironmentIntegration import AnimParamFrame
from src.systems.ReflexSystem import AnimEvent, AnimEventType

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level debug flags
# ---------------------------------------------------------------------------

DEBUG_DRAW_SKELETON: bool     = False
DEBUG_DRAW_FOOT_TARGETS: bool = False
DEBUG_SHOW_GAIT_PHASE: bool   = False
DEBUG_ACTIVE_LAYERS: bool     = False

# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

# 2-bone IK solver tolerances
_IK_EPSILON: float       = 1e-9   # near-zero length guard
_IK_REACH_MARGIN: float  = 1e-4   # margin to keep IK within achievable range

# Spatial partitioning for RNG seeding
_LAT_CELL_DIVISIONS: int = 16     # latitude cells (π rad / 16 ≈ 11° per cell)
_LON_CELL_DIVISIONS: int = 8      # longitude cells (2π rad / 8 = 45° per cell)

# Hand-side selection hash parameters (seeded per trigger, not per frame)
_HAND_TIME_SCALE: int    = 10     # converts game_time to integer ticks
_HAND_HASH_MOD:   int    = 97     # prime modulus for index into RNG

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * _clamp(t, 0.0, 1.0)


def _cfg(config, section: str, key: str, default: float) -> float:
    if config is None:
        return default
    return config.get(section, key, default=default)


# ---------------------------------------------------------------------------
# Skeleton definition
# ---------------------------------------------------------------------------

BONE_NAMES: List[str] = [
    "root",
    "pelvis",
    "spine1", "spine2", "spine3",
    "head",
    "upper_arm_l", "lower_arm_l",
    "upper_arm_r", "lower_arm_r",
    "upper_leg_l", "lower_leg_l", "foot_l",
    "upper_leg_r", "lower_leg_r", "foot_r",
]

BONE_PARENT: Dict[str, str] = {
    "pelvis":      "root",
    "spine1":      "pelvis",
    "spine2":      "spine1",
    "spine3":      "spine2",
    "head":        "spine3",
    "upper_arm_l": "spine3",
    "lower_arm_l": "upper_arm_l",
    "upper_arm_r": "spine3",
    "lower_arm_r": "upper_arm_r",
    "upper_leg_l": "pelvis",
    "lower_leg_l": "upper_leg_l",
    "foot_l":      "lower_leg_l",
    "upper_leg_r": "pelvis",
    "lower_leg_r": "upper_leg_r",
    "foot_r":      "lower_leg_r",
}

# Bind-pose local positions (metres from parent joint).
# Character frame: +Y = up (away from planet), +Z = forward, +X = right.
# Total character height ≈ 1.8 m.
BIND_LOCAL_POS: Dict[str, Vec3] = {
    "root":        Vec3( 0.00,  0.00,  0.00),
    "pelvis":      Vec3( 0.00,  0.90,  0.00),  # ~half-height above feet
    "spine1":      Vec3( 0.00,  0.22,  0.00),
    "spine2":      Vec3( 0.00,  0.22,  0.00),
    "spine3":      Vec3( 0.00,  0.18,  0.00),  # upper chest
    "head":        Vec3( 0.00,  0.20,  0.00),
    "upper_arm_l": Vec3(-0.20,  0.00,  0.00),
    "lower_arm_l": Vec3(-0.28,  0.00,  0.00),
    "upper_arm_r": Vec3( 0.20,  0.00,  0.00),
    "lower_arm_r": Vec3( 0.28,  0.00,  0.00),
    "upper_leg_l": Vec3(-0.15, -0.10,  0.00),  # hip joint
    "lower_leg_l": Vec3( 0.00, -0.40,  0.00),  # knee
    "foot_l":      Vec3( 0.00, -0.40,  0.00),  # ankle
    "upper_leg_r": Vec3( 0.15, -0.10,  0.00),
    "lower_leg_r": Vec3( 0.00, -0.40,  0.00),
    "foot_r":      Vec3( 0.00, -0.40,  0.00),
}

# Hip-joint offsets from root (sum of pelvis + upper_leg bind positions).
# Used by FootPlacement IK.
_HIP_LOCAL_L = Vec3(
    BIND_LOCAL_POS["pelvis"].x + BIND_LOCAL_POS["upper_leg_l"].x,
    BIND_LOCAL_POS["pelvis"].y + BIND_LOCAL_POS["upper_leg_l"].y,
    0.0,
)  # ≈ (-0.15, 0.80, 0)
_HIP_LOCAL_R = Vec3(
    BIND_LOCAL_POS["pelvis"].x + BIND_LOCAL_POS["upper_leg_r"].x,
    BIND_LOCAL_POS["pelvis"].y + BIND_LOCAL_POS["upper_leg_r"].y,
    0.0,
)  # ≈ (+0.15, 0.80, 0)

_THIGH_LEN  = abs(BIND_LOCAL_POS["lower_leg_l"].y)  # 0.40
_SHIN_LEN   = abs(BIND_LOCAL_POS["foot_l"].y)        # 0.40
_UPPER_ARM  = abs(BIND_LOCAL_POS["lower_arm_l"].x)   # 0.28
_LOWER_ARM  = 0.25  # forearm (shorter than upper arm segment)


# ---------------------------------------------------------------------------
# BonePose
# ---------------------------------------------------------------------------

@dataclass
class BonePose:
    """Additive pose delta for one bone, relative to bind pose.

    Rotations are in radians and applied around local axes (FK order X→Y→Z).
    ``dy`` is a local Y position offset (e.g. for pelvis bob).
    ``scale`` is a uniform scale factor (1.0 = no squash/stretch).
    """
    rx:    float = 0.0   # flex / extend (sagittal)
    ry:    float = 0.0   # twist / ab-adduction
    rz:    float = 0.0   # lateral lean
    dy:    float = 0.0   # local Y position offset from bind
    scale: float = 1.0   # squash/stretch scale


# ---------------------------------------------------------------------------
# _SeededRng — deterministic per-bucket hash RNG
# ---------------------------------------------------------------------------

class _SeededRng:
    """Hash-based pseudo-random number generator.

    Values are stable for a given (seed, idx) pair and change only when
    the seed changes (i.e. at time-bucket boundaries).
    """

    def __init__(self, seed: int) -> None:
        self._seed: int = int(seed) & 0xFFFF_FFFF

    def float(self, idx: int) -> float:
        """Return a deterministic value in [0, 1)."""
        h = (self._seed ^ (idx * 2_246_822_519)) & 0xFFFF_FFFF
        h = (h * 2_654_435_761) & 0xFFFF_FFFF
        h = (h ^ (h >> 16)) & 0xFFFF_FFFF
        return (h >> 8) / float(0xFF_FFFF)

    def signed(self, idx: int) -> float:
        """Return a deterministic value in [-1, 1)."""
        return self.float(idx) * 2.0 - 1.0

    @staticmethod
    def make(global_seed: int, character_id: int, time_bucket: int,
             lat_cell: int = 0, lon_cell: int = 0) -> "_SeededRng":
        """Create an RNG seeded by position + time + character identity."""
        h = (global_seed * 2_654_435_761
             ^ character_id * 40_503
             ^ time_bucket * 22_695_477
             ^ lat_cell * 6_364_136_223_846_793_005
             ^ lon_cell * 1_442_695_040_888_963_407)
        return _SeededRng(h & 0xFFFF_FFFF)


# ---------------------------------------------------------------------------
# 2-bone IK — analytical solver
# ---------------------------------------------------------------------------

def _two_bone_ik(
    root_pos: Vec3,
    target_pos: Vec3,
    upper_len: float,
    lower_len: float,
    pole_dir: Vec3,
) -> Tuple[Vec3, Vec3]:
    """Solve 2-bone IK in any coordinate frame.

    Returns (mid_joint_pos, end_pos) where end_pos is clamped to reach.
    ``pole_dir`` biases the mid-joint toward the indicated direction
    (e.g. forward for elbows, backward for knees).
    """
    to_target = target_pos - root_pos
    dist = to_target.length()

    max_reach = upper_len + lower_len - _IK_REACH_MARGIN
    min_reach = abs(upper_len - lower_len) + _IK_REACH_MARGIN
    dist_c = _clamp(dist, min_reach, max_reach)

    # Law of cosines — angle at root joint
    cos_a = _clamp(
        (upper_len ** 2 + dist_c ** 2 - lower_len ** 2) / (2.0 * upper_len * dist_c),
        -1.0, 1.0,
    )
    sin_a = math.sqrt(max(0.0, 1.0 - cos_a ** 2))

    dir_t = to_target.normalized() if dist > _IK_EPSILON else Vec3(0.0, -1.0, 0.0)

    # Perpendicular component of pole toward which knee bends
    pole_proj = pole_dir - dir_t * dir_t.dot(pole_dir)
    pole_n = pole_proj.normalized() if pole_proj.length() > _IK_EPSILON else Vec3(0.0, 0.0, 1.0)

    mid_pos = root_pos + (dir_t * (upper_len * cos_a) + pole_n * (upper_len * sin_a))
    end_pos = root_pos + dir_t * dist_c
    return mid_pos, end_pos


# ---------------------------------------------------------------------------
# GaitGenerator
# ---------------------------------------------------------------------------

class GaitGenerator:
    """Continuous gait phase and analytic joint angles.

    The gait cycle is parametrised by a single phase ∈ [0, 1):
      • Left foot  STANCE [0.00, 0.50)  SWING [0.50, 1.00)
      • Right foot STANCE [0.50, 1.00)  SWING [0.00, 0.50)

    Joint angles are computed analytically; no animation clips are used.
    """

    # Hip flexion amplitude (radians) — half of total stride angle range
    _HIP_FLEX: float    = 0.32
    # Maximum knee flexion during swing (radians)
    _KNEE_FLEX: float   = 0.50
    # Lateral pelvis sway amplitude (radians)
    _PELVIS_SWAY: float = 0.06
    # Arm counter-swing amplitude base (scaled by arm_swing_amp)
    _ARM_BASE: float    = 0.35

    def __init__(self, config=None, rng: Optional[_SeededRng] = None) -> None:
        self._phase:             float        = 0.0
        self._base_cadence:      float        = _cfg(config, "anim", "base_cadence",       1.0)
        self._base_stride:       float        = _cfg(config, "anim", "base_stride",        0.8)
        self._step_height_base:  float        = _cfg(config, "anim", "step_height_base",   0.15)
        self._stance_width_base: float        = _cfg(config, "anim", "stance_width_base",  0.15)
        self._timing_var:        float        = _cfg(config, "anim", "timing_var_strength", 0.06)
        self._rng: _SeededRng                 = rng or _SeededRng(42)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(
        self,
        dt: float,
        anim_frame: AnimParamFrame,
        grounded: bool,
        sliding: bool,
    ) -> None:
        """Advance gait phase by one timestep."""
        if not grounded:
            return  # freeze gait while airborne

        cadence = anim_frame.cadence * self._base_cadence
        # Stable per-bucket timing variation (not per-frame)
        cadence *= 1.0 + self._rng.signed(7) * self._timing_var

        self._phase = math.fmod(self._phase + cadence * dt, 1.0)
        if self._phase < 0.0:
            self._phase += 1.0

    # ------------------------------------------------------------------
    # Foot target (local character frame)
    # ------------------------------------------------------------------

    def foot_target_local(
        self,
        side: int,               # 0 = left, 1 = right
        anim_frame: AnimParamFrame,
    ) -> Tuple[Vec3, float]:
        """Return (foot_local_position, stance_weight ∈ {0, 1}).

        Local frame: +Y up, +Z forward, +X right.
        Foot position is relative to the root bone (character origin at feet).
        """
        p      = self._phase
        stride = anim_frame.stride_length * self._base_stride
        step_h = self._step_height_base * anim_frame.step_height_scale
        sw     = self._stance_width_base

        # Side-dependent phase offset: right foot leads by half-cycle
        if side == 0:   # left
            in_stance = (p < 0.5)
            swing_t   = (p - 0.5) * 2.0 if not in_stance else 0.0
            stance_t  = p * 2.0          if in_stance     else 0.0
        else:           # right
            in_stance = (p >= 0.5)
            swing_t   = p * 2.0          if not in_stance else 0.0
            stance_t  = (p - 0.5) * 2.0 if in_stance     else 0.0

        x = -sw if side == 0 else sw

        if in_stance:
            # Foot slides from +stride/2 (forward) to -stride/2 (behind) during stance
            z = stride * 0.5 * (1.0 - 2.0 * stance_t)
            return Vec3(x, 0.0, z), 1.0
        else:
            # S-curve in Z, arch in Y
            z = stride * 0.5 * math.sin(math.pi * (swing_t - 0.5))
            y = step_h * (math.sin(math.pi * swing_t) ** 2)
            # Small seeded lateral nudge for variation
            x += self._rng.signed(side + 10) * 0.02
            return Vec3(x, y, z), 0.0

    # ------------------------------------------------------------------
    # Analytic joint angles
    # ------------------------------------------------------------------

    def compute_pose(self, anim_frame: AnimParamFrame) -> Dict[str, BonePose]:
        """Return per-bone pose deltas for the current gait phase."""
        p       = self._phase
        stride  = anim_frame.stride_length  # normalised 0..1.5
        arm_amp = anim_frame.arm_swing_amp

        hip_flex = self._HIP_FLEX * stride

        # ---- Left leg -----------------------------------------------
        if p < 0.5:
            stance_t     = p * 2.0
            ul_rx_l      = _lerp(+hip_flex, -hip_flex, stance_t)
            ll_rx_l      = -0.08 * math.sin(math.pi * stance_t)   # slight bend
        else:
            swing_t      = (p - 0.5) * 2.0
            ul_rx_l      = _lerp(-hip_flex, +hip_flex, swing_t)
            ll_rx_l      = -self._KNEE_FLEX * math.sin(math.pi * swing_t)

        # ---- Right leg (half-cycle offset) ---------------------------
        if p >= 0.5:
            stance_t     = (p - 0.5) * 2.0
            ul_rx_r      = _lerp(+hip_flex, -hip_flex, stance_t)
            ll_rx_r      = -0.08 * math.sin(math.pi * stance_t)
        else:
            swing_t      = p * 2.0
            ul_rx_r      = _lerp(-hip_flex, +hip_flex, swing_t)
            ll_rx_r      = -self._KNEE_FLEX * math.sin(math.pi * swing_t)

        # Foot follows ankle: small dorsiflexion counter to shin angle
        foot_rx_l = ul_rx_l * 0.25
        foot_rx_r = ul_rx_r * 0.25

        # ---- Arms (counter to opposite leg) -------------------------
        arm_rx_l = -ul_rx_r * arm_amp * self._ARM_BASE
        arm_rx_r = -ul_rx_l * arm_amp * self._ARM_BASE
        arm_rz_l =  0.05    # natural resting abduction
        arm_rz_r = -0.05
        elbow_rx = -0.18    # constant slight elbow bend

        # ---- Pelvis lateral sway ------------------------------------
        pelvis_rz = self._PELVIS_SWAY * math.sin(2.0 * math.pi * p)

        # ---- Spine counter-rotation --------------------------------
        spine2_ry = anim_frame.torso_twist * math.cos(2.0 * math.pi * p) * 0.4
        spine3_ry = -spine2_ry * 0.5

        # ---- Head bob (vertical) -----------------------------------
        head_bob_rx = anim_frame.head_bob * 0.03 * math.sin(4.0 * math.pi * p)

        return {
            "pelvis":      BonePose(rz=pelvis_rz),
            "spine2":      BonePose(ry=spine2_ry),
            "spine3":      BonePose(ry=spine3_ry),
            "head":        BonePose(rx=head_bob_rx),
            "upper_arm_l": BonePose(rx=arm_rx_l, rz=arm_rz_l),
            "lower_arm_l": BonePose(rx=elbow_rx),
            "upper_arm_r": BonePose(rx=arm_rx_r, rz=arm_rz_r),
            "lower_arm_r": BonePose(rx=elbow_rx),
            "upper_leg_l": BonePose(rx=ul_rx_l),
            "lower_leg_l": BonePose(rx=ll_rx_l),
            "foot_l":      BonePose(rx=foot_rx_l),
            "upper_leg_r": BonePose(rx=ul_rx_r),
            "lower_leg_r": BonePose(rx=ll_rx_r),
            "foot_r":      BonePose(rx=foot_rx_r),
        }

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def phase(self) -> float:
        return self._phase


# ---------------------------------------------------------------------------
# FootPlacement — stance locking + IK correction
# ---------------------------------------------------------------------------

@dataclass
class _FootState:
    lock_world: Optional[Vec3] = None
    is_locked:  bool           = False


class FootPlacement:
    """Locks feet to world during stance and applies minimal foot IK.

    Foot locking prevents "skating" while the character is grounded and
    not sliding.  During swing the procedural target from GaitGenerator
    is used directly.
    """

    def __init__(self, config=None) -> None:
        self._feet:            List[_FootState]  = [_FootState(), _FootState()]
        self._foot_align:      float             = _cfg(config, "anim", "foot_align_strength", 0.5)

    def update(
        self,
        side:           int,
        gait_target:    Vec3,
        stance_weight:  float,
        ctrl_pos:       Vec3,
        up:             Vec3,
        fwd:            Vec3,
        right:          Vec3,
        sliding:        bool,
    ) -> Vec3:
        """Return the world-space foot position for this tick.

        During stance (stance_weight > 0.5 and not sliding) the foot is
        locked to its initial contact point.  During swing the procedural
        target is converted to world space and returned.
        """
        # Convert local gait target to world space
        world = (ctrl_pos
                 + up    * gait_target.y
                 + fwd   * gait_target.z
                 + right * gait_target.x)

        foot = self._feet[side]

        if stance_weight > 0.5 and not sliding:
            if not foot.is_locked:
                foot.lock_world = Vec3(world.x, world.y, world.z)
                foot.is_locked  = True
            return Vec3(foot.lock_world.x, foot.lock_world.y, foot.lock_world.z)
        else:
            foot.is_locked  = False
            foot.lock_world = None
            return world

    def get_foot_world(self, side: int) -> Optional[Vec3]:
        """Return the locked world position of a foot, or None if swinging."""
        f = self._feet[side]
        return f.lock_world if f.is_locked else None

    def is_locked(self, side: int) -> bool:
        return self._feet[side].is_locked


# ---------------------------------------------------------------------------
# BodyBalancer
# ---------------------------------------------------------------------------

class BodyBalancer:
    """Generates realistic body balance adjustments.

    Uses a spring-damper on the pelvis for weight-shift inertia, then
    derives spine and head lean from velocity, wind, and slope.
    """

    def __init__(self, config=None) -> None:
        self._pelvis_vel:    float = 0.0
        self._pelvis_offset: float = 0.0
        self._spring_k:      float = 12.0
        self._spring_d:      float =  4.0

    def update(
        self,
        dt:          float,
        anim_frame:  AnimParamFrame,
        lean:        float,       # 0..1 lateral lean (wind/slope)
        slope_angle: float,       # radians, forward slope
        grounded:    bool,
        gait_phase:  float,
    ) -> Dict[str, BonePose]:
        """Return additive bone adjustments for body balance."""
        # Pelvis vertical bob (spring-damped toward gait-cycle target)
        pelvis_target = -0.04 * math.sin(2.0 * math.pi * gait_phase)
        if not grounded:
            pelvis_target = -0.06

        spring_f = (pelvis_target - self._pelvis_offset) * self._spring_k
        self._pelvis_vel     += spring_f * dt
        self._pelvis_vel     *= max(0.0, 1.0 - self._spring_d * dt)
        self._pelvis_offset  += self._pelvis_vel * dt
        self._pelvis_offset   = _clamp(self._pelvis_offset, -0.18, 0.12)

        # Forward lean from effort + slope compensation
        fwd_lean   = anim_frame.effort * 0.18 + slope_angle * 0.25
        lat_lean   = lean * 0.22   # lateral lean into wind

        return {
            "pelvis": BonePose(dy=self._pelvis_offset, rz=lean * 0.14),
            "spine1": BonePose(rx=fwd_lean + slope_angle * 0.15, rz=lat_lean),
            "spine2": BonePose(rx=fwd_lean * 0.6,               rz=lat_lean * 0.6),
            "spine3": BonePose(rx=fwd_lean * 0.35),
            "head":   BonePose(rx=-(fwd_lean + slope_angle * 0.15) * 0.4),
        }


# ---------------------------------------------------------------------------
# ActionLayers
# ---------------------------------------------------------------------------

class _ActionLayerType(Enum):
    BRACE         = auto()
    STUMBLE       = auto()
    CROUCH_GUST   = auto()
    SLIP_RECOVER  = auto()
    GRAB_LEDGE    = auto()
    FALL          = auto()


@dataclass
class _ActiveLayer:
    layer_type:    _ActionLayerType
    trigger_time:  float
    duration:      float           # 0 = sustained until cancelled
    elapsed:       float           = 0.0
    weight:        float           = 0.0
    contact_point: Optional[Vec3]  = None
    hand_side:     int             = 0   # 0=left, 1=right


_LAYER_DURATIONS: Dict[_ActionLayerType, float] = {
    _ActionLayerType.BRACE:        0.8,
    _ActionLayerType.STUMBLE:      0.4,
    _ActionLayerType.CROUCH_GUST:  0.0,   # sustained
    _ActionLayerType.SLIP_RECOVER: 0.55,
    _ActionLayerType.GRAB_LEDGE:   0.0,   # sustained
    _ActionLayerType.FALL:         0.0,   # sustained
}

# Mapping from AnimEventType to internal layer type
_EVENT_TO_LAYER: Dict[AnimEventType, _ActionLayerType] = {
    AnimEventType.ON_BRACE:        _ActionLayerType.BRACE,
    AnimEventType.ON_STUMBLE_STEP: _ActionLayerType.STUMBLE,
    AnimEventType.ON_KNEEL_IN_GUST: _ActionLayerType.CROUCH_GUST,
    AnimEventType.ON_SLIP_RECOVER: _ActionLayerType.SLIP_RECOVER,
    AnimEventType.ON_GRAB_LEDGE:   _ActionLayerType.GRAB_LEDGE,
    AnimEventType.ON_FALL:         _ActionLayerType.FALL,
}


class ActionLayers:
    """Manages reactive action-layer overlays.

    Each layer is triggered by an ``AnimEvent``, runs for its duration
    (or until cancelled for sustained layers), and blends its pose
    contribution using an ease-in/ease-out weight curve.
    """

    def __init__(self, config=None, rng: Optional[_SeededRng] = None) -> None:
        self._layers:     List[_ActiveLayer] = []
        self._hand_bias:  float              = _cfg(config, "anim",
                                                    "brace_hand_preference_bias", 0.5)
        self._rng: _SeededRng                = rng or _SeededRng(42)

    # ------------------------------------------------------------------
    # Event ingestion
    # ------------------------------------------------------------------

    def process_events(
        self,
        events:    List[AnimEvent],
        game_time: float,
    ) -> None:
        """Translate animation events into active layers."""
        for event in events:
            lt = _EVENT_TO_LAYER.get(event.type)
            if lt is None:
                continue
            self._trigger(lt, game_time, event.contact_point)

    def _trigger(
        self,
        lt:            _ActionLayerType,
        game_time:     float,
        contact_point: Optional[Vec3] = None,
    ) -> None:
        # Remove any existing layer of the same type
        self._layers = [l for l in self._layers if l.layer_type != lt]

        # Deterministic hand side selection (seeded, stable within RNG bucket)
        hand_side = 0 if self._rng.float(
            int(game_time * _HAND_TIME_SCALE) % _HAND_HASH_MOD
        ) < self._hand_bias else 1

        self._layers.append(_ActiveLayer(
            layer_type    = lt,
            trigger_time  = game_time,
            duration      = _LAYER_DURATIONS[lt],
            contact_point = contact_point,
            hand_side     = hand_side,
        ))

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, dt: float, game_time: float) -> None:
        """Advance layer timers and remove expired non-sustained layers."""
        surviving: List[_ActiveLayer] = []
        for layer in self._layers:
            layer.elapsed += dt
            layer.weight   = self._weight(layer)
            if layer.duration == 0.0 or layer.elapsed < layer.duration:
                surviving.append(layer)
        self._layers = surviving

    @staticmethod
    def _weight(layer: _ActiveLayer) -> float:
        """Ease-in/ease-out weight (sin²(π·t)) for timed layers."""
        if layer.duration == 0.0:
            return 1.0
        t = _clamp(layer.elapsed / layer.duration, 0.0, 1.0)
        return math.sin(math.pi * t) ** 2

    # ------------------------------------------------------------------
    # Cancel a sustained layer by type name
    # ------------------------------------------------------------------

    def cancel(self, layer_type: _ActionLayerType) -> None:
        self._layers = [l for l in self._layers if l.layer_type != layer_type]

    # ------------------------------------------------------------------
    # Pose contribution
    # ------------------------------------------------------------------

    def compute_pose(self, anim_frame: AnimParamFrame) -> Dict[str, BonePose]:
        """Compute additive pose delta from all active layers."""
        result: Dict[str, BonePose] = {}
        for layer in self._layers:
            w = layer.weight
            if w < 1e-4:
                continue
            if layer.layer_type == _ActionLayerType.BRACE:
                self._apply_brace(result, layer, w)
            elif layer.layer_type == _ActionLayerType.STUMBLE:
                self._apply_stumble(result, w)
            elif layer.layer_type == _ActionLayerType.CROUCH_GUST:
                self._apply_crouch(result, w)
            elif layer.layer_type == _ActionLayerType.SLIP_RECOVER:
                self._apply_slip(result, w)
            elif layer.layer_type == _ActionLayerType.GRAB_LEDGE:
                self._apply_grab(result, w)
            elif layer.layer_type == _ActionLayerType.FALL:
                self._apply_fall(result, w)
        return result

    # ------------------------------------------------------------------
    # Per-layer pose contributions
    # ------------------------------------------------------------------

    @staticmethod
    def _add(
        result: Dict[str, BonePose], bone: str,
        rx: float = 0.0, ry: float = 0.0, rz: float = 0.0,
        dy: float = 0.0, scale: float = 1.0,
    ) -> None:
        b  = result.get(bone, BonePose())
        result[bone] = BonePose(
            rx    = b.rx + rx,
            ry    = b.ry + ry,
            rz    = b.rz + rz,
            dy    = b.dy + dy,
            scale = b.scale * scale,
        )

    def _apply_brace(
        self,
        result: Dict[str, BonePose],
        layer:  _ActiveLayer,
        w:      float,
    ) -> None:
        """One arm reaches to contact point; torso leans toward support."""
        s  = "_l" if layer.hand_side == 0 else "_r"
        sg = -1.0 if layer.hand_side == 0 else 1.0
        self._add(result, f"upper_arm{s}", rx=-w * 0.85, rz= sg * w * 0.40)
        self._add(result, f"lower_arm{s}", rx=-w * 0.60)
        self._add(result, "spine1",        rx=-w * 0.18, rz= sg * w * 0.22)
        self._add(result, "spine2",        rz= sg * w * 0.12)

    def _apply_stumble(self, result: Dict[str, BonePose], w: float) -> None:
        """Torso dips forward; arms reach out for balance."""
        self._add(result, "spine1",      rx=-w * 0.28)
        self._add(result, "pelvis",      dy=-w * 0.05)
        self._add(result, "upper_arm_l", rx=-w * 0.40)
        self._add(result, "upper_arm_r", rx=-w * 0.40)

    def _apply_crouch(self, result: Dict[str, BonePose], w: float) -> None:
        """Pelvis drops; arms draw inward; head tucks."""
        self._add(result, "pelvis",      dy=-w * 0.14)
        self._add(result, "spine1",      rx=-w * 0.38)
        self._add(result, "head",        rx= w * 0.18)
        self._add(result, "upper_arm_l", rz= w * 0.28)
        self._add(result, "upper_arm_r", rz=-w * 0.28)

    def _apply_slip(self, result: Dict[str, BonePose], w: float) -> None:
        """Lean back; arms spread wide for balance."""
        self._add(result, "spine1",      rx= w * 0.40)
        self._add(result, "upper_arm_l", rz=-w * 0.75)
        self._add(result, "upper_arm_r", rz= w * 0.75)

    def _apply_grab(self, result: Dict[str, BonePose], w: float) -> None:
        """Both arms overhead; legs loosely hang."""
        self._add(result, "upper_arm_l", rx=-w * 1.20, rz= w * 0.20)
        self._add(result, "lower_arm_l", rx=-w * 0.50)
        self._add(result, "upper_arm_r", rx=-w * 1.20, rz=-w * 0.20)
        self._add(result, "lower_arm_r", rx=-w * 0.50)
        self._add(result, "upper_leg_l", rx= w * 0.25)
        self._add(result, "upper_leg_r", rx= w * 0.25)

    def _apply_fall(self, result: Dict[str, BonePose], w: float) -> None:
        """Arms flail; spine relaxes."""
        self._add(result, "upper_arm_l", rz=-w * 0.55)
        self._add(result, "upper_arm_r", rz= w * 0.55)
        self._add(result, "spine1",      rx=-w * 0.22)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def active_layer_names(self) -> List[str]:
        """Names of currently active (weight > 0.01) layers."""
        return [l.layer_type.name for l in self._layers if l.weight > 0.01]

    def has_layer(self, lt: _ActionLayerType) -> bool:
        """Return True if a layer of the given type is active (weight > 0)."""
        return any(l.layer_type == lt and l.weight > 0.0 for l in self._layers)


# ---------------------------------------------------------------------------
# StyleController — cartoon exaggeration + squash/stretch + timing variation
# ---------------------------------------------------------------------------

class StyleController:
    """Applies cartoon-friendly motion enhancement.

    Controlled exaggeration scales arm swing and torso twist at high effort.
    Squash/stretch is applied to the pelvis scale (~±7% max) in sync with
    the gait phase landing/lift-off rhythm.
    Micro-jitter (wind/rumble) adds sub-degree tremor to the head.
    """

    def __init__(self, config=None) -> None:
        self._exag_max:  float = _cfg(config, "anim", "style_exaggeration_max", 0.30)
        self._ss_max:    float = _cfg(config, "anim", "squash_stretch_max",     0.07)
        self._jitter:    float = _cfg(config, "anim", "micro_jitter_strength",  0.02)

    def apply(
        self,
        pose:       Dict[str, BonePose],
        anim_frame: AnimParamFrame,
        gait_phase: float,
        rng:        _SeededRng,
    ) -> Dict[str, BonePose]:
        """Return a new pose dict with style enhancements applied."""
        # Copy pose so we don't mutate the input
        result: Dict[str, BonePose] = {
            k: BonePose(v.rx, v.ry, v.rz, v.dy, v.scale) for k, v in pose.items()
        }
        # Ensure all bone slots exist
        for b in BONE_NAMES:
            if b not in result:
                result[b] = BonePose()

        style  = _clamp(anim_frame.effort * 2.0, 0.0, 1.0)
        exag   = style * self._exag_max

        # Amplified arm swing
        for arm in ("upper_arm_l", "upper_arm_r"):
            p = result[arm]
            result[arm] = BonePose(
                rx=p.rx * (1.0 + exag),
                ry=p.ry,
                rz=p.rz * (1.0 + exag * 0.5),
                dy=p.dy,
                scale=p.scale,
            )

        # Torso twist exaggeration
        for bone in ("spine2", "spine3"):
            p = result[bone]
            twist_boost = anim_frame.torso_twist * exag * 0.4
            result[bone] = BonePose(p.rx, p.ry + twist_boost, p.rz, p.dy, p.scale)

        # Squash / stretch on pelvis (2 peaks per gait cycle)
        sq_factor = math.sin(2.0 * math.pi * gait_phase) * self._ss_max * style
        p = result["pelvis"]
        result["pelvis"] = BonePose(p.rx, p.ry, p.rz, p.dy, scale=p.scale * (1.0 + sq_factor))

        # Micro-jitter (head, seeded per bucket)
        jit_x = rng.signed(20) * anim_frame.micro_jitter * self._jitter
        jit_z = rng.signed(21) * anim_frame.micro_jitter * self._jitter
        h = result["head"]
        result["head"] = BonePose(h.rx + jit_x, h.ry, h.rz + jit_z, h.dy, h.scale)

        return result


# ---------------------------------------------------------------------------
# PoseComposer
# ---------------------------------------------------------------------------

class PoseComposer:
    """Assembles the final pose from ordered additive contributions.

    Layer order (matching spec §11):
      1. Gait base (GaitGenerator analytic angles)
      2. Body balancer (BodyBalancer spring/lean deltas)
      3. Action layers (ActionLayers overlays)
      4. Style (StyleController exaggeration / squash-stretch)
    """

    def compose(
        self,
        gait:   Dict[str, BonePose],
        body:   Dict[str, BonePose],
        action: Dict[str, BonePose],
        style:  Dict[str, BonePose],
    ) -> Dict[str, BonePose]:
        """Blend layers and return the complete pose for all bones."""
        result: Dict[str, BonePose] = {b: BonePose() for b in BONE_NAMES}

        # 1. Gait
        for b, v in gait.items():
            result[b] = BonePose(v.rx, v.ry, v.rz, v.dy, v.scale)

        # 2. Body balancer (additive)
        for b, v in body.items():
            if b in result:
                r = result[b]
                result[b] = BonePose(
                    r.rx + v.rx, r.ry + v.ry, r.rz + v.rz,
                    r.dy + v.dy, r.scale * v.scale,
                )

        # 3. Action layers (additive)
        for b, v in action.items():
            if b in result:
                r = result[b]
                result[b] = BonePose(
                    r.rx + v.rx, r.ry + v.ry, r.rz + v.rz,
                    r.dy + v.dy, r.scale * v.scale,
                )

        # 4. Style replaces the full pose (it already incorporates all prev layers)
        for b, v in style.items():
            result[b] = v

        return result


# ---------------------------------------------------------------------------
# ProceduralAnimationSystem — main API
# ---------------------------------------------------------------------------

class ProceduralAnimationSystem:
    """Main procedural animation system (Stage 17).

    Reads each tick from:
      * ctrl       — CharacterPhysicalController  (pos, velocity, state)
      * anim_frame — AnimParamFrame               (stride, cadence, effort …)
      * events     — List[AnimEvent]              (from AnimationEventStream)

    Outputs:
      * .pose        — Dict[str, BonePose]  (all bones, each tick)
      * .gait_phase  — float [0..1)
      * .foot_world  — Optional[Vec3]       (locked world pos or None)
      * .pose_hash() — str SHA-256
    """

    def __init__(
        self,
        config=None,
        global_seed:    int = 42,
        character_id:   int = 0,
        variation_window_sec: float = 2.0,
    ) -> None:
        self._config            = config
        self._global_seed       = global_seed
        self._character_id      = character_id
        self._var_window        = variation_window_sec

        # Shared seeded RNG (updated each variation bucket)
        self._rng: _SeededRng   = _SeededRng.make(global_seed, character_id, 0)

        # Subsystems
        self._gait   = GaitGenerator(config, self._rng)
        self._foot   = FootPlacement(config)
        self._body   = BodyBalancer(config)
        self._layers = ActionLayers(config, self._rng)
        self._style  = StyleController(config)
        self._composer = PoseComposer()

        # Current output state
        self._pose: Dict[str, BonePose]       = {b: BonePose() for b in BONE_NAMES}
        self._foot_world: List[Optional[Vec3]] = [None, None]
        self._game_time: float                 = 0.0

    # ------------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------------

    def update(
        self,
        ctrl,            # CharacterPhysicalController
        anim_frame: AnimParamFrame,
        events:     List[AnimEvent],
        dt:         float,
        game_time:  float,
    ) -> None:
        """Advance the animation system by one timestep."""
        self._game_time = game_time

        # Refresh RNG at bucket boundaries (determinism: stable per window)
        bucket     = int(game_time / max(self._var_window, 1e-6))
        self._rng  = _SeededRng.make(
            self._global_seed, self._character_id, bucket,
            lat_cell=self._lat_cell(ctrl.position),
            lon_cell=self._lon_cell(ctrl.position),
        )
        self._gait._rng   = self._rng
        self._layers._rng = self._rng

        # Character frame vectors
        up    = ctrl.position.normalized()
        vel   = ctrl.velocity
        fwd   = self._make_fwd(vel, up)
        right = up.cross(fwd).normalized()

        grounded = ctrl.state in (
            CharacterState.GROUNDED, CharacterState.BRACED, CharacterState.CROUCHED)
        sliding  = (ctrl.state == CharacterState.SLIDING)

        # Slope angle: derive from debug_info if available, else 0
        slope_angle = 0.0
        try:
            slope_angle = ctrl.debug_info().get("slope_angle", 0.0)
        except Exception:
            pass

        # 1. Advance gait phase
        self._gait.update(dt, anim_frame, grounded, sliding)

        # 2. Foot placement & locking
        for side in (0, 1):
            tgt, stance_w = self._gait.foot_target_local(side, anim_frame)
            fw = self._foot.update(side, tgt, stance_w,
                                   ctrl.position, up, fwd, right, sliding)
            self._foot_world[side] = fw if self._foot.is_locked(side) else None

        # 3. Process events → action layer triggers
        self._layers.process_events(events, game_time)
        self._layers.update(dt, game_time)

        # 4. Compute each layer's pose contribution
        gait_p   = self._gait.compute_pose(anim_frame)
        body_p   = self._body.update(dt, anim_frame, anim_frame.lean,
                                     slope_angle, grounded, self._gait.phase)
        action_p = self._layers.compute_pose(anim_frame)

        # 5. Style exaggeration (takes combined pose as base)
        combined = self._composer.compose(gait_p, body_p, action_p, {})
        style_p  = self._style.apply(combined, anim_frame, self._gait.phase, self._rng)

        # 6. Final pose (style already has everything)
        self._pose = style_p

        # 7. Debug output
        if DEBUG_SHOW_GAIT_PHASE:
            _log.debug("gait_phase=%.4f", self._gait.phase)
        if DEBUG_ACTIVE_LAYERS:
            _log.debug("active_layers=%s", self._layers.active_layer_names)
        if DEBUG_DRAW_FOOT_TARGETS:
            for s in (0, 1):
                tgt, _ = self._gait.foot_target_local(s, anim_frame)
                _log.debug("foot_target[%d] local=%s", s, tgt)
        if DEBUG_DRAW_SKELETON:
            info = self._compute_bone_world_positions(ctrl.position, up, fwd, right)
            for b, pos in info.items():
                _log.debug("bone[%s] world=%s", b, pos)

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def pose(self) -> Dict[str, BonePose]:
        """Current complete pose (all bones)."""
        return dict(self._pose)

    @property
    def gait_phase(self) -> float:
        """Current gait phase ∈ [0, 1)."""
        return self._gait.phase

    def foot_world(self, side: int) -> Optional[Vec3]:
        """Return the locked world position of a foot, or None if swinging."""
        return self._foot_world[side]

    @property
    def action_layers(self) -> ActionLayers:
        """Direct access to the ActionLayers subsystem (for tests/dev)."""
        return self._layers

    def pose_hash(self) -> str:
        """SHA-256 of all bone angles (determinism check)."""
        parts: List[str] = []
        for b in BONE_NAMES:
            p = self._pose.get(b, BonePose())
            parts.append(f"{p.rx:.7f},{p.ry:.7f},{p.rz:.7f},{p.dy:.7f},{p.scale:.7f}")
        return hashlib.sha256(";".join(parts).encode()).hexdigest()

    def get_debug_info(self) -> dict:
        """Return a snapshot of internal state for dev tooling."""
        return {
            "gait_phase":    self._gait.phase,
            "active_layers": self._layers.active_layer_names,
            "foot_world_l":  self._foot_world[0],
            "foot_world_r":  self._foot_world[1],
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_fwd(velocity: Vec3, up: Vec3) -> Vec3:
        """Project velocity onto the tangent plane to get a forward vector."""
        v_tang = velocity - up * velocity.dot(up)
        if v_tang.length() < 0.01:
            # Default forward: arbitrary tangent perpendicular to up
            arb = Vec3(0.0, 0.0, 1.0) if abs(up.y) < 0.9 else Vec3(1.0, 0.0, 0.0)
            return (arb - up * arb.dot(up)).normalized()
        return v_tang.normalized()

    @staticmethod
    def _lat_cell(pos: Vec3) -> int:
        """Coarse latitude cell for RNG spatial variation."""
        up = pos.normalized()
        lat = math.asin(_clamp(up.y, -1.0, 1.0))
        return int(lat * _LAT_CELL_DIVISIONS) & 0xFF

    @staticmethod
    def _lon_cell(pos: Vec3) -> int:
        """Coarse longitude cell for RNG spatial variation."""
        up  = pos.normalized()
        lon = math.atan2(up.x, up.z)
        return int(lon * _LON_CELL_DIVISIONS) & 0xFF

    def _compute_bone_world_positions(
        self, root_world: Vec3, up: Vec3, fwd: Vec3, right: Vec3,
    ) -> Dict[str, Vec3]:
        """Compute approximate world positions of all bones for debug drawing.

        Uses a simplified FK (translation only, no rotations) sufficient
        for gizmo / skeleton-line rendering.
        """
        # Build a local→world function for a bind-space position
        def to_world(local: Vec3) -> Vec3:
            return root_world + right * local.x + up * local.y + fwd * local.z

        # Accumulate bind positions up the hierarchy
        bind_acc: Dict[str, Vec3] = {"root": Vec3.zero()}
        for b in BONE_NAMES:
            if b == "root":
                continue
            parent     = BONE_PARENT[b]
            parent_pos = bind_acc.get(parent, Vec3.zero())
            bind_acc[b] = parent_pos + BIND_LOCAL_POS[b]

        return {b: to_world(p) for b, p in bind_acc.items()}
