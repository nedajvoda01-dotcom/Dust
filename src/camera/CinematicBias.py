"""CinematicBias — Stage 41 camera bias computation (sections 6–10).

Aggregates game-state signals into a :class:`CameraIntent` each tick.
No smoothing happens here; smoothing is the responsibility of
:class:`~src.camera.StabilityCameraController.StabilityCameraController`.

Bias sources applied in order:
    1. Stability (section 6): balance margin, slip risk, global risk
    2. Wind (section 7): wind_load → pressure sway + height reduction
    3. Social / Grasp (section 8): parallel walk, grasp constraint
    4. Macro events (section 9): rift, dust wall via MacroEventLens
    5. Attention (section 10): attentionDir look-aside (max 10–15°)

Public API
----------
StabilityInput (dataclass)
CinematicBias
  .compute(inp, config) → CameraIntent
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from src.math.Vec3 import Vec3
from src.systems.CharacterPhysicalController import CharacterState
from src.camera.CameraConfig import CameraConfig
from src.camera.CameraIntent import CameraIntent
from src.camera.MacroEventLens import MacroEventLens, MacroEventInput


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


# ---------------------------------------------------------------------------
# StabilityInput
# ---------------------------------------------------------------------------

@dataclass
class StabilityInput:
    """Combined input for CinematicBias.compute (section 2).

    Aggregates data from MotorState (stage 34), PerceptionState (stage 37),
    Micro-Intent mode (stage 38), Social/Grasp state (stages 39–40), and
    macro-event proximity (stages 29–33).

    Camera state function (section 2)::

        CameraState = f(
            balanceRisk, slipState, globalRisk, attentionDir,
            graspState, macroEventProximity, velocity, slope
        )
    """

    # --- Character position / motion ---
    position: Vec3
    velocity: Vec3
    up: Vec3                          # planet up at character position
    char_state: CharacterState = CharacterState.GROUNDED

    # --- Balance / stability (stage 34 MotorCore: COM, margin, slip) ---
    balance_margin: float = 1.0       # 0 = near fall, 1 = fully stable
    slip_risk: float = 0.0            # 0–1

    # --- Perception (stage 37: globalRisk, attentionDir, vibration) ---
    global_risk: float = 0.0          # 0–1
    attention_dir: Vec3 = field(default_factory=Vec3.zero)
    wind_load: float = 0.0            # 0–1
    vibration_level: float = 0.0      # 0–1

    # --- Social / Grasp (stages 39–40: cooperative events) ---
    grasp_active: bool = False
    grasp_point: Vec3 = field(default_factory=Vec3.zero)
    social_parallel_walk: bool = False  # stage 39 SocialMode.ParallelWalk
    social_assist_prep: bool = False    # stage 39 SocialMode.AssistPreparation
    other_position: Vec3 = field(default_factory=Vec3.zero)

    # --- Macro events (stages 29–33: fronts, rifts, scale) ---
    macro_proximity: float = 0.0       # 0 = distant, 1 = nearby
    macro_epicenter: Vec3 = field(default_factory=Vec3.zero)
    macro_is_rift: bool = False
    macro_is_dust_wall: bool = False

    # --- Landing impulse for one-frame shake spike ---
    landing_impulse: float = 0.0       # 0–1


# ---------------------------------------------------------------------------
# CinematicBias
# ---------------------------------------------------------------------------

class CinematicBias:
    """Computes CameraIntent from the full game state (sections 6–10).

    Each bias source contributes additively.  The resulting intent is a
    *request*, not a final camera state; the controller applies spring
    smoothing and enforces cinematic constraints (section 11).
    """

    def __init__(self) -> None:
        self._macro_lens = MacroEventLens()

    def compute(self, inp: StabilityInput, cfg: CameraConfig) -> CameraIntent:
        """Return a blended CameraIntent for the current frame."""
        intent = CameraIntent()

        self._apply_stability_bias(inp, cfg, intent)
        self._apply_wind_bias(inp, cfg, intent)
        self._apply_social_bias(inp, cfg, intent)
        self._apply_macro_bias(inp, cfg, intent)
        self._apply_attention_bias(inp, cfg, intent)

        return intent

    # ------------------------------------------------------------------
    # Section 6 — Stability
    # ------------------------------------------------------------------

    def _apply_stability_bias(
        self, inp: StabilityInput, cfg: CameraConfig, intent: CameraIntent
    ) -> None:
        low_balance = _clamp(1.0 - inp.balance_margin, 0.0, 1.0)  # 0=stable, 1=near-fall

        is_falling = inp.char_state in (
            CharacterState.FALLING_CONTROLLED,
            CharacterState.SLIDING,
            CharacterState.STUMBLING,
            CharacterState.AIRBORNE,
        )

        near_fall_strength = _clamp(max(low_balance, inp.slip_risk), 0.0, 1.0)
        is_near_fall = near_fall_strength > 0.4

        if is_falling:
            # Section 6.2: actual fall — roll + shake + lower
            fall_strength = _clamp(inp.slip_risk * 0.6 + low_balance * 0.4, 0.0, 1.0)
            intent.roll_bias += fall_strength * cfg.fall_roll_bias_deg
            intent.height_bias -= fall_strength * 0.25
            intent.shake_level = _clamp(intent.shake_level + fall_strength * 0.4, 0.0, 1.0)
        elif is_near_fall:
            # Section 6.1: almost falling — closer, lower, wider FOV
            intent.distance_scale = min(
                intent.distance_scale, 1.0 - near_fall_strength * 0.25
            )
            intent.height_bias -= near_fall_strength * 0.3
            intent.fov_bias += near_fall_strength * 4.0  # +2–5°
            # Focus on fall / slip direction (section 6.1)
            if inp.velocity.length() > 0.05:
                intent.focus_dir = inp.velocity.normalized()

        # High global_risk → pull back (section 6, balanceRisk)
        if inp.global_risk > 0.3:
            risk_excess = (inp.global_risk - 0.3) / 0.7
            intent.distance_scale = max(
                intent.distance_scale,
                1.0 + risk_excess * (cfg.risk_distance_scale - 1.0) * 0.5,
            )

    # ------------------------------------------------------------------
    # Section 7 — Wind
    # ------------------------------------------------------------------

    def _apply_wind_bias(
        self, inp: StabilityInput, cfg: CameraConfig, intent: CameraIntent
    ) -> None:
        if inp.wind_load < 0.1:
            return
        w = (inp.wind_load - 0.1) / 0.9  # normalise above threshold
        # Pressure sway: camera shifts slightly along wind direction (section 7)
        intent.lateral_sway += w * 0.3
        # Slight height reduction ("прижатость" / pressed-down feel)
        intent.height_bias -= w * 0.2
        intent.shake_level = _clamp(intent.shake_level + w * 0.15, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Section 8 — Social / Grasp
    # ------------------------------------------------------------------

    def _apply_social_bias(
        self, inp: StabilityInput, cfg: CameraConfig, intent: CameraIntent
    ) -> None:
        if inp.grasp_active:
            # Section 8.2: grasp active — closer + higher, centre on constraint
            intent.distance_scale = min(intent.distance_scale, 0.7)
            intent.height_bias += 0.4
            to_grasp = inp.grasp_point - inp.position
            if not to_grasp.is_near_zero():
                intent.focus_dir = to_grasp.normalized()
        elif inp.social_parallel_walk or inp.social_assist_prep:
            # Section 8.1: two players nearby — widen to frame both
            intent.distance_scale = max(intent.distance_scale, 1.3)
            if not inp.other_position.is_near_zero():
                to_other = inp.other_position - inp.position
                if not to_other.is_near_zero():
                    # Focus between self and other (0.5 = midpoint bias)
                    intent.focus_dir = to_other.normalized() * 0.5

    # ------------------------------------------------------------------
    # Section 9 — Macro events
    # ------------------------------------------------------------------

    def _apply_macro_bias(
        self, inp: StabilityInput, cfg: CameraConfig, intent: CameraIntent
    ) -> None:
        if inp.macro_proximity <= 0.0:
            return
        macro_input = MacroEventInput(
            proximity=inp.macro_proximity,
            epicenter=inp.macro_epicenter,
            is_rift=inp.macro_is_rift,
            is_dust_wall=inp.macro_is_dust_wall,
        )
        m = self._macro_lens.compute(macro_input)
        # Accumulate: distance uses max (pullback dominates), others add
        intent.distance_scale = max(intent.distance_scale, m.distance_scale)
        intent.height_bias += m.height_bias
        intent.fov_bias += m.fov_bias
        intent.tilt_bias += m.tilt_bias

    # ------------------------------------------------------------------
    # Section 10 — Attention
    # ------------------------------------------------------------------

    def _apply_attention_bias(
        self, inp: StabilityInput, cfg: CameraConfig, intent: CameraIntent
    ) -> None:
        if inp.attention_dir.is_near_zero():
            return
        # Shift look-at subtly toward saliency source (max 10–15°)
        attn_strength = _clamp(inp.global_risk * 0.5 + 0.5, 0.0, 1.0)
        intent.attention_offset_deg = attn_strength * cfg.attention_max_deg * 0.5
        if intent.focus_dir.is_near_zero():
            intent.focus_dir = inp.attention_dir
