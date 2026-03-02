"""ReflexOverlaySystem — Stage 43 perception-driven reflex biases.

Converts a :class:`~src.perception.PerceptionSystem.PerceptionState` into a
short-lived :class:`~src.input.PlayerIntent.ReflexOverlay` that is blended
into the player's primary input — but **never replaces it**.

Architecture
------------
::

    PerceptionState  ──►  ReflexOverlaySystem  ──►  ReflexOverlay
                                                          ↓
                                                    PlayerIntent.reflex

The system maintains one weight ``r`` per reflex type.  Each tick:

1. Strong stimuli *push* ``r`` toward ``r_max`` (with hysteresis to avoid
   jitter on repeated small stimuli).
2. ``r`` decays exponentially with time constant ``decay_tau_sec``.
3. Primary targets always dominate because ``r`` is bounded and temporary.

Stimulus sources (from PerceptionState)
-----------------------------------------
* ``audioUrgency``        → micro-look reflex
* ``vibrationLevel``      → brace + slowdown reflex
* ``presenceNear``        → micro-look toward presence dir
* ``globalRisk``          → brace + slowdown reflex (high risk)

Public API
----------
ReflexOverlaySystem(config=None)
  .update(dt, perception_state) → ReflexOverlay
  .current_overlay  → ReflexOverlay
  .debug_info()     → dict
"""
from __future__ import annotations

import math
from typing import Optional

from src.math.Vec3 import Vec3
from src.input.PlayerIntent import ReflexOverlay
from src.perception.PerceptionSystem import PerceptionState


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


def _normalize(v: Vec3) -> Vec3:
    l = v.length()
    return v * (1.0 / l) if l > 1e-9 else Vec3.zero()


class ReflexOverlaySystem:
    """Converts PerceptionState into a ReflexOverlay.

    Parameters
    ----------
    config :
        Optional dict; reads ``reflex.*`` sub-keys.
    """

    _DEFAULT_R_MAX_DEFAULT  = 0.15   # normal r ceiling
    _DEFAULT_R_MAX_CRITICAL = 0.35   # ceiling for very strong stimuli
    _DEFAULT_DECAY_TAU      = 0.4    # seconds; r decays to 1/e in this time
    _DEFAULT_BRACE_GAIN     = 0.6    # vibration → brace conversion
    _DEFAULT_SLOWDOWN_GAIN  = 0.4    # risk → slowdown conversion
    _DEFAULT_HYSTERESIS     = 0.05   # minimum Δ to update r upward

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg  = config or {}
        rcfg = cfg.get("reflex", {}) or {}

        self._r_max_default  = float(rcfg.get("r_max_default",  self._DEFAULT_R_MAX_DEFAULT))
        self._r_max_critical = float(rcfg.get("r_max_critical", self._DEFAULT_R_MAX_CRITICAL))
        self._decay_tau      = float(rcfg.get("decay_tau_sec",  self._DEFAULT_DECAY_TAU))
        self._brace_gain     = float(rcfg.get("brace_gain",     self._DEFAULT_BRACE_GAIN))
        self._slowdown_gain  = float(rcfg.get("slowdown_gain",  self._DEFAULT_SLOWDOWN_GAIN))
        self._hysteresis     = float(rcfg.get("hysteresis",     self._DEFAULT_HYSTERESIS))

        # Current running state
        self._r_look:     float = 0.0
        self._r_brace:    float = 0.0
        self._r_slow:     float = 0.0
        self._look_dir:   Vec3  = Vec3.zero()

        self._current_overlay = ReflexOverlay()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    @property
    def current_overlay(self) -> ReflexOverlay:
        """Most recently computed ReflexOverlay."""
        return self._current_overlay

    def update(
        self,
        dt:               float,
        perception_state: PerceptionState,
    ) -> ReflexOverlay:
        """Compute and return a ReflexOverlay for this tick.

        Parameters
        ----------
        dt :
            Elapsed time since last call [s].
        perception_state :
            Latest PerceptionState from PerceptionSystem.
        """
        p = perception_state

        # --- Decay all r values ---
        if self._decay_tau > 1e-9:
            decay = math.exp(-dt / self._decay_tau)
        else:
            decay = 0.0

        self._r_look  *= decay
        self._r_brace *= decay
        self._r_slow  *= decay

        # --- Stimulus: audio urgency → micro-look ---
        audio_stim = _clamp(p.audioUrgency, 0.0, 1.0)
        if audio_stim > 0.3:
            r_max = (self._r_max_critical if audio_stim > 0.75
                     else self._r_max_default)
            target_r = audio_stim * r_max
            if target_r > self._r_look + self._hysteresis:
                self._r_look = _clamp(target_r, 0.0, r_max)
            # Attention dir from audio
            if p.attentionDir.length() > 1e-9:
                self._look_dir = _normalize(p.attentionDir)

        # --- Stimulus: presence near → micro-look toward presence ---
        if p.presenceNear > 0.5 and p.attentionDir.length() > 1e-9:
            pres_r = p.presenceNear * self._r_max_default
            if pres_r > self._r_look + self._hysteresis:
                self._r_look  = _clamp(pres_r, 0.0, self._r_max_default)
                self._look_dir = _normalize(p.attentionDir)

        # --- Stimulus: vibration → brace ---
        vibr_stim = _clamp(p.vibrationLevel, 0.0, 1.0)
        if vibr_stim > 0.2:
            brace_target = vibr_stim * self._brace_gain
            if brace_target > self._r_brace + self._hysteresis:
                self._r_brace = _clamp(brace_target, 0.0, 1.0)

        # --- Stimulus: global risk → brace + slowdown ---
        risk = _clamp(p.globalRisk, 0.0, 1.0)
        if risk > 0.4:
            slow_target = risk * self._slowdown_gain
            if slow_target > self._r_slow + self._hysteresis:
                self._r_slow = _clamp(slow_target, 0.0, 1.0)
            brace_from_risk = risk * self._brace_gain * 0.5
            if brace_from_risk > self._r_brace + self._hysteresis:
                self._r_brace = _clamp(brace_from_risk, 0.0, 1.0)

        # --- Build overlay ---
        overlay = ReflexOverlay(
            lookBias_dir      = self._look_dir if self._r_look > 1e-9 else Vec3.zero(),
            lookBias_strength = self._r_look,
            braceBias         = self._r_brace,
            slowdownBias      = self._r_slow,
        )
        self._current_overlay = overlay
        return overlay

    def debug_info(self) -> dict:
        """Current reflex state snapshot for logging."""
        return {
            "r_look":       self._r_look,
            "r_brace":      self._r_brace,
            "r_slow":       self._r_slow,
            "look_dir":     (self._look_dir.x, self._look_dir.y, self._look_dir.z),
            "braceBias":    self._current_overlay.braceBias,
            "slowdownBias": self._current_overlay.slowdownBias,
        }
