"""ThreatAggregator — Stage 37 Threat & Salience Map.

Combines all perception sub-fields into the five key scalars that
MotorStack consumes:

* ``globalRisk``          (0..1)
* ``attentionDir``        (Vec3, unit) — where to orient head/torso
* ``movementConfidence``  (0..1)
* ``braceBias``           (0..1)
* ``assistBias``          (0..1)

Public API
----------
ThreatAggregator()
  .update(slip_risk, sink_risk, vibration_level, wind_load, visibility,
          audio_salience, audio_urgency, audio_dir, vibration_dir,
          presence_near, assist_opportunity, presence_dir) → None
  .global_risk          → float
  .attention_dir        → Vec3
  .movement_confidence  → float
  .brace_bias           → float
  .assist_bias          → float
"""
from __future__ import annotations

import math
from typing import Optional

from src.math.Vec3 import Vec3


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


class ThreatAggregator:
    """Aggregates perception sub-field outputs into MotorStack signals.

    No configuration required; all weights are based on the design spec.
    """

    def __init__(self) -> None:
        self._global_risk:         float = 0.0
        self._movement_confidence: float = 1.0
        self._brace_bias:          float = 0.0
        self._assist_bias:         float = 0.0
        self._attention_dir:       Vec3  = Vec3(0.0, 0.0, 0.0)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def update(
        self,
        slip_risk:         float = 0.0,
        sink_risk:         float = 0.0,
        vibration_level:   float = 0.0,
        wind_load:         float = 0.0,
        visibility:        float = 1.0,
        audio_salience:    float = 0.0,
        audio_urgency:     float = 0.0,
        audio_dir:         Optional[Vec3] = None,
        vibration_dir:     Optional[Vec3] = None,
        presence_near:     float = 0.0,
        assist_opportunity: float = 0.0,
        presence_dir:      Optional[Vec3] = None,
    ) -> None:
        """Compute aggregated threat and salience signals.

        Parameters match the outputs of the individual perception sub-fields.
        All scalars are 0..1 (visibility should be passed as-is: 1 = clear).
        """
        # --- globalRisk  (§4.1) -----------------------------------------
        threat_vis = 1.0 - _clamp(visibility, 0.0, 1.0)
        raw_risk = max(
            _clamp(slip_risk,       0.0, 1.0),
            _clamp(vibration_level, 0.0, 1.0),
            _clamp(wind_load,       0.0, 1.0),
            threat_vis,
            _clamp(sink_risk,       0.0, 1.0),
        )
        self._global_risk = _clamp(raw_risk, 0.0, 1.0)

        # --- movementConfidence  (§4.1) -----------------------------------
        # Falls when visibility is poor, wind is high, or footing is bad
        self._movement_confidence = _clamp(
            visibility
            * (1.0 - wind_load  * 0.6)
            * (1.0 - slip_risk  * 0.7),
            0.0, 1.0,
        )

        # --- braceBias  ---------------------------------------------------
        # High when wind, vibration, slip all combine
        self._brace_bias = _clamp(
            wind_load * 0.5 + vibration_level * 0.3 + slip_risk * 0.3,
            0.0, 1.0,
        )

        # --- assistBias  --------------------------------------------------
        self._assist_bias = _clamp(assist_opportunity, 0.0, 1.0)

        # --- attentionDir  ------------------------------------------------
        # Weighted sum of directional cues; highest saliency wins
        if audio_dir is None:
            audio_dir = Vec3(0.0, 0.0, 0.0)
        if vibration_dir is None:
            vibration_dir = Vec3(0.0, 0.0, 0.0)
        if presence_dir is None:
            presence_dir = Vec3(0.0, 0.0, 0.0)

        w_audio   = audio_salience + audio_urgency * 0.5
        w_vibr    = vibration_level
        w_pres    = presence_near

        attn = (
            audio_dir     * w_audio
            + vibration_dir * w_vibr
            + presence_dir  * w_pres
        )
        attn_len = attn.length()
        if attn_len > 1e-6:
            self._attention_dir = attn * (1.0 / attn_len)
        # else keep previous direction

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------

    @property
    def global_risk(self) -> float:
        """Overall threat level [0..1]."""
        return self._global_risk

    @property
    def attention_dir(self) -> Vec3:
        """Unit vector indicating where character should attend."""
        return self._attention_dir

    @property
    def movement_confidence(self) -> float:
        """Confidence in executing movement intent [0..1]."""
        return self._movement_confidence

    @property
    def brace_bias(self) -> float:
        """Tendency to adopt a wide/braced stance [0..1]."""
        return self._brace_bias

    @property
    def assist_bias(self) -> float:
        """Tendency to assist a nearby player [0..1]; no action yet (§14)."""
        return self._assist_bias
