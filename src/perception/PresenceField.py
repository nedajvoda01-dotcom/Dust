"""PresenceField — Stage 37 other-body presence perception field.

Aggregates nearby players into:

* ``presenceNear``        (0..1) — how close / strongly felt another body is
* ``presenceDir``         (Vec3, unit) — direction toward nearest/strongest presence
* ``assistOpportunity``   (0..1) — chance to help without self-risk

Inputs:
* List of ``OtherPlayerState`` (replicated motor data from PerceptionNetInputs)

Public API
----------
PresenceField(config=None)
  .update(listener_pos, others, self_global_risk, dt) → None
  .presence_near        → float
  .presence_dir         → Vec3
  .assist_opportunity   → float
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

from src.math.Vec3 import Vec3


@dataclass
class OtherPlayerState:
    """Replicated state of a remote player relevant for perception.

    Attributes
    ----------
    position :
        World-space position.
    velocity :
        Current velocity vector.
    is_slipping :
        True when the remote player is in a slipping / stumbling state.
    """
    position:    Vec3
    velocity:    Vec3     = field(default_factory=lambda: Vec3(0.0, 0.0, 0.0))
    is_slipping: bool     = False


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


class PresenceField:
    """Perception sub-field: other-player proximity and assist opportunity.

    Parameters
    ----------
    config :
        Optional dict; reads ``perception.presence.*`` keys.
    """

    _DEFAULT_PRESENCE_RADIUS        = 20.0  # metres
    _DEFAULT_ASSIST_RADIUS          = 8.0   # metres — close enough to help
    _DEFAULT_SMOOTHING_TAU          = 0.15  # seconds

    def __init__(self, config: Optional[dict] = None) -> None:
        pcfg = ((config or {}).get("perception", {}) or {}).get("presence", {}) or {}
        self._radius: float = float(pcfg.get("radius", self._DEFAULT_PRESENCE_RADIUS))
        acfg = ((config or {}).get("perception", {}) or {}).get("assist", {}) or {}
        self._assist_radius: float = float(
            acfg.get("opportunity_radius", self._DEFAULT_ASSIST_RADIUS)
        )
        tau = float(
            ((config or {}).get("perception", {}) or {}).get(
                "smoothing_tau_sec", self._DEFAULT_SMOOTHING_TAU
            )
        )
        self._tau: float = max(1e-3, tau)

        self._presence:  float = 0.0
        self._assist:    float = 0.0
        self._dir:       Vec3  = Vec3(0.0, 0.0, 0.0)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def update(
        self,
        listener_pos:     Vec3,
        others:           List[OtherPlayerState],
        self_global_risk: float = 0.0,
        dt:               float = 1.0 / 20.0,
    ) -> None:
        """Advance presence field one tick.

        Parameters
        ----------
        listener_pos :
            Character world position.
        others :
            Replicated states of other players.
        self_global_risk :
            Own global risk scalar from ThreatAggregator; reduces
            ``assistOpportunity`` when self is in danger.
        dt :
            Elapsed simulation time [s].
        """
        total_weight = 0.0
        dir_acc      = Vec3(0.0, 0.0, 0.0)
        raw_assist   = 0.0

        for other in others:
            diff = other.position - listener_pos
            dist = diff.length()
            if dist > self._radius or dist < 1e-6:
                continue

            # Presence fades with distance
            w = (1.0 - dist / self._radius) ** 2
            total_weight += w

            unit = diff * (1.0 / dist)
            dir_acc = dir_acc + unit * w

            # Assist opportunity: other is in distress, within help radius,
            # and self is not in too much danger
            if other.is_slipping and dist <= self._assist_radius:
                assist_w = w * _clamp(1.0 - self_global_risk, 0.0, 1.0)
                raw_assist = max(raw_assist, assist_w)

        raw_presence = _clamp(total_weight, 0.0, 1.0)
        dir_len = dir_acc.length()
        raw_dir = dir_acc * (1.0 / dir_len) if dir_len > 1e-6 else Vec3(0.0, 0.0, 0.0)

        # Exponential smoothing
        alpha = 1.0 - math.exp(-dt / self._tau)
        self._presence = self._presence + alpha * (raw_presence - self._presence)
        self._assist   = self._assist   + alpha * (raw_assist   - self._assist)

        prev_len = self._dir.length()
        if prev_len < 1e-6:
            self._dir = raw_dir
        else:
            blended = self._dir * (1.0 - alpha) + raw_dir * alpha
            bl = blended.length()
            self._dir = blended * (1.0 / bl) if bl > 1e-6 else raw_dir

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------

    @property
    def presence_near(self) -> float:
        """Proximity of nearest other player [0..1]."""
        return self._presence

    @property
    def presence_dir(self) -> Vec3:
        """Unit vector pointing toward dominant presence."""
        return self._dir

    @property
    def assist_opportunity(self) -> float:
        """Opportunity to assist another player safely [0..1]."""
        return self._assist
