"""CollisionYield — Stage 39 reciprocal collision-avoidance yield (§9).

Predicts whether two agents' trajectories will intersect within a
configurable horizon and returns a *yield bias* scalar for the agent
that should give way.

The agent with higher ``yield_bias`` yields (slows / sidesteps); the agent
in greater danger (higher ``self_risk``) is protected — a riskier mover
yields less because it cannot afford to manoeuvre.

Public API
----------
CollisionYield(config=None)
  .compute(self_pos, self_vel, other_pos, other_vel,
           self_risk, other_risk, caution) → (yield_bias, avoid_dir)

  yield_bias   float [0..1]  — how much *this* agent should yield
  avoid_dir    Vec3          — suggested lateral offset direction
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

from src.math.Vec3 import Vec3


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


class CollisionYield:
    """Reciprocal trajectory prediction and yield-bias computation.

    Parameters
    ----------
    config :
        Optional dict; reads ``social.yield_prediction_horizon_sec``.
    """

    _DEFAULT_HORIZON = 1.5   # seconds
    _DEFAULT_MIN_DIST = 1.0  # metres — minimum safe separation

    def __init__(self, config: Optional[dict] = None) -> None:
        scfg = ((config or {}).get("social", {})) or {}
        self._horizon:  float = float(
            scfg.get("yield_prediction_horizon_sec", self._DEFAULT_HORIZON)
        )
        self._min_dist: float = float(
            scfg.get("collision_min_dist_m", self._DEFAULT_MIN_DIST)
        )

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def compute(
        self,
        self_pos:   Vec3,
        self_vel:   Vec3,
        other_pos:  Vec3,
        other_vel:  Vec3,
        self_risk:  float = 0.0,
        other_risk: float = 0.0,
        caution:    float = 0.5,
    ) -> Tuple[float, Vec3]:
        """Compute yield bias and avoidance direction for *this* agent.

        Parameters
        ----------
        self_pos, self_vel :
            Own world position and current velocity.
        other_pos, other_vel :
            Remote agent's world position and velocity.
        self_risk :
            Own global_risk [0..1]; higher → yields less (cannot manoeuvre).
        other_risk :
            Remote agent's global_risk [0..1].
        caution :
            Personality caution scalar [0..1]; cautious agents yield more.

        Returns
        -------
        (yield_bias, avoid_dir)
            ``yield_bias`` in [0..1]; ``avoid_dir`` is a unit Vec3 (or zero).
        """
        # --- Closest-point-of-approach in [0, horizon] ---
        rel_pos = other_pos - self_pos
        rel_vel = other_vel - self_vel

        rv_sq = rel_vel.length_sq()
        if rv_sq < 1e-8:
            # Essentially same velocity — check current separation
            dist = rel_pos.length()
            if dist < self._min_dist:
                return self._yield_response(rel_pos, caution, self_risk)
            return 0.0, Vec3.zero()

        # Time of closest approach: t* = -dot(rel_pos, rel_vel) / |rel_vel|²
        t_star = -rel_pos.dot(rel_vel) / rv_sq
        t_star = _clamp(t_star, 0.0, self._horizon)

        # Position at closest approach
        close_rel = rel_pos + rel_vel * t_star
        close_dist = close_rel.length()

        if close_dist >= self._min_dist:
            return 0.0, Vec3.zero()

        # Overlap fraction: how much inside the safe radius?
        overlap = _clamp(1.0 - close_dist / self._min_dist, 0.0, 1.0)
        urgency = overlap * _clamp(1.0 - t_star / max(self._horizon, 1e-6), 0.0, 1.0)

        # When agents collide exactly (close_rel ≈ zero), fall back to the
        # current separation vector as the lateral reference direction.
        lateral_ref = close_rel if close_rel.length() > 1e-6 else rel_pos
        return self._yield_response(lateral_ref, caution, self_risk, urgency)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _yield_response(
        self,
        lateral_ref: Vec3,
        caution:     float,
        self_risk:   float,
        urgency:     float = 1.0,
    ) -> Tuple[float, Vec3]:
        """Compute yield bias and avoidance direction."""
        # Riskier agents yield less — they cannot afford to manoeuvre
        safety_headroom = _clamp(1.0 - self_risk, 0.0, 1.0)
        bias = _clamp(urgency * (0.5 + caution * 0.5) * safety_headroom, 0.0, 1.0)

        # Avoid direction: perpendicular to lateral_ref in the XZ plane
        ref_len = lateral_ref.length()
        if ref_len < 1e-6:
            return bias, Vec3.zero()

        ref_unit = lateral_ref * (1.0 / ref_len)
        # Rotate 90° in XZ: (-z, 0, x)
        avoid = Vec3(-ref_unit.z, 0.0, ref_unit.x)
        al = avoid.length()
        if al < 1e-6:
            return bias, Vec3.zero()
        return bias, avoid * (1.0 / al)
