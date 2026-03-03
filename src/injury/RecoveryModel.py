"""RecoveryModel — Stage 48 injury recovery rate computation.

Determines the per-tick recovery rate modifier for the injury integrator
based on the character's current activity and environment.

Recovery is fastest when:
* the character is standing still or walking slowly,
* wind load is low,
* ground support is stable,
* the character is not grasping another body.

Public API
----------
RecoveryCondition (dataclass)
RecoveryModel(config=None)
  .evaluate(env) → RecoveryCondition
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.injury.InjurySystem import InjuryEnvInput


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


# ---------------------------------------------------------------------------
# RecoveryCondition
# ---------------------------------------------------------------------------

@dataclass
class RecoveryCondition:
    """Output of :class:`RecoveryModel`.

    Attributes
    ----------
    is_resting :
        True when conditions allow meaningful injury recovery.
    recovery_multiplier :
        Rate multiplier applied to ``recover_k`` in the injury integrator
        [0..1].  1.0 = full recovery rate; 0.0 = no recovery.
    """
    is_resting:          bool  = False
    recovery_multiplier: float = 0.0


# ---------------------------------------------------------------------------
# RecoveryModel
# ---------------------------------------------------------------------------

class RecoveryModel:
    """Evaluates environmental conditions and returns a recovery modifier.

    Parameters
    ----------
    config :
        Optional dict; reads ``injury.*`` keys.
    """

    _DEFAULT_REST_SPEED_MAX   = 0.5
    _DEFAULT_REST_WIND_MAX    = 0.25
    _DEFAULT_REST_SUPPORT_MIN = 0.65

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg  = config or {}
        icfg = cfg.get("injury", {}) or {}

        rest = icfg.get("rest_condition_thresholds", {}) or {}
        self._speed_max   = float(rest.get("speed",   self._DEFAULT_REST_SPEED_MAX))
        self._wind_max    = float(rest.get("wind",    self._DEFAULT_REST_WIND_MAX))
        self._support_min = float(rest.get("support", self._DEFAULT_REST_SUPPORT_MIN))

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def evaluate(self, env: InjuryEnvInput) -> RecoveryCondition:
        """Evaluate recovery conditions.

        Parameters
        ----------
        env :
            Current environmental and motion state.

        Returns
        -------
        RecoveryCondition
        """
        speed_ok   = env.speed          <= self._speed_max
        wind_ok    = env.windLoad       <= self._wind_max
        support_ok = env.supportQuality >= self._support_min
        no_grasp   = not env.isGrasping

        is_resting = speed_ok and wind_ok and support_ok and no_grasp
        if not is_resting:
            return RecoveryCondition(is_resting=False, recovery_multiplier=0.0)

        # Partial multiplier: better support and calmer wind → faster recovery
        speed_factor   = 1.0 - _clamp(env.speed / max(self._speed_max, 1e-6), 0.0, 1.0)
        support_factor = _clamp(
            (env.supportQuality - self._support_min) / (1.0 - self._support_min + 1e-6),
            0.0, 1.0,
        )
        wind_factor  = 1.0 - _clamp(env.windLoad / max(self._wind_max, 1e-6), 0.0, 1.0)
        multiplier   = _clamp(
            (speed_factor + support_factor + wind_factor) / 3.0,
            0.1, 1.0,
        )
        return RecoveryCondition(is_resting=True, recovery_multiplier=multiplier)
