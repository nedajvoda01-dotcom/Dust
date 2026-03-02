"""RecoveryDetector — Stage 44 safe-condition detector for fatigue recovery.

Decides, per-tick, whether the player is in conditions conducive to
physiological recovery.  This is a pure function of environmental inputs;
no internal state is required beyond what is passed in.

Public API
----------
RecoveryConditions (dataclass)
RecoveryDetector(config=None)
  .evaluate(env) → RecoveryConditions
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.fatigue.FatigueSystem import EnvInput


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


@dataclass
class RecoveryConditions:
    """Output of :class:`RecoveryDetector`.

    Attributes
    ----------
    is_resting :
        True when all thresholds are satisfied (full recovery mode).
    recovery_rate :
        Normalised recovery rate multiplier [0..1].  1.0 = optimal rest,
        0.0 = no recovery possible.
    """
    is_resting:    bool  = False
    recovery_rate: float = 0.0


class RecoveryDetector:
    """Evaluates whether the current environment permits fatigue recovery.

    Parameters
    ----------
    config :
        Optional dict; reads ``fatigue.rest_condition_thresholds`` sub-keys.
    """

    _DEFAULT_REST_WIND_MAX    = 0.2
    _DEFAULT_REST_SLOPE_MAX   = 5.0    # degrees
    _DEFAULT_REST_SUPPORT_MIN = 0.7
    _DEFAULT_REST_SPEED_MAX   = 0.3    # m/s

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg   = config or {}
        fcfg  = cfg.get("fatigue", {}) or {}
        rest  = fcfg.get("rest_condition_thresholds", {}) or {}

        self._wind_max     = float(rest.get("wind",    self._DEFAULT_REST_WIND_MAX))
        self._slope_max    = float(rest.get("slope",   self._DEFAULT_REST_SLOPE_MAX))
        self._support_min  = float(rest.get("support", self._DEFAULT_REST_SUPPORT_MIN))
        self._speed_max    = float(rest.get("speed",   self._DEFAULT_REST_SPEED_MAX))

    def evaluate(self, env: EnvInput) -> RecoveryConditions:
        """Evaluate recovery conditions from the current environment.

        Parameters
        ----------
        env :
            Current environmental snapshot.

        Returns
        -------
        RecoveryConditions
        """
        wind_ok    = env.windLoad       <= self._wind_max
        slope_ok   = abs(env.slopeDeg)  <= self._slope_max
        support_ok = env.supportQuality >= self._support_min
        speed_ok   = env.speed          <= self._speed_max

        is_resting = wind_ok and slope_ok and support_ok and speed_ok

        # Graded recovery rate: each condition contributes proportionally
        rate_wind    = _clamp(1.0 - env.windLoad / max(self._wind_max, 1e-6), 0.0, 1.0)
        rate_slope   = _clamp(1.0 - abs(env.slopeDeg) / max(self._slope_max, 1e-6), 0.0, 1.0)
        rate_support = _clamp((env.supportQuality - self._support_min) / max(1.0 - self._support_min, 1e-6), 0.0, 1.0)
        rate_speed   = _clamp(1.0 - env.speed / max(self._speed_max, 1e-6), 0.0, 1.0)

        recovery_rate = rate_wind * rate_slope * rate_support * rate_speed

        return RecoveryConditions(
            is_resting=is_resting,
            recovery_rate=recovery_rate,
        )
