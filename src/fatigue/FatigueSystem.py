"""FatigueSystem — Stage 44 Embodied Fatigue & Recovery integrator.

Maintains :class:`FatigueState` which represents the player's physiological
state as a set of internal motor regulators.  There is **no UI**: fatigue
is expressed entirely through changes to motor parameters.

Design
------
* A deterministic tick (2–10 Hz) integrates work inputs into ``energy``.
* Derived fields (``neuromuscularNoise``, ``coordination``, ``tremor``,
  ``gripReserve``, ``thermalLoad``) are computed from ``energy`` and
  recent environmental load.
* Recovery happens automatically under safe conditions (low speed, low
  wind, good support, flat surface).

Public API
----------
FatigueState (dataclass)
FatigueSystem(config=None)
  .tick(dt, work_input, env_input) → FatigueState
  .state → FatigueState
  .force_set(energy)              — dev / test helper
  .debug_info()                   → dict
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * _clamp(t, 0.0, 1.0)


# ---------------------------------------------------------------------------
# FatigueState — internal motor regulator
# ---------------------------------------------------------------------------

@dataclass
class FatigueState:
    """Internal motor-regulation state.  All fields are in [0..1].

    Attributes
    ----------
    energy :
        Physiological energy reserve (1 = fully rested, 0 = exhausted).
    neuromuscularNoise :
        Signal noise in motor pathways (0 = crisp, 1 = very noisy).
    coordination :
        Motor coordination quality (1 = ideal, 0 = severely degraded).
    tremor :
        Involuntary micro-oscillation level (0 = none, 1 = severe).
    gripReserve :
        Grip strength reserve (1 = full, 0 = near-failure).
    thermalLoad :
        Combined thermal/cold stress (0 = comfortable, 1 = extreme).
    """
    energy:              float = 1.0
    neuromuscularNoise:  float = 0.0
    coordination:        float = 1.0
    tremor:              float = 0.0
    gripReserve:         float = 1.0
    thermalLoad:         float = 0.0


# ---------------------------------------------------------------------------
# WorkInput — mechanical load for one fatigue tick
# ---------------------------------------------------------------------------

@dataclass
class WorkInput:
    """Per-tick mechanical work estimate for the fatigue integrator.

    Attributes
    ----------
    mechWork :
        Average joint mechanical power [0..1 normalised] over the last tick
        (computed by :class:`~src.fatigue.WorkEstimator.WorkEstimator`).
    windWork :
        Work done against wind load [0..1].
    recoveryCost :
        Cost accumulated from balance-recovery events [0..1].
    isHoldingOther :
        True when the player is currently sustaining a grasp on another body.
    isBeingHeld :
        True when the player is being held by another agent.
    """
    mechWork:      float = 0.0
    windWork:      float = 0.0
    recoveryCost:  float = 0.0
    isHoldingOther: bool = False
    isBeingHeld:   bool = False


# ---------------------------------------------------------------------------
# EnvInput — environmental conditions for one fatigue tick
# ---------------------------------------------------------------------------

@dataclass
class EnvInput:
    """Environmental parameters consumed by the fatigue integrator.

    Attributes
    ----------
    windLoad :
        Perceived wind load from PerceptionSystem [0..1].
    slopeDeg :
        Terrain slope in degrees.
    supportQuality :
        Ground support quality from GroundStabilityField [0..1].
    temperature :
        Cold-stress proxy (0 = warm, 1 = extreme cold/heat).
    dustResistance :
        Movement resistance from dust/snow [0..1].
    visibility :
        Visibility level (low visibility adds cognitive load) [0..1].
    speed :
        Current movement speed [m/s]; used for rest detection.
    """
    windLoad:        float = 0.0
    slopeDeg:        float = 0.0
    supportQuality:  float = 1.0
    temperature:     float = 0.0
    dustResistance:  float = 0.0
    visibility:      float = 1.0
    speed:           float = 0.0


# ---------------------------------------------------------------------------
# FatigueSystem
# ---------------------------------------------------------------------------

class FatigueSystem:
    """Integrates work and environment inputs into :class:`FatigueState`.

    Parameters
    ----------
    config :
        Optional dict; reads ``fatigue.*`` keys.
    """

    _DEFAULT_TICK_HZ            = 5.0
    _DEFAULT_K_WORK             = 0.08
    _DEFAULT_K_WIND             = 0.06
    _DEFAULT_K_RECOVERY         = 0.12   # cost of a near-fall recovery
    _DEFAULT_K_RECOVER          = 0.04   # resting recovery rate
    _DEFAULT_NOISE_MAX          = 0.6
    _DEFAULT_GRIP_SCALE_MIN     = 0.5
    _DEFAULT_REST_WIND_MAX      = 0.2
    _DEFAULT_REST_SLOPE_MAX     = 5.0    # degrees
    _DEFAULT_REST_SUPPORT_MIN   = 0.7
    _DEFAULT_REST_SPEED_MAX     = 0.3    # m/s
    _DEFAULT_GRASP_DRAIN_RATE   = 0.05   # gripReserve drain per second when holding
    _DEFAULT_GRIP_RECOVER_RATE  = 0.08   # gripReserve recovery per second at rest

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg  = config or {}
        fcfg = cfg.get("fatigue", {}) or {}

        self._enabled           = bool(fcfg.get("enable", True))
        self._tick_hz           = float(fcfg.get("tick_hz", self._DEFAULT_TICK_HZ))
        self._k_work            = float(fcfg.get("k_work", self._DEFAULT_K_WORK))
        self._k_wind            = float(fcfg.get("k_wind", self._DEFAULT_K_WIND))
        self._k_recovery        = float(fcfg.get("k_recovery", self._DEFAULT_K_RECOVERY))
        self._k_recover         = float(fcfg.get("k_recover", self._DEFAULT_K_RECOVER))
        self._noise_max         = float(fcfg.get("noise_max", self._DEFAULT_NOISE_MAX))
        self._grip_scale_min    = float(fcfg.get("grip_scale_min", self._DEFAULT_GRIP_SCALE_MIN))

        rest = fcfg.get("rest_condition_thresholds", {}) or {}
        self._rest_wind_max     = float(rest.get("wind", self._DEFAULT_REST_WIND_MAX))
        self._rest_slope_max    = float(rest.get("slope", self._DEFAULT_REST_SLOPE_MAX))
        self._rest_support_min  = float(rest.get("support", self._DEFAULT_REST_SUPPORT_MIN))
        self._rest_speed_max    = float(rest.get("speed", self._DEFAULT_REST_SPEED_MAX))

        self._grasp_drain_rate  = float(fcfg.get("grasp_drain_rate", self._DEFAULT_GRASP_DRAIN_RATE))
        self._grip_recover_rate = float(fcfg.get("grip_recover_rate", self._DEFAULT_GRIP_RECOVER_RATE))

        # Accumulated load duration for noise onset (seconds above baseline)
        self._load_duration: float = 0.0

        self._state = FatigueState()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    @property
    def state(self) -> FatigueState:
        """Current :class:`FatigueState`."""
        return self._state

    def tick(
        self,
        dt:         float,
        work:       WorkInput,
        env:        EnvInput,
    ) -> FatigueState:
        """Advance the fatigue integrator by ``dt`` seconds.

        Parameters
        ----------
        dt :
            Elapsed simulation time [s] since last tick.
        work :
            Mechanical work estimate for this tick.
        env :
            Environmental conditions this tick.

        Returns
        -------
        FatigueState
            Updated fatigue state.
        """
        if not self._enabled or dt <= 0.0:
            return self._state

        s = self._state

        # --- 1. Energy drain from mechanical work ---
        drain = (
            self._k_work     * work.mechWork
            + self._k_wind   * work.windWork
            + self._k_recovery * work.recoveryCost
        )

        # Additional drain from slope / dust resistance
        slope_factor = _clamp(env.slopeDeg / 45.0, 0.0, 1.0)
        dust_factor  = env.dustResistance
        env_drain    = 0.03 * slope_factor + 0.02 * dust_factor

        # Cognitive load: low visibility → coordination degrades faster
        cog_drain = 0.015 * (1.0 - env.visibility)

        total_drain = (drain + env_drain + cog_drain) * dt
        new_energy  = _clamp(s.energy - total_drain, 0.0, 1.0)

        # --- 2. Track load duration (for noise onset model) ---
        if drain > 0.01:
            self._load_duration += dt
        else:
            self._load_duration = max(0.0, self._load_duration - dt * 0.5)

        # --- 3. Thermal load ---
        new_thermal = _clamp(
            s.thermalLoad + 0.02 * env.temperature * dt - 0.01 * dt,
            0.0, 1.0,
        )

        # --- 4. Recovery under safe conditions ---
        resting = self._is_resting(env)
        if resting:
            new_energy = _clamp(new_energy + self._k_recover * dt, 0.0, 1.0)

        # --- 5. Neuromuscular noise ---
        # Rises with fatigue + thermal, then falls slowly when resting
        fatigue_factor = 1.0 - new_energy
        onset_factor   = _clamp(self._load_duration / 30.0, 0.0, 1.0)
        target_noise   = fatigue_factor * onset_factor * (1.0 + new_thermal) * self._noise_max
        target_noise   = _clamp(target_noise, 0.0, self._noise_max)
        # Slow smoothing
        noise_tau = 8.0 if resting else 20.0
        alpha_noise = 1.0 - math.exp(-dt / noise_tau)
        new_noise = _clamp(
            s.neuromuscularNoise + (target_noise - s.neuromuscularNoise) * alpha_noise,
            0.0, self._noise_max,
        )

        # --- 6. Coordination ---
        # Coordination degrades with noise + cognitive load
        target_coord = 1.0 - new_noise - cog_drain * 5.0
        target_coord = _clamp(target_coord, 0.0, 1.0)
        coord_tau    = 12.0 if resting else 25.0
        alpha_coord  = 1.0 - math.exp(-dt / coord_tau)
        new_coord    = _clamp(
            s.coordination + (target_coord - s.coordination) * alpha_coord,
            0.0, 1.0,
        )

        # --- 7. Tremor ---
        target_tremor = new_noise * new_thermal * 0.5 + new_noise * 0.5
        target_tremor = _clamp(target_tremor, 0.0, 1.0)
        tremor_tau    = 15.0 if resting else 30.0
        alpha_tremor  = 1.0 - math.exp(-dt / tremor_tau)
        new_tremor    = _clamp(
            s.tremor + (target_tremor - s.tremor) * alpha_tremor,
            0.0, 1.0,
        )

        # --- 8. Grip reserve ---
        if work.isHoldingOther:
            grip_drain = self._grasp_drain_rate * dt
            new_grip   = _clamp(s.gripReserve - grip_drain - fatigue_factor * 0.01 * dt, 0.0, 1.0)
        elif resting and not work.isBeingHeld:
            new_grip = _clamp(s.gripReserve + self._grip_recover_rate * dt, 0.0, 1.0)
        else:
            new_grip = _clamp(s.gripReserve - fatigue_factor * 0.005 * dt, 0.0, 1.0)

        self._state = FatigueState(
            energy             = new_energy,
            neuromuscularNoise = new_noise,
            coordination       = new_coord,
            tremor             = new_tremor,
            gripReserve        = new_grip,
            thermalLoad        = new_thermal,
        )
        return self._state

    def force_set(self, energy: float) -> None:
        """Dev helper: force energy level and recompute derived fields.

        Parameters
        ----------
        energy :
            Desired energy level [0..1].
        """
        energy = _clamp(energy, 0.0, 1.0)
        fatigue_factor = 1.0 - energy
        self._state = FatigueState(
            energy             = energy,
            neuromuscularNoise = fatigue_factor * self._noise_max,
            coordination       = 1.0 - fatigue_factor * self._noise_max,
            tremor             = fatigue_factor * 0.5,
            gripReserve        = _clamp(energy, self._grip_scale_min, 1.0),
            thermalLoad        = self._state.thermalLoad,
        )
        self._load_duration = (1.0 - energy) * 60.0

    def debug_info(self) -> dict:
        """Return current fatigue scalars for logging / dev output."""
        s = self._state
        return {
            "energy":             s.energy,
            "neuromuscularNoise": s.neuromuscularNoise,
            "coordination":       s.coordination,
            "tremor":             s.tremor,
            "gripReserve":        s.gripReserve,
            "thermalLoad":        s.thermalLoad,
            "load_duration_s":    self._load_duration,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _is_resting(self, env: EnvInput) -> bool:
        """Return True when environmental conditions are safe for recovery."""
        return (
            env.windLoad       <= self._rest_wind_max
            and abs(env.slopeDeg) <= self._rest_slope_max
            and env.supportQuality >= self._rest_support_min
            and env.speed          <= self._rest_speed_max
        )
