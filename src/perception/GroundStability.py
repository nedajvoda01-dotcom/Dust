"""GroundStability — Stage 37 ground stability perception field.

Produces three scalars that describe how trustworthy the current footing
is, consumed by the Threat/Salience aggregator and MotorStack:

* ``supportQuality``  (0..1) — overall support quality (1 = perfect)
* ``slipRisk``        (0..1) — probability of slipping
* ``sinkRisk``        (0..1) — probability of sinking / surface collapse

Inputs (all caller-supplied):
* ``friction``     — surface friction coefficient proxy [0..1]
* ``softness``     — surface softness / sink proxy [0..1] (from Stage 35)
* ``slope_deg``    — surface slope angle [degrees]
* ``roughness``    — surface macro-roughness [0..1]

Public API
----------
GroundStabilityField(config=None)
  .update(friction, softness, slope_deg, roughness, dt) → None
  .support_quality  → float
  .slip_risk        → float
  .sink_risk        → float
"""
from __future__ import annotations

import math
from typing import Optional


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


class GroundStabilityField:
    """Perception sub-field: ground stability.

    Parameters
    ----------
    config :
        Optional dict; reads ``perception.ground.*`` keys.
    """

    _DEFAULT_SLIP_MU_WEIGHT = 1.5
    _DEFAULT_SINK_WEIGHT    = 1.2
    _DEFAULT_SMOOTHING_TAU  = 0.10  # seconds

    # Maximum slope considered "normal" before slip risk rises
    _SLOPE_SLIP_START_DEG   = 20.0
    _SLOPE_SLIP_FULL_DEG    = 45.0

    def __init__(self, config: Optional[dict] = None) -> None:
        pcfg = ((config or {}).get("perception", {}) or {}).get("ground", {}) or {}
        self._slip_mu_w: float = float(pcfg.get("slip_mu_weight", self._DEFAULT_SLIP_MU_WEIGHT))
        self._sink_w:    float = float(pcfg.get("sink_weight",    self._DEFAULT_SINK_WEIGHT))
        tau = float(
            ((config or {}).get("perception", {}) or {}).get(
                "smoothing_tau_sec", self._DEFAULT_SMOOTHING_TAU
            )
        )
        self._tau: float = max(1e-3, tau)

        self._support: float = 1.0
        self._slip:    float = 0.0
        self._sink:    float = 0.0

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def update(
        self,
        friction:  float = 1.0,
        softness:  float = 0.0,
        slope_deg: float = 0.0,
        roughness: float = 0.5,
        dt:        float = 1.0 / 10.0,
    ) -> None:
        """Advance the ground stability field one tick.

        Parameters
        ----------
        friction :
            Surface friction coefficient [0..1]; low = icy/slippery.
        softness :
            Surface softness proxy [0..1]; high = loose dust / deep sink.
        slope_deg :
            Current surface slope angle in degrees.
        roughness :
            Macro-roughness [0..1]; improves mechanical grip slightly.
        dt :
            Elapsed simulation time [s].
        """
        friction  = _clamp(friction,  0.0, 1.0)
        softness  = _clamp(softness,  0.0, 1.0)
        slope_deg = max(0.0, slope_deg)
        roughness = _clamp(roughness, 0.0, 1.0)

        # Slip risk: driven by low friction, steep slope
        slope_factor = _clamp(
            (slope_deg - self._SLOPE_SLIP_START_DEG)
            / max(1.0, self._SLOPE_SLIP_FULL_DEG - self._SLOPE_SLIP_START_DEG),
            0.0, 1.0,
        )
        raw_slip = _clamp(
            (1.0 - friction) * self._slip_mu_w + slope_factor * 0.5
            - roughness * 0.15,
            0.0, 1.0,
        )

        # Sink risk: driven by high softness
        raw_sink = _clamp(softness * self._sink_w, 0.0, 1.0)

        # Support quality: high when low slip/sink risk
        raw_support = _clamp(1.0 - max(raw_slip, raw_sink), 0.0, 1.0)

        # Exponential smoothing
        alpha = 1.0 - math.exp(-dt / self._tau)
        self._slip    = self._slip    + alpha * (raw_slip    - self._slip)
        self._sink    = self._sink    + alpha * (raw_sink    - self._sink)
        self._support = self._support + alpha * (raw_support - self._support)

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------

    @property
    def support_quality(self) -> float:
        """Overall ground support quality [0..1]; 1 = perfect footing."""
        return self._support

    @property
    def slip_risk(self) -> float:
        """Risk of slipping [0..1]."""
        return self._slip

    @property
    def sink_risk(self) -> float:
        """Risk of sinking into the surface [0..1]."""
        return self._sink
