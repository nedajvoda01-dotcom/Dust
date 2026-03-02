"""ExcitationGenerator — Stage 36 contact impulse → excitation signal.

Converts a :class:`~audio.ContactImpulseCollector.ContactImpulse` into a
one-dimensional excitation time-series that feeds the modal resonators.

Three excitation modes
----------------------
IMPACT :
    Short, high-energy impulse.
    ``excitation(t) = impulse_magnitude * exp(-t / tau_impact)``

SLIDING :
    Continuous friction noise while slip_ratio is high.
    ``excitation(t) = noise(seed, v_roughness) * ft * friction_k``
    Noise is deterministic: seeded from the material pair and a time-bucket.

CRUMBLE :
    Pressure-exceeded granular event (series of micro-impulse bursts).
    Modelled as a train of N short Gaussian-ish peaks with jittered timing.

The :class:`ExcitationGenerator` is stateless per call (returns a list of
``(time_offset, amplitude)`` pairs) so that resonators can accumulate
contributions cheaply.

Public API
----------
ExcitationGenerator(config=None)
  .generate(impulse, tick_index, dt) -> list[ExcitationSample]
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional

from src.audio.ContactImpulseCollector import ContactImpulse


# ---------------------------------------------------------------------------
# ExcitationType
# ---------------------------------------------------------------------------

class ExcitationType(Enum):
    IMPACT   = auto()
    SLIDING  = auto()
    CRUMBLE  = auto()


# ---------------------------------------------------------------------------
# ExcitationSample — one point on the excitation envelope
# ---------------------------------------------------------------------------

@dataclass
class ExcitationSample:
    """A scalar excitation value at a time offset within the current tick."""
    time_offset: float   # [s] from start of tick
    amplitude:   float   # scalar excitation amplitude


# ---------------------------------------------------------------------------
# Seeded deterministic noise (no external dependency)
# ---------------------------------------------------------------------------

def _lcg(state: int) -> int:
    return (1664525 * state + 1013904223) & 0xFFFFFFFF


def _det_noise(seed: int, n: int) -> List[float]:
    """Return *n* deterministic pseudo-random values in (-1, 1)."""
    state = seed & 0xFFFFFFFF
    out = []
    for _ in range(n):
        state = _lcg(state)
        out.append(state / 0x80000000 - 1.0)
    return out


# ---------------------------------------------------------------------------
# ExcitationGenerator
# ---------------------------------------------------------------------------

_IMPACT_THRESHOLD = 0.15   # slip_ratio ≤ this → IMPACT
_CRUMBLE_SLIP_MIN = 0.05   # if slip + graininess triggers crumble
_CRUMBLE_GRAIN_MIN = 0.60  # graininess threshold for crumble
_SLIDING_SLIP_MIN  = 0.30  # slip_ratio ≥ this → SLIDING component


class ExcitationGenerator:
    """Converts one :class:`ContactImpulse` into excitation samples.

    Parameters
    ----------
    config :
        Optional dict; reads ``audio.friction_noise_k`` and
        ``audio.granular_burst_k``.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        audio = (config or {}).get("audio", {})
        self._friction_k: float = float(audio.get("friction_noise_k", 1.0))
        self._granular_k: float = float(audio.get("granular_burst_k", 1.0))
        # tau_impact controls how fast the impact envelope decays (s)
        self._tau_impact: float = 0.008

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, impulse: ContactImpulse) -> ExcitationType:
        """Classify the dominant excitation type for *impulse*.

        Returns IMPACT or SLIDING.  CRUMBLE is not a primary classification;
        it is an overlay that ``generate`` adds on top when graininess is high
        (regardless of the primary type).
        """
        if impulse.slip_ratio <= _IMPACT_THRESHOLD:
            return ExcitationType.IMPACT
        if impulse.slip_ratio >= _SLIDING_SLIP_MIN:
            return ExcitationType.SLIDING
        return ExcitationType.IMPACT

    def generate(
        self,
        impulse:    ContactImpulse,
        tick_index: int,
        dt:         float,
        graininess: float = 0.0,
    ) -> List[ExcitationSample]:
        """Generate excitation samples for *impulse*.

        Parameters
        ----------
        impulse :
            The contact impulse to convert.
        tick_index :
            Current simulation tick (for deterministic seeding).
        dt :
            Duration of the current audio tick [s].
        graininess :
            Material graininess from :class:`~audio.MaterialAcousticDB`;
            increases crumble component when high.

        Returns
        -------
        list[ExcitationSample]
            Time-offset / amplitude pairs spanning the current tick window.
        """
        exc_type = self.classify(impulse)
        samples: List[ExcitationSample] = []

        if exc_type == ExcitationType.IMPACT:
            samples.extend(self._impact(impulse, dt))
        else:
            samples.extend(self._sliding(impulse, tick_index, dt))

        # Crumble overlay when graininess is high (pressure-crumble)
        if graininess >= _CRUMBLE_GRAIN_MIN and impulse.impulse_magnitude > 0.1:
            samples.extend(self._crumble(impulse, tick_index, dt, graininess))

        return samples

    # ------------------------------------------------------------------
    # Private: impact
    # ------------------------------------------------------------------

    def _impact(
        self,
        impulse: ContactImpulse,
        dt:      float,
    ) -> List[ExcitationSample]:
        """Exponential-decay impact envelope sampled at ~8 points."""
        n       = 8
        tau     = max(1e-6, self._tau_impact)
        samples = []
        for i in range(n):
            t   = (i / n) * dt
            amp = impulse.impulse_magnitude * math.exp(-t / tau)
            samples.append(ExcitationSample(time_offset=t, amplitude=amp))
        return samples

    # ------------------------------------------------------------------
    # Private: sliding
    # ------------------------------------------------------------------

    def _sliding(
        self,
        impulse:    ContactImpulse,
        tick_index: int,
        dt:         float,
    ) -> List[ExcitationSample]:
        """Deterministic friction noise proportional to Ft × friction_k."""
        pair_seed = (
            (impulse.material_pair[0] * 7919 + impulse.material_pair[1] * 6271)
            ^ (tick_index * 1013)
        ) & 0xFFFFFFFF
        n_pts   = 8
        noise   = _det_noise(pair_seed, n_pts)
        # scale: tangential component * roughness proxy * config_k
        # slip_ratio acts as a weight for the sliding contribution
        gain    = impulse.slip_ratio * impulse.impulse_magnitude * self._friction_k
        samples = []
        for i, n_val in enumerate(noise):
            t   = (i / n_pts) * dt
            amp = n_val * gain
            samples.append(ExcitationSample(time_offset=t, amplitude=amp))
        return samples

    # ------------------------------------------------------------------
    # Private: crumble
    # ------------------------------------------------------------------

    def _crumble(
        self,
        impulse:    ContactImpulse,
        tick_index: int,
        dt:         float,
        graininess: float,
    ) -> List[ExcitationSample]:
        """Granular micro-burst train — deterministic micro-impulse peaks."""
        seed = (
            (impulse.material_pair[0] * 3571 + impulse.material_pair[1] * 2897)
            ^ (tick_index * 541)
        ) & 0xFFFFFFFF
        # Number of micro-bursts scales with graininess and impulse magnitude
        n_bursts = max(2, int(graininess * 10 * impulse.impulse_magnitude * self._granular_k))
        n_bursts = min(n_bursts, 20)   # hard cap

        offsets_raw = _det_noise(seed, n_bursts)
        amps_raw    = _det_noise(seed ^ 0xBEEF, n_bursts)
        samples     = []
        base_gain   = impulse.impulse_magnitude * graininess * self._granular_k * 0.5

        for i in range(n_bursts):
            t   = ((offsets_raw[i] + 1.0) / 2.0) * dt   # map (-1,1) → (0, dt)
            amp = abs(amps_raw[i]) * base_gain
            samples.append(ExcitationSample(time_offset=t, amplitude=amp))
        return samples
