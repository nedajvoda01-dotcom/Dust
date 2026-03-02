"""AudioSalience — Stage 37 audio-salience perception field.

Aggregates nearby sound sources into three scalars consumed by the
Threat/Salience aggregator:

* ``audioSalience``  (0..1) — overall loudness relevance
* ``audioDir``       (Vec3, unit) — weighted direction toward sources
* ``audioUrgency``   (0..1) — low-frequency / impact urgency

Environmental modifiers
-----------------------
* In dust / storm: high-frequency content is cut → salience drops.
* In cave (cave_factor > 0): reverb amplifies low-frequency urgency.

Public API
----------
AudioSalienceField(config=None)
  .update(listener_pos, sources, dust_density, cave_factor, dt) → None
  .audio_salience  → float
  .audio_dir       → Vec3
  .audio_urgency   → float
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

from src.math.Vec3 import Vec3


@dataclass
class AudioSource:
    """A single active sound source visible to the perception system.

    Attributes
    ----------
    position :
        World-space position of the emitter.
    energy :
        Total acoustic energy proxy [0..1].
    low_freq_ratio :
        Fraction of energy in low frequencies (0 = pure HF, 1 = pure LF).
        Impacts ``audioUrgency``.
    """
    position:      Vec3
    energy:        float = 0.0
    low_freq_ratio: float = 0.0


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


class AudioSalienceField:
    """Perception sub-field: audio salience and directionality.

    Parameters
    ----------
    config :
        Optional dict; reads ``perception.audio.*`` keys.
    """

    _DEFAULT_SALIENCE_RADIUS   = 120.0
    _DEFAULT_LOWFREQ_URGENCY_K = 2.0
    _DEFAULT_SMOOTHING_TAU     = 0.15  # seconds

    def __init__(self, config: Optional[dict] = None) -> None:
        pcfg = ((config or {}).get("perception", {}) or {}).get("audio", {}) or {}
        self._radius:     float = float(pcfg.get("salience_radius",   self._DEFAULT_SALIENCE_RADIUS))
        self._lf_k:       float = float(pcfg.get("lowfreq_urgency_k", self._DEFAULT_LOWFREQ_URGENCY_K))
        tau = float(
            ((config or {}).get("perception", {}) or {}).get(
                "smoothing_tau_sec", self._DEFAULT_SMOOTHING_TAU
            )
        )
        self._tau: float = max(1e-3, tau)

        self._salience: float = 0.0
        self._urgency:  float = 0.0
        self._dir:      Vec3  = Vec3(0.0, 0.0, 0.0)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def update(
        self,
        listener_pos:  Vec3,
        sources:       List[AudioSource],
        dust_density:  float = 0.0,
        cave_factor:   float = 0.0,
        dt:            float = 1.0 / 20.0,
    ) -> None:
        """Recompute audio salience from *sources* this tick.

        Parameters
        ----------
        listener_pos :
            World position of the character (listener).
        sources :
            Active sound sources this tick (only those within radius
            are considered).
        dust_density :
            0..1 dust suspension; reduces high-frequency salience.
        cave_factor :
            0..1 cave enclosure factor; amplifies low-frequency urgency.
        dt :
            Elapsed simulation time [s].
        """
        total_weight = 0.0
        dir_acc      = Vec3(0.0, 0.0, 0.0)
        urgency_acc  = 0.0

        for src in sources:
            diff = src.position - listener_pos
            dist = diff.length()
            if dist > self._radius or dist < 1e-6:
                continue

            # Inverse-square attenuation
            gain = (self._radius / max(1.0, dist)) ** 2

            # Dust cuts high-freq → reduce HF portion of energy
            hf_ratio = 1.0 - src.low_freq_ratio
            atmo_cut = 1.0 - dust_density * 0.7
            effective_hf = hf_ratio * max(0.0, atmo_cut)
            effective_energy = src.energy * (src.low_freq_ratio + effective_hf)

            w = effective_energy * gain
            total_weight += w

            unit = diff * (1.0 / dist)
            dir_acc = dir_acc + unit * w

            # Urgency from low-frequency content; cave amplifies it
            lf_urgency = src.low_freq_ratio * self._lf_k
            if cave_factor > 0.0:
                lf_urgency *= 1.0 + cave_factor
            urgency_acc += lf_urgency * w

        # Normalise
        if total_weight > 1e-9:
            raw_salience = _clamp(total_weight / (self._radius ** 2), 0.0, 1.0)
            dir_len = dir_acc.length()
            raw_dir = dir_acc * (1.0 / dir_len) if dir_len > 1e-6 else Vec3(0.0, 0.0, 0.0)
            raw_urgency = _clamp(urgency_acc / total_weight, 0.0, 1.0)
        else:
            raw_salience = 0.0
            raw_dir      = Vec3(0.0, 0.0, 0.0)
            raw_urgency  = 0.0

        # Exponential smoothing
        alpha = 1.0 - math.exp(-dt / self._tau)
        self._salience = self._salience + alpha * (raw_salience - self._salience)
        self._urgency  = self._urgency  + alpha * (raw_urgency  - self._urgency)
        # Blend direction
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
    def audio_salience(self) -> float:
        """Overall audio salience [0..1]."""
        return self._salience

    @property
    def audio_dir(self) -> Vec3:
        """Unit vector pointing toward dominant audio source."""
        return self._dir

    @property
    def audio_urgency(self) -> float:
        """Low-frequency urgency [0..1]; higher in caves and after impacts."""
        return self._urgency
