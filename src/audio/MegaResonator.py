"""MegaResonator — Stage 36 bulk plate resonator for rifts and avalanches.

Models a large rock plate (or geological slab) that resonates at infrasound
frequencies (10–80 Hz) with a very long decay.  It is distinct from the
standard :class:`~audio.ModalResonator.ModalResonatorPool` which handles
per-contact micro-sounds.

A single :class:`MegaResonator` represents the entire body of a large
geological structure.  It accumulates stress and emits low-frequency output
that represents the ``BulkPlateResonator`` described in the problem statement.

Architecture
------------
Excitation:
    stress_input → integrated into per-mode amplitude:
        A_i += stress * modal_weight_i * bulk_plate_gain
Output:
    output = Σ A_i * sin(2π f_i t)
    A_i   *= exp(-dt / decay_i)

The four default modes cover 12–68 Hz (infrasound range).  The resonator
activates when ``stress`` exceeds ``activation_threshold``.

Public API
----------
MegaResonator(config=None)
  .apply_stress(stress)        → None
  .tick(dt)                    → float
  .is_active                   → bool
  .peak_amplitude              → float
"""
from __future__ import annotations

import math
from typing import Optional

from src.audio.MaterialAcousticDB import MaterialAcousticDB, MAT_PLATE


# ---------------------------------------------------------------------------
# MegaResonator
# ---------------------------------------------------------------------------

class MegaResonator:
    """Infrasound resonator for large geological plates.

    Parameters
    ----------
    config :
        Optional dict; reads ``audio.bulk_plate_lowfreq_gain`` (default 1.0).
    activation_threshold :
        Minimum cumulative stress to trigger resonance.
    """

    _DEFAULT_GAIN         = 1.0
    _DEFAULT_THRESHOLD    = 0.1

    def __init__(
        self,
        config:               Optional[dict] = None,
        activation_threshold: float          = _DEFAULT_THRESHOLD,
    ) -> None:
        audio = (config or {}).get("audio", {})
        self._gain:      float = float(
            audio.get("bulk_plate_lowfreq_gain", self._DEFAULT_GAIN)
        )
        self._threshold: float = activation_threshold

        # Load MAT_PLATE profile (infrasound modes)
        db      = MaterialAcousticDB()
        profile = db.get(MAT_PLATE)
        n       = len(profile.modal_frequencies)

        self._freqs:   list = list(profile.modal_frequencies)
        self._weights: list = list(profile.modal_weights)
        self._decays:  list = list(profile.modal_decay)
        self._amps:    list = [0.0] * n
        self._phases:  list = [0.0] * n
        self._n              = n
        self._t:       float = 0.0
        self._activated:     bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply_stress(self, stress: float) -> None:
        """Inject cumulative geological stress into the resonator.

        Parameters
        ----------
        stress :
            Normalised stress value [0..∞); values above ``activation_threshold``
            excite the plate modes.
        """
        if stress < self._threshold:
            return
        self._activated = True
        excitation = (stress - self._threshold) * self._gain
        for i in range(self._n):
            self._amps[i] += excitation * self._weights[i]

    def tick(self, dt: float) -> float:
        """Advance all plate modes and return the summed output sample."""
        self._t += dt
        out = 0.0
        for i in range(self._n):
            if abs(self._amps[i]) < 1e-12:
                continue
            self._phases[i] = (
                self._phases[i] + self._freqs[i] * dt
            ) % 1.0
            out += self._amps[i] * math.sin(2.0 * math.pi * self._phases[i])
            decay = max(1e-6, self._decays[i])
            self._amps[i] *= math.exp(-dt / decay)
        return out

    @property
    def is_active(self) -> bool:
        """True if any mode amplitude is non-negligible."""
        return any(abs(a) > 1e-9 for a in self._amps)

    @property
    def peak_amplitude(self) -> float:
        """Sum of absolute mode amplitudes (useful for mixing / gating)."""
        return sum(abs(a) for a in self._amps)
