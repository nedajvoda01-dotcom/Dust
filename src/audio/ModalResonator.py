"""ModalResonator — Stage 36 modal-synthesis resonator (budget-capped pool).

Each active :class:`ModalResonatorVoice` maintains per-mode amplitude state:

    for each mode i:
        A_i += excitation * modal_weight_i
        output += A_i * sin(2π f_i * t)
        A_i  *= exp(-dt / decay_i)

The :class:`ModalResonatorPool` manages up to *max_voices* concurrent voices
with LRU eviction for the quietest entries.

Public API
----------
ModalResonatorPool(config=None)
  .trigger(profile, excitation_samples, world_pos, tick_time) → None
  .tick(dt) → float        # synthesise one audio tick → scalar mix
  .active_count → int
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from src.audio.MaterialAcousticDB import MaterialAcousticProfile
from src.audio.ExcitationGenerator import ExcitationSample


# ---------------------------------------------------------------------------
# ModalResonatorVoice
# ---------------------------------------------------------------------------

class ModalResonatorVoice:
    """Single resonator driven by a :class:`MaterialAcousticProfile`.

    Maintains per-mode amplitude state and advances it each tick.
    """

    def __init__(
        self,
        profile:     MaterialAcousticProfile,
        max_modes:   int,
        tick_time:   float,
    ) -> None:
        n = min(len(profile.modal_frequencies), max(1, max_modes))
        self._freqs:   List[float] = list(profile.modal_frequencies[:n])
        self._weights: List[float] = list(profile.modal_weights[:n])
        self._decays:  List[float] = list(profile.modal_decay[:n])
        self._amps:    List[float] = [0.0] * n
        self._n        = n
        self._phase:   List[float] = [0.0] * n   # per-mode phase accumulator
        self._t        = tick_time
        self._alive    = True

    # ------------------------------------------------------------------

    def excite(self, samples: List[ExcitationSample]) -> None:
        """Apply excitation samples to all modal amplitudes."""
        if not samples:
            return
        # Sum all sample amplitudes (cheap approximation for a batch)
        total = sum(abs(s.amplitude) for s in samples) / max(1, len(samples))
        for i in range(self._n):
            self._amps[i] += total * self._weights[i]

    def tick(self, dt: float) -> float:
        """Advance all modes and return the summed output sample."""
        out = 0.0
        all_silent = True
        for i in range(self._n):
            if abs(self._amps[i]) < 1e-9:
                continue
            all_silent = False
            # Phase advance
            self._phase[i] = (
                self._phase[i] + self._freqs[i] * dt
            ) % 1.0
            out += self._amps[i] * math.sin(2.0 * math.pi * self._phase[i])
            # Exponential decay
            decay = max(1e-6, self._decays[i])
            self._amps[i] *= math.exp(-dt / decay)
        self._t     += dt
        if all_silent:
            self._alive = False
        return out

    @property
    def amplitude(self) -> float:
        """Approximate current peak amplitude (for LRU eviction)."""
        return sum(abs(a) for a in self._amps)

    @property
    def is_alive(self) -> bool:
        return self._alive


# ---------------------------------------------------------------------------
# ModalResonatorPool
# ---------------------------------------------------------------------------

class ModalResonatorPool:
    """Budget-capped pool of :class:`ModalResonatorVoice` instances.

    Parameters
    ----------
    config :
        Optional dict; reads:
        - ``audio.max_active_resonators`` (default 128)
        - ``audio.modal_max_modes``       (default 8)
    """

    _DEFAULT_MAX_RESONATORS = 128
    _DEFAULT_MAX_MODES      = 8

    def __init__(self, config: Optional[dict] = None) -> None:
        audio = (config or {}).get("audio", {})
        self._max_voices: int = int(
            audio.get("max_active_resonators", self._DEFAULT_MAX_RESONATORS)
        )
        self._max_modes: int = int(
            audio.get("modal_max_modes", self._DEFAULT_MAX_MODES)
        )
        self._voices: List[ModalResonatorVoice] = []
        self._tick_time: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def trigger(
        self,
        profile:  MaterialAcousticProfile,
        samples:  List[ExcitationSample],
        world_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0),  # reserved for spatial audio
    ) -> None:
        """Spawn or re-excite a resonator voice for *profile*.

        If the pool is full, the quietest active voice is evicted (LRU-ish).
        """
        # Remove silent voices first
        self._voices = [v for v in self._voices if v.is_alive]

        if len(self._voices) >= self._max_voices:
            self._evict_quietest()

        voice = ModalResonatorVoice(profile, self._max_modes, self._tick_time)
        voice.excite(samples)
        self._voices.append(voice)

    def tick(self, dt: float) -> float:
        """Advance all voices and return the summed output sample."""
        self._tick_time += dt
        out = 0.0
        alive: List[ModalResonatorVoice] = []
        for v in self._voices:
            out += v.tick(dt)
            if v.is_alive:
                alive.append(v)
        self._voices = alive
        return out

    @property
    def active_count(self) -> int:
        return len(self._voices)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _evict_quietest(self) -> None:
        """Remove the voice with the lowest current amplitude."""
        if not self._voices:
            return
        idx = min(range(len(self._voices)), key=lambda i: self._voices[i].amplitude)
        self._voices.pop(idx)
