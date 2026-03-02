"""AtmosphericPropagation — Stage 36 dust low-pass + cave reverb.

Applies two atmospheric effects to a mixed audio signal:

1. **Dust low-pass** — at high ``dust_density`` the high-frequency content is
   increasingly attenuated (dust particles absorb high-frequency sound).
2. **Cave reverb** — inside enclosed spaces a simple comb-filter delay network
   adds early reflections and modal room resonance.

Both effects are stateful (filter state, delay buffer) and advance each tick.

Public API
----------
AtmosphericPropagation(config=None)
  .process(sample, dust_density, in_cave, dt) → float
"""
from __future__ import annotations

import math
from collections import deque
from typing import Deque, Optional


class AtmosphericPropagation:
    """Atmospheric post-processor for the final mixed audio signal.

    Parameters
    ----------
    config :
        Optional dict; reads:
        - ``audio.atmo_lowpass_dust_k``  (default 1.0)
        - ``audio.cave_reverb_mix``      (default 0.35)
    """

    _DEFAULT_DUST_K      = 1.0
    _DEFAULT_REVERB_MIX  = 0.35
    # Cave delay length in ticks at 60 Hz ≈ 500 ms → 30 ticks
    _CAVE_DELAY_TICKS    = 30
    _CAVE_FEEDBACK       = 0.45

    def __init__(self, config: Optional[dict] = None) -> None:
        audio = (config or {}).get("audio", {})
        self._dust_k:     float = float(audio.get("atmo_lowpass_dust_k",  self._DEFAULT_DUST_K))
        self._reverb_mix: float = float(audio.get("cave_reverb_mix",      self._DEFAULT_REVERB_MIX))

        # Dust low-pass (one-pole)
        self._lp_state: float = 0.0

        # Cave delay line (circular buffer of tick samples)
        self._delay: Deque[float] = deque(
            [0.0] * self._CAVE_DELAY_TICKS,
            maxlen=self._CAVE_DELAY_TICKS,
        )
        # Second comb tap at a prime offset (gives richer reverb texture)
        _tap2 = max(1, int(self._CAVE_DELAY_TICKS * 0.61))
        self._delay2: Deque[float] = deque(
            [0.0] * _tap2,
            maxlen=_tap2,
        )

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def process(
        self,
        sample:       float,
        dust_density: float = 0.0,
        in_cave:      bool  = False,
        dt:           float = 1.0 / 60.0,
    ) -> float:
        """Apply dust LP and optional cave reverb to *sample*.

        Parameters
        ----------
        sample :
            Input audio sample (scalar).
        dust_density :
            Normalised dust suspension [0..1].
        in_cave :
            True when the listener is inside a cave / enclosed space.
        dt :
            Elapsed time this tick [s] (currently unused but reserved for
            variable-rate operation).
        """
        # 1. Dust low-pass: higher dust → lower cutoff
        #    cutoff_norm ≈ (1 - dust_density * dust_k) * base_cutoff
        base_cutoff   = 0.45
        dust_cutoff   = max(0.02, base_cutoff * (1.0 - dust_density * self._dust_k))
        self._lp_state = self._lp_state + dust_cutoff * (sample - self._lp_state)
        out = self._lp_state

        # 2. Cave reverb
        if in_cave:
            delayed1 = self._delay[0]
            delayed2 = self._delay2[0]
            reverb   = (delayed1 + delayed2) * self._CAVE_FEEDBACK * 0.5
            wet      = out + reverb * self._reverb_mix
            self._delay.appendleft(wet)
            self._delay2.appendleft(wet)
            out = wet

        return out
