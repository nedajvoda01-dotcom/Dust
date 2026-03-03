"""InstabilityToAudioAdapter — Stage 52 instability → audio emitter bridge.

Creates :class:`~audio.audio_world.EmitterAggregator.AcousticEmitterRecord`
objects from instability events.  All structural instability events produce
``STRUCTURAL`` emitters with:

* high ``band_energy_infra`` (infrasound)
* moderate ``band_energy_audible``
* long TTL (long decay)

Public API
----------
InstabilityToAudioAdapter(config=None)
  .emitter_from_event(pos, intensity, sim_tick) → AcousticEmitterRecord
"""
from __future__ import annotations

from typing import Optional

from src.audio.audio_world.EmitterAggregator import (
    AcousticEmitterRecord,
    EmitterType,
)
from src.math.Vec3 import Vec3


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


class InstabilityToAudioAdapter:
    """Converts instability discharge events to STRUCTURAL acoustic emitters.

    Parameters
    ----------
    config :
        Optional dict; reads ``instability.*`` keys.
    """

    _DEFAULT_AUDIO_K     = 0.6    # scale intensity → audible energy
    _DEFAULT_INFRA_K     = 0.9    # scale intensity → infra energy
    _DEFAULT_TTL         = 180    # ticks (~3 s at 60 Hz)

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = (config or {}).get("instability", {}) or {}
        self._audio_k: float = float(cfg.get("energy_to_audio_k", self._DEFAULT_AUDIO_K))
        self._infra_k: float = float(cfg.get("energy_to_infra_k", self._DEFAULT_INFRA_K))
        self._ttl:     int   = int(  cfg.get("structural_ttl",    self._DEFAULT_TTL))

    def emitter_from_event(
        self,
        pos:       Vec3,
        intensity: float,
        sim_tick:  int = 0,
    ) -> AcousticEmitterRecord:
        """Build a STRUCTURAL AcousticEmitterRecord for the given event.

        Parameters
        ----------
        pos       : World-space position of the instability event.
        intensity : Discharge energy [0..1+].
        sim_tick  : Current simulation tick (for TTL bookkeeping).

        Returns
        -------
        AcousticEmitterRecord ready to pass to EmitterAggregator.add().
        """
        return AcousticEmitterRecord(
            id=0,   # aggregator assigns a unique id
            pos=pos,
            band_energy_audible=_clamp(intensity * self._audio_k),
            band_energy_infra=_clamp(intensity * self._infra_k),
            directivity=0.0,   # omnidirectional structural wave
            emitter_type=EmitterType.STRUCTURAL,
            created_tick=sim_tick,
            ttl=self._ttl,
        )
