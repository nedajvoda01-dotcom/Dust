"""SubsurfaceToAudio — Stage 67 adapter: fracture/vent → audio signals.

Translates subsurface crack events and vent activations into audio
modulation parameters consumed by the procedural audio system (Stage 46).

Audio signals produced (§5.1, §7)
-----------------------------------
vent_rumble :
    Low-frequency rumble proportional to vent intensity.
crack_impact :
    Impulse energy from fracture events.
lava_hiss :
    Continuous high-frequency hiss from active lava surface.

Public API
----------
SubsurfaceAudioSignal (dataclass)
  .vent_rumble  float [0..1]
  .crack_impact float [0..1]
  .lava_hiss    float [0..1]

SubsurfaceToAudio(config=None)
  .signals(vent_intensity, crack_energy, lava_volume) → SubsurfaceAudioSignal
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


@dataclass
class SubsurfaceAudioSignal:
    """Audio modulation parameters from subsurface events."""

    vent_rumble:  float = 0.0
    crack_impact: float = 0.0
    lava_hiss:    float = 0.0


class SubsurfaceToAudio:
    """Map subsurface events to audio modulation parameters.

    Parameters
    ----------
    config :
        Optional dict; reads ``subsurface67.*`` keys.
    """

    _DEFAULT_RUMBLE_SCALE  = 0.8
    _DEFAULT_IMPACT_SCALE  = 1.0
    _DEFAULT_HISS_SCALE    = 0.5

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = (config or {}).get("subsurface67", {}) or {}
        self._rumble_scale = float(cfg.get("audio_rumble_scale",  self._DEFAULT_RUMBLE_SCALE))
        self._impact_scale = float(cfg.get("audio_impact_scale",  self._DEFAULT_IMPACT_SCALE))
        self._hiss_scale   = float(cfg.get("audio_hiss_scale",    self._DEFAULT_HISS_SCALE))

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def signals(
        self,
        vent_intensity: float = 0.0,
        crack_energy:   float = 0.0,
        lava_volume:    float = 0.0,
    ) -> SubsurfaceAudioSignal:
        """Return audio signals for current subsurface activity.

        Parameters
        ----------
        vent_intensity :
            Intensity of the most recent vent event [0, 1].
        crack_energy :
            Energy released by fracture events [0, 1].
        lava_volume :
            Mean active lava surface volume [0, 1].
        """
        return SubsurfaceAudioSignal(
            vent_rumble  = _clamp(vent_intensity * self._rumble_scale),
            crack_impact = _clamp(crack_energy   * self._impact_scale),
            lava_hiss    = _clamp(lava_volume    * self._hiss_scale),
        )
