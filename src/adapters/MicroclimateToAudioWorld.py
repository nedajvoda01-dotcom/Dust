"""MicroclimateToAudioWorld — Stage 49 microclimate → Audio World (46) adapter.

Translates :class:`~src.microclimate.LocalClimateComposer.LocalClimate`
into audio-world parameters:

* **Occlusion boost** — sheltered locations improve terrain occlusion for
  distant sources (they "hide" behind the same geometry).
* **Reverb mix** — high ``echo_potential`` raises late-reflection gain.
* **SNR penalty** — local dust and wind channel reduce signal-to-noise.

Public API
----------
MicroclimateToAudioWorld()
  .audio_modifiers(local_climate) → AudioModifiers
"""
from __future__ import annotations

from dataclasses import dataclass

from src.microclimate.LocalClimateComposer import LocalClimate


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


@dataclass
class AudioModifiers:
    """Per-listener audio environment modifiers.

    Attributes
    ----------
    occlusion_boost : float
        Additional occlusion applied to all distant sources [0..1].
        Represents the shelter geometry blocking far sounds.
    reverb_mix : float
        Additive reverb mix gain [0..1].  Higher in enclosed spaces.
    snr_penalty : float
        Reduction in signal-to-noise ratio [0..1].  Higher in dusty/windy channels.
    """
    occlusion_boost: float = 0.0
    reverb_mix:      float = 0.0
    snr_penalty:     float = 0.0


class MicroclimateToAudioWorld:
    """Translates LocalClimate to AudioModifiers for the audio world model.

    Parameters
    ----------
    reverb_gain_max :
        Maximum reverb mix added at full echo potential.
    occlusion_shelter_k :
        How strongly windShelter contributes to extra occlusion for distant sounds.
    """

    def __init__(
        self,
        reverb_gain_max:    float = 0.8,
        occlusion_shelter_k: float = 0.5,
    ) -> None:
        self._reverb_max      = reverb_gain_max
        self._occlusion_k     = occlusion_shelter_k

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def audio_modifiers(self, local_climate: LocalClimate) -> AudioModifiers:
        """Compute audio environment modifiers from local climate.

        Parameters
        ----------
        local_climate :
            Locally-adjusted climate from LocalClimateComposer.

        Returns
        -------
        AudioModifiers
        """
        # Reverb: driven by echo potential
        reverb = _clamp(local_climate.echo_potential) * self._reverb_max

        # Occlusion boost: shelter geometry blocks far sources
        occlusion = _clamp(local_climate.shelter * self._occlusion_k)

        # SNR penalty: dust + channeled wind adds acoustic noise
        snr_pen = _clamp(
            local_climate.dust_density * 0.4
            + local_climate.wind_channel * 0.3
        )

        return AudioModifiers(
            occlusion_boost=occlusion,
            reverb_mix=reverb,
            snr_penalty=snr_pen,
        )
