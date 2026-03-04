"""AtmosphereToAudio — Stage 64 adapter: atmosphere → audio parameter bundle.

Translates LocalAtmoParams into audio modulation parameters consumed by the
audio/procedural-foley layers (Stages 46/77/78).

Output parameters
-----------------
wind_spectrum_gain    float [0..1]  — overall wind noise level
wind_bandpass_freq    float [Hz]    — dominant wind noise frequency
infrasound_gain       float [0..1]  — low-frequency storm rumble
fog_dampening         float [0..1]  — high-frequency attenuation in fog
distant_rumble_gain   float [0..1]  — thunder/front rumble
electro_click_rate    float [0..1]  — probability of electrical crackle
storm_modulation      float [0..1]  — modulation depth of wind gusts

Public API
----------
AtmosphereToAudio(config=None)
  .audio_params(local_params) -> AtmoAudioParams
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.atmo.AtmosphereSystem       import LocalAtmoParams
from src.atmo.WeatherRegimeDetector  import WeatherRegime


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


@dataclass
class AtmoAudioParams:
    wind_spectrum_gain:  float = 0.0
    wind_bandpass_freq:  float = 200.0   # Hz
    infrasound_gain:     float = 0.0
    fog_dampening:       float = 0.0
    distant_rumble_gain: float = 0.0
    electro_click_rate:  float = 0.0
    storm_modulation:    float = 0.0


class AtmosphereToAudio:
    """Map LocalAtmoParams to audio modulation parameters.

    Parameters
    ----------
    config :
        Optional dict; reads ``atmo64.audio.*`` keys.
    """

    _DEFAULT_WIND_BASE_STRENGTH = 0.6
    _DEFAULT_BASE_FREQ          = 150.0   # Hz

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg  = config or {}
        acfg = cfg.get("atmo64", {}).get("audio", {}) or {}
        self._wind_base = float(acfg.get("wind_base_strength", self._DEFAULT_WIND_BASE_STRENGTH))
        self._base_freq = float(acfg.get("base_freq_hz",       self._DEFAULT_BASE_FREQ))

    def audio_params(self, local_params: LocalAtmoParams) -> AtmoAudioParams:
        """Compute audio modulation parameters from *local_params*."""
        regime = local_params.regime
        ws     = local_params.wind_speed
        sp     = local_params.storm_potential
        fp     = local_params.fog_potential

        # Wind noise: proportional to local wind speed × base strength
        wind_gain = _clamp(ws * self._wind_base)
        # Bandpass frequency rises with wind speed (higher pitch in strong wind)
        bp_freq   = self._base_freq + ws * 400.0   # 150 – 550 Hz

        # Infrasound from storm fronts
        infra = _clamp(sp * 0.8) if regime in (WeatherRegime.DUST_STORM, WeatherRegime.ELECTRICAL) else 0.0

        # Fog dampening attenuates high frequencies
        fog_damp = _clamp(fp * 0.9) if regime == WeatherRegime.FOG else 0.0

        # Distant rumble at fronts (pressure gradients)
        rumble = _clamp(sp * 0.6)

        # Electrical crackle rate
        click = _clamp(local_params.electro) if regime == WeatherRegime.ELECTRICAL else 0.0

        # Gust modulation depth
        mod = _clamp(ws * 0.5 + sp * 0.5)

        return AtmoAudioParams(
            wind_spectrum_gain  = wind_gain,
            wind_bandpass_freq  = bp_freq,
            infrasound_gain     = infra,
            fog_dampening       = fog_damp,
            distant_rumble_gain = rumble,
            electro_click_rate  = click,
            storm_modulation    = mod,
        )
