"""PropagationModel — Stage 46 acoustic propagation with atmospheric filtering.

Computes how a sound source at distance *d* is perceived by a listener,
accounting for:

* Inverse-square attenuation with a characteristic scale ``d0``.
* Additional exponential high-frequency air absorption.
* Atmospheric modifiers (dust density, wind speed) reducing SNR / range.
* LOD: update rate decreases with distance.
* Valley gain multiplier supplied by :class:`ValleyConcavityProxy`.

Public API
----------
PropagationModel(config=None)
  .propagate(distance, band_energy_audible, band_energy_infra,
             dust_density, wind_speed, occlusion, valley_gain)
    → PropagationResult
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class PropagationResult:
    """Outputs of the propagation model for a single source.

    Attributes
    ----------
    gain_audible :
        Net gain for the audible band [0..1].
    gain_infra :
        Net gain for the infrasound band [0..1].
    lp_cutoff_norm :
        Normalised low-pass cutoff [0..1]; lower = more muffled.
    snr :
        Signal-to-noise ratio proxy [0..1]; drops in wind/dust.
    """
    gain_audible:   float = 0.0
    gain_infra:     float = 0.0
    lp_cutoff_norm: float = 1.0
    snr:            float = 1.0


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


class PropagationModel:
    """Compute propagation from emitter to listener.

    Parameters
    ----------
    config :
        Optional dict; reads ``audio_world.*`` keys.
    """

    # Reference distance (m) at which gain = 1.0
    _DEFAULT_D0            = 10.0
    # Air absorption coefficient (per metre) for the audible band
    _DEFAULT_K_AIR         = 0.004
    # Dust low-pass strength (higher → more muffling per unit dust)
    _DEFAULT_DUST_LOWPASS_K = 2.0
    # Wind SNR degradation coefficient
    _DEFAULT_WIND_SNR_K    = 1.5
    # Max effective distance for audible band
    _DEFAULT_MAX_DIST      = 2000.0
    # Infra band has longer range; lose only half the HF air absorption
    _INFRA_RANGE_FACTOR    = 3.0

    def __init__(self, config: Optional[dict] = None) -> None:
        awcfg = (config or {}).get("audio_world", {}) or {}
        self._d0:            float = float(awcfg.get("d0",              self._DEFAULT_D0))
        self._k_air:         float = float(awcfg.get("k_air",           self._DEFAULT_K_AIR))
        self._dust_lp_k:     float = float(awcfg.get("dust_lowpass_k",  self._DEFAULT_DUST_LOWPASS_K))
        self._wind_snr_k:    float = float(awcfg.get("wind_snr_k",      self._DEFAULT_WIND_SNR_K))
        self._max_dist:      float = float(awcfg.get("max_dist",        self._DEFAULT_MAX_DIST))

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def propagate(
        self,
        distance:            float,
        band_energy_audible: float = 1.0,
        band_energy_infra:   float = 0.0,
        dust_density:        float = 0.0,
        wind_speed:          float = 0.0,
        occlusion:           float = 0.0,
        valley_gain:         float = 1.0,
    ) -> PropagationResult:
        """Compute propagation result for a source at *distance*.

        Parameters
        ----------
        distance :
            Metres from listener to source.
        band_energy_audible :
            Source audible energy [0..1+].
        band_energy_infra :
            Source infrasound energy [0..1+].
        dust_density :
            Atmospheric dust [0..1].
        wind_speed :
            Wind speed (normalised 0..1 proxy).
        occlusion :
            Terrain occlusion [0..1]; 1 = fully occluded.
        valley_gain :
            Terrain channelling multiplier [0.8..1.3].
        """
        d = max(0.0, distance)

        # --- Inverse-square base attenuation ---
        #   gain = 1 / (1 + (d/d0)^2)
        inv_sq = 1.0 / (1.0 + (d / self._d0) ** 2)

        # --- Exponential HF air absorption ---
        hi_loss = math.exp(-self._k_air * d)

        # --- Audible gain ---
        audible_atten = inv_sq * hi_loss
        # Occlusion hits audible band hard
        audible_atten *= (1.0 - occlusion * 0.85)
        # Valley channelling
        audible_atten *= valley_gain
        gain_audible = _clamp(band_energy_audible * audible_atten, 0.0, 1.0)

        # --- Infra gain (longer range, less affected by occlusion) ---
        infra_range = self._INFRA_RANGE_FACTOR
        inv_sq_infra = 1.0 / (1.0 + (d / (self._d0 * infra_range)) ** 2)
        hi_loss_infra = math.exp(-self._k_air * 0.1 * d)  # infra barely absorbed by air
        infra_atten = inv_sq_infra * hi_loss_infra
        infra_atten *= (1.0 - occlusion * 0.25)  # occlusion barely affects infra
        infra_atten *= valley_gain
        gain_infra = _clamp(band_energy_infra * infra_atten, 0.0, 1.0)

        # --- Low-pass cutoff (dust muffling) ---
        lp_cutoff = _clamp(1.0 - dust_density * self._dust_lp_k * 0.5, 0.05, 1.0)

        # --- SNR (wind + dust degrade detectability) ---
        wind_noise = _clamp(wind_speed * self._wind_snr_k * 0.4, 0.0, 0.9)
        dust_noise = _clamp(dust_density * 0.3, 0.0, 0.7)
        snr = _clamp(1.0 - wind_noise - dust_noise, 0.05, 1.0)

        return PropagationResult(
            gain_audible=gain_audible,
            gain_infra=gain_infra,
            lp_cutoff_norm=lp_cutoff,
            snr=snr,
        )
