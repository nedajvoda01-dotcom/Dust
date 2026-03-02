"""AtmosphereAcousticAdapter — Stage 46 climate-to-acoustics translation.

Converts real-time atmospheric parameters (from Stage 29–33 climate / dust /
storm systems) into acoustic modifier structs consumed by
:class:`~audio_world.PropagationModel`.

Public API
----------
AtmosphereAcousticAdapter(config=None)
  .compute(dust_density, wind_speed, temp_gradient) → AtmoAcousticMods
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


@dataclass
class AtmoAcousticMods:
    """Acoustic modifiers derived from atmospheric conditions.

    Attributes
    ----------
    dust_lowpass_k :
        Multiplier applied on top of the propagation model's ``dust_lowpass_k``
        to strengthen muffling in extreme dust conditions.
    wind_snr_k :
        Multiplier applied to wind-related SNR degradation.
    range_factor :
        Overall range multiplier (< 1 in storms, > 1 in calm valleys).
    """
    dust_lowpass_k: float = 1.0
    wind_snr_k:     float = 1.0
    range_factor:   float = 1.0


class AtmosphereAcousticAdapter:
    """Translate atmospheric conditions into acoustic modifiers.

    Parameters
    ----------
    config :
        Optional dict; reads ``audio_world.*`` keys.
    """

    _DEFAULT_DUST_LP_K  = 2.0
    _DEFAULT_WIND_SNR_K = 1.5

    def __init__(self, config: Optional[dict] = None) -> None:
        awcfg = (config or {}).get("audio_world", {}) or {}
        self._dust_lp_k:  float = float(awcfg.get("dust_lowpass_k",  self._DEFAULT_DUST_LP_K))
        self._wind_snr_k: float = float(awcfg.get("wind_snr_k",      self._DEFAULT_WIND_SNR_K))

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def compute(
        self,
        dust_density:   float = 0.0,
        wind_speed:     float = 0.0,
        temp_gradient:  float = 0.0,
    ) -> AtmoAcousticMods:
        """Compute acoustic modifier set from atmospheric parameters.

        Parameters
        ----------
        dust_density :
            Normalised dust suspension [0..1].
        wind_speed :
            Normalised wind speed [0..1].
        temp_gradient :
            Temperature gradient proxy [0..1]; positive = inversion
            (traps sound near ground, increases range slightly).
        """
        # Dust → stronger low-pass in heavy dust
        dust_lp = 1.0 + dust_density * (self._dust_lp_k - 1.0)

        # Wind → more SNR noise
        wind_snr = 1.0 + wind_speed * (self._wind_snr_k - 1.0)

        # Range factor: storm halves audible range; temperature inversion boosts it
        storm = _clamp(dust_density * 0.6 + wind_speed * 0.4, 0.0, 1.0)
        range_f = _clamp(1.0 - storm * 0.5 + temp_gradient * 0.15, 0.3, 1.3)

        return AtmoAcousticMods(
            dust_lowpass_k=_clamp(dust_lp,  1.0, self._dust_lp_k * 2.0),
            wind_snr_k=_clamp(wind_snr,     1.0, self._wind_snr_k * 2.0),
            range_factor=range_f,
        )
