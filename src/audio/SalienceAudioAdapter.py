"""SalienceAudioAdapter — Stage 55 salience-driven audio parameter modifier.

Maps a :class:`~src.perception.SalienceSystem.PerceptualState` to subtle
audio mixing parameters that modulate the generative audio layer (Stage 46)
**without** introducing music or scripted triggers.

Behaviour (per §4.3):
* ``riskSalience``    ↑  →  dynamic range compression ↑ (tension feel),
                             low-frequency emphasis ↑.
* ``scaleSalience``   ↑  →  spatial width ↑ (wider stereo field),
                             LF emphasis ↑ (grandeur).
* ``environmentalSalience`` ↑ → LF emphasis ↑ (infrasound / unusual echo).

All modifiers are bounded by ``max_audio_gain`` from config.

Public API
----------
AudioModifiers (dataclass)
SalienceAudioAdapter(config=None)
  .compute(perceptual_state) → AudioModifiers
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.perception.SalienceSystem import PerceptualState


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


# ---------------------------------------------------------------------------
# AudioModifiers
# ---------------------------------------------------------------------------

@dataclass
class AudioModifiers:
    """Salience-derived audio mixing adjustments.

    Attributes
    ----------
    dynamic_range_compression :
        Additional dynamic range compression [0..1] (0 = none).
    lf_emphasis :
        Low-frequency boost [0..1] (0 = none).
    spatial_width :
        Stereo/spatial width scale [0..1] (0 = mono, 1 = full).
    """
    dynamic_range_compression: float = 0.0
    lf_emphasis:               float = 0.0
    spatial_width:             float = 0.5   # neutral = 0.5


# ---------------------------------------------------------------------------
# SalienceAudioAdapter
# ---------------------------------------------------------------------------

class SalienceAudioAdapter:
    """Maps PerceptualState → AudioModifiers.

    Parameters
    ----------
    config :
        Optional dict; reads ``salience.*`` keys.
    """

    _DEFAULT_MAX_AUDIO_GAIN    = 0.4
    _DEFAULT_RISK_COMPRESS     = 0.6   # compression at full risk
    _DEFAULT_RISK_LF           = 0.5   # LF emphasis at full risk
    _DEFAULT_SCALE_WIDTH       = 0.4   # extra width at full scale
    _DEFAULT_SCALE_LF          = 0.3   # LF at full scale
    _DEFAULT_ENV_LF            = 0.4   # LF at full env salience

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = (config or {}).get("salience", {}) or {}
        self._max_gain: float = float(cfg.get("max_audio_gain", self._DEFAULT_MAX_AUDIO_GAIN))

        self._risk_compress: float = float(cfg.get("audio_risk_compress", self._DEFAULT_RISK_COMPRESS))
        self._risk_lf:       float = float(cfg.get("audio_risk_lf",       self._DEFAULT_RISK_LF))
        self._scale_width:   float = float(cfg.get("audio_scale_width",   self._DEFAULT_SCALE_WIDTH))
        self._scale_lf:      float = float(cfg.get("audio_scale_lf",      self._DEFAULT_SCALE_LF))
        self._env_lf:        float = float(cfg.get("audio_env_lf",        self._DEFAULT_ENV_LF))

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def compute(self, perceptual_state: PerceptualState) -> AudioModifiers:
        """Compute audio modifiers from the current perceptual state.

        Parameters
        ----------
        perceptual_state :
            Current :class:`~src.perception.SalienceSystem.PerceptualState`.
        """
        risk  = _clamp(perceptual_state.riskSalience,          0.0, 1.0)
        scale = _clamp(perceptual_state.scaleSalience,         0.0, 1.0)
        env   = _clamp(perceptual_state.environmentalSalience, 0.0, 1.0)

        compress = _clamp(risk * self._risk_compress, 0.0, self._max_gain)

        lf_raw = risk * self._risk_lf + scale * self._scale_lf + env * self._env_lf
        lf_emphasis = _clamp(lf_raw, 0.0, self._max_gain)

        # Spatial width: neutral 0.5, scale pushes it wider (max 0.5 + max_gain)
        width = _clamp(0.5 + scale * self._scale_width * self._max_gain, 0.0, 1.0)

        return AudioModifiers(
            dynamic_range_compression=compress,
            lf_emphasis=lf_emphasis,
            spatial_width=width,
        )
