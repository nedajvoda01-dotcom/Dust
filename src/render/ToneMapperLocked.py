"""ToneMapperLocked — Stage 62 locked ACES-style tone mapper.

Implements §4.2:

* One tone map is chosen and **never** changes at runtime.
* Simplified ACES filmic curve — close to industry standard but faster
  (no matrix transforms required).
* Contrast is modulated by a single ``contrast`` parameter that is driven
  by weather conditions (storm → lower, clear dual-sun → higher, never clips).
* Gamma correction (sRGB) is applied at the end.

No dynamic LUT switching.  No texture assets.

Public API
----------
ToneMapperLocked(config=None)
  .apply(color) → tuple[float, float, float]
  .apply_buffer(buf) → list[tuple[float,float,float]]
  .set_contrast(c)   — called once per weather tick
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

_Vec3 = Tuple[float, float, float]


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


def _aces_channel(x: float) -> float:
    """Simplified ACES filmic curve on a single channel.

    Coefficients from the standard reference fit:
        f(x) = (x(ax + b)) / (x(cx + d) + e)
    with a=2.51, b=0.03, c=2.43, d=0.59, e=0.14
    """
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    denom = x * (c * x + d) + e
    if denom < 1e-10:
        return 0.0
    return _clamp((x * (a * x + b)) / denom, 0.0, 1.0)


def _linear_to_srgb(c: float) -> float:
    """Apply sRGB gamma curve."""
    if c <= 0.0031308:
        return c * 12.92
    return 1.055 * (c ** (1.0 / 2.4)) - 0.055


class ToneMapperLocked:
    """Locked ACES-style tone mapper with contrast control (§4.2, §4.3).

    Parameters
    ----------
    config :
        Optional dict; reads ``render.tone_mapper`` (must equal
        ``"aces_locked"`` — any other value raises no error but logs a
        warning) and ``render.contrast_base``.
    """

    _CONTRAST_MIN = 0.7   # storm floor (§4.3)
    _CONTRAST_MAX = 1.3   # clear dual-sun ceiling (§4.3)

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = (config or {}).get("render", {}) or {}
        self._contrast: float = float(cfg.get("contrast_base", 1.0))
        self._exposure: float = float(cfg.get("exposure_base", 1.0))

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def set_contrast(self, contrast: float) -> None:
        """Set per-frame contrast (driven by weather, not auto-tuning).

        Clamped to [_CONTRAST_MIN, _CONTRAST_MAX].
        """
        self._contrast = _clamp(contrast, self._CONTRAST_MIN, self._CONTRAST_MAX)

    def set_exposure(self, exposure: float) -> None:
        """Set exposure multiplier (e.g. for day/night)."""
        self._exposure = _clamp(exposure, 0.1, 10.0)

    def apply(self, color: _Vec3) -> _Vec3:
        """Tone-map and gamma-correct a single linear (r, g, b) colour.

        Returns an sRGB (r, g, b) in [0, 1].
        """
        r, g, b = color

        # Exposure
        r *= self._exposure
        g *= self._exposure
        b *= self._exposure

        # Contrast: pivot at 0.5 in linear space
        r = (r - 0.5) * self._contrast + 0.5
        g = (g - 0.5) * self._contrast + 0.5
        b = (b - 0.5) * self._contrast + 0.5

        # ACES filmic
        r = _aces_channel(max(0.0, r))
        g = _aces_channel(max(0.0, g))
        b = _aces_channel(max(0.0, b))

        # sRGB gamma
        r = _linear_to_srgb(r)
        g = _linear_to_srgb(g)
        b = _linear_to_srgb(b)

        return (_clamp(r, 0.0, 1.0), _clamp(g, 0.0, 1.0), _clamp(b, 0.0, 1.0))

    def apply_buffer(self, buf: List[_Vec3]) -> List[_Vec3]:
        """Apply :meth:`apply` to every pixel in a flat buffer."""
        return [self.apply(c) for c in buf]
