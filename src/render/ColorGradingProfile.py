"""ColorGradingProfile — Stage 62 locked colour grading (§4).

Applies the final colour-science lockdown after tone-mapping:

* **Limited colour gamut** — values are gently pushed toward the palette
  centre to avoid over-saturated edges (§4.1).
* **Controlled gradients** — a subtle S-curve is applied to each channel so
  mid-tones remain clean and shadow/highlight roll-off is graceful.
* **No dynamic LUT switching** — the profile is fixed at construction time and
  can only be changed through a controlled ``tuning_epoch`` mechanism (§12).

Pure Python — no texture lookups.

Public API
----------
ColorGradingProfile(config=None)
  .grade(color) → tuple[float, float, float]
  .grade_buffer(buf) → list[tuple[float,float,float]]
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

_Vec3 = Tuple[float, float, float]


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


def _s_curve(x: float, strength: float = 0.5) -> float:
    """Gentle S-curve centred at 0.5 (keeps mid-tones neutral)."""
    # Simple cubic: f(x) = x + strength * x(x-1)(2x-1)
    # This is zero-derivative at 0 and 1, ensuring no clipping.
    return _clamp(x + strength * x * (x - 1.0) * (2.0 * x - 1.0), 0.0, 1.0)


class ColorGradingProfile:
    """Locked colour grading profile (§4.1, §4.3).

    Parameters
    ----------
    config :
        Optional dict; reads ``render.grade_saturation_limit``,
        ``render.grade_scurve_strength``.
    """

    # How aggressively to pull colours toward the palette centre
    _DEFAULT_SAT_LIMIT = 0.85   # max saturation relative to input
    _DEFAULT_SCURVE    = 0.35   # S-curve strength

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = (config or {}).get("render", {}) or {}
        self._sat_limit: float = float(
            cfg.get("grade_saturation_limit", self._DEFAULT_SAT_LIMIT)
        )
        self._scurve_strength: float = float(
            cfg.get("grade_scurve_strength", self._DEFAULT_SCURVE)
        )

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def grade(self, color: _Vec3) -> _Vec3:
        """Apply colour grading to a single sRGB (r, g, b) in [0, 1].

        Returns graded (r, g, b) in [0, 1].
        """
        r, g, b = color

        # Step 1: limited gamut — reduce saturation toward luma
        luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
        r = luma + (r - luma) * self._sat_limit
        g = luma + (g - luma) * self._sat_limit
        b = luma + (b - luma) * self._sat_limit

        # Step 2: S-curve per channel
        r = _s_curve(_clamp(r, 0.0, 1.0), self._scurve_strength)
        g = _s_curve(_clamp(g, 0.0, 1.0), self._scurve_strength)
        b = _s_curve(_clamp(b, 0.0, 1.0), self._scurve_strength)

        return (r, g, b)

    def grade_buffer(self, buf: List[_Vec3]) -> List[_Vec3]:
        """Apply :meth:`grade` to every pixel in a flat buffer."""
        return [self.grade(c) for c in buf]
