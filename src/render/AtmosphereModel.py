"""AtmosphereModel — Stage 62 simplified atmosphere model (§6).

Provides a lightweight, **stable** atmosphere computation for use inside the
PixelPipeline.  It complements the heavier
:class:`~src.render.AtmosphereRenderer.AtmosphereRenderer` (which handles
full CPU ray-marching for sky colour) by exposing only the parameters that
affect per-surface shading:

* **Height-based fog** (§6.1): exponential density that increases near the
  surface and thickens with dust.
* **Depth-based light scattering hint** (§6.2): screen-space approximation of
  volumetric inscattering along the view ray.
* **Storm mode** (§6.3): reduced visibility + dust colour shift, no sprites.

This module is pure Python maths — no OpenGL.

Public API
----------
AtmosphereParams (dataclass)
AtmosphereModel(config=None)
  .update(dust_density, storm_active, altitude)
  .fog_factor(distance)  → float  [0 = no fog, 1 = full fog]
  .scatter_hint(distance, depth) → tuple[float,float,float]
  .color_shift            → tuple[float,float,float]
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

_Vec3 = Tuple[float, float, float]


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * _clamp(t, 0.0, 1.0)


def _lerp3(a: _Vec3, b: _Vec3, t: float) -> _Vec3:
    t = _clamp(t, 0.0, 1.0)
    return (a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t,
            a[2] + (b[2] - a[2]) * t)


# ---------------------------------------------------------------------------
# AtmosphereParams
# ---------------------------------------------------------------------------

@dataclass
class AtmosphereParams:
    """Live atmospheric state passed to :meth:`AtmosphereModel.update`."""
    dust_density: float = 0.0    # 0 = clear, 1 = maximum dust
    storm_active: bool = False
    altitude: float = 0.0        # observer height above surface (sim units)


# ---------------------------------------------------------------------------
# AtmosphereModel
# ---------------------------------------------------------------------------

class AtmosphereModel:
    """Height fog + scattering hint + storm colour shift (§6).

    Parameters
    ----------
    config :
        Optional dict; reads ``render.fog_density_base``,
        ``render.dust_color_shift``.
    """

    # Dust colour shift (warm ochre) applied in storm mode (§6.3)
    _DUST_SHIFT_COLOR: _Vec3 = (0.82, 0.64, 0.38)
    # Clear-sky inscattering tint (cool blue horizon)
    _SCATTER_COLOR: _Vec3 = (0.52, 0.70, 0.90)

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = (config or {}).get("render", {}) or {}
        self._fog_density_base: float = float(cfg.get("fog_density_base", 0.003))
        self._dust_shift_strength: float = float(cfg.get("dust_color_shift", 0.45))

        # Runtime state
        self._dust: float = 0.0
        self._storm: bool = False
        self._altitude: float = 0.0

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def update(
        self,
        dust_density: float,
        storm_active: bool = False,
        altitude: float = 0.0,
    ) -> None:
        """Update live atmospheric state once per weather tick."""
        self._dust = _clamp(dust_density, 0.0, 1.0)
        self._storm = storm_active
        self._altitude = max(0.0, altitude)

    @property
    def color_shift(self) -> _Vec3:
        """Current dust colour shift tint (white = no shift)."""
        t = self._dust * self._dust_shift_strength
        if self._storm:
            t = _clamp(t + 0.25, 0.0, 1.0)
        white: _Vec3 = (1.0, 1.0, 1.0)
        return _lerp3(white, self._DUST_SHIFT_COLOR, t)

    def fog_factor(self, distance: float) -> float:
        """Compute fog opacity for a surface at *distance* from camera.

        Returns a value in [0, 1]:
        * 0 = fully clear
        * 1 = completely fogged out

        Density is:
        * increased by dust (multiplicative)
        * increased in storm mode
        * reduced at high altitude (height falloff)
        """
        if distance <= 0.0:
            return 0.0

        # Height decay: density halves every 500 sim units above surface
        height_scale = math.exp(-self._altitude * 0.002)

        # Dust multiplier
        dust_scale = 1.0 + self._dust * 3.0
        if self._storm:
            dust_scale += 2.0

        density = self._fog_density_base * dust_scale * height_scale
        # Beer-Lambert exponential fog
        fog = 1.0 - math.exp(-density * distance)
        return _clamp(fog, 0.0, 1.0)

    def scatter_hint(self, distance: float, depth: float = 1.0) -> _Vec3:
        """Screen-space depth-based inscattering approximation (§6.2).

        Parameters
        ----------
        distance :
            View-ray distance to surface (sim units).
        depth :
            Normalised depth buffer value [0, 1]; 1 = far plane.

        Returns
        -------
        (r, g, b) inscattering contribution to add to the surface colour.
        """
        fog = self.fog_factor(distance)
        # Inscattering grows with fog, modulated by depth
        s = fog * _clamp(depth, 0.0, 1.0) * 0.4
        # In storm mode shift toward dust colour instead of sky blue
        if self._storm:
            base = self._DUST_SHIFT_COLOR
        else:
            base = self._SCATTER_COLOR
        return (base[0] * s, base[1] * s, base[2] * s)
