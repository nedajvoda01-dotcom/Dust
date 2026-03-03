"""SalienceCameraAdapter — Stage 55 salience-driven camera modifier.

Maps a :class:`~src.perception.SalienceSystem.PerceptualState` to subtle
camera parameter modifiers that are blended into
:class:`~src.camera.CinematicBias.StabilityInput` **without** creating
scripted moments or directional hints.

Behaviour (per §4.1):
* ``riskSalience``    ↑  →  FOV narrows slightly, sway reduces, horizon
                             stabilises (negative fov_bias, reduced sway).
* ``scaleSalience``   ↑  →  FOV widens slightly, vertical framing rises,
                             turn speed slows (positive fov_bias, height_bias).
* ``structuralSalience`` ↑ → short micro-lag via increased spring damping
                             (expressed as a rotation_lag scalar [0..1]).

All modifiers are bounded by ``max_camera_mod`` from config so they remain
*subtle*.  Input priority: caller can suppress salience mods by passing
``player_input_active=True``.

Public API
----------
CameraModifiers (dataclass)
SalienceCameraAdapter(config=None)
  .compute(perceptual_state, player_input_active=False) → CameraModifiers
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.perception.SalienceSystem import PerceptualState


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


# ---------------------------------------------------------------------------
# CameraModifiers
# ---------------------------------------------------------------------------

@dataclass
class CameraModifiers:
    """Subtle salience-driven camera adjustments.

    All values are deltas / scale factors to be **additively** or
    **multiplicatively** applied to the base camera intent.

    Attributes
    ----------
    fov_bias_deg :
        FOV offset in degrees (negative = narrow, positive = wide).
    height_bias :
        Additional camera height offset [world units].
    sway_scale :
        Multiplier on lateral sway (0 = none, 1 = full).
    rotation_lag :
        Extra smoothing on camera rotation [0..1] (0 = none).
    """
    fov_bias_deg:  float = 0.0
    height_bias:   float = 0.0
    sway_scale:    float = 1.0
    rotation_lag:  float = 0.0


# ---------------------------------------------------------------------------
# SalienceCameraAdapter
# ---------------------------------------------------------------------------

class SalienceCameraAdapter:
    """Maps PerceptualState → CameraModifiers.

    Parameters
    ----------
    config :
        Optional dict; reads ``salience.*`` keys.
    """

    _DEFAULT_MAX_MOD        = 0.3    # maximum absolute modifier magnitude
    _DEFAULT_RISK_FOV_DEG   = -3.0   # FOV delta at full riskSalience
    _DEFAULT_SCALE_FOV_DEG  =  4.0   # FOV delta at full scaleSalience
    _DEFAULT_SCALE_HEIGHT   =  0.25  # height delta at full scaleSalience
    _DEFAULT_STRUCT_LAG     =  0.25  # rotation_lag at full structuralSalience

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = (config or {}).get("salience", {}) or {}
        max_mod = float(cfg.get("max_camera_mod", self._DEFAULT_MAX_MOD))
        self._max_mod: float = max(0.0, max_mod)

        self._risk_fov_deg:  float = float(cfg.get("cam_risk_fov_deg",   self._DEFAULT_RISK_FOV_DEG))
        self._scale_fov_deg: float = float(cfg.get("cam_scale_fov_deg",  self._DEFAULT_SCALE_FOV_DEG))
        self._scale_height:  float = float(cfg.get("cam_scale_height",   self._DEFAULT_SCALE_HEIGHT))
        self._struct_lag:    float = float(cfg.get("cam_struct_lag",     self._DEFAULT_STRUCT_LAG))

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def compute(
        self,
        perceptual_state: PerceptualState,
        player_input_active: bool = False,
    ) -> CameraModifiers:
        """Compute camera modifiers from the current perceptual state.

        Parameters
        ----------
        perceptual_state :
            Current :class:`~src.perception.SalienceSystem.PerceptualState`.
        player_input_active :
            When ``True`` the player is actively controlling the camera;
            salience modifiers are suppressed (§8).
        """
        if player_input_active:
            return CameraModifiers()

        risk   = _clamp(perceptual_state.riskSalience,   0.0, 1.0)
        scale  = _clamp(perceptual_state.scaleSalience,  0.0, 1.0)
        struct = _clamp(perceptual_state.structuralSalience, 0.0, 1.0)

        # FOV: risk narrows, scale widens
        fov_bias = risk * self._risk_fov_deg + scale * self._scale_fov_deg
        fov_bias = _clamp(fov_bias, -self._max_mod * 10.0, self._max_mod * 10.0)

        # Height: scale raises framing
        height_bias = scale * self._scale_height
        height_bias = _clamp(height_bias, 0.0, self._max_mod)

        # Sway: risk reduces sway (1 → 1-risk*max)
        sway_scale = _clamp(1.0 - risk * self._max_mod, 0.0, 1.0)

        # Rotation lag: structural events add micro-lag
        rotation_lag = _clamp(struct * self._struct_lag, 0.0, self._max_mod)

        return CameraModifiers(
            fov_bias_deg=fov_bias,
            height_bias=height_bias,
            sway_scale=sway_scale,
            rotation_lag=rotation_lag,
        )
