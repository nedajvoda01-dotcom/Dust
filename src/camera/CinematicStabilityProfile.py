"""CinematicStabilityProfile — Stage 62 camera inertia + sway profile (§8).

Encodes the cinematic camera behaviour parameters for Stage 62:

* **Soft inertia** (§8.1): position and orientation lag (critically damped
  spring constants).
* **Vertical sway** (§8.1): gentle breathing/idle sway that does not conflict
  with the pixel grid.
* **No cinematic shake** (§8.1): the profile explicitly zeroes shake after the
  pixel pass.
* **Salience FOV** (§8.2): very gentle FOV modulation from Salience (stage 55)
  that never exceeds ``fov_salience_max_deg`` so the pixel grid remains stable.

This dataclass is the **single source of truth** for cinematic camera
parameters; other camera modules read from it so tuning is centralised.

Public API
----------
CinematicStabilityProfile(config=None)
  .position_spring_k  : float
  .position_spring_damp : float
  .orientation_spring_k : float
  .orientation_spring_damp : float
  .sway_amplitude     : float
  .sway_frequency_hz  : float
  .fov_salience_max_deg : float
  .fov_base_deg       : float
  .camera_inertia     : float
  .as_dict() → dict
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Optional


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


@dataclass
class CinematicStabilityProfile:
    """Locked cinematic camera parameters (§8, §12).

    Construct via :meth:`from_config` (reads ``camera.*`` and ``render.*``)
    or use the dataclass directly with defaults.

    Attributes
    ----------
    position_spring_k :
        Spring constant for position lag (critically damped recommended ≈ 25).
    position_spring_damp :
        Damping ratio for position spring (1.0 = critically damped).
    orientation_spring_k :
        Spring constant for orientation / look-at lag.
    orientation_spring_damp :
        Damping ratio for orientation spring.
    sway_amplitude :
        Peak vertical sway in world units (§8.1: лёгкий вертикальный sway).
    sway_frequency_hz :
        Breathing cycle frequency in Hz (keep < 0.5 to avoid flicker).
    fov_base_deg :
        Base field-of-view in degrees (§12: render.fov_base).
    fov_salience_max_deg :
        Maximum FOV deviation allowed from salience (§8.2; keep ≤ 3°).
    camera_inertia :
        General inertia scalar [0, 1]; 0 = instant, 1 = very heavy.
    cinematic_shake_enabled :
        Must be False after the pixel quantization pass (§8.1, §9).
    """

    position_spring_k: float = 25.0
    position_spring_damp: float = 1.0
    orientation_spring_k: float = 20.0
    orientation_spring_damp: float = 1.0
    sway_amplitude: float = 0.04
    sway_frequency_hz: float = 0.25
    fov_base_deg: float = 68.0
    fov_salience_max_deg: float = 2.5
    camera_inertia: float = 0.18
    cinematic_shake_enabled: bool = False

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: Optional[dict] = None) -> "CinematicStabilityProfile":
        """Create from a config dict (reads ``render.*`` and ``camera.*``)."""
        cfg_render = (config or {}).get("render", {}) or {}
        cfg_camera = (config or {}).get("camera", {}) or {}

        return cls(
            fov_base_deg=float(cfg_render.get("fov_base", 68.0)),
            camera_inertia=float(cfg_render.get("camera_inertia", 0.18)),
            # camera sub-keys
            position_spring_k=float(cfg_camera.get("position_spring_k", 25.0)),
            position_spring_damp=float(cfg_camera.get("position_spring_damp", 1.0)),
            orientation_spring_k=float(cfg_camera.get("orientation_spring_k", 20.0)),
            orientation_spring_damp=float(cfg_camera.get("orientation_spring_damp", 1.0)),
            sway_amplitude=float(cfg_camera.get("sway_amplitude", 0.04)),
            sway_frequency_hz=float(cfg_camera.get("sway_frequency_hz", 0.25)),
            fov_salience_max_deg=float(cfg_camera.get("fov_salience_max_deg", 2.5)),
            cinematic_shake_enabled=bool(cfg_camera.get("cinematic_shake_enabled", False)),
        )

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def sway_offset(self, time: float) -> float:
        """Return the vertical sway offset at *time* seconds.

        The sway is a simple sinusoid — deterministic, no random().
        """
        return self.sway_amplitude * math.sin(
            2.0 * math.pi * self.sway_frequency_hz * time
        )

    def fov_with_salience(self, salience_fov_bias_deg: float) -> float:
        """Return the actual FOV clamped so the pixel grid stays stable (§8.2)."""
        bias = _clamp(
            salience_fov_bias_deg,
            -self.fov_salience_max_deg,
            self.fov_salience_max_deg,
        )
        return self.fov_base_deg + bias

    def as_dict(self) -> dict:
        """Serialise all parameters to a plain dict."""
        return asdict(self)
