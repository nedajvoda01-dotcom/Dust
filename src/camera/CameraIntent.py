"""CameraIntent — Stage 41 desired camera parameters (section 4).

CameraIntent is computed each tick by CinematicBias and then smoothed and
applied by StabilityCameraController.

Fields (section 4):
    targetOffset      → distance_scale + height_bias
    distanceScale     → distance_scale
    heightBias        → height_bias
    focusDir          → focus_dir
    shakeLevel        → shake_level
    tiltBias          → tilt_bias
    fovBias           → fov_bias
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.math.Vec3 import Vec3


@dataclass
class CameraIntent:
    """Desired camera parameters produced by CinematicBias each tick.

    CameraController smooths every field through spring integrators and
    enforces the cinematic constraints from section 11 before applying.
    """

    distance_scale: float = 1.0
    """Multiplier on ``base_distance``.  1 = neutral; < 1 = closer; > 1 = pulled back."""

    height_bias: float = 0.0
    """Additive metres relative to ``base_height``."""

    fov_bias: float = 0.0
    """Additive degrees relative to ``base_fov``.  Clamped to [min_fov, max_fov]."""

    roll_bias: float = 0.0
    """Degrees of camera roll around the forward axis (section 6.2: 1–3°)."""

    tilt_bias: float = 0.0
    """Degrees of world tilt toward a macro-event epicenter (section 9.1)."""

    shake_level: float = 0.0
    """Shake amplitude request 0–1 (section 14)."""

    lateral_sway: float = 0.0
    """Metres of lateral offset in wind-pressure direction (section 7)."""

    attention_offset_deg: float = 0.0
    """Degrees to shift look-at toward attention target (section 10, max 10–15°)."""

    focus_dir: Vec3 = field(default_factory=Vec3.zero)
    """World-space direction bias for the look-at point."""
