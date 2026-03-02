"""CameraConfig — Stage 41 camera configuration.

Config keys (section 18):
    camera.base_distance, camera.base_height, camera.risk_distance_scale,
    camera.max_fov, camera.min_fov, camera.tilt_max, camera.roll_max,
    camera.shake_k, camera.smoothing_tau, camera.macro_pullback_k
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CameraConfig:
    """Configuration for the Stage 41 Stability-Driven Cinematic Camera.

    All fields correspond to the config keys listed in section 18 of the
    Stage 41 specification.  Defaults match the recommended design values.
    """

    # --- Core geometry (section 18) ---
    base_distance: float = 4.5        # metres behind character
    base_height: float = 1.8          # metres above character position
    risk_distance_scale: float = 1.5  # max distance multiplier at globalRisk=1
    max_fov: float = 75.0             # degrees (section 11 constraint)
    min_fov: float = 60.0             # degrees (section 11 constraint)
    base_fov: float = 65.0            # neutral FOV
    tilt_max_deg: float = 5.0         # max world tilt (section 11)
    roll_max_deg: float = 5.0         # max camera roll (section 11)
    shake_k: float = 0.01             # shake amplitude scale (section 14)
    smoothing_tau: float = 0.5        # smoothing time constant (section 11: 0.3–1.0 s)
    macro_pullback_k: float = 2.0     # pullback multiplier when proximity=1 (section 9)

    # --- Framing ---
    shoulder_offset: float = 0.35     # lateral shoulder offset metres
    head_height: float = 1.6          # character eye-level above position

    # --- Attention look-aside (section 10) ---
    attention_max_deg: float = 12.0   # max look-aside degrees (section 10: 10–15°)

    # --- Spring parameters (section 13) ---
    spring_freq: float = 3.0          # natural frequency Hz
    spring_damp: float = 0.9          # damping ratio (≈1 = critically damped)
    fov_freq: float = 1.5             # FOV spring frequency Hz
    roll_freq: float = 2.0            # roll spring frequency Hz

    # --- Collision ---
    collision_radius: float = 0.25    # minimum metres above planet surface

    # --- Predictive offset (section 13) ---
    predictive_velocity_scale: float = 0.05  # look-ahead metres per (m/s)
    predictive_offset_max: float = 0.30      # hard cap on predictive offset metres

    # --- Attention look-aside internal weights (section 10) ---
    attention_rotation_weight: float = 0.30  # rotation scale applied to attn angle
    attention_position_weight: float = 0.40  # metres shifted in attention direction

    # --- Fall roll bias amplitude (section 6.2, before spring smoothing) ---
    fall_roll_bias_deg: float = 2.5   # degrees; clamped by roll_max_deg
