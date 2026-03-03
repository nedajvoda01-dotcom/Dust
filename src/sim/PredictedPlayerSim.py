"""PredictedPlayerSim — Stage 58 client-side predicted player simulation.

Runs a lightweight forward simulation of the local player's root motion and
look rig based on the latest accepted :class:`~src.net.InputSender.InputFrame`,
so that the player's movement feels instantaneous even under network latency.

The simulation uses a simple Euler integrator that mirrors the server's motor
model closely enough for reconciliation errors to stay small.

Conventions
-----------
* ``pos`` — world-space (x, y, z) in simulation units.
* ``vel`` — world-space linear velocity (m/s).
* ``yaw`` — horizontal look angle (radians).
* ``pitch`` — vertical look angle (radians, clamped to ±π/2).
* ``speed_intent`` — normalised [0, 1] run speed request.

Public API
----------
PredictedPlayerSim(config)
    .apply_input(frame)
        Feed the latest :class:`~src.net.InputSender.InputFrame`.
    .tick(dt)
        Advance the simulation by *dt* seconds.
    .pos, .vel, .yaw, .pitch
        Current predicted state (read-only properties).
    .snap_to(pos, vel, yaw)
        Hard-reset state (called on hard-snap reconciliation).
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

from src.net.InputSender import InputFrame, dequantise_dir, dequantise_angle

_HALF_PI = math.pi / 2.0
_TWO_PI  = math.pi * 2.0

# Simple ground-movement constants (tunable via config in the future)
_MAX_SPEED      = 5.0   # m/s at speed_intent = 1.0
_ACCEL          = 20.0  # m/s² horizontal acceleration
_FRICTION       = 8.0   # m/s² horizontal friction
_GRAVITY        = 9.8   # m/s²
_GROUND_Y       = 0.0   # placeholder floor level


class PredictedPlayerSim:
    """Lightweight predicted player simulation for client-side latency hiding.

    Parameters
    ----------
    config : dict
        Full game config dict (reserved for future tuning; unused now).
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        self._pos:   Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._vel:   Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._yaw:   float = 0.0
        self._pitch: float = 0.0

        # Desired movement from latest input
        self._move_x:     float = 0.0
        self._move_z:     float = 0.0
        self._speed_intent: float = 0.0
        self._target_yaw:   float = 0.0
        self._target_pitch: float = 0.0

        # Camera yaw tracks instantly; body yaw catches up
        self._cam_yaw:   float = 0.0
        self._cam_pitch: float = 0.0

    # ------------------------------------------------------------------
    # Input application
    # ------------------------------------------------------------------

    def apply_input(self, frame: InputFrame) -> None:
        """Update desired motion from an :class:`InputFrame`."""
        mx, mz = dequantise_dir(frame.move_dir_qx, frame.move_dir_qz)
        self._move_x      = mx
        self._move_z      = mz
        self._speed_intent = frame.speed_intent
        self._target_yaw   = dequantise_angle(frame.look_yaw_q)
        self._target_pitch = dequantise_angle(frame.look_pitch_q)
        if self._target_pitch > math.pi:
            self._target_pitch -= _TWO_PI

        # Camera follows mouse immediately (look smoothing section 8.1)
        self._cam_yaw   = self._target_yaw
        self._cam_pitch = max(-_HALF_PI, min(self._target_pitch, _HALF_PI))

    # ------------------------------------------------------------------
    # Tick
    # ------------------------------------------------------------------

    def tick(self, dt: float) -> None:
        """Advance simulation by *dt* seconds."""
        # Desired world-space horizontal direction rotated by yaw
        cos_y = math.cos(self._yaw)
        sin_y = math.sin(self._yaw)
        world_x = self._move_x * cos_y - self._move_z * sin_y
        world_z = self._move_x * sin_y + self._move_z * cos_y

        target_vx = world_x * _MAX_SPEED * self._speed_intent
        target_vz = world_z * _MAX_SPEED * self._speed_intent

        vx, vy, vz = self._vel

        # Accelerate toward target
        vx += (target_vx - vx) * min(1.0, _ACCEL * dt)
        vz += (target_vz - vz) * min(1.0, _ACCEL * dt)

        # Gravity
        vy -= _GRAVITY * dt

        px, py, pz = self._pos
        px += vx * dt
        py += vy * dt
        pz += vz * dt

        # Simple floor collision
        if py < _GROUND_Y:
            py = _GROUND_Y
            vy = 0.0

        self._pos = (px, py, pz)
        self._vel = (vx, vy, vz)

        # Body yaw tracks camera yaw with a lag
        yaw_err = (self._cam_yaw - self._yaw + math.pi) % _TWO_PI - math.pi
        self._yaw += yaw_err * min(1.0, 10.0 * dt)

    # ------------------------------------------------------------------
    # Hard snap (reconciliation)
    # ------------------------------------------------------------------

    def snap_to(
        self,
        pos: Tuple[float, float, float],
        vel: Tuple[float, float, float],
        yaw: float,
    ) -> None:
        """Immediately adopt the given state (hard-snap reconciliation)."""
        self._pos = tuple(pos)  # type: ignore[assignment]
        self._vel = tuple(vel)  # type: ignore[assignment]
        self._yaw = float(yaw)

    # ------------------------------------------------------------------
    # State properties
    # ------------------------------------------------------------------

    @property
    def pos(self) -> Tuple[float, float, float]:
        return self._pos

    @property
    def vel(self) -> Tuple[float, float, float]:
        return self._vel

    @property
    def yaw(self) -> float:
        return self._yaw

    @property
    def pitch(self) -> float:
        return self._cam_pitch

    @property
    def cam_yaw(self) -> float:
        """Camera yaw — instantaneous, follows mouse directly."""
        return self._cam_yaw

    @property
    def cam_pitch(self) -> float:
        """Camera pitch — instantaneous, follows mouse directly."""
        return self._cam_pitch
