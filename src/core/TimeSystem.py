"""TimeSystem — real-time and game-time tracking."""
from __future__ import annotations

import time


class TimeSystem:
    """Tracks wall-clock dt, game dt, and accumulated game time."""

    def __init__(self, game_time_scale: float = 1.0) -> None:
        self.game_time_scale: float = game_time_scale
        self.real_dt: float = 0.0
        self.game_dt: float = 0.0
        self.game_time_accum: float = 0.0
        self._last: float = time.monotonic()

    def tick(self) -> None:
        now = time.monotonic()
        self.real_dt = now - self._last
        # Clamp to prevent spiral of death on very slow frames
        self.real_dt = min(self.real_dt, 0.1)
        self._last = now
        self.game_dt = self.real_dt * self.game_time_scale
        self.game_time_accum += self.game_dt

    def reset(self) -> None:
        self._last = time.monotonic()
        self.real_dt = 0.0
        self.game_dt = 0.0
        self.game_time_accum = 0.0
