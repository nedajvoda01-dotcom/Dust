"""WorldClock — Stage 16 multi-scale time management.

Manages three time streams for the simulation:
  * real_time  — wall-clock seconds (OS time, for profiling)
  * game_time  — player-visible world time (real_time * time_scale, clamped)
  * sim_time   — simulation accumulator (advances with game_time; fixed-step
                 sub-division is handled by SimulationScheduler)

Dev controls (pause / step) affect game_time and sim_time only; real_time
always advances.

Usage
-----
clock = WorldClock(time_scale=1.0, max_frame_dt_clamp=0.1)
clock.tick(real_dt)   # call once per frame with OS-measured frame dt
dt = clock.game_dt    # scaled, clamped dt ready for this frame
"""
from __future__ import annotations


class WorldClock:
    """Multi-scale clock for Dust simulation.

    Parameters
    ----------
    time_scale:
        Ratio of game seconds to real seconds (e.g. 5.0 → 5× faster).
        Controlled via ``time.scale`` in CONFIG_DEFAULTS.json.
    max_frame_dt_clamp:
        Maximum real-seconds per frame before clamping.  Prevents the
        "spiral of death" when a frame takes unexpectedly long.
        Controlled via ``time.max_frame_dt_clamp`` in CONFIG_DEFAULTS.json.
    """

    def __init__(
        self,
        time_scale: float = 1.0,
        max_frame_dt_clamp: float = 0.1,
    ) -> None:
        self.time_scale: float = time_scale
        self.max_frame_dt_clamp: float = max_frame_dt_clamp

        # Accumulated totals
        self.real_time: float = 0.0
        self.game_time: float = 0.0
        self.sim_time: float = 0.0

        # Per-tick deltas (set by tick())
        self.real_dt: float = 0.0
        self.game_dt: float = 0.0

        # Dev controls
        self._paused: bool = False
        self._pending_step: bool = False

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def tick(self, real_dt: float) -> None:
        """Advance the clock by *real_dt* wall-clock seconds.

        * Clamps *real_dt* to ``max_frame_dt_clamp`` to prevent instability.
        * When paused, ``game_dt`` and ``sim_time`` do **not** advance unless
          a ``step()`` was requested.
        """
        clamped = min(real_dt, self.max_frame_dt_clamp)
        self.real_dt = clamped
        self.real_time += clamped

        if self._paused and not self._pending_step:
            self.game_dt = 0.0
            return

        self.game_dt = clamped * self.time_scale
        self.game_time += self.game_dt
        self.sim_time += self.game_dt
        self._pending_step = False

    # ------------------------------------------------------------------
    # Dev controls (pause / step)
    # ------------------------------------------------------------------

    def pause(self) -> None:
        """Freeze game_time / sim_time advancement (dev only)."""
        self._paused = True

    def resume(self) -> None:
        """Resume normal time advancement."""
        self._paused = False
        self._pending_step = False

    def step(self) -> None:
        """Advance exactly one frame even while paused (dev only).

        Must be called *before* the next ``tick()`` to take effect.
        """
        self._pending_step = True

    @property
    def is_paused(self) -> bool:
        """True when the clock is in paused state."""
        return self._paused
