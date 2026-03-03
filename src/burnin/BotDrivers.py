"""BotDrivers — Stage 56 deterministic simulation load-generators.

Each bot is a seeded, stateless-per-tick state machine that returns a
``BotInput`` each tick.  Bots are NOT AI characters; they exist solely to
stress-test materials, physics, and memory systems over long runs.

Bot types
---------
* :class:`WalkerBot`   — walks along a heading, turns, occasionally stops.
* :class:`SlopeBot`    — repeatedly climbs/descends a fixed slope heading.
* :class:`BuddyBot`    — walks close to another position, occasionally grasps.
* :class:`ShelterBot`  — moves away from high wind-load toward a shelter point.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

from src.core.DetRng import DetRng
from src.core.Logger import Logger

_TAG = "BotDrivers"


# ---------------------------------------------------------------------------
# Shared input type
# ---------------------------------------------------------------------------

@dataclass
class BotInput:
    """Normalised per-tick input produced by a bot driver."""
    move_x: float = 0.0          # lateral  (-1..1)
    move_z: float = 0.0          # forward  (-1..1)
    jump: bool    = False
    grasp: bool   = False
    shelter_seek: bool = False    # ShelterBot: actively seeking shelter

    @property
    def is_moving(self) -> bool:
        return abs(self.move_x) > 0.01 or abs(self.move_z) > 0.01


# ---------------------------------------------------------------------------
# WalkerBot
# ---------------------------------------------------------------------------

class WalkerBot:
    """Walks straight, turns at random intervals, occasionally stops.

    Parameters
    ----------
    seed:        Deterministic seed.
    turn_every:  Average ticks between direction changes.
    stop_chance: Probability [0, 1] of stopping for one tick.
    """

    def __init__(
        self,
        seed: int = 0,
        turn_every: int = 120,
        stop_chance: float = 0.02,
    ) -> None:
        self._rng = DetRng(seed)
        self._turn_every = max(1, turn_every)
        self._stop_chance = stop_chance
        self._heading: float = 0.0   # radians
        self._ticks_since_turn: int = 0

    def tick(self) -> BotInput:
        self._ticks_since_turn += 1
        # Random turn
        if self._ticks_since_turn >= self._turn_every:
            self._heading += self._rng.next_range(-math.pi, math.pi) * 0.5
            self._ticks_since_turn = 0

        # Random stop
        if self._rng.next_float01() < self._stop_chance:
            return BotInput()

        return BotInput(
            move_x=math.sin(self._heading),
            move_z=math.cos(self._heading),
        )


# ---------------------------------------------------------------------------
# SlopeBot
# ---------------------------------------------------------------------------

class SlopeBot:
    """Repeatedly climbs and descends a fixed slope heading.

    Parameters
    ----------
    seed:          Deterministic seed.
    climb_ticks:   How many ticks to walk up before reversing.
    """

    def __init__(self, seed: int = 1, climb_ticks: int = 200) -> None:
        self._rng = DetRng(seed)
        self._climb_ticks = max(1, climb_ticks)
        self._phase: int = 0          # 0 = up, 1 = down
        self._elapsed: int = 0
        self._heading: float = self._rng.next_range(0.0, math.pi * 2.0)

    def tick(self) -> BotInput:
        self._elapsed += 1
        if self._elapsed >= self._climb_ticks:
            self._elapsed = 0
            self._phase = 1 - self._phase
            # Vary heading slightly each pass
            self._heading += self._rng.next_range(-0.3, 0.3)

        direction = 1.0 if self._phase == 0 else -1.0
        return BotInput(
            move_x=math.sin(self._heading) * direction,
            move_z=math.cos(self._heading) * direction,
        )


# ---------------------------------------------------------------------------
# BuddyBot
# ---------------------------------------------------------------------------

class BuddyBot:
    """Walks near a companion position; occasionally initiates grasp.

    Parameters
    ----------
    seed:          Deterministic seed.
    grasp_chance:  Per-tick probability of initiating a grasp action.
    """

    def __init__(
        self,
        seed: int = 2,
        grasp_chance: float = 0.005,
    ) -> None:
        self._rng = DetRng(seed)
        self._grasp_chance = grasp_chance
        self._heading: float = 0.0
        self._turn_counter: int = 0

    def tick(self, companion_pos: Optional[Tuple[float, float, float]] = None) -> BotInput:
        self._turn_counter += 1
        if self._turn_counter % 90 == 0:
            self._heading += self._rng.next_range(-0.8, 0.8)

        grasp = self._rng.next_float01() < self._grasp_chance

        return BotInput(
            move_x=math.sin(self._heading) * 0.6,
            move_z=math.cos(self._heading) * 0.6,
            grasp=grasp,
        )


# ---------------------------------------------------------------------------
# ShelterBot
# ---------------------------------------------------------------------------

class ShelterBot:
    """Moves toward a shelter point when wind_load exceeds threshold.

    Parameters
    ----------
    seed:             Deterministic seed.
    wind_threshold:   wind_load above which the bot seeks shelter (0..1).
    shelter_pos:      World-space destination considered "safe".
    """

    def __init__(
        self,
        seed: int = 3,
        wind_threshold: float = 0.6,
        shelter_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        self._rng = DetRng(seed)
        self._wind_threshold = wind_threshold
        self._shelter_pos = shelter_pos
        self._wander_heading: float = 0.0
        self._wander_counter: int = 0

    def tick(
        self,
        current_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        wind_load: float = 0.0,
    ) -> BotInput:
        seeking = wind_load > self._wind_threshold

        if seeking:
            # Move toward shelter
            dx = self._shelter_pos[0] - current_pos[0]
            dz = self._shelter_pos[2] - current_pos[2]
            dist = math.sqrt(dx * dx + dz * dz)
            if dist > 0.1:
                return BotInput(
                    move_x=dx / dist,
                    move_z=dz / dist,
                    shelter_seek=True,
                )
            return BotInput(shelter_seek=True)

        # Wander normally
        self._wander_counter += 1
        if self._wander_counter % 150 == 0:
            self._wander_heading += self._rng.next_range(-1.2, 1.2)

        return BotInput(
            move_x=math.sin(self._wander_heading) * 0.8,
            move_z=math.cos(self._wander_heading) * 0.8,
        )
