"""DetRng — Stage 42 §2.2  Deterministic seeded RNG.

Provides a fully deterministic pseudo-random stream whose seed is derived
from a structured key:

    seed = hash(worldSeed, playerId, systemId, tickIndex, regionId)

Only **step-based** generation is allowed.  Callers must not use
``random.random()``, ``time``-seeded sources, or any other non-deterministic
entropy inside deterministic systems.

Public API
----------
DetRng(world_seed, player_id, system_id, tick_index, region_id)
  .rand_float01()      → float in [0, 1)
  .rand_range(a, b)    → float in [a, b)
  .rand_int(a, b)      → int in [a, b] inclusive
  .rand_unit_vec3()    → (float, float, float) unit vector
  .fork(sub_id)        → new DetRng with sub_id mixed into the seed
  .seed               → int (derived seed)
"""
from __future__ import annotations

import hashlib
import math
import random
import struct
from typing import Tuple


def _derive_seed(
    world_seed: int,
    player_id: int,
    system_id: str,
    tick_index: int,
    region_id: int,
) -> int:
    """Deterministically derive a 64-bit seed from the five components."""
    blob = struct.pack(">qqq", world_seed, player_id, tick_index)
    blob += struct.pack(">q", region_id)
    blob += system_id.encode("utf-8")
    digest = hashlib.sha256(blob).digest()
    return int.from_bytes(digest[:8], "big")


class DetRng:
    """Deterministic step-based RNG stream.

    Two ``DetRng`` objects with identical construction arguments will
    produce identical sequences regardless of platform.

    Parameters
    ----------
    world_seed:
        Global world seed.
    player_id:
        Numeric player (or entity) identifier.  Use ``0`` for world-global
        streams not tied to a specific player.
    system_id:
        Unique name string identifying the subsystem (e.g. ``"audio"``).
    tick_index:
        Current simulation tick counter.
    region_id:
        Sector / region index; use ``0`` for world-global streams.
    """

    def __init__(
        self,
        world_seed: int = 42,
        player_id: int = 0,
        system_id: str = "default",
        tick_index: int = 0,
        region_id: int = 0,
    ) -> None:
        seed = _derive_seed(world_seed, player_id, system_id, tick_index, region_id)
        self._rng = random.Random(seed)
        self._seed = seed

    # ------------------------------------------------------------------
    # Core generators
    # ------------------------------------------------------------------

    def rand_float01(self) -> float:
        """Return a float uniformly distributed in ``[0, 1)``."""
        return self._rng.random()

    def rand_range(self, a: float, b: float) -> float:
        """Return a float uniformly distributed in ``[a, b)``."""
        return self._rng.uniform(a, b)

    def rand_int(self, a: int, b: int) -> int:
        """Return an int uniformly distributed in ``[a, b]`` (inclusive)."""
        return self._rng.randint(a, b)

    def rand_unit_vec3(self) -> Tuple[float, float, float]:
        """Return a uniformly distributed unit vector on the sphere."""
        for _ in range(100):
            x = self._rng.uniform(-1.0, 1.0)
            y = self._rng.uniform(-1.0, 1.0)
            z = self._rng.uniform(-1.0, 1.0)
            r2 = x * x + y * y + z * z
            if 1e-10 < r2 < 1.0:
                r = math.sqrt(r2)
                return (x / r, y / r, z / r)
        return (1.0, 0.0, 0.0)  # fallback (never reached in practice)

    # ------------------------------------------------------------------
    # Fork for sub-streams
    # ------------------------------------------------------------------

    def fork(self, sub_id: str) -> "DetRng":
        """Return a new ``DetRng`` with ``sub_id`` mixed into the seed."""
        raw = struct.pack(">q", self._seed) + sub_id.encode("utf-8")
        digest = hashlib.sha256(raw).digest()
        child_seed = int.from_bytes(digest[:8], "big")
        inst = object.__new__(DetRng)
        inst._rng = random.Random(child_seed)
        inst._seed = child_seed
        return inst

    # ------------------------------------------------------------------
    # Seed introspection
    # ------------------------------------------------------------------

    @property
    def seed(self) -> int:
        """The derived 64-bit seed used to initialise this stream."""
        return self._seed
