"""DetRng — deterministic, step-based RNG for Stage 42.

All generative systems that must produce identical results across clients
and across re-runs use ``DetRng`` instead of ``random`` / ``Math.random()``.

Seed derivation
---------------
``seed = hash(worldSeed, playerId, systemId, tickIndex, regionId)``

Each domain (motor, audio, deform, …) obtains its own ``DetRng`` by calling
``DetRng.for_domain(world_seed, player_id, system_id, tick_index, region_id)``.
Calling :meth:`next_float01`, :meth:`next_range`, or :meth:`next_int` advances
an internal step counter — the same sequence of calls always yields the same
values for a given seed.

Usage
-----
rng = DetRng.for_domain(world_seed=42, player_id=1, system_id="audio",
                        tick_index=100, region_id=0)
v = rng.next_float01()   # → [0, 1)
"""
from __future__ import annotations

import hashlib
import struct


class DetRng:
    """Deterministic step-based RNG backed by SHA-256 counter mode."""

    def __init__(self, seed: int) -> None:
        self._seed: int = seed & 0xFFFFFFFFFFFFFFFF  # keep 64-bit
        self._step: int = 0

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @staticmethod
    def for_domain(
        world_seed: int,
        player_id: int,
        system_id: str,
        tick_index: int,
        region_id: int = 0,
    ) -> "DetRng":
        """Derive a seed from the five-tuple and return a fresh ``DetRng``."""
        raw = struct.pack(">qqq", world_seed, player_id, tick_index)
        raw += system_id.encode("utf-8")
        raw += struct.pack(">q", region_id)
        digest = hashlib.sha256(raw).digest()
        seed = int.from_bytes(digest[:8], "big")
        return DetRng(seed)

    # ------------------------------------------------------------------
    # Core generation  (counter-mode: each step hashes seed || counter)
    # ------------------------------------------------------------------

    def _next_raw(self) -> int:
        """Return next 64-bit value, advance step."""
        data = struct.pack(">QQ", self._seed, self._step)
        digest = hashlib.sha256(data).digest()
        self._step += 1
        return int.from_bytes(digest[:8], "big")

    def next_float01(self) -> float:
        """Return a float in [0, 1)."""
        raw = self._next_raw()
        return (raw >> 11) * (1.0 / (1 << 53))

    def next_range(self, a: float, b: float) -> float:
        """Return a float in [a, b)."""
        return a + self.next_float01() * (b - a)

    def next_int(self, a: int, b: int) -> int:
        """Return an int in [a, b] (inclusive)."""
        span = b - a + 1
        if span <= 0:
            return a
        raw = self._next_raw()
        return a + int(raw % span)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def step(self) -> int:
        """Current step count (number of values generated)."""
        return self._step

    @property
    def seed(self) -> int:
        return self._seed
