"""RngSystem — deterministic RNG with named sub-streams."""
from __future__ import annotations

import random
import struct
import hashlib


class RngStream:
    """Independent RNG stream derived from a base seed and a stream name."""

    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def rand_float01(self) -> float:
        return self._rng.random()

    def rand_range(self, a: float, b: float) -> float:
        return self._rng.uniform(a, b)

    def rand_int(self, a: int, b: int) -> int:
        return self._rng.randint(a, b)


class RngSystem:
    """Single global RNG plus named sub-streams. All randomness goes here."""

    def __init__(self, seed: int = 42) -> None:
        self._seed: int = seed
        self._streams: dict[str, RngStream] = {}
        self._global = RngStream(seed)

    def set_seed(self, seed: int) -> None:
        self._seed = seed
        self._streams.clear()
        self._global = RngStream(seed)

    @property
    def seed(self) -> int:
        return self._seed

    # --- Global stream ---
    def rand_float01(self) -> float:
        return self._global.rand_float01()

    def rand_range(self, a: float, b: float) -> float:
        return self._global.rand_range(a, b)

    def rand_int(self, a: int, b: int) -> int:
        return self._global.rand_int(a, b)

    # --- Named sub-streams ---
    def get_stream(self, name: str) -> RngStream:
        if name not in self._streams:
            # Derive a deterministic seed from base seed + stream name
            raw = struct.pack(">q", self._seed) + name.encode("utf-8")
            digest = hashlib.sha256(raw).digest()
            stream_seed = int.from_bytes(digest[:8], "big")
            self._streams[name] = RngStream(stream_seed)
        return self._streams[name]
