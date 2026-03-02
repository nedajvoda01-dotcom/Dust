"""DeterminismReplayHarness — Stage 42 CI replay harness.

Provides infrastructure for determinism testing:

1. **Input recording** — :class:`InputRecorder` captures a stream of
   ``(tick_index, input_dict)`` tuples.

2. **Replay** — :class:`ReplayRunner` drives a headless simulation from
   a recorded input stream, returning a state hash at each tick.

3. **Hash comparison** — two independent replay runs with the same
   seed + input stream must produce identical per-tick hashes.

Typical CI usage
----------------
    recorder = InputRecorder()
    runner   = ReplayRunner(world_seed=42, tick_hz=60)

    # Record 60 seconds of inputs
    for tick in range(3600):
        recorder.record(tick, {"move_x": 0.1, "jump": False})

    # Run twice, compare
    hashes_a = runner.run(recorder.stream())
    hashes_b = runner.run(recorder.stream())
    assert hashes_a == hashes_b, "Determinism broken!"
"""
from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Sequence, Tuple

from src.core.DetRng import DetRng
from src.core.DeterminismContract import (
    TICK_DT, sort_events, quantise_position_mm,
)
from src.core.Logger import Logger

_TAG = "DetReplay"


# ---------------------------------------------------------------------------
# Input types
# ---------------------------------------------------------------------------

@dataclass
class InputEvent:
    """A single deterministic input event for one tick."""
    tick_index: int
    entity_id: int = 0
    payload: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# InputRecorder
# ---------------------------------------------------------------------------

class InputRecorder:
    """Records a sequence of :class:`InputEvent` objects."""

    def __init__(self) -> None:
        self._events: List[InputEvent] = []

    def record(self, tick_index: int, payload: Dict[str, Any], entity_id: int = 0) -> None:
        """Append one input event."""
        self._events.append(InputEvent(tick_index=tick_index, entity_id=entity_id, payload=payload))

    def stream(self) -> List[InputEvent]:
        """Return the recorded events sorted deterministically."""
        return sort_events(list(self._events), tick_key="tick_index", id_key="entity_id")

    def clear(self) -> None:
        self._events.clear()

    def __len__(self) -> int:
        return len(self._events)


# ---------------------------------------------------------------------------
# Minimal headless sim state (used by ReplayRunner)
# ---------------------------------------------------------------------------

@dataclass
class _SimState:
    """Minimal simulation state for determinism hashing."""
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    stance: str = "grounded"
    rng_step: int = 0


def _hash_state(state: _SimState, tick_index: int) -> int:
    """Return a 64-bit hash of the sim state at *tick_index*."""
    qx = quantise_position_mm(state.position[0])
    qy = quantise_position_mm(state.position[1])
    qz = quantise_position_mm(state.position[2])
    stance_b = state.stance.encode("utf-8")
    data = struct.pack(">iiiQQ", qx, qy, qz, tick_index, state.rng_step) + stance_b
    digest = hashlib.sha256(data).digest()
    return int.from_bytes(digest[:8], "big")


# ---------------------------------------------------------------------------
# ReplayRunner
# ---------------------------------------------------------------------------

class ReplayRunner:
    """Runs a minimal headless simulation from a recorded input stream.

    The simulation is intentionally simple (deterministic integration of
    player inputs) so that the harness itself stays dependency-free and fast.
    Real subsystems wire in their own hash contributions via
    :meth:`register_hash_hook`.

    Parameters
    ----------
    world_seed: Master seed for DetRng.
    tick_hz:    Simulation tick rate (default 60 Hz).
    """

    def __init__(self, world_seed: int = 42, tick_hz: float = 60.0) -> None:
        self._world_seed = world_seed
        self._tick_hz = tick_hz
        self._dt = 1.0 / tick_hz
        self._hash_hooks: List[Any] = []

    def register_hash_hook(self, hook: Any) -> None:
        """Register a callable ``hook(state, tick_index) -> int`` that
        contributes an extra hash component to each tick's fingerprint."""
        self._hash_hooks.append(hook)

    def run(self, events: List[InputEvent]) -> List[int]:
        """Run the replay and return the per-tick state hash list.

        Parameters
        ----------
        events: Sorted list of input events (from ``InputRecorder.stream()``).

        Returns
        -------
        List of one 64-bit integer hash per tick (length = max tick index + 1).
        """
        if not events:
            return []

        max_tick = max(e.tick_index for e in events)
        # Build a tick → events map
        tick_map: Dict[int, List[InputEvent]] = {}
        for ev in events:
            tick_map.setdefault(ev.tick_index, []).append(ev)

        state = _SimState()
        rng = DetRng.for_domain(self._world_seed, 0, "replay", 0, 0)
        hashes: List[int] = []

        for tick_idx in range(max_tick + 1):
            tick_events = tick_map.get(tick_idx, [])
            # Apply inputs deterministically
            for ev in tick_events:
                mv_x = float(ev.payload.get("move_x", 0.0))
                mv_z = float(ev.payload.get("move_z", 0.0))
                jump = bool(ev.payload.get("jump", False))

                x, y, z = state.position
                vx, vy, vz = state.velocity

                # Simple Euler integration (deterministic)
                vx = vx * 0.9 + mv_x * 5.0 * self._dt
                vz = vz * 0.9 + mv_z * 5.0 * self._dt
                if jump:
                    vy = 5.0
                else:
                    vy = max(vy - 9.8 * self._dt, 0.0)

                x += vx * self._dt
                y += vy * self._dt
                z += vz * self._dt

                state.position = (x, y, z)
                state.velocity = (vx, vy, vz)
                state.stance = "airborne" if vy > 0.01 else "grounded"

            # Advance RNG one step per tick (deterministic)
            rng.next_float01()
            state.rng_step = rng.step

            # Base state hash
            h = _hash_state(state, tick_idx)

            # Mix in hook contributions
            for hook in self._hash_hooks:
                try:
                    extra = int(hook(state, tick_idx))
                    h = _sha256_mix(h, extra)
                except Exception as exc:  # noqa: BLE001
                    hook_name = getattr(hook, "__name__", repr(hook))
                    Logger.error(_TAG, f"Hash hook '{hook_name}' error at tick {tick_idx}: {exc}")

            hashes.append(h)

        return hashes


def _sha256_mix(a: int, b: int) -> int:
    """Mix two 64-bit hash values deterministically."""
    data = struct.pack(">QQ", a & 0xFFFFFFFFFFFFFFFF, b & 0xFFFFFFFFFFFFFFFF)
    digest = hashlib.sha256(data).digest()
    return int.from_bytes(digest[:8], "big")
