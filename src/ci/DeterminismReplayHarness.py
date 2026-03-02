"""DeterminismReplayHarness — Stage 42 §8.  CI determinism test harness.

Records a stream of simulation inputs, replays it twice, and asserts that
the resulting state hashes are identical.  Also supports multiplayer
join-sync hash checks.

Public API
----------
InputEvent(tick, entity_id, kind, payload)

DeterminismReplayHarness(world_seed, config_dict)
  .record_event(tick, entity_id, kind, payload)
  .clear_events()
  .run_replay(n_ticks)                            → (hash_run1, hash_run2)
  .assert_deterministic(n_ticks)
  .run_join_sync_test(join_tick, total_ticks)     → (server_hash, client_hash)
  .assert_join_sync(join_tick, total_ticks)
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.core.DetRng import DetRng
from src.core.DeterminismContract import DeterminismContract


@dataclass
class InputEvent:
    """A single recorded simulation input."""
    tick: int
    entity_id: int
    kind: str
    payload: Any = None


class _SimState:
    """Minimal deterministic simulation state for hash testing."""

    def __init__(self, world_seed: int) -> None:
        self.world_seed = world_seed
        self.tick: int = 0
        self._accum: int = 0

    def apply_event(self, event: InputEvent) -> None:
        rng = DetRng(
            world_seed=self.world_seed,
            player_id=event.entity_id,
            system_id=event.kind,
            tick_index=event.tick,
            region_id=0,
        )
        sample = int(rng.rand_float01() * 1_000_000)
        self._accum = (self._accum * 31 + sample) & 0xFFFFFFFF

    def step(self, tick: int) -> None:
        self.tick = tick

    def state_hash(self) -> int:
        buf = struct.pack(">II", self.tick, self._accum)
        h = 0x811C9DC5
        for b in buf:
            h ^= b
            h = (h * 0x01000193) & 0xFFFFFFFF
        return h


class DeterminismReplayHarness:
    """Records inputs, replays twice, and asserts hash equality.

    Parameters
    ----------
    world_seed:
        Seed for the simulation under test.
    config:
        Optional config dict.
    """

    def __init__(
        self,
        world_seed: int = 42,
        config: Optional[Dict] = None,
    ) -> None:
        self.world_seed = world_seed
        self._events: List[InputEvent] = []

    def record_event(
        self,
        tick: int,
        entity_id: int,
        kind: str,
        payload: Any = None,
    ) -> None:
        """Record a single input event for replay."""
        self._events.append(InputEvent(tick=tick, entity_id=entity_id,
                                       kind=kind, payload=payload))

    def clear_events(self) -> None:
        """Discard all recorded events."""
        self._events.clear()

    # ------------------------------------------------------------------
    # Replay
    # ------------------------------------------------------------------

    def _build_sorted_queue(self) -> List[InputEvent]:
        wrapped = [{"tick_index": e.tick, "entity_id": e.entity_id, "_ev": e}
                   for e in self._events]
        return [item["_ev"] for item in DeterminismContract.sort_events(wrapped)]

    def _replay_once(self, n_ticks: int) -> int:
        state = _SimState(self.world_seed)
        event_queue = self._build_sorted_queue()
        ev_idx = 0
        for tick in range(n_ticks):
            state.step(tick)
            while ev_idx < len(event_queue) and event_queue[ev_idx].tick == tick:
                state.apply_event(event_queue[ev_idx])
                ev_idx += 1
        return state.state_hash()

    def run_replay(self, n_ticks: int = 3600) -> Tuple[int, int]:
        """Run the recorded input stream twice and return both hashes."""
        h1 = self._replay_once(n_ticks)
        h2 = self._replay_once(n_ticks)
        return h1, h2

    def assert_deterministic(self, n_ticks: int = 3600) -> None:
        """Replay twice and raise ``AssertionError`` if hashes differ."""
        h1, h2 = self.run_replay(n_ticks)
        if h1 != h2:
            raise AssertionError(
                f"Determinism violation: run1={h1:#010x} run2={h2:#010x}"
            )

    # ------------------------------------------------------------------
    # Multiplayer join-sync test
    # ------------------------------------------------------------------

    def run_join_sync_test(
        self,
        join_tick: int = 60,
        total_ticks: int = 180,
    ) -> Tuple[int, int]:
        """Simulate a late-joining client and compare hashes with server.

        Server runs all *total_ticks*.  Client starts from a baseline
        snapshot taken at *join_tick* and replays from *join_tick+1*.

        Returns (server_hash, client_hash) — should be equal.
        """
        event_queue = self._build_sorted_queue()

        # --- Server run ---
        server_state = _SimState(self.world_seed)
        ev_idx = 0
        snapshot_accum: int = 0
        snapshot_tick: int = 0

        for tick in range(total_ticks):
            server_state.step(tick)
            while ev_idx < len(event_queue) and event_queue[ev_idx].tick == tick:
                server_state.apply_event(event_queue[ev_idx])
                ev_idx += 1
            if tick == join_tick:
                snapshot_accum = server_state._accum
                snapshot_tick = tick

        server_hash = server_state.state_hash()

        # --- Client run (starts from snapshot) ---
        client_state = _SimState(self.world_seed)
        client_state._accum = snapshot_accum
        client_state.tick = snapshot_tick

        # Advance past events up to and including join_tick
        ev_idx_c = 0
        while ev_idx_c < len(event_queue) and event_queue[ev_idx_c].tick <= join_tick:
            ev_idx_c += 1

        for tick in range(join_tick + 1, total_ticks):
            client_state.step(tick)
            while (ev_idx_c < len(event_queue)
                   and event_queue[ev_idx_c].tick == tick):
                client_state.apply_event(event_queue[ev_idx_c])
                ev_idx_c += 1

        client_hash = client_state.state_hash()
        return server_hash, client_hash

    def assert_join_sync(
        self,
        join_tick: int = 60,
        total_ticks: int = 180,
    ) -> None:
        """Assert that a late-joining client reaches the same hash as server."""
        sh, ch = self.run_join_sync_test(join_tick, total_ticks)
        if sh != ch:
            raise AssertionError(
                f"Join-sync mismatch: server={sh:#010x} client={ch:#010x}"
            )
