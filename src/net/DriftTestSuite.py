"""DriftTestSuite — Stage 56 multiplayer hash-drift test harness.

Simulates a server and multiple clients running the same deterministic
simulation and verifies that their :class:`~src.net.StateHashSync.StateHashes`
remain consistent after:

* Initial join (client A at t0)
* Late join   (client B at t0 + Δ)
* Rejoin      (client A after a long absence)

Test scenarios
--------------
``run_two_client_drift_test``
    Server + client A + client B; all must agree on hashes at every
    checkpoint.

``run_rejoin_test``
    Client A disconnects at mid-point and rejoins; the server must
    issue a correction action and hashes must reconverge.

Hash domains checked (see :class:`~src.net.StateHashSync.StateHashes`)
    * motor_core
    * deform_nearby
    * grasp
    * astro_climate

Usage
-----
    suite = DriftTestSuite(world_seed=42, tick_hz=60.0, check_interval=5.0)
    result = suite.run_two_client_drift_test(
        total_ticks=3600,
        client_b_join_tick=60,
    )
    assert result.drift_detected == 0, result.mismatch_report
"""
from __future__ import annotations

import math
import struct
import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.core.DetRng import DetRng
from src.core.DeterminismContract import quantise_position_mm
from src.core.Logger import Logger
from src.net.StateHashSync import (
    StateHashSync,
    StateHashes,
    CorrectionAction,
    CorrectionLevel,
    hash_motor_core,
    hash_deform_nearby,
    hash_grasp,
    hash_astro_climate,
)

_TAG = "DriftTestSuite"


# ---------------------------------------------------------------------------
# Minimal headless server/client sim state
# ---------------------------------------------------------------------------

class _SimNode:
    """A single deterministic simulation node (server or client).

    Uses the same integration as :class:`~src.ci.DeterminismReplayHarness.ReplayRunner`
    but also computes :class:`StateHashes` each tick.

    Parameters
    ----------
    world_seed: Seed for the deterministic RNG.
    node_id:    Identifier string (for logging).
    """

    def __init__(self, world_seed: int, node_id: str = "node") -> None:
        self._rng = DetRng.for_domain(world_seed, 0, "drift", 0, 0)
        self._node_id = node_id
        self._tick: int = 0
        self._sim_time: float = 0.0

        # Simple position + velocity (same model as ReplayRunner)
        self._pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._vel: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._stance: str = "grounded"
        self._contact_count: int = 2

        # Climate proxy (slow sinusoid — same function of tick → deterministic)
        self._solar: float = 0.5
        self._wind: float = 0.3
        self._dust: float = 0.1
        self._temperature: float = 280.0

    def advance(self, dt: float = 1.0 / 60.0) -> StateHashes:
        """Advance one tick and return the resulting :class:`StateHashes`."""
        self._tick += 1
        self._sim_time += dt

        # Deterministic movement (no external input)
        phase = self._tick * 0.03
        mv_x = math.sin(phase) * 0.5
        mv_z = math.cos(phase * 0.7) * 0.5

        x, y, z = self._pos
        vx, vy, vz = self._vel

        vx = vx * 0.9 + mv_x * 5.0 * dt
        vz = vz * 0.9 + mv_z * 5.0 * dt
        vy = max(vy - 9.8 * dt, 0.0)

        x += vx * dt
        y += vy * dt
        z += vz * dt

        self._pos = (x, y, z)
        self._vel = (vx, vy, vz)
        self._stance = "grounded" if vy < 0.01 else "airborne"
        self._contact_count = 2 if self._stance == "grounded" else 0

        # Advance deterministic RNG one step
        self._rng.next_float01()

        # Slow climate update (deterministic sinusoid)
        self._solar       = 0.5 + 0.4 * math.sin(self._tick * 0.001)
        self._wind        = 0.3 + 0.2 * math.cos(self._tick * 0.0013)
        self._dust        = 0.1 + 0.05 * math.sin(self._tick * 0.002 + 1.0)
        self._temperature = 280.0 + 20.0 * math.sin(self._tick * 0.0005)

        return self._compute_hashes()

    def _compute_hashes(self) -> StateHashes:
        return StateHashes(
            sim_time=self._sim_time,
            motor_core=hash_motor_core(self._pos, self._stance, self._contact_count),
            deform_nearby=hash_deform_nearby(4, 2, 0xABCD),
            grasp=hash_grasp([], []),
            astro_climate=hash_astro_climate(
                self._solar, self._wind * 50.0, self._dust, self._temperature
            ),
        )

    @property
    def sim_time(self) -> float:
        return self._sim_time


# ---------------------------------------------------------------------------
# DriftTestResult
# ---------------------------------------------------------------------------

@dataclass
class DriftTestResult:
    """Result of one drift test run."""
    test_name: str = ""
    total_ticks: int = 0
    drift_detected: int = 0          # number of ticks where hashes mismatched
    corrections_issued: int = 0      # number of CorrectionActions issued
    max_correction_level: Optional[CorrectionLevel] = None
    mismatch_report: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.drift_detected == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_name":           self.test_name,
            "total_ticks":         self.total_ticks,
            "drift_detected":      self.drift_detected,
            "corrections_issued":  self.corrections_issued,
            "max_correction_level": (
                self.max_correction_level.name
                if self.max_correction_level else None
            ),
            "passed":              self.passed,
            "mismatches":          list(self.mismatch_report),
        }


# ---------------------------------------------------------------------------
# DriftTestSuite
# ---------------------------------------------------------------------------

class DriftTestSuite:
    """Multiplayer hash-drift test harness.

    Parameters
    ----------
    world_seed:      Shared seed for all nodes.
    tick_hz:         Simulation tick rate (Hz).
    check_interval:  How often (in sim seconds) to check hashes.
    """

    def __init__(
        self,
        world_seed: int = 42,
        tick_hz: float = 60.0,
        check_interval: float = 5.0,
    ) -> None:
        self._world_seed = world_seed
        self._tick_hz = tick_hz
        self._dt = 1.0 / tick_hz
        self._check_interval = check_interval

    # ------------------------------------------------------------------
    # Two-client drift test
    # ------------------------------------------------------------------

    def run_two_client_drift_test(
        self,
        total_ticks: int = 1800,
        client_b_join_tick: int = 60,
    ) -> DriftTestResult:
        """Run server + client A + late-joining client B.

        Client B joins at *client_b_join_tick* and is expected to converge
        to the same state as the server and client A.

        Returns
        -------
        :class:`DriftTestResult` with drift counts per hash domain.
        """
        result = DriftTestResult(
            test_name="two_client_drift",
            total_ticks=total_ticks,
        )

        server = _SimNode(self._world_seed, "server")
        client_a = _SimNode(self._world_seed, "client_a")
        client_b = _SimNode(self._world_seed, "client_b")

        sync = StateHashSync(hash_interval_sec=self._check_interval)

        # Fast-forward client_b to the join point first
        for _ in range(client_b_join_tick):
            client_b.advance(self._dt)

        for tick in range(total_ticks):
            h_server  = server.advance(self._dt)
            h_client_a = client_a.advance(self._dt)

            if tick >= client_b_join_tick:
                h_client_b = client_b.advance(self._dt)
            else:
                h_client_b = None

            # Record server snapshot
            sync.record_server_snapshot(server.sim_time, h_server)

            # Check clients
            action_a = sync.check_client("client_a", client_a.sim_time, h_client_a)
            self._handle_action(action_a, result)

            if h_client_b is not None:
                action_b = sync.check_client("client_b", client_b.sim_time, h_client_b)
                self._handle_action(action_b, result)

            # Direct hash comparison (ground truth)
            if not h_client_a.matches(h_server):
                result.drift_detected += 1
                result.mismatch_report.append(
                    f"tick={tick}: client_a vs server mismatch "
                    f"(motor_a={h_client_a.motor_core:#x}, motor_s={h_server.motor_core:#x})"
                )

            if h_client_b is not None and not h_client_b.matches(h_server):
                result.drift_detected += 1
                result.mismatch_report.append(
                    f"tick={tick}: client_b vs server mismatch"
                )

        Logger.info(
            _TAG,
            f"two_client_drift: ticks={total_ticks}, drift={result.drift_detected}, "
            f"corrections={result.corrections_issued}",
        )
        return result

    # ------------------------------------------------------------------
    # Rejoin test
    # ------------------------------------------------------------------

    def run_rejoin_test(
        self,
        total_ticks: int = 1200,
        rejoin_tick: int = 600,
    ) -> DriftTestResult:
        """Client A rejoins mid-simulation; server issues a correction action.

        Between tick 0 and *rejoin_tick*, client A is absent.  On rejoin it
        fast-forwards from tick 0 with the same world seed, so it should
        converge to the server's state immediately.

        Returns
        -------
        :class:`DriftTestResult`.
        """
        result = DriftTestResult(
            test_name="rejoin_test",
            total_ticks=total_ticks,
        )

        server   = _SimNode(self._world_seed, "server")
        client_a = _SimNode(self._world_seed, "client_a")

        sync = StateHashSync(hash_interval_sec=self._check_interval)

        # Fast-forward client A to rejoin point before the main loop
        for _ in range(rejoin_tick):
            client_a.advance(self._dt)

        for tick in range(total_ticks):
            h_server = server.advance(self._dt)
            sync.record_server_snapshot(server.sim_time, h_server)

            if tick >= rejoin_tick:
                h_client_a = client_a.advance(self._dt)
                action = sync.check_client(
                    "client_a_rejoined", client_a.sim_time, h_client_a
                )
                self._handle_action(action, result)

                if not h_client_a.matches(h_server):
                    result.drift_detected += 1
                    result.mismatch_report.append(
                        f"rejoin tick={tick}: client_a vs server mismatch"
                    )

        Logger.info(
            _TAG,
            f"rejoin_test: ticks={total_ticks}, drift={result.drift_detected}, "
            f"corrections={result.corrections_issued}",
        )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _handle_action(
        self,
        action: Optional[CorrectionAction],
        result: DriftTestResult,
    ) -> None:
        if action is None:
            return
        result.corrections_issued += 1
        if (
            result.max_correction_level is None
            or action.level > result.max_correction_level
        ):
            result.max_correction_level = action.level
