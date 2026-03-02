"""StateHashSync — Stage 42 hash-based network consistency layer.

The server (or a dedicated sync manager) periodically computes lightweight
hash fingerprints over authoritative subsystem state and either:

* broadcasts them so clients can self-check, or
* compares client-reported hashes and triggers graduated autocorrection.

Hash domains
------------
``hashMotorCore``    — COM position (quantised), stance, contact count
``hashDeformNearby`` — H/M deformation chunk count + material overlay hash
``hashGrasp``        — active constraint ids XOR-reduced + anchor positions
``hashAstroClimate`` — key astro/climate coefficients (quantised)

Autocorrection levels
---------------------
1. ``SIMTIME_OFFSET`` — correct simTime offset only
2. ``KEY_STATE``      — re-sync stance / contacts
3. ``SECTOR_SNAPSHOT`` — resend full baseline snapshot for the player's sector
4. ``SOFT_RESET``     — soft-reset local generative systems (audio, camera)

The :class:`StateHashSync` instance lives on the server.  Clients send their
own hashes; :meth:`check_client` returns the appropriate action.

Usage
-----
sync = StateHashSync(hash_interval_sec=5.0)

# Server frame update:
server_hashes = sync.compute_server_hashes(motor_state, deform_state,
                                           grasp_state, astro_state)
sync.record_server_snapshot(sim_time, server_hashes)

# When a client reports its hashes:
action = sync.check_client(client_id, sim_time, client_hashes)
# action.level → CorrectionLevel enum (or None if in sync)
"""
from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple

from src.core.DeterminismContract import quantise_position_mm, quantise_direction_int16
from src.core.Logger import Logger

_TAG = "StateHashSync"


# ---------------------------------------------------------------------------
# Correction levels
# ---------------------------------------------------------------------------

class CorrectionLevel(IntEnum):
    SIMTIME_OFFSET  = 1
    KEY_STATE       = 2
    SECTOR_SNAPSHOT = 3
    SOFT_RESET      = 4


@dataclass
class CorrectionAction:
    level: CorrectionLevel
    client_id: str
    sim_time: float
    detail: str = ""


# ---------------------------------------------------------------------------
# Hash helpers
# ---------------------------------------------------------------------------

def _sha256_int(data: bytes) -> int:
    """Return the first 8 bytes of SHA-256 as an unsigned 64-bit integer."""
    digest = hashlib.sha256(data).digest()
    return int.from_bytes(digest[:8], "big")


@dataclass
class StateHashes:
    """A bundle of hash fingerprints for one sim-tick snapshot."""
    sim_time: float = 0.0
    motor_core: int = 0
    deform_nearby: int = 0
    grasp: int = 0
    astro_climate: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sim_time": self.sim_time,
            "motor_core": self.motor_core,
            "deform_nearby": self.deform_nearby,
            "grasp": self.grasp,
            "astro_climate": self.astro_climate,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "StateHashes":
        return StateHashes(
            sim_time=float(d.get("sim_time", 0.0)),
            motor_core=int(d.get("motor_core", 0)),
            deform_nearby=int(d.get("deform_nearby", 0)),
            grasp=int(d.get("grasp", 0)),
            astro_climate=int(d.get("astro_climate", 0)),
        )

    def matches(self, other: "StateHashes") -> bool:
        return (
            self.motor_core   == other.motor_core
            and self.deform_nearby == other.deform_nearby
            and self.grasp         == other.grasp
            and self.astro_climate == other.astro_climate
        )


# ---------------------------------------------------------------------------
# Hash computation helpers (stateless)
# ---------------------------------------------------------------------------

def hash_motor_core(
    com_pos_m: Tuple[float, float, float],
    stance: str,
    contact_count: int,
) -> int:
    """Hash the motor/physics core state."""
    qx = quantise_position_mm(com_pos_m[0])
    qy = quantise_position_mm(com_pos_m[1])
    qz = quantise_position_mm(com_pos_m[2])
    stance_bytes = stance.encode("utf-8")
    data = struct.pack(">iii", qx, qy, qz) + struct.pack(">H", contact_count & 0xFFFF) + stance_bytes
    return _sha256_int(data)


def hash_deform_nearby(
    active_h_chunks: int,
    active_m_chunks: int,
    material_hash: int,
) -> int:
    """Hash deformation state near the player."""
    data = struct.pack(">iiQ", active_h_chunks, active_m_chunks, material_hash & 0xFFFFFFFFFFFFFFFF)
    return _sha256_int(data)


def hash_grasp(
    constraint_ids: List[int],
    anchor_positions_m: List[Tuple[float, float, float]],
) -> int:
    """Hash active grasp constraints."""
    # Sort ids for determinism
    ids_sorted = sorted(constraint_ids)
    buf = struct.pack(f">{len(ids_sorted)}Q", *[i & 0xFFFFFFFFFFFFFFFF for i in ids_sorted])
    for pos in anchor_positions_m:
        buf += struct.pack(">iii",
                           quantise_position_mm(pos[0]),
                           quantise_position_mm(pos[1]),
                           quantise_position_mm(pos[2]))
    return _sha256_int(buf)


def hash_astro_climate(
    solar_intensity: float,
    wind_speed: float,
    dust_level: float,
    temperature: float,
) -> int:
    """Hash key astro/climate coefficients (quantised to int16)."""
    qi = quantise_direction_int16(min(solar_intensity, 1.0))
    qw = quantise_direction_int16(min(wind_speed / 50.0, 1.0))
    qd = quantise_direction_int16(min(dust_level, 1.0))
    qt = quantise_direction_int16(min(max((temperature - 200.0) / 400.0, -1.0), 1.0))
    data = struct.pack(">hhhh", qi, qw, qd, qt)
    return _sha256_int(data)


# ---------------------------------------------------------------------------
# StateHashSync
# ---------------------------------------------------------------------------

class StateHashSync:
    """Server-side hash snapshot store and client drift detector."""

    def __init__(self, hash_interval_sec: float = 5.0) -> None:
        self._interval: float = hash_interval_sec
        self._server_snapshots: List[StateHashes] = []
        self._client_last: Dict[str, StateHashes] = {}
        self._next_check: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Server-side recording
    # ------------------------------------------------------------------

    def record_server_snapshot(self, sim_time: float, hashes: StateHashes) -> None:
        """Store a server-authoritative hash snapshot."""
        hashes.sim_time = sim_time
        self._server_snapshots.append(hashes)
        # Ring-buffer: keep last 60 snapshots
        if len(self._server_snapshots) > 60:
            self._server_snapshots.pop(0)

    def latest_server_hashes(self) -> Optional[StateHashes]:
        """Return the most recent server snapshot, or None."""
        return self._server_snapshots[-1] if self._server_snapshots else None

    # ------------------------------------------------------------------
    # Client check
    # ------------------------------------------------------------------

    def check_client(
        self,
        client_id: str,
        sim_time: float,
        client_hashes: StateHashes,
    ) -> Optional[CorrectionAction]:
        """Compare *client_hashes* against the server snapshot nearest in time.

        Returns a :class:`CorrectionAction` if drift is detected, else None.
        Throttles checks to at most once per ``hash_interval_sec``.
        """
        next_t = self._next_check.get(client_id, 0.0)
        if sim_time < next_t:
            return None
        self._next_check[client_id] = sim_time + self._interval

        server = self._find_snapshot_at(sim_time)
        if server is None:
            Logger.debug(_TAG, f"No server snapshot yet for t={sim_time:.2f}")
            return None

        if client_hashes.matches(server):
            Logger.debug(_TAG, f"Client '{client_id}' in sync at t={sim_time:.2f}")
            return None

        level = self._classify_drift(client_hashes, server)
        Logger.warn(
            _TAG,
            f"Client '{client_id}' drift at t={sim_time:.2f} → level {level.name}",
        )
        return CorrectionAction(level=level, client_id=client_id, sim_time=sim_time)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_snapshot_at(self, sim_time: float) -> Optional[StateHashes]:
        """Return the snapshot with sim_time closest to *sim_time*."""
        if not self._server_snapshots:
            return None
        return min(self._server_snapshots, key=lambda s: abs(s.sim_time - sim_time))

    def _classify_drift(
        self, client: StateHashes, server: StateHashes
    ) -> CorrectionLevel:
        """Choose the least-invasive correction level for the observed drift."""
        mismatches = sum([
            client.motor_core   != server.motor_core,
            client.deform_nearby != server.deform_nearby,
            client.grasp        != server.grasp,
            client.astro_climate != server.astro_climate,
        ])
        if mismatches == 1 and client.astro_climate != server.astro_climate:
            # Only ephemeral generative state differs → soft reset
            return CorrectionLevel.SOFT_RESET
        if mismatches <= 1:
            return CorrectionLevel.SIMTIME_OFFSET
        if mismatches == 2:
            return CorrectionLevel.KEY_STATE
        if mismatches == 3:
            return CorrectionLevel.SECTOR_SNAPSHOT
        return CorrectionLevel.SOFT_RESET
