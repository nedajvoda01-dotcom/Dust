"""DeformNetReplicator — Stage 35 interest-based network deformation sync.

Aggregates ContactSamples from clients into DeformStamps and broadcasts
them to nearby players at a controlled rate.

Architecture
------------
* Client → Server: compact "ContactSample summary" at ``contact_send_hz`` (5–10 Hz).
* Server → Clients: ``DeformStampBatch`` at ``stamp_broadcast_hz`` (2–5 Hz),
  interest-based (only to players within ``interest_radius_m``).

DeformNetReplicator
    Server-side: accumulates pending stamps, broadcasts on timer.
    Client-side: receives batches and applies to local DeformationIntegrator.

All synchronisation is deterministic given the same tick_index ordering.

Public API (server)
-------------------
DeformNetReplicator.ingest_contact_summary(player_id, samples, sim_time)
DeformNetReplicator.tick(dt, sim_time, player_positions) -> Dict[player_id, bytes]

Public API (client)
-------------------
DeformNetReplicator.apply_received_batch(data, integrator, dt)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.physics.MaterialYieldModel import MaterialClass, MaterialYieldModel
from src.surface.DeformationField import ContactSample, DeformationField
from src.surface.DeformStampCodec import (
    DeformStamp,
    DeformStampBatch,
    DeformStampCodec,
)


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


# ---------------------------------------------------------------------------
# DeformNetReplicator
# ---------------------------------------------------------------------------

class DeformNetReplicator:
    """Manages deformation network replication.

    Parameters
    ----------
    config:
        Optional Config with ``deform.net.*`` keys.
    is_server:
        True when running on the authoritative server.
    """

    def __init__(self, config=None, is_server: bool = True) -> None:
        self._cfg       = config
        self._is_server = is_server

        self._contact_send_hz:    float = self._get("net.contact_send_hz",    5.0)
        self._stamp_broadcast_hz: float = self._get("net.stamp_broadcast_hz", 2.0)
        self._interest_radius_m:  float = self._get("net.interest_radius_m", 200.0)

        # Metres per degree — derived from planet radius so the interest
        # culling works on non-Earth planets.  Default uses a Mars-like radius.
        planet_radius_m: float = self._get("planet_radius_m", 3_390_000.0)
        import math as _math
        self._m_per_deg: float = (2.0 * _math.pi * planet_radius_m) / 360.0

        # Server state
        self._pending_stamps: List[DeformStamp]                   = []
        self._broadcast_timer: float                               = 0.0
        self._player_contacts: Dict[str, List[ContactSample]]     = {}

        # Ring-buffer: last N seconds of stamps per sector for rejoin
        self._stamp_history: List[Tuple[float, DeformStamp]]      = []
        self._history_window_sec: float                            = self._get(
            "net.history_window_sec", 120.0)

    # ------------------------------------------------------------------
    # Server: ingest
    # ------------------------------------------------------------------

    def ingest_contact_summary(
        self,
        player_id: str,
        samples: List[ContactSample],
        sim_time: float,
    ) -> None:
        """Accept a batch of ContactSamples from a client (server only)."""
        if not self._is_server:
            return
        # Aggregate: keep only the last batch per player
        self._player_contacts[player_id] = list(samples)

    # ------------------------------------------------------------------
    # Server: tick → produce outbound batches
    # ------------------------------------------------------------------

    def tick(
        self,
        dt: float,
        sim_time: float,
        player_positions: Dict[str, Tuple[float, float]],
    ) -> Dict[str, bytes]:
        """Advance the replicator; return per-player encoded batches.

        Parameters
        ----------
        dt:
            Elapsed time [s].
        sim_time:
            Current simulation time [s] (for history pruning).
        player_positions:
            Map of player_id → (lat, lon) for interest culling.

        Returns
        -------
        Dict[player_id, bytes]
            Encoded DeformStampBatch for each player that has nearby stamps.
            Empty dict when nothing to send.
        """
        if not self._is_server:
            return {}

        self._broadcast_timer += dt

        # Convert accumulated contacts to stamps
        self._digest_pending_contacts(sim_time)

        broadcast_interval = 1.0 / max(self._stamp_broadcast_hz, 0.1)
        if self._broadcast_timer < broadcast_interval:
            return {}

        self._broadcast_timer = 0.0

        if not self._pending_stamps:
            return {}

        # Prune history
        cutoff = sim_time - self._history_window_sec
        self._stamp_history = [
            (t, s) for (t, s) in self._stamp_history if t >= cutoff
        ]

        # Add pending to history
        for s in self._pending_stamps:
            self._stamp_history.append((sim_time, s))

        # Per-player interest culling
        outbound: Dict[str, bytes] = {}
        for pid, ppos in player_positions.items():
            nearby = self._filter_by_interest(self._pending_stamps, ppos)
            if nearby:
                batch = DeformStampBatch(stamps=nearby)
                outbound[pid] = DeformStampCodec.encode_batch(batch)

        self._pending_stamps.clear()
        return outbound

    # ------------------------------------------------------------------
    # Client: apply received bytes
    # ------------------------------------------------------------------

    @staticmethod
    def apply_received_batch(data: bytes, integrator, dt: float) -> int:
        """Decode *data* and apply stamps to *integrator*.

        Parameters
        ----------
        integrator:
            DeformationIntegrator instance on the client.
        dt:
            Elapsed time to use when applying the stamps.

        Returns
        -------
        int
            Number of stamps applied.
        """
        stamps = DeformStampCodec.decode_batch(data)
        if not stamps:
            return 0

        from src.physics.MaterialYieldModel import MaterialYieldModel
        model = MaterialYieldModel()

        count = 0
        for s in stamps:
            params = model.get(s.material)
            grid_res = getattr(integrator, '_grid_res', 64)
            centre   = grid_res // 2
            # Convert stamp to ContactSample for integrator
            sample = ContactSample(
                world_ix   = centre,
                world_iy   = centre,
                fn         = abs(s.depth_m) * params.yield_strength * 2.0,
                ft_x       = s.push_dir_x * s.push_amount * 10.0,
                ft_y       = s.push_dir_y * s.push_amount * 10.0,
                v_rel_x    = s.push_dir_x * s.push_amount * 0.5,
                v_rel_y    = s.push_dir_y * s.push_amount * 0.5,
                area       = math.pi * (max(s.radius_m, 0.1) ** 2) * 0.01,
                material   = s.material,
                tick_index = s.tick_index,
            )
            chunk_id = (int(s.lat * 10), int(s.lon * 10))
            f        = integrator.get_field(chunk_id)
            f.apply_contact_sample(sample, params, dt)
            count += 1

        return count

    def get_history_for_rejoin(
        self,
        lat: float,
        lon: float,
    ) -> bytes:
        """Return encoded recent stamp history near *lat/lon* for a rejoining player."""
        nearby = self._filter_by_interest(
            [s for (_, s) in self._stamp_history],
            (lat, lon),
        )
        if not nearby:
            return b""
        return DeformStampCodec.encode_batch(DeformStampBatch(stamps=nearby))

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _digest_pending_contacts(self, sim_time: float) -> None:
        """Convert pending player contacts into aggregated stamps."""
        # Import here to avoid circular dependency
        tick_index = int(sim_time * 10) & 0xFFFFFFFF

        for player_id, samples in self._player_contacts.items():
            for sample in samples:
                stamp = DeformStamp(
                    lat        = sample.world_ix * 0.001,   # proxy lat from grid
                    lon        = sample.world_iy * 0.001,
                    radius_m   = max(math.sqrt(max(getattr(sample, 'area', 0.1), 0.0001)) * 3.0, 0.1),
                    depth_m    = abs(sample.fn) * 1e-6,     # scale Fn → depth proxy
                    push_dir_x = math.copysign(1.0, sample.ft_x) if abs(sample.ft_x) > 0 else 0.0,
                    push_dir_y = math.copysign(1.0, sample.ft_y) if abs(sample.ft_y) > 0 else 0.0,
                    push_amount= _clamp(
                        math.sqrt(sample.ft_x ** 2 + sample.ft_y ** 2) * 0.01,
                        0.0, 1.0),
                    material   = sample.material,
                    tick_index = tick_index,
                )
                self._pending_stamps.append(stamp)

        self._player_contacts.clear()

    def _filter_by_interest(
        self,
        stamps: List[DeformStamp],
        player_pos: Tuple[float, float],
    ) -> List[DeformStamp]:
        """Return stamps within interest radius of player."""
        lat, lon = player_pos
        r        = self._interest_radius_m

        # Convert interest radius in metres to rough degree tolerance
        deg_tol = r / self._m_per_deg

        return [
            s for s in stamps
            if abs(s.lat - lat) <= deg_tol and abs(s.lon - lon) <= deg_tol
        ]

    def _get(self, key: str, default) -> float:
        if self._cfg is None:
            return default
        v = self._cfg.get("deform", key, default=None)
        return v if v is not None else default
