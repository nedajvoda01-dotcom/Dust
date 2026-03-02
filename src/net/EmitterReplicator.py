"""EmitterReplicator — Stage 46 quantised acoustic emitter replication.

Serialises :class:`~audio_world.EmitterAggregator.AcousticEmitterRecord`
objects into compact wire records for network transmission, and
deserialises them back.

Design choices (per §9 / §5 of the spec)
-----------------------------------------
* Only emitter **parameters** are sent — not waveforms.
* Positions are quantised to 4 m grid (matches
  :func:`~audio_world.EmitterAggregator._quantise`).
* Energy values are quantised to 8-bit unsigned (256 steps).
* The batch size is interest-based: only emitters within
  ``interest_radius`` of the target client position are included.

Public API
----------
EmitterReplicator(config=None)
  .serialise(emitters, client_pos)  → List[dict]   (wire records)
  .deserialise(records)             → List[AcousticEmitterRecord]
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from src.math.Vec3 import Vec3
from src.audio.audio_world.EmitterAggregator import (
    AcousticEmitterRecord,
    EmitterType,
)


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


def _quantise_energy(v: float) -> int:
    """Encode [0..1+] energy to 8-bit unsigned (0–255)."""
    return int(_clamp(v, 0.0, 1.0) * 255.0 + 0.5)


def _dequantise_energy(b: int) -> float:
    return b / 255.0


def _quantise_pos_coord(v: float, grid: float = 4.0) -> int:
    """Snap coordinate to *grid* and encode as int."""
    return int(round(v / grid))


def _dequantise_pos_coord(i: int, grid: float = 4.0) -> float:
    return float(i) * grid


class EmitterReplicator:
    """Serialises / deserialises acoustic emitter records for the network.

    Parameters
    ----------
    config :
        Optional dict; reads ``audio_world.*`` keys.
    """

    _DEFAULT_INTEREST_RADIUS = 800.0  # metres
    _POS_GRID = 4.0  # must match EmitterAggregator._QUANTISE_GRID

    def __init__(self, config: Optional[dict] = None) -> None:
        awcfg = (config or {}).get("audio_world", {}) or {}
        self._interest_radius: float = float(
            awcfg.get("replication_interest_radius", self._DEFAULT_INTEREST_RADIUS)
        )

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def serialise(
        self,
        emitters:   List[AcousticEmitterRecord],
        client_pos: Vec3,
    ) -> List[Dict[str, Any]]:
        """Build a list of wire-format dicts for *client_pos*.

        Only emitters within ``interest_radius`` are included.

        Parameters
        ----------
        emitters :
            Active emitters from :class:`~audio_world.EmitterAggregator`.
        client_pos :
            Recipient client's world position (interest filtering).
        """
        records = []
        for e in emitters:
            diff = e.pos - client_pos
            if diff.length() > self._interest_radius:
                continue
            records.append({
                "id":  e.id,
                "px":  _quantise_pos_coord(e.pos.x, self._POS_GRID),
                "py":  _quantise_pos_coord(e.pos.y, self._POS_GRID),
                "pz":  _quantise_pos_coord(e.pos.z, self._POS_GRID),
                "ea":  _quantise_energy(e.band_energy_audible),
                "ei":  _quantise_energy(e.band_energy_infra),
                "dir": int(_clamp(e.directivity, 0.0, 1.0) * 15.0 + 0.5),  # 4-bit
                "typ": int(e.emitter_type),
                "ttl": max(0, e.ttl),
            })
        return records

    def deserialise(
        self,
        records: List[Dict[str, Any]],
    ) -> List[AcousticEmitterRecord]:
        """Reconstruct emitter records from wire-format dicts.

        Parameters
        ----------
        records :
            List of wire-format dicts produced by :meth:`serialise`.
        """
        out = []
        for r in records:
            pos = Vec3(
                _dequantise_pos_coord(r["px"], self._POS_GRID),
                _dequantise_pos_coord(r["py"], self._POS_GRID),
                _dequantise_pos_coord(r["pz"], self._POS_GRID),
            )
            out.append(AcousticEmitterRecord(
                id=r["id"],
                pos=pos,
                band_energy_audible=_dequantise_energy(r["ea"]),
                band_energy_infra=_dequantise_energy(r["ei"]),
                directivity=r["dir"] / 15.0,
                emitter_type=EmitterType(r["typ"]),
                ttl=r["ttl"],
            ))
        return out
