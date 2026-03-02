"""StateHashSync — Stage 42 §3.2–3.3  Hash-based network consistency.

Computes lightweight state hashes that the server and clients use to detect
simulation drift, and provides an auto-correction escalation ladder.

Hash domains
------------
``hash_motor_core``     COM position + contact count + stance id.
``hash_deform_nearby``  Stamp sequence numbers for H/M deformation chunks.
``hash_grasp``          Active constraint ids + quantised anchor positions.
``hash_astro_climate``  Key astro/climate coefficients.

Drift correction ladder
-----------------------
Level 1  Adjust simTime offset on client.
Level 2  Resync stance / contacts key state.
Level 3  Resend sector baseline snapshot.
Level 4  Soft-reset local generative systems (audio/camera).

Public API
----------
StateHashSync(config_dict)
  .hash_motor_core(com_mm, contact_count, stance_id) → int
  .hash_deform_nearby(stamp_seq_list)                → int
  .hash_grasp(constraint_ids, anchor_mm_list)        → int
  .hash_astro_climate(coeff_dict)                    → int
  .compute_full_hash(motor, deform, grasp, astro)    → int
  .check_drift(local_hash, remote_hash)              → DriftLevel | None
  .reset_mismatch_counter()
  .mismatch_count                                    → int
  .hash_interval_sec                                 → float

DriftLevel
  LEVEL_1_TIME_OFFSET  = 1
  LEVEL_2_KEY_STATE    = 2
  LEVEL_3_BASELINE     = 3
  LEVEL_4_SOFT_RESET   = 4
"""
from __future__ import annotations

import struct
from enum import IntEnum
from typing import Dict, Iterable, List, Optional, Sequence

from src.core.Logger import Logger

_TAG = "StateHashSync"


class DriftLevel(IntEnum):
    """Severity of detected state drift."""
    LEVEL_1_TIME_OFFSET = 1
    LEVEL_2_KEY_STATE   = 2
    LEVEL_3_BASELINE    = 3
    LEVEL_4_SOFT_RESET  = 4


def _fnv32(data: bytes) -> int:
    """FNV-1a 32-bit hash — fast, portable, deterministic."""
    h = 0x811C9DC5
    for byte in data:
        h ^= byte
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


class StateHashSync:
    """Computes and compares state hashes for multiplayer consistency.

    Parameters
    ----------
    config:
        Dict expected to contain ``determinism.hash_interval_sec`` and
        optionally ``determinism.drift_thresholds``.
    """

    def __init__(self, config: Optional[Dict] = None) -> None:
        cfg = config or {}
        det_cfg = cfg.get("determinism", {})
        self.hash_interval_sec: float = float(
            det_cfg.get("hash_interval_sec", 5.0)
        )
        thresholds = det_cfg.get("drift_thresholds", {})
        self._thresh: Dict[DriftLevel, int] = {
            DriftLevel.LEVEL_1_TIME_OFFSET: int(thresholds.get("level1", 1)),
            DriftLevel.LEVEL_2_KEY_STATE:   int(thresholds.get("level2", 3)),
            DriftLevel.LEVEL_3_BASELINE:    int(thresholds.get("level3", 6)),
            DriftLevel.LEVEL_4_SOFT_RESET:  int(thresholds.get("level4", 10)),
        }
        self._mismatch_count: int = 0

    # ------------------------------------------------------------------
    # Hash components
    # ------------------------------------------------------------------

    @staticmethod
    def hash_motor_core(
        com_mm: Sequence[int],
        contact_count: int,
        stance_id: int,
    ) -> int:
        """Hash centre-of-mass (mm fixed-point), contact count, and stance."""
        buf = struct.pack(
            ">iiiiB",
            int(com_mm[0]), int(com_mm[1]), int(com_mm[2]),
            int(contact_count), int(stance_id) & 0xFF,
        )
        return _fnv32(buf)

    @staticmethod
    def hash_deform_nearby(stamp_seq_list: Iterable[int]) -> int:
        """Hash the ordered list of deformation stamp sequence numbers."""
        stamps = [int(s) for s in stamp_seq_list]
        buf = struct.pack(">" + "I" * len(stamps), *stamps)
        return _fnv32(buf)

    @staticmethod
    def hash_grasp(
        constraint_ids: Iterable[int],
        anchor_mm_list: Iterable[Sequence[int]],
    ) -> int:
        """Hash active constraint ids and quantised anchor positions."""
        ids = [int(c) for c in constraint_ids]
        anchors = [tuple(int(v) for v in a) for a in anchor_mm_list]
        buf = struct.pack(">" + "I" * len(ids), *ids)
        for ax, ay, az in anchors:
            buf += struct.pack(">iii", ax, ay, az)
        return _fnv32(buf)

    @staticmethod
    def hash_astro_climate(coeff_dict: Dict[str, float]) -> int:
        """Hash key astro/climate coefficients (sorted by key)."""
        buf = b""
        for k in sorted(coeff_dict.keys()):
            v_int = int(round(float(coeff_dict[k]) * 1_000_000))
            buf += k.encode("utf-8") + struct.pack(">i", v_int)
        return _fnv32(buf)

    @staticmethod
    def compute_full_hash(
        motor: int,
        deform: int,
        grasp: int,
        astro: int,
    ) -> int:
        """Combine four domain hashes into a single composite hash."""
        buf = struct.pack(
            ">IIII",
            motor & 0xFFFFFFFF, deform & 0xFFFFFFFF,
            grasp & 0xFFFFFFFF, astro & 0xFFFFFFFF,
        )
        return _fnv32(buf)

    # ------------------------------------------------------------------
    # Drift detection
    # ------------------------------------------------------------------

    def check_drift(
        self,
        local_hash: int,
        remote_hash: int,
    ) -> Optional[DriftLevel]:
        """Compare hashes; return correction level or None on match."""
        if local_hash == remote_hash:
            self._mismatch_count = 0
            return None

        self._mismatch_count += 1
        n = self._mismatch_count
        Logger.warn(
            _TAG,
            f"hash mismatch #{n}: local={local_hash:#010x} "
            f"remote={remote_hash:#010x}",
        )

        if n >= self._thresh[DriftLevel.LEVEL_4_SOFT_RESET]:
            return DriftLevel.LEVEL_4_SOFT_RESET
        if n >= self._thresh[DriftLevel.LEVEL_3_BASELINE]:
            return DriftLevel.LEVEL_3_BASELINE
        if n >= self._thresh[DriftLevel.LEVEL_2_KEY_STATE]:
            return DriftLevel.LEVEL_2_KEY_STATE
        return DriftLevel.LEVEL_1_TIME_OFFSET

    def reset_mismatch_counter(self) -> None:
        """Reset the consecutive-mismatch counter (e.g. after resync)."""
        self._mismatch_count = 0

    @property
    def mismatch_count(self) -> int:
        """Number of consecutive hash mismatches since last reset."""
        return self._mismatch_count
