"""EnergySnapshot — Stage 54 binary save / restore for GlobalEnergySystem state.

Format
------
Magic   : b"GES4"  (4 bytes)
Header  : sim_time(8, double) + planet_entropy(4, float) +
          dissipation_scale(4, float)          = 16 bytes
Reservoirs: 7 × 4 bytes (float32) — in canonical order:
            solar, atmospheric, gravitational, thermal,
            mechanical, phase, acoustic

Total: 4 + 16 + 28 = 48 bytes

Public API
----------
EnergySnapshot()
  .save(energy_system, sim_time=0.0) → bytes
  .load(data: bytes) → (dict, meta_dict)
      dict keys: "type", "planet_entropy", "dissipation_scale", "reservoirs"
      meta keys: "sim_time"
"""
from __future__ import annotations

import struct
from typing import Dict, Tuple

from src.energy.GlobalEnergySystem import GlobalEnergySystem

_MAGIC       = b"GES4"
_RESERVOIR_ORDER = (
    "solar", "atmospheric", "gravitational", "thermal",
    "mechanical", "phase", "acoustic",
)

# Header: sim_time (d=8) + entropy (f=4) + dissipation (f=4) = 16 bytes
_HDR_FMT  = "!dff"
_HDR_SIZE = struct.calcsize(_HDR_FMT)   # 16

# Reservoirs: 7 floats = 28 bytes
_RES_FMT  = "!" + "f" * len(_RESERVOIR_ORDER)
_RES_SIZE = struct.calcsize(_RES_FMT)   # 28

_TOTAL_SIZE = 4 + _HDR_SIZE + _RES_SIZE  # 48


class EnergySnapshot:
    """Binary serialiser for :class:`GlobalEnergySystem` state.

    Usage::

        snap = EnergySnapshot()
        blob = snap.save(energy_system, sim_time=42.0)
        state_dict, meta = snap.load(blob)
        energy_system.load_state_dict(state_dict)
        # meta["sim_time"]
    """

    def save(
        self,
        energy_system: GlobalEnergySystem,
        sim_time: float = 0.0,
    ) -> bytes:
        """Serialise *energy_system* to bytes.

        Parameters
        ----------
        energy_system : GlobalEnergySystem to persist.
        sim_time      : Simulation time to embed.
        """
        d = energy_system.state_dict()

        header = struct.pack(
            _HDR_FMT,
            sim_time,
            float(d["planet_entropy"]),
            float(d["dissipation_scale"]),
        )

        res = d.get("reservoirs", {})
        res_vals = [float(res.get(name, 0.0)) for name in _RESERVOIR_ORDER]
        res_bytes = struct.pack(_RES_FMT, *res_vals)

        return _MAGIC + header + res_bytes

    def load(
        self,
        data: bytes,
    ) -> Tuple[dict, Dict]:
        """Deserialise bytes to a state dict.

        Parameters
        ----------
        data : bytes produced by :meth:`save`.

        Returns
        -------
        (state_dict, meta_dict)
            *state_dict* can be passed to
            :meth:`GlobalEnergySystem.load_state_dict`.
            *meta_dict* contains ``sim_time``.

        Raises
        ------
        ValueError
            If magic bytes do not match.
        """
        if len(data) < _TOTAL_SIZE or data[:4] != _MAGIC:
            raise ValueError(
                f"EnergySnapshot.load: bad magic or truncated data "
                f"(got {data[:4]!r}, expected {_MAGIC!r})"
            )

        offset = 4
        sim_time, entropy, dissipation = struct.unpack_from(_HDR_FMT, data, offset)
        offset += _HDR_SIZE

        res_vals = struct.unpack_from(_RES_FMT, data, offset)
        reservoirs = {
            name: float(val)
            for name, val in zip(_RESERVOIR_ORDER, res_vals)
        }

        state_dict = {
            "type":              "GLOBAL_ENERGY_STATE_54",
            "planet_entropy":    float(entropy),
            "dissipation_scale": float(dissipation),
            "reservoirs":        reservoirs,
        }
        meta = {"sim_time": float(sim_time)}

        return state_dict, meta
