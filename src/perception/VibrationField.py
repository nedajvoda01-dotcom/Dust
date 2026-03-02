"""VibrationField — Stage 37 ground vibration / infrasound perception field.

Aggregates geological and acoustic low-frequency energy into:

* ``vibrationLevel``  (0..1) — magnitude of felt vibration
* ``vibrationDir``    (Vec3, unit) — gradient direction (toward epicentre)

Inputs:
* Geo/subsurface event signals (stress energy, affected position)
* Bulk low-frequency audio energy (from MegaResonator proxy)

Public API
----------
VibrationField(config=None)
  .update(listener_pos, geo_signals, bulk_lf_energy, dt) → None
  .vibration_level  → float
  .vibration_dir    → Vec3
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

from src.math.Vec3 import Vec3


@dataclass
class GeoVibrationSignal:
    """A single ground-vibration signal emitted by a geological event.

    Attributes
    ----------
    position :
        Epicentre of the event in world space.
    energy :
        Normalised energy proxy [0..1].
    """
    position: Vec3
    energy:   float = 0.0


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


class VibrationField:
    """Perception sub-field: ground vibration and infrasound.

    Parameters
    ----------
    config :
        Optional dict; reads ``perception.vibration.*`` keys.
    """

    _DEFAULT_WEIGHT         = 1.0
    _DEFAULT_SMOOTHING_TAU  = 0.20  # seconds
    # Influence radius for a vibration event
    _INFLUENCE_RADIUS       = 400.0

    def __init__(self, config: Optional[dict] = None) -> None:
        pcfg = ((config or {}).get("perception", {}) or {}).get("vibration", {}) or {}
        self._weight: float = float(pcfg.get("weight", self._DEFAULT_WEIGHT))
        tau = float(
            ((config or {}).get("perception", {}) or {}).get(
                "smoothing_tau_sec", self._DEFAULT_SMOOTHING_TAU
            )
        )
        self._tau: float = max(1e-3, tau)

        self._level: float = 0.0
        self._dir:   Vec3  = Vec3(0.0, 0.0, 0.0)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def update(
        self,
        listener_pos:   Vec3,
        geo_signals:    List[GeoVibrationSignal],
        bulk_lf_energy: float = 0.0,
        dt:             float = 1.0 / 10.0,
    ) -> None:
        """Advance vibration field one tick.

        Parameters
        ----------
        listener_pos :
            Character world position.
        geo_signals :
            Active geological vibration signals this tick.
        bulk_lf_energy :
            0..1 bulk low-frequency resonator energy (from audio MegaResonator).
        dt :
            Elapsed simulation time [s].
        """
        total_weight = 0.0
        dir_acc      = Vec3(0.0, 0.0, 0.0)

        for sig in geo_signals:
            diff = sig.position - listener_pos
            dist = diff.length()
            if dist > self._INFLUENCE_RADIUS or dist < 1e-6:
                continue

            # Vibration attenuates linearly with distance
            atten = 1.0 - dist / self._INFLUENCE_RADIUS
            w = sig.energy * atten
            total_weight += w

            # Gradient points toward epicentre
            unit = diff * (1.0 / dist)
            dir_acc = dir_acc + unit * w

        # Combine with bulk audio low-frequency energy
        raw_level = _clamp(
            (total_weight / max(1.0, len(geo_signals))) * self._weight
            + bulk_lf_energy * 0.5,
            0.0, 1.0,
        ) if geo_signals else _clamp(bulk_lf_energy * 0.5, 0.0, 1.0)

        dir_len = dir_acc.length()
        raw_dir = dir_acc * (1.0 / dir_len) if dir_len > 1e-6 else Vec3(0.0, 0.0, 0.0)

        # Exponential smoothing
        alpha = 1.0 - math.exp(-dt / self._tau)
        self._level = self._level + alpha * (raw_level - self._level)

        prev_len = self._dir.length()
        if prev_len < 1e-6:
            self._dir = raw_dir
        else:
            blended = self._dir * (1.0 - alpha) + raw_dir * alpha
            bl = blended.length()
            self._dir = blended * (1.0 / bl) if bl > 1e-6 else raw_dir

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------

    @property
    def vibration_level(self) -> float:
        """Felt ground vibration intensity [0..1]."""
        return self._level

    @property
    def vibration_dir(self) -> Vec3:
        """Unit vector pointing toward vibration epicentre."""
        return self._dir
