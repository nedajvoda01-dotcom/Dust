"""InfraField — Stage 46 infrasound / low-frequency energy field.

Stores low-frequency energy density per world sector.  Energy is injected
by structural events (rifts, cave-ins) and decays exponentially with time
constant ``infra_decay_tau``.

The field deliberately does **not** model wave propagation precisely —
instead it provides a plausible "pressure has been felt here recently" cue
that Perception consumes as ``vibrationLevel`` and ``infraUrgency``.

Public API
----------
InfraField(config=None)
  .inject(pos, energy)                          → None
  .tick(dt)                                     → None
  .sample(pos)                                  → float   (0..1 energy)
  .urgency(pos)                                 → float   (0..1)
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

from src.math.Vec3 import Vec3


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


class InfraField:
    """Sector-based infrasound energy store with exponential decay.

    Parameters
    ----------
    config :
        Optional dict; reads ``audio_world.*`` keys.
    """

    _DEFAULT_DECAY_TAU  = 4.0    # seconds; energy halves every ~2.77 s
    _DEFAULT_SECTOR_SZ  = 200.0  # metres per sector cell
    # Maximum injection energy accepted in one call
    _MAX_INJECT         = 5.0

    def __init__(self, config: Optional[dict] = None) -> None:
        awcfg = (config or {}).get("audio_world", {}) or {}
        self._tau: float = float(awcfg.get("infra_decay_tau", self._DEFAULT_DECAY_TAU))
        self._sector_sz: float = float(awcfg.get("infra_sector_size", self._DEFAULT_SECTOR_SZ))
        # {sector_key: energy}
        self._sectors: Dict[Tuple[int, int], float] = {}

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def inject(self, pos: Vec3, energy: float) -> None:
        """Add infrasound energy at *pos*.

        Energy is distributed across the sector containing *pos* and
        the four adjacent sectors (cross-shaped diffusion).

        Parameters
        ----------
        pos :
            World-space position of the source.
        energy :
            Energy to inject [0..]. Capped internally.
        """
        energy = _clamp(energy, 0.0, self._MAX_INJECT)
        cx, cz = self._sector(pos)
        # Central sector gets the most; neighbours get 20 % each
        centres = [(cx, cz, 1.0), (cx+1, cz, 0.2), (cx-1, cz, 0.2),
                   (cx, cz+1, 0.2), (cx, cz-1, 0.2)]
        for sx, sz, w in centres:
            key = (sx, sz)
            self._sectors[key] = self._sectors.get(key, 0.0) + energy * w

    def tick(self, dt: float) -> None:
        """Decay all stored energies by one timestep.

        Parameters
        ----------
        dt :
            Elapsed simulation time [s].
        """
        if self._tau <= 1e-6:
            self._sectors.clear()
            return
        decay = math.exp(-dt / self._tau)
        self._sectors = {
            k: v * decay
            for k, v in self._sectors.items()
            if v * decay > 1e-6
        }

    def sample(self, pos: Vec3) -> float:
        """Return normalised infra energy [0..1] at *pos*.

        Parameters
        ----------
        pos :
            World-space listener position.
        """
        key = self._sector(pos)
        raw = self._sectors.get(key, 0.0)
        # Soft-clip to [0, 1]
        return _clamp(raw / (1.0 + raw), 0.0, 1.0)

    def urgency(self, pos: Vec3) -> float:
        """Return infraUrgency [0..1] — same as sample but named for clarity."""
        return self.sample(pos)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _sector(self, pos: Vec3) -> Tuple[int, int]:
        return (
            int(math.floor(pos.x / self._sector_sz)),
            int(math.floor(pos.z / self._sector_sz)),
        )
