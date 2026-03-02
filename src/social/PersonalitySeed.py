"""PersonalitySeed — Stage 39 deterministic per-player personality parameters.

Derives fixed behavioural scalars from a ``player_id`` integer using a
reproducible hash chain.  No ``random()`` is ever called — the same
``player_id`` always yields the same :class:`PersonalityParams`.

Public API
----------
PersonalityParams
  .sociability     (0..1)  — tendency to approach / engage
  .caution         (0..1)  — risk-aversion when manoeuvring
  .helpfulness     (0..1)  — propensity to enter AssistPrep
  .personalSpace   (m)     — preferred inter-player gap [personal_space_min..max]

PersonalitySeed.generate(player_id, config=None) → PersonalityParams
"""
from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import Optional


@dataclass
class PersonalityParams:
    """Deterministic per-player behavioural scalars (§5.1)."""
    sociability:   float  # 0..1 — approaches others / stays nearby
    caution:       float  # 0..1 — risk-averse manoeuvring
    helpfulness:   float  # 0..1 — willingness to assist
    personalSpace: float  # metres — preferred gap to others


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


def _hash_float(player_id: int, salt: int) -> float:
    """Return a deterministic float in [0, 1) derived from player_id + salt."""
    raw = struct.pack(">qq", int(player_id), int(salt))
    digest = hashlib.sha256(raw).digest()
    # Take first 4 bytes as a uint32, normalise to [0, 1)
    u32 = struct.unpack(">I", digest[:4])[0]
    return u32 / 4_294_967_296.0


class PersonalitySeed:
    """Factory that deterministically generates :class:`PersonalityParams`.

    Parameters
    ----------
    config :
        Optional dict; reads ``social.personal_space_min`` and
        ``social.personal_space_max``.
    """

    _DEFAULT_PS_MIN = 1.2  # metres
    _DEFAULT_PS_MAX = 3.5  # metres

    def __init__(self, config: Optional[dict] = None) -> None:
        scfg = ((config or {}).get("social", {})) or {}
        self._ps_min = float(scfg.get("personal_space_min", self._DEFAULT_PS_MIN))
        self._ps_max = float(scfg.get("personal_space_max", self._DEFAULT_PS_MAX))

    def generate(self, player_id: int) -> PersonalityParams:
        """Return deterministic :class:`PersonalityParams` for *player_id*.

        Parameters
        ----------
        player_id :
            Stable integer player identifier.
        """
        pid = int(player_id)
        soc  = _hash_float(pid, 0)
        caut = _hash_float(pid, 1)
        help_= _hash_float(pid, 2)
        ps_t = _hash_float(pid, 3)  # 0..1, mapped to [ps_min, ps_max]
        ps   = self._ps_min + ps_t * (self._ps_max - self._ps_min)
        return PersonalityParams(
            sociability=soc,
            caution=caut,
            helpfulness=help_,
            personalSpace=_clamp(ps, self._ps_min, self._ps_max),
        )
