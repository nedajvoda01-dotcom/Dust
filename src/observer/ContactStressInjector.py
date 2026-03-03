"""ContactStressInjector — Stage 53 player mass / contact → stress fields.

The player's body weight and dynamic contact events (footsteps, falls,
slides) inject a small, capped amount of stress into:

* ``WorldMemoryState.stressAccumulationField`` (Stage 51)
* ``WorldMemoryState.compactionHistoryField``   (Stage 51)

Model
-----
Per contact event at a tile::

    stress_delta     = k_player_stress * contact_force * dt
    compaction_delta = k_player_stress * 0.5 * contact_force * dt

Both contributions are clipped by :class:`InfluenceLimiter` before being
applied, so the player cannot indefinitely raise stress in a stable zone.

Config keys (under ``observer.*``)
-----------------------------------
enable           : bool  — master switch   (default True)
k_player_stress  : float — stress gain per unit force per second (default 0.02)

Public API
----------
ContactStressInjector(config=None, limiter=None)
  .inject(memory_state, tile_idx, contact_force, dt) -> None
"""
from __future__ import annotations

from typing import Optional

from src.memory.WorldMemoryState    import WorldMemoryState
from src.observer.InfluenceLimiter  import InfluenceLimiter


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


class ContactStressInjector:
    """Injects player contact forces into WorldMemoryState stress fields.

    Parameters
    ----------
    config :
        Optional dict; reads ``observer.*`` keys.
    limiter :
        Shared :class:`InfluenceLimiter` instance.  A private one is created
        if ``None`` is supplied.
    """

    _DEFAULTS = {
        "enable":          True,
        "k_player_stress": 0.02,
    }

    def __init__(
        self,
        config:  Optional[dict] = None,
        limiter: Optional[InfluenceLimiter] = None,
    ) -> None:
        cfg = dict(self._DEFAULTS)
        if isinstance(config, dict):
            src = config.get("observer", config)
            for k in self._DEFAULTS:
                if k in src:
                    cfg[k] = src[k]

        self._enabled:  bool  = bool(cfg["enable"])
        self._k_stress: float = float(cfg["k_player_stress"])
        self._limiter:  InfluenceLimiter = limiter or InfluenceLimiter(config)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def inject(
        self,
        memory_state:  WorldMemoryState,
        tile_idx:      int,
        contact_force: float,
        dt:            float,
    ) -> None:
        """Apply a contact contribution to *tile_idx*.

        Parameters
        ----------
        memory_state  : WorldMemoryState updated in-place.
        tile_idx      : Flat tile index.
        contact_force : Normalised contact force [0, 1].
        dt            : Time step (seconds).
        """
        if not self._enabled or dt <= 0.0 or contact_force <= 0.0:
            return

        # --- stress accumulation ---
        stress_raw = self._k_stress * contact_force * dt
        stress_clipped = self._limiter.clip(tile_idx, stress_raw)
        if stress_clipped > 0.0:
            f = memory_state.stressAccumulationField
            f[tile_idx] = _clamp(f[tile_idx] + stress_clipped)
            self._limiter.record(tile_idx, stress_clipped)

        # --- compaction history (half the gain) ---
        compact_raw = self._k_stress * 0.5 * contact_force * dt
        compact_clipped = self._limiter.clip(tile_idx, compact_raw)
        if compact_clipped > 0.0:
            c = memory_state.compactionHistoryField
            c[tile_idx] = _clamp(c[tile_idx] + compact_clipped)
            self._limiter.record(tile_idx, compact_clipped)
