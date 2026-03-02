"""ModeHysteresis — per-mode activation hysteresis for IntentArbitrator.

Prevents rapid oscillation between motor modes (Stage 38, §11).

Each mode has an *enterThreshold* and *exitThreshold*.  The mode
activates once its score crosses ``enterThreshold`` and only
deactivates after its score falls below ``exitThreshold``.  The gap
between the two thresholds forms the hysteresis band.

Example (BraceMode from the spec)::

    h = ModeHysteresis(enter_threshold=0.6, exit_threshold=0.4)
    h.update(0.7)   # → True  (activated)
    h.update(0.5)   # → True  (still active: 0.5 >= 0.4)
    h.update(0.35)  # → False (deactivated: 0.35 < 0.4)

Public API
----------
ModeHysteresis(enter_threshold, exit_threshold)
  .update(score) → bool
  .is_active     → bool
  .reset()       → None
"""
from __future__ import annotations


class ModeHysteresis:
    """Hysteresis filter for a single motor mode.

    Parameters
    ----------
    enter_threshold :
        Minimum score required to activate the mode.
    exit_threshold :
        Score below which the mode deactivates.  Must be <= ``enter_threshold``.
    """

    def __init__(self, enter_threshold: float, exit_threshold: float) -> None:
        if exit_threshold > enter_threshold:
            raise ValueError(
                f"exit_threshold ({exit_threshold}) must be <= "
                f"enter_threshold ({enter_threshold})"
            )
        self._enter   = float(enter_threshold)
        self._exit    = float(exit_threshold)
        self._active  = False

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    @property
    def is_active(self) -> bool:
        """Whether the mode is currently activated."""
        return self._active

    @property
    def enter_threshold(self) -> float:
        return self._enter

    @property
    def exit_threshold(self) -> float:
        return self._exit

    def update(self, score: float) -> bool:
        """Advance hysteresis state and return whether the mode is active.

        Parameters
        ----------
        score :
            Current activation score for the mode [0..1].

        Returns
        -------
        bool
            ``True`` if the mode is active after this update.
        """
        if not self._active:
            if score >= self._enter:
                self._active = True
        else:
            if score < self._exit:
                self._active = False
        return self._active

    def reset(self) -> None:
        """Force the mode back to inactive."""
        self._active = False
