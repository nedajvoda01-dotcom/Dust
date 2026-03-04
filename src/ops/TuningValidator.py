"""TuningValidator — Stage 61 tuning change validation.

Validates proposed tuning deltas against:
* an allowlist of permitted parameter names
* per-parameter value ranges

Any parameter not in the allowlist is rejected.
Any value outside its [min, max] range is rejected.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Default allowlist with [min, max] ranges
# ---------------------------------------------------------------------------

DEFAULT_ALLOWLIST: Dict[str, Tuple[float, float]] = {
    # LOD / budgets
    "max_raycasts_per_sec":      (0,      100_000),
    "qp_max_iters":              (1,      500),
    "net_update_hz":             (1,      60),
    "chunk_update_interval":     (1,      120),
    # Instability thresholds
    "instability_threshold":     (0.0,    1.0),
    "instability_decay_rate":    (0.0,    1.0),
    # Decay rates
    "dust_decay_rate":           (0.0,    1.0),
    "ice_decay_rate":            (0.0,    1.0),
    # Camera / salience smoothing
    "max_camera_mod":            (0.0,    5.0),
    "smoothing_tau":             (0.01,   10.0),
}


class TuningValidationError(ValueError):
    """Raised when a tuning delta fails validation."""


class TuningValidator:
    """Validates tuning deltas against an allowlist with ranges.

    Parameters
    ----------
    allowlist:
        Mapping of ``param_name -> (min_value, max_value)``.  If *None*,
        :data:`DEFAULT_ALLOWLIST` is used.
    """

    def __init__(
        self,
        allowlist: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> None:
        self._allowlist = dict(allowlist) if allowlist is not None else dict(DEFAULT_ALLOWLIST)

    def validate(self, delta: Dict[str, Any]) -> List[str]:
        """Validate a proposed config delta.

        Returns
        -------
        list[str]
            Empty list on success; list of error messages on failure.
        """
        errors: List[str] = []
        for key, value in delta.items():
            if key not in self._allowlist:
                errors.append(f"parameter '{key}' is not in the tuning allowlist")
                continue
            lo, hi = self._allowlist[key]
            try:
                v = float(value)
            except (TypeError, ValueError):
                errors.append(f"parameter '{key}' has non-numeric value: {value!r}")
                continue
            if not (lo <= v <= hi):
                errors.append(
                    f"parameter '{key}' value {v} is out of range [{lo}, {hi}]"
                )
        return errors

    def is_valid(self, delta: Dict[str, Any]) -> bool:
        """Return True iff the delta passes all validation checks."""
        return len(self.validate(delta)) == 0

    @property
    def allowlist(self) -> Dict[str, Tuple[float, float]]:
        """Read-only view of the current allowlist."""
        return dict(self._allowlist)
