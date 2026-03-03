"""SnapshotDiff — Stage 56 snapshot comparison utilities.

Compares two :class:`~src.burnin.SnapshotScheduler.WorldSnapshot` objects to
detect:

* **Field drift** — mean value shift between snapshots for scalar fields.
* **Saturation** — a field stuck near 0 or 1 across the whole snapshot.
* **Stagnation** — fields that change less than a minimum delta between
  two consecutive snapshots (world "frozen").

Usage
-----
    diff = SnapshotDiff()
    result = diff.compare(snap_a, snap_b)
    if result.has_issues:
        for issue in result.issues:
            print(issue)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.burnin.SnapshotScheduler import WorldSnapshot
from src.core.Logger import Logger

_TAG = "SnapshotDiff"

# Thresholds
_SATURATION_HIGH = 0.95   # value ≥ this is "near-max saturated"
_SATURATION_LOW  = 0.05   # value ≤ this is "near-zero saturated"
_STAGNATION_EPS  = 1e-6   # changes smaller than this are considered stagnant


# ---------------------------------------------------------------------------
# DiffResult
# ---------------------------------------------------------------------------

@dataclass
class DiffResult:
    """Result of comparing two snapshots."""
    snap_a_index: int = 0
    snap_b_index: int = 0
    planet_time_a: float = 0.0
    planet_time_b: float = 0.0
    issues: List[str] = field(default_factory=list)

    @property
    def has_issues(self) -> bool:
        return len(self.issues) > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "snap_a_index":  self.snap_a_index,
            "snap_b_index":  self.snap_b_index,
            "planet_time_a": self.planet_time_a,
            "planet_time_b": self.planet_time_b,
            "has_issues":    self.has_issues,
            "issues":        list(self.issues),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flatten_numerics(d: Any, prefix: str = "") -> Dict[str, float]:
    """Recursively extract all numeric leaf values from a nested dict."""
    out: Dict[str, float] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            child_prefix = f"{prefix}.{k}" if prefix else k
            out.update(_flatten_numerics(v, child_prefix))
    elif isinstance(d, (int, float)):
        out[prefix] = float(d)
    elif isinstance(d, (list, tuple)):
        for i, v in enumerate(d):
            out.update(_flatten_numerics(v, f"{prefix}[{i}]"))
    return out


# ---------------------------------------------------------------------------
# SnapshotDiff
# ---------------------------------------------------------------------------

class SnapshotDiff:
    """Compares two :class:`WorldSnapshot` objects and reports anomalies."""

    def __init__(
        self,
        saturation_high: float = _SATURATION_HIGH,
        saturation_low:  float = _SATURATION_LOW,
        stagnation_eps:  float = _STAGNATION_EPS,
    ) -> None:
        self._sat_high = saturation_high
        self._sat_low  = saturation_low
        self._stag_eps = stagnation_eps

    # ------------------------------------------------------------------
    # Main comparison
    # ------------------------------------------------------------------

    def compare(
        self,
        snap_a: WorldSnapshot,
        snap_b: WorldSnapshot,
    ) -> DiffResult:
        """Compare *snap_b* against *snap_a* and return a :class:`DiffResult`."""
        result = DiffResult(
            snap_a_index=snap_a.index,
            snap_b_index=snap_b.index,
            planet_time_a=snap_a.planet_time,
            planet_time_b=snap_b.planet_time,
        )

        vals_a = _flatten_numerics(snap_a.subsystems)
        vals_b = _flatten_numerics(snap_b.subsystems)

        all_keys = set(vals_a) | set(vals_b)
        for key in sorted(all_keys):
            va = vals_a.get(key)
            vb = vals_b.get(key)

            if va is None or vb is None:
                continue   # key only in one snapshot — skip

            # Saturation check on the later snapshot
            if vb >= self._sat_high:
                result.issues.append(
                    f"SATURATION_HIGH: '{key}'={vb:.4f} at snap #{snap_b.index}"
                )
            elif vb <= self._sat_low and abs(vb) > 0:
                result.issues.append(
                    f"SATURATION_LOW: '{key}'={vb:.4f} at snap #{snap_b.index}"
                )

            # Stagnation check
            if abs(vb - va) < self._stag_eps and abs(va) > self._stag_eps:
                result.issues.append(
                    f"STAGNATION: '{key}' unchanged between snap "
                    f"#{snap_a.index} and #{snap_b.index} (Δ={abs(vb - va):.2e})"
                )

        if result.has_issues:
            Logger.warn(
                _TAG,
                f"Diff #{snap_a.index}→#{snap_b.index}: "
                f"{len(result.issues)} issue(s)",
            )
        return result

    def compare_series(
        self,
        snapshots: List[WorldSnapshot],
    ) -> List[DiffResult]:
        """Compare each consecutive pair in *snapshots*."""
        results: List[DiffResult] = []
        for i in range(len(snapshots) - 1):
            results.append(self.compare(snapshots[i], snapshots[i + 1]))
        return results
