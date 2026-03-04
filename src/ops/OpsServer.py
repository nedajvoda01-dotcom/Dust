"""OpsServer — Stage 61 lightweight HTTP ops API.

Provides protected ops endpoints (no player-facing UI):

    GET  /ops/health
    GET  /ops/metrics
    POST /ops/tuning/propose
    POST /ops/tuning/rollback
    POST /ops/snapshot/force
    POST /ops/world/reset   (requires confirmation key)
    POST /ops/region/migrate

Authentication modes
--------------------
* ``token``  — Bearer token checked against ``ops_token``
* ``open``   — no auth (dev/test only)

This module has **no asyncio dependency** so it can be tested
synchronously.  The actual HTTP server can wrap these handlers.
"""
from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any, Callable, Dict, Optional, Tuple

from src.obs.MetricsRegistry import MetricsRegistry
from src.obs.WorldHealthScorer import WorldHealthScorer, HealthInputs
from src.ops.TuningManager import TuningManager


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------

def _ok(body: Any) -> Tuple[int, Dict[str, Any]]:
    return 200, {"status": "ok", "data": body}


def _err(code: int, msg: str) -> Tuple[int, Dict[str, Any]]:
    return code, {"status": "error", "message": msg}


# ---------------------------------------------------------------------------
# OpsServer
# ---------------------------------------------------------------------------

class OpsServer:
    """Lightweight ops API handler.

    Parameters
    ----------
    metrics:
        Live :class:`~src.obs.MetricsRegistry.MetricsRegistry`.
    tuning_manager:
        Live :class:`~src.ops.TuningManager.TuningManager`.
    health_scorer:
        Live :class:`~src.obs.WorldHealthScorer.WorldHealthScorer`.
    auth_mode:
        ``"token"`` or ``"open"``.
    ops_token:
        Secret token for ``"token"`` auth mode.
    reset_confirm_key:
        Additional confirmation key required for ``POST /ops/world/reset``.
    """

    def __init__(
        self,
        metrics:           Optional[MetricsRegistry]   = None,
        tuning_manager:    Optional[TuningManager]     = None,
        health_scorer:     Optional[WorldHealthScorer] = None,
        auth_mode:         str                         = "open",
        ops_token:         str                         = "",
        reset_confirm_key: str                         = "",
    ) -> None:
        self._metrics           = metrics or MetricsRegistry()
        self._tuning            = tuning_manager or TuningManager()
        self._health_scorer     = health_scorer or WorldHealthScorer()
        self._auth_mode         = auth_mode
        self._ops_token         = ops_token
        self._reset_confirm_key = reset_confirm_key

        # Callbacks for side-effecting ops
        self.on_force_snapshot: Optional[Callable] = None
        self.on_world_reset:    Optional[Callable] = None
        self.on_region_migrate: Optional[Callable] = None

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    def _check_auth(self, headers: Dict[str, str]) -> bool:
        if self._auth_mode == "open":
            return True
        token = headers.get("Authorization", "")
        if token.startswith("Bearer "):
            token = token[len("Bearer "):]
        if not self._ops_token:
            return False
        return hmac.compare_digest(token.encode(), self._ops_token.encode())

    # ------------------------------------------------------------------
    # Request dispatch
    # ------------------------------------------------------------------

    def handle(
        self,
        method:  str,
        path:    str,
        headers: Optional[Dict[str, str]] = None,
        body:    Optional[bytes]          = None,
    ) -> Tuple[int, Dict[str, Any]]:
        """Dispatch an ops request.

        Returns
        -------
        (status_code, response_dict)
        """
        headers = headers or {}
        if not self._check_auth(headers):
            return _err(401, "unauthorized")

        if method == "GET" and path == "/ops/health":
            return self._handle_health()

        if method == "GET" and path == "/ops/metrics":
            return self._handle_metrics()

        if method == "POST" and path == "/ops/tuning/propose":
            return self._handle_tuning_propose(body)

        if method == "POST" and path == "/ops/tuning/rollback":
            return self._handle_tuning_rollback(body)

        if method == "POST" and path == "/ops/snapshot/force":
            return self._handle_snapshot_force()

        if method == "POST" and path == "/ops/world/reset":
            return self._handle_world_reset(body)

        if method == "POST" and path == "/ops/region/migrate":
            return self._handle_region_migrate(body)

        return _err(404, f"unknown ops endpoint: {method} {path}")

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def _handle_health(self) -> Tuple[int, Dict[str, Any]]:
        snap = self._metrics.export()
        m    = snap.get("metrics", {})
        inputs = HealthInputs(
            entropy              = m.get("entropy",                   0.5),
            dust_conservation_err= m.get("dust_conservation_err",     0.0),
            instability_rate     = m.get("instability_events_per_hour", 0.0),
            energy_balance_err   = m.get("energy_balance_error",      0.0),
            snapshot_fail_rate   = m.get("snapshot_fail_rate",        0.0),
            tick_lag_ticks       = int(m.get("sim_lag_ticks",         0)),
        )
        hs = self._health_scorer.score(inputs)
        return _ok({
            "worldHealthScore": hs.score,
            "components":       hs.components,
            "alerts":           hs.alerts,
            "tuning":           self._tuning.snapshot_meta(),
        })

    def _handle_metrics(self) -> Tuple[int, Dict[str, Any]]:
        return _ok(self._metrics.export())

    def _handle_tuning_propose(
        self, body: Optional[bytes]
    ) -> Tuple[int, Dict[str, Any]]:
        if not body:
            return _err(400, "request body required")
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            return _err(400, f"invalid JSON: {exc}")
        delta = payload.get("delta", {})
        if not isinstance(delta, dict):
            return _err(400, "'delta' must be an object")
        accepted, errors = self._tuning.propose(delta)
        if not accepted:
            return _err(400, "; ".join(errors))
        return _ok({
            "accepted":    True,
            "pendingEpoch": self._tuning.current_epoch + 1,
        })

    def _handle_tuning_rollback(
        self, body: Optional[bytes]
    ) -> Tuple[int, Dict[str, Any]]:
        if not body:
            return _err(400, "request body required")
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            return _err(400, f"invalid JSON: {exc}")
        to_epoch = payload.get("toEpoch")
        if to_epoch is None:
            return _err(400, "'toEpoch' required")
        ok, msg = self._tuning.rollback(int(to_epoch))
        if not ok:
            return _err(400, msg)
        return _ok({"rolledBackTo": to_epoch, "message": msg})

    def _handle_snapshot_force(self) -> Tuple[int, Dict[str, Any]]:
        if self.on_force_snapshot:
            self.on_force_snapshot()
        return _ok({"snapshot": "requested"})

    def _handle_world_reset(
        self, body: Optional[bytes]
    ) -> Tuple[int, Dict[str, Any]]:
        if self._reset_confirm_key:
            try:
                payload = json.loads(body or b"{}")
            except json.JSONDecodeError:
                payload = {}
            provided = payload.get("confirmKey", "")
            if not hmac.compare_digest(
                provided.encode(), self._reset_confirm_key.encode()
            ):
                return _err(403, "invalid confirmKey")
        if self.on_world_reset:
            self.on_world_reset()
        return _ok({"reset": "requested"})

    def _handle_region_migrate(
        self, body: Optional[bytes]
    ) -> Tuple[int, Dict[str, Any]]:
        try:
            payload = json.loads(body or b"{}")
        except json.JSONDecodeError:
            payload = {}
        if self.on_region_migrate:
            self.on_region_migrate(payload)
        return _ok({"migrate": "requested", "params": payload})
