"""OpsLayer — Stage 24 operational layer.

Provides:
* Health & metrics (``health()`` / ``metrics()``)
* Structured JSON-lines logging with size-based rotation + pruning
* ``world_state/`` compaction: baseline snapshot creation + delta pruning
* Soft world reset without restarting the process
* Safety guards: geo-event rate cap, patch-batch rate cap, NaN/Inf detection,
  storage-size cap

This module has *no* asyncio dependency on its own so that it can be tested
synchronously.  The only coroutine is ``maybe_reset()`` which the caller awaits
inside the running event-loop.
"""
from __future__ import annotations

import asyncio
import gzip
import json
import math
import os
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# ---------------------------------------------------------------------------
# Log-category constants (used in structured records)
# ---------------------------------------------------------------------------

CAT_BOOT    = "BOOT"
CAT_NET     = "NET"
CAT_WORLD   = "WORLD"
CAT_CLIMATE = "CLIMATE"
CAT_GEO     = "GEO"
CAT_PATCH   = "PATCH"
CAT_SAVE    = "SAVE"
CAT_OPS     = "OPS"
CAT_ERROR   = "ERROR"

# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

_LOG_DIR            = "logs"
_LOG_FILENAME       = "server.log.jsonl"
_ARCHIVE_DIR        = "archive"
_BASELINE_PREFIX    = "baseline_"
_DELTA_PREFIX       = "geo_events.delta_"

_DEFAULT_LOG_ROTATE_MB              = 100.0
_DEFAULT_LOG_KEEP_FILES             = 10
_DEFAULT_BASELINE_INTERVAL_MIN      = 60.0
_DEFAULT_BASELINE_SIZE_TRIGGER_MB   = 200.0
_DEFAULT_WORLD_STATE_SIZE_CAP_MB    = 500.0
_DEFAULT_MAX_GEO_EVENTS_PER_HOUR    = 10_000
_DEFAULT_MAX_PATCH_BATCHES_PER_MIN  = 1_000
_DEFAULT_RESET_FLAG_PATH            = "world_state/RESET_NOW"

_DEGRADED_TICK_AGE_MS   = 5_000
_DEGRADED_JOB_DEPTH     = 100


# ===========================================================================
# OpsLayer
# ===========================================================================

class OpsLayer:
    """Operational layer for the Dust multiplayer server.

    Parameters
    ----------
    world_state:
        Live :class:`~src.net.WorldState.WorldState` instance.
    config:
        Loaded :class:`~src.core.Config.Config` (or *None* → defaults).
    registry:
        Live :class:`~src.net.PlayerRegistry.PlayerRegistry` (or *None*).
    state_dir:
        Path to the persistent world-state directory (default ``world_state``).
    build_id:
        Build identifier string injected from NetworkServer.
    """

    def __init__(
        self,
        world_state                 = None,
        config                      = None,
        registry                    = None,
        state_dir: str              = "world_state",
        build_id:  str              = "",
    ) -> None:
        self._world_state = world_state
        self._config      = config
        self._registry    = registry
        self._state_dir   = Path(state_dir)
        self._build_id    = build_id
        self._start_time  = time.monotonic()

        # ---- Config helpers ------------------------------------------------
        def _cfg(section: str, key: str, default: Any) -> Any:
            if config is None:
                return default
            return config.get(section, key, default=default)

        self._log_rotate_mb            = float(_cfg("ops", "log_rotate_mb",
                                                     _DEFAULT_LOG_ROTATE_MB))
        self._log_keep_files           = int(  _cfg("ops", "log_keep_files",
                                                     _DEFAULT_LOG_KEEP_FILES))
        self._baseline_interval_min    = float(_cfg("ops", "baseline_interval_min",
                                                     _DEFAULT_BASELINE_INTERVAL_MIN))
        self._baseline_size_trigger_mb = float(_cfg("ops", "baseline_size_trigger_mb",
                                                     _DEFAULT_BASELINE_SIZE_TRIGGER_MB))
        self._world_state_cap_mb       = float(_cfg("ops", "world_state_size_cap_mb",
                                                     _DEFAULT_WORLD_STATE_SIZE_CAP_MB))
        self._max_geo_per_hour         = int(  _cfg("ops", "max_geo_events_per_hour",
                                                     _DEFAULT_MAX_GEO_EVENTS_PER_HOUR))
        self._max_patch_per_min        = int(  _cfg("ops", "max_patch_batches_per_min",
                                                     _DEFAULT_MAX_PATCH_BATCHES_PER_MIN))
        reset_flag_raw                 = _cfg("ops", "reset_flag_path",
                                              _DEFAULT_RESET_FLAG_PATH)
        self._reset_flag_path          = Path(str(reset_flag_raw))

        # ---- Metrics -------------------------------------------------------
        self._ws_msgs_in:         int   = 0
        self._ws_msgs_out:        int   = 0
        self._climate_tick_count: int   = 0
        self._geo_event_total:    int   = 0
        self._patch_count_total:  int   = 0
        self._snapshot_count:     int   = 0
        self._disk_write_failures: int  = 0
        self._last_world_tick_ts: float = time.monotonic()

        # External: set by NetworkServer to expose job-queue depth
        self.job_queue_depth: int = 0

        # ---- Guard state ---------------------------------------------------
        self._geo_hour_count:   int   = 0
        self._geo_hour_start:   float = time.monotonic()
        self._patch_min_count:  int   = 0
        self._patch_min_start:  float = time.monotonic()
        self._event_cap_active: bool  = False

        # ---- Reset state ---------------------------------------------------
        self._reset_pending: bool                 = False
        self._on_reset: Optional[Callable]        = None   # coroutine callback

        # ---- Compaction state ----------------------------------------------
        self._last_baseline_ts: float = time.monotonic()

        # ---- Structured log ------------------------------------------------
        self._log_dir  = Path(state_dir).parent / _LOG_DIR
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._log_path: Optional[Path] = None
        self._log_file = None
        self._open_log()

    # ==========================================================================
    # Structured logging
    # ==========================================================================

    def _open_log(self) -> None:
        self._log_path = self._log_dir / _LOG_FILENAME
        try:
            self._log_file = open(self._log_path, "a", encoding="utf-8")
        except OSError:
            self._log_file = None

    def log(
        self,
        cat:   str,
        msg:   str,
        level: str = "INFO",
        data:  Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append a JSON-lines structured log entry."""
        if self._log_file is None:
            return
        record: Dict[str, Any] = {
            "ts":      datetime.now(timezone.utc).isoformat(),
            "level":   level,
            "cat":     cat,
            "worldId": self._world_state.world_id if self._world_state else "",
            "msg":     msg,
        }
        if data:
            record["data"] = data
        try:
            self._log_file.write(json.dumps(record) + "\n")
            self._log_file.flush()
        except OSError:
            self._disk_write_failures += 1
        self._maybe_rotate_log()

    def _maybe_rotate_log(self) -> None:
        if self._log_path is None or not self._log_path.exists():
            return
        try:
            size_mb = self._log_path.stat().st_size / (1024 * 1024)
        except OSError:
            return
        if size_mb < self._log_rotate_mb:
            return

        # Close
        if self._log_file is not None:
            try:
                self._log_file.close()
            except OSError:
                pass
            self._log_file = None

        # Compress + rename
        ts_str   = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        rotated  = self._log_dir / f"server.log.{ts_str}.jsonl.gz"
        try:
            with open(self._log_path, "rb") as fin, \
                 gzip.open(rotated, "wb") as fout:
                shutil.copyfileobj(fin, fout)
            self._log_path.unlink()
        except OSError:
            self._disk_write_failures += 1

        self._prune_log_files()
        self._open_log()

    def _prune_log_files(self) -> None:
        """Delete oldest rotated log files beyond the keep limit."""
        rotated = sorted(self._log_dir.glob("server.log.*.jsonl.gz"))
        excess  = rotated[: max(0, len(rotated) - self._log_keep_files)]
        for f in excess:
            try:
                f.unlink()
            except OSError:
                pass

    def prune_logs(self) -> None:
        """Public: prune excess rotated log files now."""
        self._prune_log_files()

    # ==========================================================================
    # Health
    # ==========================================================================

    def health(self) -> Dict[str, Any]:
        """Return a health-status dict suitable for the ``/health`` endpoint."""
        uptime_sec    = time.monotonic() - self._start_time
        players       = (
            len(self._registry.all_players()) if self._registry else 0
        )
        tick_age_ms   = (time.monotonic() - self._last_world_tick_ts) * 1000.0
        degraded      = (
            tick_age_ms > _DEGRADED_TICK_AGE_MS
            or self.job_queue_depth > _DEGRADED_JOB_DEPTH
            or self._disk_write_failures > 0
        )
        return {
            "status":             "degraded" if degraded else "ok",
            "uptimeSec":          round(uptime_sec, 1),
            "buildId":            self._build_id,
            "worldId":            self._world_state.world_id if self._world_state else "",
            "playersConnected":   players,
            "wsBacklog":          0,
            "jobQueueDepth":      self.job_queue_depth,
            "lastWorldTickAgeMs": round(tick_age_ms, 1),
        }

    # ==========================================================================
    # Metrics
    # ==========================================================================

    def metrics(self) -> str:
        """Return Prometheus-like text metrics for the ``/metrics`` endpoint."""
        players  = (
            len(self._registry.all_players()) if self._registry else 0
        )
        ws_bytes = self._world_state_bytes()
        lines: List[str] = [
            f"players_connected {players}",
            f"ws_msgs_in_total {self._ws_msgs_in}",
            f"ws_msgs_out_total {self._ws_msgs_out}",
            f"climate_tick_count_total {self._climate_tick_count}",
            f"geo_event_count_total {self._geo_event_total}",
            f"patch_count_total {self._patch_count_total}",
            f"snapshot_count_total {self._snapshot_count}",
            f"world_state_bytes_total {ws_bytes}",
            f"job_queue_depth {self.job_queue_depth}",
            f"disk_write_failures_total {self._disk_write_failures}",
        ]
        return "\n".join(lines) + "\n"

    def _world_state_bytes(self) -> int:
        total = 0
        if self._state_dir.is_dir():
            for p in self._state_dir.rglob("*"):
                if p.is_file():
                    try:
                        total += p.stat().st_size
                    except OSError:
                        pass
        return total

    # ==========================================================================
    # Tick / event notifications (called by NetworkServer)
    # ==========================================================================

    def on_world_tick(self) -> None:
        self._last_world_tick_ts = time.monotonic()

    def on_ws_msg_in(self) -> None:
        self._ws_msgs_in += 1

    def on_ws_msg_out(self) -> None:
        self._ws_msgs_out += 1

    def on_climate_tick(self) -> None:
        self._climate_tick_count += 1

    def on_snapshot(self) -> None:
        self._snapshot_count += 1

    # ==========================================================================
    # Safety guards
    # ==========================================================================

    def check_geo_event_cap(self) -> bool:
        """Return *True* if this geo event is within the hourly rate cap."""
        now = time.monotonic()
        if now - self._geo_hour_start >= 3600.0:
            self._geo_hour_count  = 0
            self._geo_hour_start  = now
            self._event_cap_active = False

        self._geo_hour_count  += 1
        self._geo_event_total += 1

        if self._geo_hour_count > self._max_geo_per_hour:
            if not self._event_cap_active:
                self._event_cap_active = True
                self.log(CAT_OPS, "EVENT_CAP_TRIGGERED", level="WARN",
                         data={"geo_per_hour": self._geo_hour_count})
            return False
        return True

    def check_patch_batch_cap(self) -> bool:
        """Return *True* if this patch batch is within the per-minute cap."""
        now = time.monotonic()
        if now - self._patch_min_start >= 60.0:
            self._patch_min_count = 0
            self._patch_min_start = now

        self._patch_min_count   += 1
        self._patch_count_total += 1

        if self._patch_min_count > self._max_patch_per_min:
            self.log(CAT_OPS, "PATCH_CAP_TRIGGERED", level="WARN",
                     data={"patch_per_min": self._patch_min_count})
            return False
        return True

    def check_nan_in_value(self, value: float, context: str) -> bool:
        """Return *True* if *value* is finite; *False* and log on NaN/Inf."""
        if math.isfinite(value):
            return True
        self.log(
            CAT_ERROR,
            f"NaN/Inf detected in {context}",
            level="ERROR",
            data={"value": str(value), "context": context},
        )
        return False

    def check_storage_cap(self) -> bool:
        """Return *False* and trigger compaction when storage exceeds the cap."""
        size_mb = self._world_state_bytes() / (1024 * 1024)
        if size_mb > self._world_state_cap_mb:
            self.log(CAT_OPS, "STORAGE_CAP_TRIGGERED", level="WARN",
                     data={"size_mb": round(size_mb, 1),
                           "cap_mb":  self._world_state_cap_mb})
            self.compact()
            return False
        return True

    # ==========================================================================
    # Compaction
    # ==========================================================================

    def maybe_compact(self) -> bool:
        """Compact if the time/size threshold is exceeded.

        Returns *True* when compaction was performed.
        """
        elapsed_min = (time.monotonic() - self._last_baseline_ts) / 60.0
        if elapsed_min >= self._baseline_interval_min:
            return self.compact()
        size_mb = self._world_state_bytes() / (1024 * 1024)
        if size_mb >= self._baseline_size_trigger_mb:
            return self.compact()
        return False

    def compact(self) -> bool:
        """Create a baseline snapshot and prune old geo-event delta logs.

        Returns *True* on success, *False* on failure.
        """
        if self._world_state is None:
            return False
        try:
            ts  = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            bld = self._state_dir / f"{_BASELINE_PREFIX}{ts}"
            bld.mkdir(parents=True, exist_ok=True)

            # world.json
            (bld / "world.json").write_text(
                json.dumps({
                    "seed":        self._world_state.seed,
                    "worldId":     self._world_state.world_id,
                    "simTime":     self._world_state.sim_time,
                    "epoch":       self._world_state.epoch,
                    "baseline_ts": ts,
                }, indent=2),
                encoding="utf-8",
            )

            # climate.snapshot
            climate_snap = self._world_state.load_climate_snapshot()
            if climate_snap is not None:
                (bld / "climate.snapshot").write_text(
                    json.dumps(climate_snap, indent=2), encoding="utf-8"
                )

            # geo_events.snapshot
            geo_events = self._world_state.geo_events()
            (bld / "geo_events.snapshot").write_text(
                json.dumps(geo_events, indent=2), encoding="utf-8"
            )

            # Archive + prune delta logs
            self._rotate_geo_delta(ts)

            self._last_baseline_ts = time.monotonic()
            self.log(CAT_SAVE, f"Baseline created: {bld.name}",
                     data={"geo_events": len(geo_events)})
            return True
        except Exception as exc:
            self._disk_write_failures += 1
            self.log(CAT_ERROR, f"Compact failed: {exc}", level="ERROR")
            return False

    def _rotate_geo_delta(self, ts: str) -> None:
        """Rename current geo_events.jsonl to a delta file; prune old deltas."""
        geo_path = self._state_dir / "geo_events.jsonl"
        if geo_path.exists():
            delta_name = f"{_DELTA_PREFIX}{ts}.jsonl"
            try:
                geo_path.rename(self._state_dir / delta_name)
            except OSError:
                pass

        deltas = sorted(self._state_dir.glob(f"{_DELTA_PREFIX}*.jsonl"))
        excess = deltas[: max(0, len(deltas) - self._log_keep_files)]
        for f in excess:
            try:
                f.unlink()
            except OSError:
                pass

    # ==========================================================================
    # Soft reset
    # ==========================================================================

    def check_reset_flag(self) -> bool:
        """Return *True* when the ``RESET_NOW`` flag file is present."""
        return self._reset_flag_path.exists()

    def trigger_reset(self) -> None:
        """Schedule a soft world reset on the next ops tick."""
        self._reset_pending = True

    def set_reset_callback(self, callback: Callable) -> None:
        """Register an async callback invoked after the world is recreated.

        Signature::

            async def on_reset(new_world_id: str, new_seed: int,
                               new_sim_time: float) -> None: ...
        """
        self._on_reset = callback

    async def maybe_reset(self) -> bool:
        """Execute a soft reset if one is pending or the flag file exists.

        Returns *True* when a reset was performed.
        """
        if not self._reset_pending and not self.check_reset_flag():
            return False

        self._reset_pending = False
        # Remove flag file
        try:
            if self._reset_flag_path.exists():
                self._reset_flag_path.unlink()
        except OSError:
            pass

        await self._do_reset()
        return True

    async def _do_reset(self) -> None:
        old_id = self._world_state.world_id if self._world_state else ""
        self.log(CAT_OPS, f"Soft reset initiated, old worldId={old_id}")

        # 1. Final baseline (best-effort)
        self.compact()

        # 2. Archive old world state (best-effort, with timeout guard)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        archive_root = self._state_dir.parent / _ARCHIVE_DIR
        archive_root.mkdir(parents=True, exist_ok=True)
        dest = archive_root / f"world_{old_id}_{ts}"
        try:
            if self._state_dir.exists():
                shutil.copytree(str(self._state_dir), str(dest))
        except Exception as exc:
            self.log(CAT_OPS,
                     f"Archive skipped (non-fatal): {exc}", level="WARN")

        # 3. Reset world state → new worldId
        if self._world_state is not None:
            old_seed = self._world_state.seed
            self._world_state.reset()
            self._world_state.seed     = old_seed
            self._world_state.world_id = str(uuid.uuid4())
            self._world_state.save()

        # 4. Notify connected clients via callback
        if self._on_reset is not None and self._world_state is not None:
            try:
                await self._on_reset(
                    self._world_state.world_id,
                    self._world_state.seed,
                    self._world_state.sim_time,
                )
            except Exception as exc:
                self.log(CAT_ERROR,
                         f"Reset callback failed: {exc}", level="ERROR")

        new_id = self._world_state.world_id if self._world_state else ""
        self.log(CAT_OPS, f"Soft reset complete, new worldId={new_id}")
