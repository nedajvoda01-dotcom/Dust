"""CrashWatchdog — Stage 59 tick-loop watchdog timer.

Monitors the simulation tick loop.  If the loop stops calling
``kick()`` within ``tick_timeout_ms`` milliseconds, the watchdog
fires a configurable callback (default: log a critical warning).

The watchdog runs as a background asyncio task and is intentionally
lightweight — it performs no I/O beyond the callback.

Public API
----------
CrashWatchdog(tick_timeout_ms=5000, on_timeout=None)
  .kick()           — call from tick loop each tick
  .start()          — start background monitoring coroutine
  .stop()           — cancel monitoring task
  .is_alive         — True if last kick was within the timeout window
  await .run()      — coroutine suitable for asyncio.create_task()
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Callable, Optional

_LOG = logging.getLogger(__name__)


class CrashWatchdog:
    """Detects tick-loop freezes and calls *on_timeout*.

    Parameters
    ----------
    tick_timeout_ms :
        Milliseconds of silence before the watchdog fires.
    on_timeout :
        Async or sync callable invoked when the watchdog fires.
        Receives no arguments.  Defaults to a ``CRITICAL`` log line.
    poll_interval_ms :
        How often (ms) the watchdog checks the last-kick timestamp.
    """

    def __init__(
        self,
        tick_timeout_ms: int = 5_000,
        on_timeout: Optional[Callable] = None,
        poll_interval_ms: int = 500,
    ) -> None:
        self._timeout_sec  = tick_timeout_ms / 1_000.0
        self._poll_sec     = poll_interval_ms / 1_000.0
        self._on_timeout   = on_timeout or self._default_timeout
        self._last_kick_ts = time.monotonic()
        self._task: Optional[asyncio.Task] = None
        self._fired        = False

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def kick(self) -> None:
        """Reset the watchdog timer.  Call from every tick."""
        self._last_kick_ts = time.monotonic()
        self._fired = False

    @property
    def is_alive(self) -> bool:
        """True if the last kick was within the timeout window."""
        return (time.monotonic() - self._last_kick_ts) < self._timeout_sec

    def start(self) -> None:
        """Schedule the watchdog coroutine on the running event loop."""
        loop = asyncio.get_running_loop()
        self._task = loop.create_task(self.run())

    def stop(self) -> None:
        """Cancel the watchdog task."""
        if self._task is not None:
            self._task.cancel()
            self._task = None

    async def run(self) -> None:
        """Watchdog monitoring coroutine.

        Polls ``_last_kick_ts`` every ``poll_interval_ms``.
        When the timeout elapses, calls ``on_timeout`` once and
        continues monitoring (so repeated firings are suppressed until
        a new kick resets ``_fired``).
        """
        try:
            while True:
                await asyncio.sleep(self._poll_sec)
                elapsed = time.monotonic() - self._last_kick_ts
                if elapsed >= self._timeout_sec and not self._fired:
                    self._fired = True
                    result = self._on_timeout()
                    if asyncio.iscoroutine(result):
                        await result
        except asyncio.CancelledError:
            pass

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    @staticmethod
    def _default_timeout() -> None:
        _LOG.critical(
            "CrashWatchdog: tick loop timeout — server may be frozen"
        )
