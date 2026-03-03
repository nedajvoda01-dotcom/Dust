"""RateLimiter — Stage 58 server-side per-client rate limiter.

Uses a token-bucket algorithm to enforce per-client message rate caps.
The bucket refills at the configured rate; each received message costs one
token.  When the bucket is empty the message is dropped.

Public API
----------
RateLimiter(rate_hz, burst_multiplier)
    .allow(player_id, now_s) → bool
        Return True and deduct a token if the client is within budget.
    .remove(player_id)
        Drop a player's bucket on disconnect.
    .reset(player_id)
        Reset a player's bucket to full (e.g. after reconnect).
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple


class RateLimiter:
    """Token-bucket rate limiter for per-client message ingestion.

    Parameters
    ----------
    rate_hz : float
        Sustained allowed rate (tokens per second).  Default 30.
    burst_multiplier : float
        Bucket capacity = rate_hz * burst_multiplier.  Default 2.0
        (allows short bursts up to 2× the sustained rate).
    """

    def __init__(
        self,
        rate_hz:          float = 30.0,
        burst_multiplier: float = 2.0,
    ) -> None:
        self._rate:      float = max(1.0, float(rate_hz))
        self._capacity:  float = self._rate * max(1.0, float(burst_multiplier))
        # (tokens, last_refill_time)
        self._buckets: Dict[str, Tuple[float, float]] = {}

    # ------------------------------------------------------------------

    def allow(self, player_id: str, now_s: float) -> bool:
        """Check and consume one token for *player_id*.

        Returns
        -------
        bool
            True if the message is allowed; False if the bucket is empty.
        """
        tokens, last_t = self._buckets.get(player_id, (self._capacity, now_s))

        # Refill
        elapsed = now_s - last_t
        tokens = min(self._capacity, tokens + elapsed * self._rate)

        if tokens < 1.0:
            self._buckets[player_id] = (tokens, now_s)
            return False

        tokens -= 1.0
        self._buckets[player_id] = (tokens, now_s)
        return True

    def remove(self, player_id: str) -> None:
        """Remove a player's bucket on disconnect."""
        self._buckets.pop(player_id, None)

    def reset(self, player_id: str) -> None:
        """Reset a player's bucket to full capacity."""
        self._buckets[player_id] = (self._capacity, 0.0)
