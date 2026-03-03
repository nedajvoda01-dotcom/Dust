"""NetDiagnostics — Stage 58 developer-mode network diagnostics.

Collects RTT samples, correction magnitudes, packet counts and remote
interpolation delay for offline analysis and in-process reporting.

In production (``dev.enable_dev = False``) all collection methods are
no-ops so there is zero runtime cost.

Public API
----------
NetDiagnostics(config)
    .record_rtt(rtt_s)
        Record one round-trip time sample.
    .record_correction(magnitude_m)
        Record one reconciliation correction magnitude.
    .record_packet_in()
        Increment the received-packet counter.
    .record_packet_out()
        Increment the sent-packet counter.
    .record_interp_delay(delay_s)
        Record the current remote interpolation delay.
    .record_dropped_frame()
        Increment the dropped-frame counter.
    .get_summary() → dict
        Return a snapshot of all diagnostics.
    .reset()
        Clear all collected samples.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Deque, List, Optional


class NetDiagnostics:
    """Developer-mode network diagnostics collector.

    Parameters
    ----------
    config : dict
        Full game config dict; reads ``dev.enable_dev`` (default True)
        and ``dev.net_diag_buffer_size`` (default 256).
    """

    _DEFAULT_BUFFER = 256

    def __init__(self, config: Optional[dict] = None) -> None:
        devcfg = (config or {}).get("dev", {}) or {}
        self._enabled   = bool(devcfg.get("enable_dev", True))
        buf             = int(devcfg.get("net_diag_buffer_size", self._DEFAULT_BUFFER))
        self._buf_size  = buf

        self._rtt_samples:        Deque[float] = deque(maxlen=buf)
        self._correction_samples: Deque[float] = deque(maxlen=buf)
        self._interp_delay_samples: Deque[float] = deque(maxlen=buf)

        self._packets_in:     int = 0
        self._packets_out:    int = 0
        self._dropped_frames: int = 0

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_rtt(self, rtt_s: float) -> None:
        """Record one RTT sample (seconds)."""
        if self._enabled:
            self._rtt_samples.append(float(rtt_s))

    def record_correction(self, magnitude_m: float) -> None:
        """Record one reconciliation correction magnitude (metres)."""
        if self._enabled:
            self._correction_samples.append(float(magnitude_m))

    def record_packet_in(self) -> None:
        """Increment the received-packet counter."""
        if self._enabled:
            self._packets_in += 1

    def record_packet_out(self) -> None:
        """Increment the sent-packet counter."""
        if self._enabled:
            self._packets_out += 1

    def record_interp_delay(self, delay_s: float) -> None:
        """Record the current remote interpolation delay (seconds)."""
        if self._enabled:
            self._interp_delay_samples.append(float(delay_s))

    def record_dropped_frame(self) -> None:
        """Increment the dropped-frame counter."""
        if self._enabled:
            self._dropped_frames += 1

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def get_summary(self) -> dict:
        """Return a snapshot dict of all collected diagnostics.

        Returns
        -------
        dict with keys:
            rtt_mean_ms, rtt_max_ms, rtt_samples,
            correction_mean_m, correction_max_m, correction_samples,
            interp_delay_mean_ms, interp_delay_samples,
            packets_in, packets_out, dropped_frames.
        """
        def _stats(samples: Deque[float], scale: float = 1.0) -> tuple:
            if not samples:
                return 0.0, 0.0, 0
            mean = sum(samples) / len(samples) * scale
            mx   = max(samples) * scale
            return round(mean, 4), round(mx, 4), len(samples)

        rtt_mean, rtt_max, rtt_n        = _stats(self._rtt_samples, 1000.0)
        cor_mean, cor_max, cor_n        = _stats(self._correction_samples)
        dly_mean, dly_max, dly_n        = _stats(self._interp_delay_samples, 1000.0)

        return {
            "rtt_mean_ms":          rtt_mean,
            "rtt_max_ms":           rtt_max,
            "rtt_samples":          rtt_n,
            "correction_mean_m":    cor_mean,
            "correction_max_m":     cor_max,
            "correction_samples":   cor_n,
            "interp_delay_mean_ms": dly_mean,
            "interp_delay_samples": dly_n,
            "packets_in":           self._packets_in,
            "packets_out":          self._packets_out,
            "dropped_frames":       self._dropped_frames,
        }

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all samples and counters."""
        self._rtt_samples.clear()
        self._correction_samples.clear()
        self._interp_delay_samples.clear()
        self._packets_in     = 0
        self._packets_out    = 0
        self._dropped_frames = 0
