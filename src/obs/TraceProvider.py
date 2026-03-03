"""TraceProvider — Stage 61 lightweight span-based tracing.

Provides a minimal tracing API that records named spans with start/end
timestamps and metadata.  No external dependency; spans are stored in memory
and exportable as plain dicts.

Canonical span names
--------------------
* ``TickLoop``
* ``QP_Solve``
* ``ChunkUpdate``
* ``MaterialPhaseTick``
* ``InstabilityTick``
* ``SnapshotWrite``
* ``NetBroadcast``
"""
from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional


CANONICAL_SPANS = (
    "TickLoop",
    "QP_Solve",
    "ChunkUpdate",
    "MaterialPhaseTick",
    "InstabilityTick",
    "SnapshotWrite",
    "NetBroadcast",
)


class Span:
    """A single trace span."""

    def __init__(
        self,
        name:       str,
        trace_id:   str,
        parent_id:  Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.name       = name
        self.trace_id   = trace_id
        self.span_id    = uuid.uuid4().hex[:16]
        self.parent_id  = parent_id
        self.attributes: Dict[str, Any] = attributes or {}
        self.start_ns   = time.monotonic_ns()
        self.end_ns:    Optional[int] = None

    def finish(self) -> None:
        if self.end_ns is None:
            self.end_ns = time.monotonic_ns()

    @property
    def duration_ms(self) -> float:
        if self.end_ns is None:
            return 0.0
        return (self.end_ns - self.start_ns) / 1_000_000.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name":        self.name,
            "trace_id":    self.trace_id,
            "span_id":     self.span_id,
            "parent_id":   self.parent_id,
            "attributes":  self.attributes,
            "start_ns":    self.start_ns,
            "end_ns":      self.end_ns,
            "duration_ms": self.duration_ms,
        }


class TraceProvider:
    """Lightweight in-process span collector.

    Parameters
    ----------
    sampling_rate:
        Fraction of traces to record (0.0–1.0).  1.0 = record all.
    max_spans:
        Maximum number of finished spans to keep in memory.
    """

    def __init__(
        self,
        sampling_rate: float = 1.0,
        max_spans:     int   = 10_000,
    ) -> None:
        self._sampling_rate = max(0.0, min(1.0, sampling_rate))
        self._max_spans     = max_spans
        self._finished:     List[Span] = []
        self._active:       Dict[str, Span] = {}

    # ------------------------------------------------------------------
    # Context-manager API
    # ------------------------------------------------------------------

    @contextmanager
    def span(
        self,
        name:       str,
        trace_id:   Optional[str]       = None,
        parent_id:  Optional[str]       = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Generator[Span, None, None]:
        """Execute a block inside a named span.

        Usage::

            with tracer.span("TickLoop", trace_id=tid) as s:
                s.attributes["world_tick"] = tick
                run_tick()
        """
        import random
        should_sample = random.random() < self._sampling_rate
        if trace_id is None:
            trace_id = uuid.uuid4().hex
        s = Span(name, trace_id, parent_id=parent_id, attributes=attributes or {})
        try:
            yield s
        finally:
            s.finish()
            if should_sample:
                self._finished.append(s)
                if len(self._finished) > self._max_spans:
                    self._finished = self._finished[-self._max_spans:]

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def recent_spans(self, name: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Return the most recent finished spans as dicts."""
        spans = self._finished if name is None else [s for s in self._finished if s.name == name]
        return [s.to_dict() for s in spans[-limit:]]

    def avg_duration_ms(self, name: str) -> float:
        """Return the average duration of recently finished spans with the given name."""
        spans = [s for s in self._finished if s.name == name]
        if not spans:
            return 0.0
        return sum(s.duration_ms for s in spans) / len(spans)

    def clear(self) -> None:
        self._finished.clear()
        self._active.clear()
