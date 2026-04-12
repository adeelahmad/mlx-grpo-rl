"""Batched generation collector for group-level generate_batch() fusion.

When multiple RolloutContexts (group members) each call ctx.generate(),
the collector accumulates their requests and flushes them as a single
generate_batch() call. Each context blocks until its result is ready.

Thread-safe: each member runs in its own thread, the collector
coordinates via a Barrier + Lock.

Deadlock-safe: members that finish early (short-circuit, error) call
mark_done() to reduce the expected count. The barrier dynamically
adjusts so remaining members aren't stuck waiting.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any


@dataclass
class _PendingRequest:
    """A single generate() call waiting for batched execution."""

    member_id: int
    prefix_tokens: list[int]
    params: Any  # GenerateParams
    result: Any = None  # GenerateOutput, set by _flush
    ready: threading.Event = field(default_factory=threading.Event)


class BatchCollector:
    """Collect generate() calls from N group members, flush as generate_batch().

    Each member thread calls submit() which blocks until the batch executes.
    When all active members have submitted (or been marked done), the batch
    fires and results are distributed.

    Args:
        backend: Must implement generate_batch().
        group_size: Number of members that will submit requests.
    """

    def __init__(self, backend: Any, group_size: int):
        self._backend = backend
        self._expected = group_size
        self._lock = threading.Lock()
        self._pending: dict[int, _PendingRequest] = {}
        self._done_members: set[int] = set()
        self._round = 0

    def submit(self, member_id: int, prefix_tokens: list[int], params: Any) -> Any:
        """Submit a generate request. Blocks until batched result is ready."""
        req = _PendingRequest(member_id=member_id, prefix_tokens=prefix_tokens, params=params)

        should_flush = False
        with self._lock:
            self._pending[member_id] = req
            if len(self._pending) + len(self._done_members) >= self._expected:
                should_flush = True

        if should_flush:
            self._flush()

        req.ready.wait()
        return req.result

    def mark_done(self, member_id: int) -> None:
        """Mark a member as done (won't submit any more requests).

        Call this when a member's rollout finishes or short-circuits
        to prevent deadlock from other members waiting for it.
        """
        should_flush = False
        with self._lock:
            self._done_members.add(member_id)
            # Check if remaining active members have all submitted
            if len(self._pending) + len(self._done_members) >= self._expected:
                should_flush = True

        if should_flush and self._pending:
            self._flush()

    def _flush(self) -> None:
        """Execute generate_batch() for all pending requests."""
        with self._lock:
            if not self._pending:
                return
            requests = dict(self._pending)
            self._pending.clear()
            self._round += 1

        ordered = sorted(requests.values(), key=lambda r: r.member_id)
        prefix_batch = [r.prefix_tokens for r in ordered]
        params = ordered[0].params

        if len(ordered) == 1:
            # Single request — use generate() directly (no batch overhead)
            result = self._backend.generate(prefix_batch[0], params)
            ordered[0].result = result
            ordered[0].ready.set()
            return

        results = self._backend.generate_batch(prefix_batch, params)

        for req, result in zip(ordered, results, strict=False):
            req.result = result
            req.ready.set()

    def reset(self) -> None:
        """Reset for the next generation round (same group, next segment)."""
        with self._lock:
            self._pending.clear()
            self._done_members.clear()
            self._round += 1
