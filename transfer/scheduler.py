"""Synchronous expert-transfer scheduler for MoEInfra.

TransferScheduler owns request ordering, deduplication, cancellation, and
transfer bookkeeping. CacheManager owns the actual expert objects and their
CPU/GPU residency.

This is intentionally synchronous. Async CUDA streams will be introduced only
after the synchronous state transitions are validated.
"""

from __future__ import annotations

import heapq
import logging
import time
from typing import Optional

from transfer.bandwidth import BandwidthMonitor
from transfer.types import (
    TransferDirection,
    TransferRequest,
    TransferResult,
)


class TransferScheduler:
    """Priority-queue scheduler for expert residency transfers."""

    def __init__(
        self,
        cache_manager,
        bandwidth_gbps: float,
        max_concurrent: int,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._cache_manager = cache_manager
        self.bandwidth_gbps = bandwidth_gbps
        self.max_concurrent = max_concurrent
        self._logger = logger or logging.getLogger(__name__)
        self._bw_monitor = BandwidthMonitor(window_s=5.0)

        # Heap entries: ((priority.value, issued_at), request)
        self._heap: list[tuple[tuple[int, float], TransferRequest]] = []

        # Kept separately for simple membership/cancellation.
        self._queue: list[TransferRequest] = []
        self._in_flight: list[TransferRequest] = []

    def _is_duplicate(self, request: TransferRequest) -> bool:
        """Return True if an equivalent request is queued or executing."""
        triple = (
            request.layer_id,
            request.expert_id,
            request.direction,
        )

        for existing in (*self._queue, *self._in_flight):
            if existing.request_id == request.request_id:
                return True
            if (
                existing.layer_id,
                existing.expert_id,
                existing.direction,
            ) == triple:
                return True

        return False

    @staticmethod
    def _sort_key(request: TransferRequest) -> tuple[int, float]:
        """Return the heap ordering key."""
        return (request.priority.value, request.issued_at)

    def submit(self, request: TransferRequest) -> bool:
        """Queue a request unless an equivalent request already exists."""
        if self._is_duplicate(request):
            self._logger.debug(
                "submit: duplicate dropped id=%s layer=%d expert=%d direction=%s",
                request.request_id,
                request.layer_id,
                request.expert_id,
                request.direction.value,
            )
            return False

        heapq.heappush(
            self._heap,
            (self._sort_key(request), request),
        )
        self._queue.append(request)

        self._logger.debug(
            "submit: queued id=%s layer=%d expert=%d direction=%s priority=%s",
            request.request_id,
            request.layer_id,
            request.expert_id,
            request.direction.value,
            request.priority.name,
        )
        return True

    def execute_next(self) -> Optional[TransferResult]:
        """Execute the highest-priority pending transfer synchronously.

        CPU→GPU delegates to CacheManager.promote_to_gpu().
        GPU→CPU delegates to CacheManager.demote_to_cpu().
        """
        if not self._heap:
            return None

        _, request = heapq.heappop(self._heap)
        self._queue.remove(request)
        self._in_flight.append(request)

        started = time.perf_counter()
        bytes_transferred = 0
        error: str | None = None
        success = False

        try:
            bytes_transferred = self._cache_manager.get_size_bytes(
                request.layer_id,
                request.expert_id,
            )

            if request.direction == TransferDirection.CPU_TO_GPU:
                success = self._cache_manager.promote_to_gpu(
                    request.layer_id,
                    request.expert_id,
                )
            elif request.direction == TransferDirection.GPU_TO_CPU:
                success = self._cache_manager.demote_to_cpu(
                    request.layer_id,
                    request.expert_id,
                )
            else:
                raise ValueError(
                    f"Unsupported transfer direction: {request.direction!r}"
                )

            if not success:
                error = (
                    f"Transfer failed for layer={request.layer_id}, "
                    f"expert={request.expert_id}, "
                    f"direction={request.direction.value}"
                )

        except Exception as exc:  # pylint: disable=broad-except
            error = str(exc)
            self._logger.error(
                "execute_next: FAILED id=%s error=%s",
                request.request_id,
                error,
            )

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._in_flight.remove(request)

        if success:
            self._bw_monitor.record(bytes_transferred)
            self._logger.debug(
                "execute_next: done id=%s bytes=%d elapsed=%.2f ms",
                request.request_id,
                bytes_transferred,
                elapsed_ms,
            )

        return TransferResult(
            request_id=request.request_id,
            success=success,
            elapsed_ms=elapsed_ms,
            bytes_transferred=bytes_transferred if success else 0,
            error=error,
        )

    def drain(self) -> list[TransferResult]:
        """Execute all pending requests in priority order."""
        results: list[TransferResult] = []

        while self._heap:
            result = self.execute_next()
            if result is not None:
                results.append(result)

        return results

    def cancel(self, request_id: str) -> bool:
        """Cancel a queued request."""
        target = next(
            (request for request in self._queue
             if request.request_id == request_id),
            None,
        )

        if target is None:
            return False

        self._queue.remove(target)
        self._heap = [
            (self._sort_key(request), request)
            for request in self._queue
        ]
        heapq.heapify(self._heap)

        self._logger.debug(
            "cancel: removed id=%s layer=%d expert=%d",
            target.request_id,
            target.layer_id,
            target.expert_id,
        )
        return True

    def pending_count(self) -> int:
        """Return the number of queued requests."""
        return len(self._queue)

    @property
    def bandwidth_monitor(self) -> BandwidthMonitor:
        """Return the rolling transfer-bandwidth monitor."""
        return self._bw_monitor

    def __repr__(self) -> str:
        return (
            "TransferScheduler("
            f"pending={len(self._queue)}, "
            f"in_flight={len(self._in_flight)}, "
            f"bandwidth={self.bandwidth_gbps:.1f} GB/s)"
        )
