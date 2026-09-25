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
import torch

from transfer.bandwidth import BandwidthMonitor
from transfer.types import (
    TransferDirection,
    TransferRequest,
    TransferResult,
)
from transfer.nf4_transfer import transfer_staged_expert_to_gpu
from transfer.nf4_reconstruct import ReconstructedNF4Expert
from transfer.types import (
    TransferDirection,
    TransferRequest,
    TransferResult,
    TransferStatus,
    TransferHandle,
)


class TransferScheduler:
    """Priority-queue scheduler for expert residency transfers."""

    def __init__(
        self,
        cache_manager,
        bandwidth_gbps: float,
        max_concurrent: int,
        staging_pool=None,
        transfer_stream=None,
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

        self._staging_pool = staging_pool
        self._transfer_stream = transfer_stream
        self._async_in_flight: dict[
            tuple[int, int], TransferHandle
        ] = {}

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

    def submit_async(self, request: TransferRequest) -> TransferHandle:
        key = (request.layer_id, request.expert_id)

        # Already resident.
        if self._cache_manager.is_gpu_resident(
            request.layer_id,
            request.expert_id,
        ):
            return TransferHandle(
                request_id=request.request_id,
                layer_id=request.layer_id,
                expert_id=request.expert_id,
                slot=None,
                event=None,
                status=TransferStatus.READY,
            )

        # Already being transferred.
        existing = self._async_in_flight.get(key)
        if existing is not None:
            return existing

        if self._staging_pool is None or self._transfer_stream is None:
            raise RuntimeError(
                "Async transfer requires staging_pool and transfer_stream"
            )

        slot = self._staging_pool.acquire()

        if slot is None:
            return TransferHandle(
                request_id=request.request_id,
                layer_id=request.layer_id,
                expert_id=request.expert_id,
                slot=None,
                event=None,
                status=TransferStatus.REJECTED,
            )

        expert = self._cache_manager.peek_cpu(
            request.layer_id,
            request.expert_id,
        )

        if expert is None:
            self._staging_pool.release(slot)

            return TransferHandle(
                request_id=request.request_id,
                layer_id=request.layer_id,
                expert_id=request.expert_id,
                slot=None,
                event=None,
                status=TransferStatus.REJECTED,
            )

        try:
            self._staging_pool.stage_expert(
                slot,
                expert,
                request.layer_id,
                request.expert_id,
            )

            gpu_state = transfer_staged_expert_to_gpu(
                slot,
                self._transfer_stream,
            )

            event = torch.cuda.Event()
            with torch.cuda.stream(self._transfer_stream):
                event.record(self._transfer_stream)

            handle = TransferHandle(
                request_id=request.request_id,
                layer_id=request.layer_id,
                expert_id=request.expert_id,
                slot=slot,
                event=event,
                status=TransferStatus.IN_FLIGHT,
            )

            # Keep gpu_state alive until reconstruction.
            handle.gpu_state = gpu_state
            handle.cpu_expert = expert

            self._async_in_flight[key] = handle

            return handle

        except Exception:
            self._staging_pool.release(slot)
            raise

    def poll_async(
        self,
        layer_id: int,
        expert_id: int,
    ) -> TransferStatus:
        key = (layer_id, expert_id)
        handle = self._async_in_flight.get(key)

        if handle is None:
            return TransferStatus.REJECTED

        if handle.status == TransferStatus.CANCELLED:
            return TransferStatus.CANCELLED

        if not handle.event.query():
            return TransferStatus.IN_FLIGHT

        # H2D has completed. Now reconstruct.
        try:
            reconstructed = ReconstructedNF4Expert(
                handle.cpu_expert,
                handle.gpu_state,
            )

            self._cache_manager.put(
                layer_id,
                expert_id,
                reconstructed,
                device="cuda",
            )

            handle.status = TransferStatus.READY

            self._staging_pool.mark_transfer_complete(
                handle.slot,
                self._transfer_stream,
            )
            self._staging_pool.release(handle.slot)

            del self._async_in_flight[key]

            return TransferStatus.READY

        except Exception:
            handle.status = TransferStatus.REJECTED
            self._staging_pool.release(handle.slot)
            del self._async_in_flight[key]
            raise