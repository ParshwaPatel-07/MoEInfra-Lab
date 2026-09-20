"""Tests for the transfer subsystem.

Covers :class:`~transfer.bandwidth.BandwidthMonitor`,
:class:`~transfer.types.TransferRequest`,
:class:`~transfer.types.TransferResult`, and
:class:`~transfer.scheduler.TransferScheduler`.

Design constraints mirrored from the production code:
- ``TransferRequest`` is a *frozen* dataclass with fields:
      request_id, layer_id, expert_id, direction, priority, issued_at
  There is intentionally **no** ``tensor`` field — the scheduler does not own
  expert objects.  Ownership belongs to :class:`~cache.manager.CacheManager`.
- ``TransferScheduler.submit()`` returns ``bool`` (True = accepted).
- ``execute_next()`` delegates to ``cache_manager.promote_to_gpu()`` or
  ``demote_to_cpu()`` and gets byte size from ``cache_manager.get_size_bytes()``.
- ``bytes_transferred`` in the result equals the value returned by
  ``get_size_bytes()`` on a successful transfer, or 0 on failure.

A lightweight :class:`FakeCacheManager` is used so that tests do not require
CUDA or bitsandbytes.  The fake manager exposes the same API subset that
``TransferScheduler.execute_next()`` actually calls.
"""
from __future__ import annotations

import time
import uuid
from typing import Optional

import pytest
import torch

from transfer.bandwidth import BandwidthMonitor
from transfer.scheduler import TransferScheduler
from transfer.types import (
    TransferDirection,
    TransferPriority,
    TransferRequest,
    TransferResult,
)

CUDA_AVAILABLE = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(
    not CUDA_AVAILABLE,
    reason="CUDA device required for GPU-transfer tests",
)


# ── Lightweight fake cache manager ───────────────────────────────────────

class FakeCacheManager:
    """Minimal stand-in for CacheManager in TransferScheduler tests.

    The scheduler only calls three methods on the cache manager:
      - get_size_bytes(layer_id, expert_id) -> int
      - promote_to_gpu(layer_id, expert_id) -> bool
      - demote_to_cpu(layer_id, expert_id) -> bool

    This fake lets individual tests control the return values and track
    which calls were made.
    """

    def __init__(self) -> None:
        # Keys that exist with known sizes
        self._sizes: dict[tuple[int, int], int] = {}
        # Keys resident on CPU (promote_to_gpu succeeds for these)
        self._cpu_residents: set[tuple[int, int]] = set()
        # Keys resident on GPU (demote_to_cpu succeeds for these)
        self._gpu_residents: set[tuple[int, int]] = set()
        # Call records for assertions
        self.promote_calls: list[tuple[int, int]] = []
        self.demote_calls: list[tuple[int, int]] = []

    def add_cpu_expert(
        self, layer_id: int, expert_id: int, size_bytes: int
    ) -> None:
        """Register an expert as CPU-resident with a known size."""
        self._sizes[(layer_id, expert_id)] = size_bytes
        self._cpu_residents.add((layer_id, expert_id))

    def add_gpu_expert(
        self, layer_id: int, expert_id: int, size_bytes: int
    ) -> None:
        """Register an expert as GPU-resident with a known size."""
        self._sizes[(layer_id, expert_id)] = size_bytes
        self._gpu_residents.add((layer_id, expert_id))

    def get_size_bytes(self, layer_id: int, expert_id: int) -> int:
        key = (layer_id, expert_id)
        if key not in self._sizes:
            raise KeyError(
                f"Expert not present in cache: layer={layer_id}, expert={expert_id}"
            )
        return self._sizes[key]

    def promote_to_gpu(self, layer_id: int, expert_id: int) -> bool:
        key = (layer_id, expert_id)
        self.promote_calls.append(key)
        if key not in self._cpu_residents:
            return False
        self._cpu_residents.discard(key)
        self._gpu_residents.add(key)
        return True

    def demote_to_cpu(self, layer_id: int, expert_id: int) -> bool:
        key = (layer_id, expert_id)
        self.demote_calls.append(key)
        if key not in self._gpu_residents:
            return False
        self._gpu_residents.discard(key)
        self._cpu_residents.add(key)
        return True


# ── Helpers ───────────────────────────────────────────────────────────────

def make_request(
    layer_id: int = 0,
    expert_id: int = 0,
    direction: TransferDirection = TransferDirection.CPU_TO_GPU,
    priority: TransferPriority = TransferPriority.NORMAL,
    request_id: Optional[str] = None,
) -> TransferRequest:
    """Return a new TransferRequest with a fresh UUID."""
    return TransferRequest(
        request_id=request_id or str(uuid.uuid4()),
        layer_id=layer_id,
        expert_id=expert_id,
        direction=direction,
        priority=priority,
        issued_at=time.monotonic(),
    )


def make_scheduler(
    fake_cm: Optional[FakeCacheManager] = None,
    bandwidth_gbps: float = 8.0,
    max_concurrent: int = 4,
) -> tuple[TransferScheduler, FakeCacheManager]:
    """Return a (scheduler, fake_cache_manager) pair."""
    if fake_cm is None:
        fake_cm = FakeCacheManager()
    sched = TransferScheduler(
        cache_manager=fake_cm,
        bandwidth_gbps=bandwidth_gbps,
        max_concurrent=max_concurrent,
    )
    return sched, fake_cm


# ── BandwidthMonitor ──────────────────────────────────────────────────────

class TestBandwidthMonitor:
    """BandwidthMonitor correctness tests (no CUDA required)."""

    def test_initial_current_gbps_is_zero(self) -> None:
        """current_gbps() must be 0.0 with fewer than two samples."""
        mon = BandwidthMonitor(window_s=5.0)
        assert mon.current_gbps() == 0.0

    def test_initial_peak_gbps_is_zero(self) -> None:
        """peak_gbps() must be 0.0 after construction."""
        mon = BandwidthMonitor(window_s=5.0)
        assert mon.peak_gbps() == 0.0

    def test_two_samples_produce_positive_throughput(self) -> None:
        """Two samples separated in time must yield current_gbps() > 0."""
        mon = BandwidthMonitor(window_s=5.0)
        mon.record(1 * 1024 ** 3)
        time.sleep(0.01)
        mon.record(1 * 1024 ** 3)
        assert mon.current_gbps() > 0.0

    def test_peak_gbps_tracks_maximum(self) -> None:
        """peak_gbps() must stay >= any previously observed current_gbps()."""
        mon = BandwidthMonitor(window_s=5.0)
        mon.record(1 * 1024 ** 3)
        time.sleep(0.01)
        mon.record(1 * 1024 ** 3)
        peak = mon.peak_gbps()
        assert peak > 0.0
        assert peak >= mon.current_gbps()

    def test_reset_clears_samples_and_peak(self) -> None:
        """reset() must zero out all samples and the peak counter."""
        mon = BandwidthMonitor(window_s=5.0)
        mon.record(1024)
        time.sleep(0.01)
        mon.record(1024)
        mon.reset()
        assert mon.current_gbps() == 0.0
        assert mon.peak_gbps() == 0.0


# ── TransferRequest type contract ─────────────────────────────────────────

class TestTransferRequestType:
    """TransferRequest must be an immutable, (layer_id, expert_id)-indexed type."""

    def test_fields_accessible(self) -> None:
        """All declared fields must be readable."""
        req = make_request(layer_id=3, expert_id=7)
        assert req.layer_id == 3
        assert req.expert_id == 7
        assert req.direction == TransferDirection.CPU_TO_GPU
        assert req.priority == TransferPriority.NORMAL
        assert isinstance(req.request_id, str)
        assert req.issued_at > 0.0

    def test_frozen_immutable(self) -> None:
        """TransferRequest must reject attribute assignment (frozen dataclass)."""
        req = make_request()
        with pytest.raises((AttributeError, TypeError)):
            req.layer_id = 999  # type: ignore[misc]

    def test_no_tensor_field(self) -> None:
        """TransferRequest must NOT carry a tensor field."""
        req = make_request()
        assert not hasattr(req, "tensor")

    def test_unique_ids_by_default(self) -> None:
        """Two requests created without an explicit request_id must be unique."""
        r1 = make_request()
        r2 = make_request()
        assert r1.request_id != r2.request_id


# ── TransferResult type contract ──────────────────────────────────────────

class TestTransferResultType:
    """TransferResult must accurately reflect the outcome of execute_next()."""

    def _build_result(self, **kwargs) -> TransferResult:
        defaults = dict(
            request_id=str(uuid.uuid4()),
            success=True,
            elapsed_ms=1.5,
            bytes_transferred=1024,
            error=None,
        )
        defaults.update(kwargs)
        return TransferResult(**defaults)

    def test_success_fields(self) -> None:
        r = self._build_result(success=True, bytes_transferred=2048)
        assert r.success is True
        assert r.bytes_transferred == 2048
        assert r.error is None

    def test_failure_fields(self) -> None:
        r = self._build_result(success=False, bytes_transferred=0, error="boom")
        assert r.success is False
        assert r.bytes_transferred == 0
        assert r.error == "boom"


# ── submit() ──────────────────────────────────────────────────────────────

class TestSubmit:
    """TransferScheduler.submit() queuing, deduplication, and return value."""

    def test_submit_returns_true_for_new_request(self) -> None:
        """submit() must return True when the request is accepted."""
        sched, _ = make_scheduler()
        assert sched.submit(make_request()) is True

    def test_submit_increments_pending_count(self) -> None:
        """submit() must increase pending_count() by 1 for each new request."""
        sched, _ = make_scheduler()
        assert sched.pending_count() == 0
        sched.submit(make_request(layer_id=0, expert_id=0))
        assert sched.pending_count() == 1
        sched.submit(make_request(layer_id=0, expert_id=1))
        assert sched.pending_count() == 2

    def test_submit_returns_false_for_duplicate_triple(self) -> None:
        """Duplicate (layer_id, expert_id, direction) must be rejected."""
        sched, _ = make_scheduler()
        req1 = make_request(layer_id=0, expert_id=0, direction=TransferDirection.CPU_TO_GPU)
        req2 = make_request(layer_id=0, expert_id=0, direction=TransferDirection.CPU_TO_GPU)
        assert sched.submit(req1) is True
        assert sched.submit(req2) is False  # duplicate triple
        assert sched.pending_count() == 1

    def test_submit_returns_false_for_duplicate_request_id(self) -> None:
        """A request with the same request_id must be rejected."""
        sched, _ = make_scheduler()
        fixed_id = str(uuid.uuid4())
        req1 = make_request(layer_id=0, expert_id=0, request_id=fixed_id)
        req2 = make_request(layer_id=0, expert_id=1, request_id=fixed_id)
        assert sched.submit(req1) is True
        assert sched.submit(req2) is False
        assert sched.pending_count() == 1

    def test_different_directions_are_not_duplicates(self) -> None:
        """CPU→GPU and GPU→CPU for the same expert are distinct requests."""
        sched, _ = make_scheduler()
        up = make_request(layer_id=0, expert_id=0, direction=TransferDirection.CPU_TO_GPU)
        down = make_request(layer_id=0, expert_id=0, direction=TransferDirection.GPU_TO_CPU)
        assert sched.submit(up) is True
        assert sched.submit(down) is True
        assert sched.pending_count() == 2

    def test_different_experts_are_not_duplicates(self) -> None:
        """Same direction for different experts must both be accepted."""
        sched, _ = make_scheduler()
        r1 = make_request(layer_id=0, expert_id=0)
        r2 = make_request(layer_id=0, expert_id=1)
        assert sched.submit(r1) is True
        assert sched.submit(r2) is True
        assert sched.pending_count() == 2


# ── Priority ordering ──────────────────────────────────────────────────────

class TestPriorityOrdering:
    """execute_next() must always pop the highest-priority request first."""

    def test_high_before_low(self) -> None:
        """A HIGH-priority request submitted after a LOW one must execute first."""
        fake_cm = FakeCacheManager()
        fake_cm.add_cpu_expert(0, 0, 512)
        fake_cm.add_cpu_expert(0, 1, 512)
        sched = TransferScheduler(fake_cm, bandwidth_gbps=8.0, max_concurrent=4)

        low_req = make_request(expert_id=0, priority=TransferPriority.LOW)
        high_req = make_request(expert_id=1, priority=TransferPriority.HIGH)
        sched.submit(low_req)
        sched.submit(high_req)

        result = sched.execute_next()
        assert result is not None
        assert result.request_id == high_req.request_id

    def test_normal_before_low(self) -> None:
        """NORMAL-priority requests must execute before LOW-priority ones."""
        fake_cm = FakeCacheManager()
        fake_cm.add_cpu_expert(0, 0, 256)
        fake_cm.add_cpu_expert(0, 1, 256)
        sched = TransferScheduler(fake_cm, bandwidth_gbps=8.0, max_concurrent=4)

        low_req = make_request(expert_id=0, priority=TransferPriority.LOW)
        norm_req = make_request(expert_id=1, priority=TransferPriority.NORMAL)
        sched.submit(low_req)
        sched.submit(norm_req)

        result = sched.execute_next()
        assert result is not None
        assert result.request_id == norm_req.request_id

    def test_fifo_within_same_priority(self) -> None:
        """Requests at identical priority must execute in FIFO (issued_at) order."""
        fake_cm = FakeCacheManager()
        fake_cm.add_cpu_expert(0, 0, 256)
        fake_cm.add_cpu_expert(0, 1, 256)
        sched = TransferScheduler(fake_cm, bandwidth_gbps=8.0, max_concurrent=4)

        first = make_request(expert_id=0, priority=TransferPriority.NORMAL)
        time.sleep(0.005)
        second = make_request(expert_id=1, priority=TransferPriority.NORMAL)
        sched.submit(first)
        sched.submit(second)

        result = sched.execute_next()
        assert result is not None
        assert result.request_id == first.request_id


# ── cancel() ──────────────────────────────────────────────────────────────

class TestCancel:
    """cancel() must remove exactly the named request and nothing else."""

    def test_cancel_existing_returns_true(self) -> None:
        """cancel() must return True for a queued request_id."""
        sched, _ = make_scheduler()
        req = make_request()
        sched.submit(req)
        assert sched.cancel(req.request_id) is True

    def test_cancel_existing_decrements_pending(self) -> None:
        """After cancel(), pending_count() must decrease by 1."""
        sched, _ = make_scheduler()
        req = make_request()
        sched.submit(req)
        sched.cancel(req.request_id)
        assert sched.pending_count() == 0

    def test_cancel_unknown_returns_false(self) -> None:
        """cancel() on an unknown request_id must return False."""
        sched, _ = make_scheduler()
        assert sched.cancel(str(uuid.uuid4())) is False

    def test_cancel_does_not_affect_other_requests(self) -> None:
        """Cancelling one request must leave others in the queue."""
        sched, _ = make_scheduler()
        r1 = make_request(expert_id=0)
        r2 = make_request(expert_id=1)
        sched.submit(r1)
        sched.submit(r2)
        sched.cancel(r1.request_id)
        assert sched.pending_count() == 1

    def test_cancel_preserves_heap_invariant(self) -> None:
        """After cancellation the heap must remain valid (no pop-order corruption)."""
        fake_cm = FakeCacheManager()
        # Register all three experts so execute_next won't raise KeyError
        fake_cm.add_cpu_expert(0, 0, 128)
        fake_cm.add_cpu_expert(0, 1, 128)
        fake_cm.add_cpu_expert(0, 2, 128)

        sched = TransferScheduler(fake_cm, bandwidth_gbps=8.0, max_concurrent=4)
        r_low  = make_request(expert_id=0, priority=TransferPriority.LOW)
        r_norm = make_request(expert_id=1, priority=TransferPriority.NORMAL)
        r_high = make_request(expert_id=2, priority=TransferPriority.HIGH)

        sched.submit(r_low)
        sched.submit(r_norm)
        sched.submit(r_high)

        sched.cancel(r_norm.request_id)  # remove the middle-priority one

        result = sched.execute_next()  # must pop r_high (priority=0 → HIGH)
        assert result is not None
        assert result.request_id == r_high.request_id


# ── execute_next() ────────────────────────────────────────────────────────

class TestExecuteNext:
    """execute_next() must delegate to CacheManager and return correct results."""

    def test_empty_queue_returns_none(self) -> None:
        """execute_next() on an empty queue must return None."""
        sched, _ = make_scheduler()
        assert sched.execute_next() is None

    def test_cpu_to_gpu_transfer_calls_promote(self) -> None:
        """execute_next() for CPU_TO_GPU must call cache_manager.promote_to_gpu()."""
        fake_cm = FakeCacheManager()
        fake_cm.add_cpu_expert(2, 5, size_bytes=2048)
        sched = TransferScheduler(fake_cm, bandwidth_gbps=8.0, max_concurrent=4)

        sched.submit(make_request(
            layer_id=2, expert_id=5,
            direction=TransferDirection.CPU_TO_GPU,
        ))
        result = sched.execute_next()

        assert (2, 5) in fake_cm.promote_calls
        assert result is not None
        assert result.success is True

    def test_gpu_to_cpu_transfer_calls_demote(self) -> None:
        """execute_next() for GPU_TO_CPU must call cache_manager.demote_to_cpu()."""
        fake_cm = FakeCacheManager()
        fake_cm.add_gpu_expert(1, 3, size_bytes=4096)
        sched = TransferScheduler(fake_cm, bandwidth_gbps=8.0, max_concurrent=4)

        sched.submit(make_request(
            layer_id=1, expert_id=3,
            direction=TransferDirection.GPU_TO_CPU,
        ))
        result = sched.execute_next()

        assert (1, 3) in fake_cm.demote_calls
        assert result is not None
        assert result.success is True

    def test_successful_result_contains_correct_bytes(self) -> None:
        """bytes_transferred must equal get_size_bytes() on success."""
        fake_cm = FakeCacheManager()
        fake_cm.add_cpu_expert(0, 0, size_bytes=8192)
        sched = TransferScheduler(fake_cm, bandwidth_gbps=8.0, max_concurrent=4)

        sched.submit(make_request(layer_id=0, expert_id=0))
        result = sched.execute_next()

        assert result is not None
        assert result.success is True
        assert result.bytes_transferred == 8192

    def test_failed_transfer_bytes_is_zero(self) -> None:
        """bytes_transferred must be 0 when the transfer fails."""
        fake_cm = FakeCacheManager()
        # Expert exists in sizes dict but NOT in cpu_residents → promote returns False
        fake_cm._sizes[(0, 0)] = 1024  # get_size_bytes will succeed
        # promote_to_gpu will return False because (0,0) not in _cpu_residents
        sched = TransferScheduler(fake_cm, bandwidth_gbps=8.0, max_concurrent=4)

        sched.submit(make_request(layer_id=0, expert_id=0,
                                  direction=TransferDirection.CPU_TO_GPU))
        result = sched.execute_next()

        assert result is not None
        assert result.success is False
        assert result.bytes_transferred == 0

    def test_missing_expert_raises_key_error_recorded_in_result(self) -> None:
        """execute_next() for an expert absent from cache must return failure result."""
        fake_cm = FakeCacheManager()
        # Don't register the expert at all → get_size_bytes raises KeyError
        sched = TransferScheduler(fake_cm, bandwidth_gbps=8.0, max_concurrent=4)

        sched.submit(make_request(layer_id=9, expert_id=9))
        result = sched.execute_next()

        assert result is not None
        assert result.success is False
        assert result.bytes_transferred == 0
        assert result.error is not None
        # The error string must mention the layer and expert
        assert "9" in result.error

    def test_successful_result_has_positive_elapsed_ms(self) -> None:
        """elapsed_ms must be a non-negative float for any result."""
        fake_cm = FakeCacheManager()
        fake_cm.add_cpu_expert(0, 0, size_bytes=512)
        sched = TransferScheduler(fake_cm, bandwidth_gbps=8.0, max_concurrent=4)

        sched.submit(make_request(layer_id=0, expert_id=0))
        result = sched.execute_next()

        assert result is not None
        assert result.elapsed_ms >= 0.0

    def test_request_removed_from_pending_after_execute(self) -> None:
        """After execute_next(), the request must no longer appear in pending."""
        fake_cm = FakeCacheManager()
        fake_cm.add_cpu_expert(0, 0, size_bytes=256)
        sched = TransferScheduler(fake_cm, bandwidth_gbps=8.0, max_concurrent=4)

        sched.submit(make_request(layer_id=0, expert_id=0))
        assert sched.pending_count() == 1
        sched.execute_next()
        assert sched.pending_count() == 0

    def test_result_request_id_matches_submitted_request(self) -> None:
        """TransferResult.request_id must equal the submitted request's ID."""
        fake_cm = FakeCacheManager()
        fake_cm.add_cpu_expert(0, 0, size_bytes=256)
        sched = TransferScheduler(fake_cm, bandwidth_gbps=8.0, max_concurrent=4)

        req = make_request(layer_id=0, expert_id=0)
        sched.submit(req)
        result = sched.execute_next()

        assert result is not None
        assert result.request_id == req.request_id


# ── drain() ───────────────────────────────────────────────────────────────

class TestDrain:
    """drain() must process all pending requests and return all results."""

    def test_drain_empty_returns_empty_list(self) -> None:
        """drain() on an empty queue must return []."""
        sched, _ = make_scheduler()
        assert sched.drain() == []

    def test_drain_returns_all_results(self) -> None:
        """drain() must return one TransferResult per queued request."""
        fake_cm = FakeCacheManager()
        n = 4
        for i in range(n):
            fake_cm.add_cpu_expert(0, i, size_bytes=256)

        sched = TransferScheduler(fake_cm, bandwidth_gbps=8.0, max_concurrent=4)
        for i in range(n):
            sched.submit(make_request(expert_id=i))

        results = sched.drain()
        assert len(results) == n

    def test_drain_leaves_queue_empty(self) -> None:
        """After drain(), pending_count() must be 0."""
        fake_cm = FakeCacheManager()
        for i in range(3):
            fake_cm.add_cpu_expert(0, i, size_bytes=128)

        sched = TransferScheduler(fake_cm, bandwidth_gbps=8.0, max_concurrent=4)
        for i in range(3):
            sched.submit(make_request(expert_id=i))

        sched.drain()
        assert sched.pending_count() == 0

    def test_drain_processes_in_priority_order(self) -> None:
        """drain() must return results in priority order (HIGH first)."""
        fake_cm = FakeCacheManager()
        for i in range(3):
            fake_cm.add_cpu_expert(0, i, size_bytes=256)

        sched = TransferScheduler(fake_cm, bandwidth_gbps=8.0, max_concurrent=4)
        r_low  = make_request(expert_id=0, priority=TransferPriority.LOW)
        r_norm = make_request(expert_id=1, priority=TransferPriority.NORMAL)
        r_high = make_request(expert_id=2, priority=TransferPriority.HIGH)

        sched.submit(r_low)
        sched.submit(r_norm)
        sched.submit(r_high)

        results = sched.drain()
        assert len(results) == 3
        assert results[0].request_id == r_high.request_id
        assert results[1].request_id == r_norm.request_id
        assert results[2].request_id == r_low.request_id

    def test_drain_updates_bandwidth_monitor_on_success(self) -> None:
        """After drain() with successful transfers, bandwidth_monitor must record >0 GB/s.

        We need at least two successful transfers with a small sleep in between
        to get a non-zero rolling throughput.
        """
        fake_cm = FakeCacheManager()
        for i in range(2):
            # Large size so throughput is measurable
            fake_cm.add_cpu_expert(0, i, size_bytes=10 * 1024 ** 2)

        sched = TransferScheduler(fake_cm, bandwidth_gbps=8.0, max_concurrent=4)
        sched.submit(make_request(expert_id=0))
        time.sleep(0.01)
        sched.submit(make_request(expert_id=1))

        sched.drain()
        assert sched.bandwidth_monitor.peak_gbps() > 0.0


# ── Full integration: CPU→GPU via real CacheManager (requires CUDA) ───────

class TestRealCacheManagerIntegration:
    """Integration tests that use the real CacheManager + real QuantizedMixtralExpert.

    These tests require an actual CUDA device.
    """

    @requires_cuda
    def test_cpu_to_gpu_via_real_promote(self) -> None:
        """TransferScheduler must successfully promote a real expert to GPU."""
        from cache.manager import CacheManager
        from cache.types import EvictionPolicy
        from model.expert import QuantizedMixtralExpert

        cm = CacheManager(gpu_slots=4, cpu_slots=8, policy=EvictionPolicy.LRU)

        # Minimal real expert
        w1 = torch.randn(16, 8, dtype=torch.bfloat16)
        w2 = torch.randn(8, 16, dtype=torch.bfloat16)
        w3 = torch.randn(16, 8, dtype=torch.bfloat16)
        expert = QuantizedMixtralExpert(w1, w2, w3)

        cm.put(0, 0, expert, "cpu")

        sched = TransferScheduler(cm, bandwidth_gbps=8.0, max_concurrent=4)
        sched.submit(make_request(layer_id=0, expert_id=0,
                                  direction=TransferDirection.CPU_TO_GPU))
        result = sched.execute_next()

        assert result is not None
        assert result.success is True
        assert result.bytes_transferred > 0
        assert (0, 0) in cm._gpu_cache

    @requires_cuda
    def test_gpu_to_cpu_via_real_demote(self) -> None:
        """TransferScheduler must successfully demote a real GPU expert to CPU."""
        from cache.manager import CacheManager
        from cache.types import EvictionPolicy
        from model.expert import QuantizedMixtralExpert

        cm = CacheManager(gpu_slots=4, cpu_slots=8, policy=EvictionPolicy.LRU)

        w1 = torch.randn(16, 8, dtype=torch.bfloat16)
        w2 = torch.randn(8, 16, dtype=torch.bfloat16)
        w3 = torch.randn(16, 8, dtype=torch.bfloat16)
        expert = QuantizedMixtralExpert(w1, w2, w3)

        cm.put(0, 0, expert, "cpu")
        cm.promote_to_gpu(0, 0)
        assert (0, 0) in cm._gpu_cache

        sched = TransferScheduler(cm, bandwidth_gbps=8.0, max_concurrent=4)
        sched.submit(make_request(layer_id=0, expert_id=0,
                                  direction=TransferDirection.GPU_TO_CPU))
        result = sched.execute_next()

        assert result is not None
        assert result.success is True
        assert result.bytes_transferred > 0
        assert (0, 0) in cm._cpu_cache
        assert (0, 0) not in cm._gpu_cache

    @requires_cuda
    def test_bytes_transferred_matches_get_size_bytes(self) -> None:
        """TransferResult.bytes_transferred must equal CacheManager.get_size_bytes()."""
        from cache.manager import CacheManager
        from cache.types import EvictionPolicy
        from model.expert import QuantizedMixtralExpert

        cm = CacheManager(gpu_slots=4, cpu_slots=8, policy=EvictionPolicy.LRU)

        w1 = torch.randn(16, 8, dtype=torch.bfloat16)
        w2 = torch.randn(8, 16, dtype=torch.bfloat16)
        w3 = torch.randn(16, 8, dtype=torch.bfloat16)
        expert = QuantizedMixtralExpert(w1, w2, w3)

        cm.put(0, 0, expert, "cpu")
        expected_bytes = cm.get_size_bytes(0, 0)

        sched = TransferScheduler(cm, bandwidth_gbps=8.0, max_concurrent=4)
        sched.submit(make_request(layer_id=0, expert_id=0))
        result = sched.execute_next()

        assert result is not None
        assert result.success is True
        assert result.bytes_transferred == expected_bytes
