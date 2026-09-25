import pytest
import torch
import copy

from cache.manager import CacheManager
from cache.types import EvictionPolicy
from tests.test_nf4_reconstruct import make_test_expert
from transfer.pinned_memory import PinnedMemoryBudget
from transfer.reusable_staging import ReusablePinnedStagingPool
from transfer.scheduler import TransferScheduler
from transfer.types import (
    TransferDirection,
    TransferPriority,
    TransferRequest,
    TransferStatus,
)


def make_request(
    layer_id=0,
    expert_id=0,
    request_id="req-0",
):
    return TransferRequest(
        request_id=request_id,
        layer_id=layer_id,
        expert_id=expert_id,
        direction=TransferDirection.CPU_TO_GPU,
        priority=TransferPriority.HIGH,
        issued_at=0.0,
    )


def make_scheduler(
    cache_manager,
):
    pool = ReusablePinnedStagingPool(
        PinnedMemoryBudget(20 * 1024 * 1024),
        slot_size_bytes=5 * 1024 * 1024,
    )

    stream = torch.cuda.Stream()

    scheduler = TransferScheduler(
        cache_manager=cache_manager,
        bandwidth_gbps=10.0,
        max_concurrent=1,
        staging_pool=pool,
        transfer_stream=stream,
    )

    return scheduler, pool, stream

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_async_submit_rejects_without_cpu_expert():
    cache = CacheManager(
        gpu_slots=1,
        cpu_slots=4,
        policy=EvictionPolicy.LRU,
    )

    scheduler, pool, stream = make_scheduler(cache)

    handle = scheduler.submit_async(
        make_request(),
    )

    assert handle.status == TransferStatus.REJECTED
    assert scheduler._async_in_flight == {}
    assert pool.active_slots == 0

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_async_submit_creates_in_flight_transfer():
    cache = CacheManager(
        gpu_slots=1,
        cpu_slots=4,
        policy=EvictionPolicy.LRU,
    )

    expert = make_test_expert()

    cache.put(
        layer_id=0,
        expert_id=0,
        expert=expert,
        device="cpu",
    )

    scheduler, pool, stream = make_scheduler(cache)

    handle = scheduler.submit_async(
        make_request(),
    )

    assert handle.status == TransferStatus.IN_FLIGHT
    assert handle.event is not None
    assert handle.slot is not None
    assert pool.active_slots == 1

    assert (0, 0) in scheduler._async_in_flight

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_duplicate_submit_returns_existing_handle():
    cache = CacheManager(
        gpu_slots=1,
        cpu_slots=4,
        policy=EvictionPolicy.LRU,
    )

    expert = make_test_expert()

    cache.put(
        0,
        0,
        expert,
        "cpu",
    )

    scheduler, pool, stream = make_scheduler(cache)

    first = scheduler.submit_async(
        make_request(request_id="req-1"),
    )

    second = scheduler.submit_async(
        make_request(request_id="req-2"),
    )

    assert first is second
    assert first.status == TransferStatus.IN_FLIGHT
    assert pool.active_slots == 1
    assert len(scheduler._async_in_flight) == 1

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_poll_async_reports_in_flight_before_completion():
    cache = CacheManager(
        gpu_slots=1,
        cpu_slots=4,
        policy=EvictionPolicy.LRU,
    )

    expert = make_test_expert()

    cache.put(0, 0, expert, "cpu")

    scheduler, pool, stream = make_scheduler(cache)

    scheduler.submit_async(
        make_request(),
    )

    status = scheduler.poll_async(0, 0)

    assert status in (
        TransferStatus.IN_FLIGHT,
        TransferStatus.READY,
    )

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_completed_transfer_becomes_ready_and_populates_gpu_cache():
    cache = CacheManager(
        gpu_slots=1,
        cpu_slots=4,
        policy=EvictionPolicy.LRU,
    )

    expert = make_test_expert()

    cache.put(0, 0, expert, "cpu")

    scheduler, pool, stream = make_scheduler(cache)

    handle = scheduler.submit_async(
        make_request(),
    )

    # Wait only for the transfer stream in the test.
    stream.synchronize()

    status = scheduler.poll_async(0, 0)

    assert status == TransferStatus.READY

    gpu_expert = cache.peek_gpu(0, 0)

    assert gpu_expert is not None
    assert gpu_expert.device.type == "cuda"

    assert pool.active_slots == 0
    assert (0, 0) not in scheduler._async_in_flight

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_async_transfer_output_matches_normal_gpu_expert():
    cache = CacheManager(
        gpu_slots=1,
        cpu_slots=4,
        policy=EvictionPolicy.LRU,
    )

    expert = make_test_expert()

    # Keep the authoritative CPU copy in the cache.
    cache.put(0, 0, expert, "cpu")

    scheduler, pool, stream = make_scheduler(cache)

    scheduler.submit_async(
        make_request(),
    )

    stream.synchronize()

    status = scheduler.poll_async(0, 0)

    assert status == TransferStatus.READY

    transferred = cache.peek_gpu(0, 0)
    assert transferred is not None

    # Separate GPU copy of the same expert used for the transfer.
    normal = copy.deepcopy(expert).cuda()
    torch.cuda.synchronize()

    x = torch.randn(
        2,
        128,
        device="cuda",
        dtype=torch.float16,
    )

    expected = normal(x)
    actual = transferred(x)

    torch.cuda.synchronize()

    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-3,
        atol=1e-3,
    )

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_async_respects_max_concurrent():
    cache = CacheManager(
        gpu_slots=2,
        cpu_slots=4,
        policy=EvictionPolicy.LRU,
    )

    expert0 = make_test_expert()
    expert1 = make_test_expert()

    cache.put(0, 0, expert0, "cpu")
    cache.put(0, 1, expert1, "cpu")

    # Two staging slots, so staging capacity is NOT the limiting factor.
    pool = ReusablePinnedStagingPool(
        PinnedMemoryBudget(20 * 1024 * 1024),
        slot_size_bytes=5 * 1024 * 1024,
    )
    stream = torch.cuda.Stream()

    scheduler = TransferScheduler(
        cache_manager=cache,
        bandwidth_gbps=10.0,
        max_concurrent=1,
        staging_pool=pool,
        transfer_stream=stream,
    )

    handle0 = scheduler.submit_async(
        make_request(expert_id=0, request_id="req-0"),
    )

    handle1 = scheduler.submit_async(
        make_request(expert_id=1, request_id="req-1"),
    )

    assert handle0.status == TransferStatus.IN_FLIGHT

    # max_concurrent=1 means the second transfer must not become
    # simultaneously in-flight.
    assert handle1.status == TransferStatus.REJECTED

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_async_completion_releases_staging_slot():
    cache = CacheManager(
        gpu_slots=1,
        cpu_slots=4,
        policy=EvictionPolicy.LRU,
    )

    cache.put(0, 0, make_test_expert(), "cpu")

    scheduler, pool, stream = make_scheduler(cache)

    handle = scheduler.submit_async(make_request())

    assert handle.status == TransferStatus.IN_FLIGHT
    assert pool.active_slots == 1

    stream.synchronize()

    status = scheduler.poll_async(0, 0)

    assert status == TransferStatus.READY
    assert pool.active_slots == 0

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_async_completion_finalizes_only_once():
    cache = CacheManager(
        gpu_slots=1,
        cpu_slots=4,
        policy=EvictionPolicy.LRU,
    )

    cache.put(0, 0, make_test_expert(), "cpu")

    scheduler, pool, stream = make_scheduler(cache)

    scheduler.submit_async(make_request())

    stream.synchronize()

    first = scheduler.poll_async(0, 0)

    assert first == TransferStatus.READY
    assert cache.is_gpu_resident(0, 0)

    # The in-flight handle should have been removed.
    assert (0, 0) not in scheduler._async_in_flight

    # A second poll must not reconstruct/publish another expert.
    second = scheduler.poll_async(0, 0)

    assert second == TransferStatus.REJECTED

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_async_cancel_does_not_publish_gpu_expert():
    cache = CacheManager(
        gpu_slots=1,
        cpu_slots=4,
        policy=EvictionPolicy.LRU,
    )

    cache.put(0, 0, make_test_expert(), "cpu")

    scheduler, pool, stream = make_scheduler(cache)

    handle = scheduler.submit_async(make_request())

    assert handle.status == TransferStatus.IN_FLIGHT

    cancelled = scheduler.cancel("req-0")

    assert cancelled is True

    # The transfer itself may already have been submitted to CUDA,
    # but cancellation must prevent publication into the GPU cache.
    stream.synchronize()

    status = scheduler.poll_async(0, 0)

    assert status == TransferStatus.CANCELLED
    assert not cache.is_gpu_resident(0, 0)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_async_cancel_releases_staging_slot_after_transfer_completion():
    cache = CacheManager(
        gpu_slots=1,
        cpu_slots=4,
        policy=EvictionPolicy.LRU,
    )

    cache.put(0, 0, make_test_expert(), "cpu")

    scheduler, pool, stream = make_scheduler(cache)

    scheduler.submit_async(make_request())

    assert pool.active_slots == 1

    cancelled = scheduler.cancel("req-0")

    assert cancelled is True

    # Cancellation must not prematurely release the slot.
    # The DMA may still be using the pinned buffers.
    assert pool.active_slots == 1

    stream.synchronize()

    status = scheduler.poll_async(0, 0)

    assert status == TransferStatus.CANCELLED
    assert pool.active_slots == 0
    assert not cache.is_gpu_resident(0, 0)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_async_rejects_gpu_to_cpu_request():
    cache = CacheManager(
        gpu_slots=1,
        cpu_slots=4,
        policy=EvictionPolicy.LRU,
    )

    cache.put(0, 0, make_test_expert(), "cpu")

    scheduler, pool, stream = make_scheduler(cache)

    request = TransferRequest(
        request_id="gpu-to-cpu",
        layer_id=0,
        expert_id=0,
        direction=TransferDirection.GPU_TO_CPU,
        priority=TransferPriority.HIGH,
        issued_at=0.0,
    )

    with pytest.raises(ValueError):
        scheduler.submit_async(request)