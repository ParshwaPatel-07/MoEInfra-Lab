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