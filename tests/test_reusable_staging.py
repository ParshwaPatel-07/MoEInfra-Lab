import pytest

from transfer.pinned_memory import PinnedMemoryBudget
from transfer.reusable_staging import ReusablePinnedStagingPool


def test_initial_state():
    budget = PinnedMemoryBudget(1000)

    pool = ReusablePinnedStagingPool(
        budget,
        slot_size_bytes=400,
    )

    assert pool.slot_count == 0
    assert pool.active_slots == 0


def test_acquire_creates_slot():
    budget = PinnedMemoryBudget(1000)

    pool = ReusablePinnedStagingPool(
        budget,
        slot_size_bytes=400,
    )

    slot = pool.acquire()

    assert slot is not None
    assert slot.slot_id == 0
    assert slot.size_bytes == 400
    assert slot.in_use is True

    assert pool.slot_count == 1
    assert pool.active_slots == 1


def test_multiple_slots():
    budget = PinnedMemoryBudget(1000)

    pool = ReusablePinnedStagingPool(
        budget,
        slot_size_bytes=400,
    )

    slot1 = pool.acquire()
    slot2 = pool.acquire()

    assert slot1 is not None
    assert slot2 is not None
    assert slot1.slot_id != slot2.slot_id

    assert pool.slot_count == 2
    assert pool.active_slots == 2


def test_budget_limits_slots():
    budget = PinnedMemoryBudget(1000)

    pool = ReusablePinnedStagingPool(
        budget,
        slot_size_bytes=400,
    )

    assert pool.acquire() is not None
    assert pool.acquire() is not None
    assert pool.acquire() is None

    assert pool.slot_count == 2
    assert budget.used_bytes == 800


def test_release_makes_slot_available():
    budget = PinnedMemoryBudget(1000)

    pool = ReusablePinnedStagingPool(
        budget,
        slot_size_bytes=400,
    )

    slot1 = pool.acquire()
    pool.release(slot1)

    assert slot1.in_use is False
    assert pool.active_slots == 0


def test_released_slot_is_reused():
    budget = PinnedMemoryBudget(1000)

    pool = ReusablePinnedStagingPool(
        budget,
        slot_size_bytes=400,
    )

    slot1 = pool.acquire()
    pool.release(slot1)

    slot2 = pool.acquire()

    assert slot2 is slot1
    assert pool.slot_count == 1


def test_double_release_fails():
    budget = PinnedMemoryBudget(1000)

    pool = ReusablePinnedStagingPool(
        budget,
        slot_size_bytes=400,
    )

    slot = pool.acquire()
    pool.release(slot)

    with pytest.raises(RuntimeError):
        pool.release(slot)

import pytest
import torch

from transfer.pinned_memory import PinnedMemoryBudget
from transfer.reusable_staging import ReusablePinnedStagingPool


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_slot_cannot_be_reused_before_transfer_completes():
    budget = PinnedMemoryBudget(2 * 1024 * 1024)
    pool = ReusablePinnedStagingPool(
        budget=budget,
        slot_size_bytes=1024 * 1024,
    )

    slot = pool.acquire()
    assert slot is not None

    stream = torch.cuda.Stream()

    # Put work on the stream so the event is not immediately complete.
    with torch.cuda.stream(stream):
        x = torch.randn(4096, 4096, device="cuda")
        y = x @ x

    pool.mark_transfer_complete(slot, stream)
    pool.release(slot)

    # The slot must not be reused while the CUDA work is still pending.
    second = pool.acquire()

    assert second is None or second.slot_id != slot.slot_id

    # Finish the stream.
    stream.synchronize()

    # Now the original slot should become reusable.
    reused = pool.acquire()

    assert reused is not None
    assert reused.slot_id == slot.slot_id


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_completed_event_allows_reuse():
    budget = PinnedMemoryBudget(1024 * 1024)
    pool = ReusablePinnedStagingPool(
        budget=budget,
        slot_size_bytes=1024 * 1024,
    )

    slot = pool.acquire()
    assert slot is not None

    stream = torch.cuda.Stream()

    pool.mark_transfer_complete(slot, stream)
    stream.synchronize()

    pool.release(slot)

    reused = pool.acquire()

    assert reused is not None
    assert reused.slot_id == slot.slot_id


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_mark_transfer_requires_active_slot():
    budget = PinnedMemoryBudget(1024 * 1024)
    pool = ReusablePinnedStagingPool(
        budget=budget,
        slot_size_bytes=1024 * 1024,
    )

    slot = pool.acquire()
    assert slot is not None

    pool.release(slot)

    stream = torch.cuda.Stream()

    with pytest.raises(RuntimeError):
        pool.mark_transfer_complete(slot, stream)

def test_acquire_returns_none_when_all_slots_in_use():
    budget = PinnedMemoryBudget(800)

    pool = ReusablePinnedStagingPool(
        budget,
        slot_size_bytes=400,
    )

    slot1 = pool.acquire()
    slot2 = pool.acquire()

    assert slot1 is not None
    assert slot2 is not None

    exhausted = pool.acquire()

    assert exhausted is None

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_acquire_returns_none_when_all_slots_have_pending_events():
    budget = PinnedMemoryBudget(800)

    pool = ReusablePinnedStagingPool(
        budget,
        slot_size_bytes=400,
    )

    slot1 = pool.acquire()
    slot2 = pool.acquire()

    assert slot1 is not None
    assert slot2 is not None

    stream = torch.cuda.Stream()

    with torch.cuda.stream(stream):
        x = torch.randn(4096, 4096, device="cuda")
        y = x @ x

    pool.mark_transfer_complete(slot1, stream)
    pool.mark_transfer_complete(slot2, stream)

    pool.release(slot1)
    pool.release(slot2)

    exhausted = pool.acquire()

    assert exhausted is None

    stream.synchronize()

    reusable = pool.acquire()

    assert reusable is not None