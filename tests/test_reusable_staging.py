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

import pytest
import torch

from model.expert import QuantizedMixtralExpert
from transfer.pinned_memory import PinnedMemoryBudget
from transfer.reusable_staging import ReusablePinnedStagingPool


def make_test_expert():
    """
    Small real NF4 expert.

    QuantizedMixtralExpert expects BF16 source weights.
    Moving to CUDA once forces bitsandbytes to create the packed NF4
    representation, then we bring it back to CPU.
    """
    hidden = 128
    intermediate = 256

    w1 = torch.randn(intermediate, hidden, dtype=torch.bfloat16)
    w2 = torch.randn(hidden, intermediate, dtype=torch.bfloat16)
    w3 = torch.randn(intermediate, hidden, dtype=torch.bfloat16)

    expert = QuantizedMixtralExpert(w1, w2, w3)

    expert = expert.cuda()
    torch.cuda.synchronize()
    expert = expert.cpu()

    return expert


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_slot_can_stage_nf4_expert():
    expert = make_test_expert()

    pool = ReusablePinnedStagingPool(
        PinnedMemoryBudget(10 * 1024 * 1024),
        slot_size_bytes=5 * 1024 * 1024,
    )

    slot = pool.acquire()
    assert slot is not None

    pool.stage_expert(
        slot,
        expert,
        layer_id=0,
        expert_id=3,
    )

    assert slot.expert_key == (0, 3)

    for name in ("w1", "w2", "w3"):
        assert name in slot.tensors


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_staging_buffers_are_pinned():
    expert = make_test_expert()

    pool = ReusablePinnedStagingPool(
        PinnedMemoryBudget(10 * 1024 * 1024),
        slot_size_bytes=5 * 1024 * 1024,
    )

    slot = pool.acquire()
    pool.stage_expert(slot, expert, layer_id=0, expert_id=0)

    for name in ("w1", "w2", "w3"):
        state = slot.tensors[name]

        for tensor in state.values():
            if tensor is not None:
                assert tensor.is_pinned()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_source_expert_remains_pageable():
    expert = make_test_expert()

    # Verify source starts pageable.
    for name in ("w1", "w2", "w3"):
        param = getattr(expert, name).weight
        assert not param.data.is_pinned()

    pool = ReusablePinnedStagingPool(
        PinnedMemoryBudget(10 * 1024 * 1024),
        slot_size_bytes=5 * 1024 * 1024,
    )

    slot = pool.acquire()
    pool.stage_expert(slot, expert, layer_id=0, expert_id=0)

    # Staging must not mutate/pin the cold expert itself.
    for name in ("w1", "w2", "w3"):
        param = getattr(expert, name).weight
        assert not param.data.is_pinned()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_all_quant_state_is_staged():
    expert = make_test_expert()

    pool = ReusablePinnedStagingPool(
        PinnedMemoryBudget(10 * 1024 * 1024),
        slot_size_bytes=5 * 1024 * 1024,
    )

    slot = pool.acquire()
    pool.stage_expert(slot, expert, layer_id=0, expert_id=0)

    for name in ("w1", "w2", "w3"):
        source = getattr(expert, name).weight
        qs = source.quant_state
        staged = slot.tensors[name]

        assert "weight" in staged
        assert "absmax" in staged
        assert "code" in staged
        assert "offset" in staged

        assert torch.equal(staged["weight"], source.data)
        assert torch.equal(staged["absmax"], qs.absmax)
        assert torch.equal(staged["code"], qs.code)

        if qs.offset is not None:
            assert staged["offset"] is not None
            assert torch.equal(staged["offset"], qs.offset)

        if qs.state2 is not None:
            assert "state2_absmax" in staged
            assert "state2_code" in staged
            assert "state2_offset" in staged

            assert torch.equal(
                staged["state2_absmax"],
                qs.state2.absmax,
            )
            assert torch.equal(
                staged["state2_code"],
                qs.state2.code,
            )

            if qs.state2.offset is not None:
                assert torch.equal(
                    staged["state2_offset"],
                    qs.state2.offset,
                )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_staged_size_matches_actual_tensors():
    expert = make_test_expert()

    pool = ReusablePinnedStagingPool(
        PinnedMemoryBudget(10 * 1024 * 1024),
        slot_size_bytes=5 * 1024 * 1024,
    )

    slot = pool.acquire()
    pool.stage_expert(slot, expert, layer_id=0, expert_id=0)

    actual_bytes = 0

    for state in slot.tensors.values():
        for tensor in state.values():
            if tensor is not None:
                actual_bytes += tensor.numel() * tensor.element_size()

    assert actual_bytes > 0
    assert actual_bytes <= slot.size_bytes

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_staging_slot_reuses_existing_buffers():
    expert1 = make_test_expert()
    expert2 = make_test_expert()

    pool = ReusablePinnedStagingPool(
        PinnedMemoryBudget(10 * 1024 * 1024),
        slot_size_bytes=5 * 1024 * 1024,
    )

    slot = pool.acquire()
    pool.stage_expert(slot, expert1, layer_id=0, expert_id=0)

    # Record physical storage addresses.
    addresses_before = {
        (layer, key): tensor.data_ptr()
        for layer, state in slot.tensors.items()
        for key, tensor in state.items()
        if tensor is not None
    }

    pool.release(slot)

    reused = pool.acquire()

    assert reused is slot

    pool.stage_expert(
        reused,
        expert2,
        layer_id=1,
        expert_id=5,
    )

    addresses_after = {
        (layer, key): tensor.data_ptr()
        for layer, state in reused.tensors.items()
        for key, tensor in state.items()
        if tensor is not None
    }

    assert addresses_before == addresses_after
    assert reused.expert_key == (1, 5)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_release_keeps_physical_buffers():
    expert = make_test_expert()

    pool = ReusablePinnedStagingPool(
        PinnedMemoryBudget(10 * 1024 * 1024),
        slot_size_bytes=5 * 1024 * 1024,
    )

    slot = pool.acquire()
    pool.stage_expert(slot, expert, layer_id=0, expert_id=2)

    pointers = {
        (layer, key): tensor.data_ptr()
        for layer, state in slot.tensors.items()
        for key, tensor in state.items()
        if tensor is not None
    }

    pool.release(slot)

    # Slot is logically free, but physical pinned allocations remain.
    assert slot.in_use is False
    assert slot.tensors

    for layer, state in slot.tensors.items():
        for key, tensor in state.items():
            if tensor is not None:
                assert tensor.data_ptr() == pointers[(layer, key)]
                assert tensor.is_pinned()