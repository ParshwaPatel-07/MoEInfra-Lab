import pytest
import torch

from model.expert import QuantizedMixtralExpert
from transfer.nf4_transfer import transfer_staged_expert_to_gpu
from transfer.pinned_memory import PinnedMemoryBudget
from transfer.reusable_staging import ReusablePinnedStagingPool


def make_test_expert():
    hidden = 128
    intermediate = 256

    w1 = torch.randn(intermediate, hidden, dtype=torch.bfloat16)
    w2 = torch.randn(hidden, intermediate, dtype=torch.bfloat16)
    w3 = torch.randn(intermediate, hidden, dtype=torch.bfloat16)

    expert = QuantizedMixtralExpert(w1, w2, w3)

    # Force bitsandbytes to materialize the actual NF4 representation.
    expert = expert.cuda()
    torch.cuda.synchronize()
    expert = expert.cpu()

    return expert


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_staged_expert_transfers_to_gpu():
    expert = make_test_expert()

    pool = ReusablePinnedStagingPool(
        PinnedMemoryBudget(20 * 1024 * 1024),
        slot_size_bytes=5 * 1024 * 1024,
    )

    slot = pool.acquire()
    assert slot is not None

    pool.stage_expert(
        slot,
        expert,
        layer_id=0,
        expert_id=0,
    )

    stream = torch.cuda.Stream()

    gpu_state = transfer_staged_expert_to_gpu(
        slot,
        stream,
    )

    # Transfer is asynchronous, so wait before inspecting results.
    stream.synchronize()

    for name in ("w1", "w2", "w3"):
        state = gpu_state[name]

        for tensor in state.values():
            if tensor is not None:
                assert tensor.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_complete_nf4_state_is_transferred():
    expert = make_test_expert()

    pool = ReusablePinnedStagingPool(
        PinnedMemoryBudget(20 * 1024 * 1024),
        slot_size_bytes=5 * 1024 * 1024,
    )

    slot = pool.acquire()
    assert slot is not None

    pool.stage_expert(
        slot,
        expert,
        layer_id=0,
        expert_id=0,
    )

    stream = torch.cuda.Stream()

    gpu_state = transfer_staged_expert_to_gpu(
        slot,
        stream,
    )

    stream.synchronize()

    for name in ("w1", "w2", "w3"):
        cpu_state = slot.tensors[name]
        gpu_state_layer = gpu_state[name]

        for key, cpu_tensor in cpu_state.items():
            gpu_tensor = gpu_state_layer[key]

            if cpu_tensor is None:
                assert gpu_tensor is None
            else:
                assert gpu_tensor is not None
                assert gpu_tensor.device.type == "cuda"
                assert gpu_tensor.shape == cpu_tensor.shape
                assert gpu_tensor.dtype == cpu_tensor.dtype
                assert torch.equal(
                    gpu_tensor.cpu(),
                    cpu_tensor,
                )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_transfer_does_not_modify_cpu_staging_buffers():
    expert = make_test_expert()

    pool = ReusablePinnedStagingPool(
        PinnedMemoryBudget(20 * 1024 * 1024),
        slot_size_bytes=5 * 1024 * 1024,
    )

    slot = pool.acquire()
    assert slot is not None

    pool.stage_expert(
        slot,
        expert,
        layer_id=0,
        expert_id=0,
    )

    # Snapshot the staged data.
    before = {
        name: {
            key: None if tensor is None else tensor.clone()
            for key, tensor in state.items()
        }
        for name, state in slot.tensors.items()
    }

    stream = torch.cuda.Stream()

    transfer_staged_expert_to_gpu(
        slot,
        stream,
    )

    stream.synchronize()

    # CPU staging buffers must remain unchanged.
    for name, state in slot.tensors.items():
        for key, tensor in state.items():
            if tensor is None:
                continue

            assert torch.equal(
                tensor,
                before[name][key],
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_transfer_requires_active_slot():
    expert = make_test_expert()

    pool = ReusablePinnedStagingPool(
        PinnedMemoryBudget(20 * 1024 * 1024),
        slot_size_bytes=5 * 1024 * 1024,
    )

    slot = pool.acquire()
    assert slot is not None

    pool.stage_expert(
        slot,
        expert,
        layer_id=0,
        expert_id=0,
    )

    pool.release(slot)

    stream = torch.cuda.Stream()

    with pytest.raises(RuntimeError):
        transfer_staged_expert_to_gpu(
            slot,
            stream,
        )