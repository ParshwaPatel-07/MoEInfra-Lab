import pytest
import torch

from model.expert import QuantizedMixtralExpert
from transfer.nf4_reconstruct import ReconstructedNF4Expert
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

    expert = expert.cuda()
    torch.cuda.synchronize()
    expert = expert.cpu()

    return expert


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_reconstructed_expert_runs_on_gpu():
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
    torch.cuda.synchronize()

    reconstructed = ReconstructedNF4Expert(
        expert,
        gpu_state,
    )

    torch.cuda.synchronize()

    x = torch.randn(
        2,
        128,
        device="cuda",
        dtype=torch.float16,
    )

    actual = reconstructed(x)

    torch.cuda.synchronize()
    stream.synchronize()

    reconstructed = ReconstructedNF4Expert(
        expert,
        gpu_state,
    )

    x = torch.randn(
        2,
        128,
        device="cuda",
        dtype=torch.float16,
    )

    output = reconstructed(x)

    assert output.shape == (2, 128)
    assert output.device.type == "cuda"
    assert output.dtype == torch.float16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_reconstructed_expert_matches_normal_gpu_expert():
    expert = make_test_expert()

    pool = ReusablePinnedStagingPool(
        PinnedMemoryBudget(20 * 1024 * 1024),
        slot_size_bytes=5 * 1024 * 1024,
    )

    slot = pool.acquire()
    assert slot is not None

    # Keep this expert CPU-resident for staging.
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

    # Now make the normal GPU version.
    normal_gpu_expert = expert.cuda()
    torch.cuda.synchronize()

    # Reconstruct from the same expert metadata/weights.
    reconstructed = ReconstructedNF4Expert(
        expert,
        gpu_state,
    )

    x = torch.randn(
        2,
        128,
        device="cuda",
        dtype=torch.float16,
    )

    expected = normal_gpu_expert(x)
    actual = reconstructed(x)

    torch.cuda.synchronize()

    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-3,
        atol=1e-3,
    )