import pytest
import torch


MODEL_PATH = (
    "/kaggle/input/models/mistral-ai/mixtral/"
    "pytorch/8x7b-instruct-v0.1-hf/1"
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_real_mixtral_expert_load_and_forward():
    from model.loader import ModelLoader

    loader = ModelLoader(
        model_name=MODEL_PATH,
        num_layers=32,
        num_experts=8,
        hidden_size=4096,
        intermediate_size=14336,
    )

    loader.load()

    # Load one real expert from the checkpoint.
    expert = loader.load_expert(
        layer_id=0,
        expert_id=0,
    )

    # ---------------------------------------------------------
    # 1. Expert should be CPU-resident after ModelLoader.load_expert()
    # ---------------------------------------------------------
    assert expert.device.type == "cpu"

    # ---------------------------------------------------------
    # 2. Verify actual NF4 packing
    # ---------------------------------------------------------
    for name in ("w1", "w2", "w3"):
        param = getattr(expert, name).weight

        assert param.dtype == torch.uint8
        assert param.quant_state is not None
        assert param.quant_state.quant_type == "nf4"

    # ---------------------------------------------------------
    # 3. Move the real expert to the T4
    # ---------------------------------------------------------
    expert = expert.cuda()
    torch.cuda.synchronize()

    assert expert.device.type == "cuda"

    for name in ("w1", "w2", "w3"):
        param = getattr(expert, name).weight

        assert param.device.type == "cuda"
        assert param.quant_state is not None
        assert param.quant_state.absmax.device.type == "cuda"
        assert param.quant_state.code.device.type == "cuda"

    # ---------------------------------------------------------
    # 4. Run an actual expert forward
    # ---------------------------------------------------------
    hidden_states = torch.randn(
        4,
        4096,
        device="cuda",
        dtype=torch.float16,
    )

    with torch.no_grad():
        output = expert(hidden_states)

    # ---------------------------------------------------------
    # 5. Basic correctness checks
    # ---------------------------------------------------------
    assert output.shape == (4, 4096)
    assert output.device.type == "cuda"
    assert output.dtype == torch.float16
    assert torch.isfinite(output).all()

    print(f"Expert device: {expert.device}")
    print(f"Expert packed size: {expert.size_bytes / 1024**2:.2f} MiB")
    print(f"Output shape: {tuple(output.shape)}")