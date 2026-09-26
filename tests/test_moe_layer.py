import torch

from model.moe_layer import MoELayer
from engine.decoder_layer import (
    MoEInfraDecoderLayer,
    MoELayerAdapter,
)
from transformers import MixtralConfig
from transformers.models.mixtral.modeling_mixtral import MixtralRotaryEmbedding


def test_moe_layer_3d_input_preserves_token_order(monkeypatch):
    moe = MoELayer(
        layer_id=0,
        router=None,
        cache_manager=None,
        model_loader=None,
        transfer_scheduler=None,
    )

    def fake_forward_tokens(hidden_states):
        return hidden_states * 2.0 + 1.0

    monkeypatch.setattr(
        moe,
        "_forward_tokens",
        fake_forward_tokens,
    )

    hidden_states = torch.arange(
        2 * 3 * 4,
        dtype=torch.float32,
    ).reshape(2, 3, 4)

    output_3d = moe.forward(hidden_states)

    expected = fake_forward_tokens(
        hidden_states.reshape(-1, 4)
    ).reshape(2, 3, 4)

    torch.testing.assert_close(
        output_3d,
        expected,
    )


def test_decoder_layer_moe_injection():
    config = MixtralConfig(
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        num_local_experts=8,
        num_experts_per_tok=2,
    )

    moe = MoELayer(
        layer_id=0,
        router=None,
        cache_manager=None,
        model_loader=None,
        transfer_scheduler=None,
    )

    layer = MoEInfraDecoderLayer(
        config=config,
        layer_idx=0,
    )

    layer.set_moe(moe)

    assert isinstance(layer.mlp, MoELayerAdapter)
    assert layer.mlp.moe_layer is moe

def test_decoder_layer_forward_with_injected_moe():
    config = MixtralConfig(
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        num_local_experts=8,
        num_experts_per_tok=2,
    )

    class IdentityMoE(torch.nn.Module):
        def forward(self, hidden_states):
            return hidden_states

    layer = MoEInfraDecoderLayer(
        config=config,
        layer_idx=0,
    )

    layer.set_moe(IdentityMoE())

    hidden_states = torch.randn(
        1,
        4,
        4096,
        dtype=torch.float32,
    )

    cache_position = torch.arange(
        4,
        dtype=torch.long,
    )

    position_ids = cache_position.unsqueeze(0)

    rotary_emb = MixtralRotaryEmbedding(config)
    position_embeddings = rotary_emb(
        hidden_states,
        position_ids=position_ids,
    )

    output = layer(
        hidden_states,
        position_embeddings=position_embeddings,
        attention_mask=None,
        position_ids=position_ids,
        cache_position=cache_position,
    )

    assert output.shape == hidden_states.shape
    assert torch.isfinite(output).all()

def test_expert_router_with_mixtral_gate():
    from engine.router import ExpertRouter

    hidden_size = 4096
    num_experts = 8
    top_k = 2

    # Deterministic Mixtral-shaped gate weight.
    torch.manual_seed(42)

    gate_weight = torch.randn(
        num_experts,
        hidden_size,
        dtype=torch.float32,
    )

    router = ExpertRouter(
        gate_weight=gate_weight,
        num_experts_per_tok=top_k,
    )

    hidden_states = torch.randn(
        6,
        hidden_size,
        dtype=torch.float32,
    )

    expert_indices, expert_weights = router.route(
        hidden_states
    )

    assert expert_indices.shape == (6, 2)
    assert expert_weights.shape == (6, 2)

    assert expert_indices.dtype == torch.long

    assert torch.all(
        (expert_indices >= 0)
        & (expert_indices < num_experts)
    )

    assert torch.isfinite(expert_weights).all()

    # Mixtral routing weights for each token should sum to 1.
    torch.testing.assert_close(
        expert_weights.sum(dim=-1),
        torch.ones(6),
        rtol=1e-5,
        atol=1e-5,
    )

def test_decoder_layer_with_real_moe_pipeline(monkeypatch):
    from engine.router import ExpertRouter

    config = MixtralConfig(
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        num_local_experts=8,
        num_experts_per_tok=2,
    )

    torch.manual_seed(42)

    gate_weight = torch.randn(
        8,
        4096,
        dtype=torch.float32,
    )

    router = ExpertRouter(
        gate_weight=gate_weight,
        num_experts_per_tok=2,
    )

    moe = MoELayer(
        layer_id=0,
        router=router,
        cache_manager=None,
        model_loader=None,
        transfer_scheduler=None,
    )

    # Replace only expert execution. Keep the REAL router and
    # MoELayer dispatch logic.
    def fake_ensure_expert_on_gpu(expert_id):
        class FakeExpert:
            def __call__(self, x):
                return x * (expert_id + 1)

        return FakeExpert()

    monkeypatch.setattr(
        moe,
        "_ensure_expert_on_gpu",
        fake_ensure_expert_on_gpu,
    )

    layer = MoEInfraDecoderLayer(
        config=config,
        layer_idx=0,
    )

    layer.set_moe(moe)

    hidden_states = torch.randn(
        1,
        4,
        4096,
        dtype=torch.float32,
    )

    cache_position = torch.arange(
        4,
        dtype=torch.long,
    )

    position_ids = cache_position.unsqueeze(0)

    rotary_emb = MixtralRotaryEmbedding(config)

    position_embeddings = rotary_emb(
        hidden_states,
        position_ids=position_ids,
    )

    output = layer(
        hidden_states,
        position_embeddings=position_embeddings,
        attention_mask=None,
        position_ids=position_ids,
        cache_position=cache_position,
    )

    assert output.shape == hidden_states.shape
    assert torch.isfinite(output).all()