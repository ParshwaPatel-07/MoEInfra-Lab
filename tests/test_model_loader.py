import json

import torch
from safetensors.torch import save_file

from model.loader import ModelLoader
from model.types import LayerWeights


HIDDEN = 8
INTERMEDIATE = 16
VOCAB = 12
NUM_LAYERS = 2
NUM_EXPERTS = 2


def make_checkpoint(tmp_path):
    """Create a tiny Mixtral-like checkpoint."""

    tensors = {
        "model.embed_tokens.weight":
            torch.randn(VOCAB, HIDDEN, dtype=torch.bfloat16),

        "model.norm.weight":
            torch.randn(HIDDEN, dtype=torch.bfloat16),

        "lm_head.weight":
            torch.randn(VOCAB, HIDDEN, dtype=torch.bfloat16),
    }

    for layer in range(NUM_LAYERS):
        prefix = f"model.layers.{layer}"

        tensors.update({
            f"{prefix}.input_layernorm.weight":
                torch.randn(HIDDEN, dtype=torch.bfloat16),

            f"{prefix}.self_attn.q_proj.weight":
                torch.randn(HIDDEN, HIDDEN, dtype=torch.bfloat16),

            f"{prefix}.self_attn.k_proj.weight":
                torch.randn(HIDDEN // 2, HIDDEN, dtype=torch.bfloat16),

            f"{prefix}.self_attn.v_proj.weight":
                torch.randn(HIDDEN // 2, HIDDEN, dtype=torch.bfloat16),

            f"{prefix}.self_attn.o_proj.weight":
                torch.randn(HIDDEN, HIDDEN, dtype=torch.bfloat16),

            f"{prefix}.post_attention_layernorm.weight":
                torch.randn(HIDDEN, dtype=torch.bfloat16),

            f"{prefix}.block_sparse_moe.gate.weight":
                torch.randn(NUM_EXPERTS, HIDDEN, dtype=torch.bfloat16),
        })

    shard_name = "model-00001-of-00001.safetensors"

    save_file(
        tensors,
        str(tmp_path / shard_name),
    )

    weight_map = {
        name: shard_name
        for name in tensors
    }

    with (tmp_path / "model.safetensors.index.json").open("w") as f:
        json.dump({"weight_map": weight_map}, f)

    return tensors


def make_loader(tmp_path):
    
    return ModelLoader(
        model_name=str(tmp_path),
        num_layers=NUM_LAYERS,
        num_experts=NUM_EXPERTS,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
    )

def test_load_embeddings(tmp_path):
    tensors = make_checkpoint(tmp_path)
    loader = make_loader(tmp_path)

    loader.load()

    result = loader.load_embeddings()

    assert torch.equal(
        result,
        tensors["model.embed_tokens.weight"],
    )
    assert result.shape == (VOCAB, HIDDEN)
    assert result.dtype == torch.bfloat16
    assert result.device.type == "cpu"


def test_load_layer(tmp_path):
    tensors = make_checkpoint(tmp_path)
    loader = make_loader(tmp_path)

    loader.load()

    layer = loader.load_layer(0)

    assert isinstance(layer, LayerWeights)

    assert torch.equal(
        layer.input_layernorm,
        tensors["model.layers.0.input_layernorm.weight"],
    )

    assert torch.equal(
        layer.q_proj,
        tensors["model.layers.0.self_attn.q_proj.weight"],
    )

    assert torch.equal(
        layer.k_proj,
        tensors["model.layers.0.self_attn.k_proj.weight"],
    )

    assert torch.equal(
        layer.v_proj,
        tensors["model.layers.0.self_attn.v_proj.weight"],
    )

    assert torch.equal(
        layer.o_proj,
        tensors["model.layers.0.self_attn.o_proj.weight"],
    )

    assert torch.equal(
        layer.post_attention_layernorm,
        tensors[
            "model.layers.0.post_attention_layernorm.weight"
        ],
    )

    assert torch.equal(
        layer.moe_gate,
        tensors[
            "model.layers.0.block_sparse_moe.gate.weight"
        ],
    )


def test_load_final_norm(tmp_path):
    tensors = make_checkpoint(tmp_path)
    loader = make_loader(tmp_path)

    loader.load()

    result = loader.load_final_norm()

    assert torch.equal(
        result,
        tensors["model.norm.weight"],
    )
    assert result.shape == (HIDDEN,)
    assert result.dtype == torch.bfloat16
    assert result.device.type == "cpu"


def test_load_lm_head(tmp_path):
    tensors = make_checkpoint(tmp_path)
    loader = make_loader(tmp_path)

    loader.load()

    result = loader.load_lm_head()

    assert torch.equal(
        result,
        tensors["lm_head.weight"],
    )
    assert result.shape == (VOCAB, HIDDEN)
    assert result.dtype == torch.bfloat16
    assert result.device.type == "cpu"


def test_load_layer_rejects_invalid_layer(tmp_path):
    make_checkpoint(tmp_path)
    loader = make_loader(tmp_path)

    loader.load()

    try:
        loader.load_layer(NUM_LAYERS)
    except IndexError:
        pass
    else:
        raise AssertionError("Expected IndexError")