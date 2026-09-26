from model.loader import ModelLoader
from model.moe_layer import MoELayer
from engine.decoder_layer import MoEInfraDecoderLayer

import torch
import pytest
from transformers import MixtralConfig

def test_real_mixtral_layer_weights_load_into_hf_decoder():

    model_path = (
        "/kaggle/input/models/mistral-ai/mixtral/"
        "pytorch/8x7b-instruct-v0.1-hf/1"
    )

    loader = ModelLoader(
        model_name=model_path,
        num_layers=32,
        num_experts=8,
        hidden_size=4096,
        intermediate_size=14336,
    )

    loader.load()

    weights = loader.load_layer(0)

    config = MixtralConfig.from_pretrained(model_path)

    layer = MoEInfraDecoderLayer(
        config=config,
        layer_idx=0,
    )

    with torch.no_grad():
        layer.input_layernorm.weight.copy_(
            weights.input_layernorm
        )

        layer.self_attn.q_proj.weight.copy_(
            weights.q_proj
        )

        layer.self_attn.k_proj.weight.copy_(
            weights.k_proj
        )

        layer.self_attn.v_proj.weight.copy_(
            weights.v_proj
        )

        layer.self_attn.o_proj.weight.copy_(
            weights.o_proj
        )

        layer.post_attention_layernorm.weight.copy_(
            weights.post_attention_layernorm
        )

    assert torch.equal(
        layer.self_attn.q_proj.weight,
        weights.q_proj,
    )

    assert torch.equal(
        layer.self_attn.k_proj.weight,
        weights.k_proj,
    )

    assert torch.equal(
        layer.self_attn.v_proj.weight,
        weights.v_proj,
    )

    assert torch.equal(
        layer.self_attn.o_proj.weight,
        weights.o_proj,
    )

    assert torch.equal(
        layer.input_layernorm.weight,
        weights.input_layernorm,
    )

    assert torch.equal(
        layer.post_attention_layernorm.weight,
        weights.post_attention_layernorm,
    )