import gc

import pytest
import torch
from transformers import MixtralConfig
from transformers.models.mixtral.modeling_mixtral import MixtralRotaryEmbedding

from cache.manager import CacheManager
from cache.types import EvictionPolicy
from engine.decoder_layer import MoEInfraDecoderLayer
from engine.router import ExpertRouter
from model.loader import ModelLoader
from model.moe_layer import MoELayer
from transfer.pinned_memory import PinnedMemoryBudget
from transfer.reusable_staging import ReusablePinnedStagingPool
from transfer.scheduler import TransferScheduler


MODEL_PATH = (
    "/kaggle/input/models/mistral-ai/mixtral/"
    "pytorch/8x7b-instruct-v0.1-hf/1"
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_real_mixtral_32_layer_chain_on_t4():
    """
    Run all 32 real Mixtral decoder layers sequentially with:

    - real checkpoint weights
    - real HF attention
    - our ExpertRouter
    - our MoELayer
    - CPU NF4 expert backing cache
    - 1-slot GPU expert cache
    - real CPU -> GPU transfer/reconstruction

    This is a lightweight cross-layer correctness test.

    It intentionally does NOT test:
    - tokenizer
    - KV cache
    - generation
    - prefetching
    - batching
    - performance
    """

    # ---------------------------------------------------------
    # 1. Load the real Mixtral checkpoint
    # ---------------------------------------------------------
    loader = ModelLoader(
        model_name=MODEL_PATH,
        num_layers=32,
        num_experts=8,
        hidden_size=4096,
        intermediate_size=14336,
    )

    loader.load()

    # ---------------------------------------------------------
    # 2. Load the real Mixtral configuration
    # ---------------------------------------------------------
    config = MixtralConfig.from_pretrained(MODEL_PATH)

    # Transformers 5.x requires an explicit attention backend
    # for standalone decoder-layer construction.
    config._attn_implementation = "eager"

    assert config.hidden_size == 4096
    assert config.num_hidden_layers == 32
    assert config.num_local_experts == 8
    assert config.num_experts_per_tok == 2

    # ---------------------------------------------------------
    # 3. Shared cache + transfer infrastructure
    #
    # IMPORTANT:
    # These are shared across ALL 32 layers.
    #
    # Therefore GPU capacity is genuinely limited to one
    # expert across the complete layer chain.
    # ---------------------------------------------------------
    cache_manager = CacheManager(
        gpu_slots=1,
        cpu_slots=8,
        policy=EvictionPolicy.LRU,
    )

    staging_budget = PinnedMemoryBudget(
        budget_bytes=512 * 1024 * 1024,
    )

    staging_pool = ReusablePinnedStagingPool(
        budget=staging_budget,
        slot_size_bytes=128 * 1024 * 1024,
    )

    transfer_stream = torch.cuda.Stream()

    transfer_scheduler = TransferScheduler(
        cache_manager=cache_manager,
        bandwidth_gbps=10.0,
        max_concurrent=1,
        staging_pool=staging_pool,
        transfer_stream=transfer_stream,
    )

    # ---------------------------------------------------------
    # 4. Synthetic hidden states
    #
    # This is the input to layer 0.
    # Every subsequent layer receives the previous layer's
    # output.
    # ---------------------------------------------------------
    batch_size = 1
    sequence_length = 4
    hidden_size = config.hidden_size

    hidden_states = torch.randn(
        batch_size,
        sequence_length,
        hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    )

    initial_hidden_states = hidden_states.clone()

    # ---------------------------------------------------------
    # 5. Shared positional information
    #
    # RoPE is computed once for this synthetic sequence.
    # Each decoder layer receives the same position embeddings,
    # just like the model-level forward path.
    # ---------------------------------------------------------
    position_ids = torch.arange(
        sequence_length,
        device="cuda",
    ).unsqueeze(0)

    cache_position = torch.arange(
        sequence_length,
        device="cuda",
    )

    rotary_emb = MixtralRotaryEmbedding(config)

    position_embeddings = rotary_emb(
        hidden_states,
        position_ids=position_ids,
    )

    # ---------------------------------------------------------
    # 6. Run all 32 real decoder layers
    # ---------------------------------------------------------
    layer_outputs = []

    for layer_id in range(config.num_hidden_layers):

        print(f"\n===== Layer {layer_id} =====")

        # -----------------------------------------------------
        # Load this layer's real dense weights from checkpoint
        # -----------------------------------------------------
        weights = loader.load_layer(layer_id)

        assert weights.input_layernorm.device.type == "cpu"
        assert weights.q_proj.device.type == "cpu"
        assert weights.k_proj.device.type == "cpu"
        assert weights.v_proj.device.type == "cpu"
        assert weights.o_proj.device.type == "cpu"
        assert weights.post_attention_layernorm.device.type == "cpu"
        assert weights.moe_gate.device.type == "cpu"

        # -----------------------------------------------------
        # Construct HF decoder layer
        # -----------------------------------------------------
        decoder_layer = MoEInfraDecoderLayer(
            config,
            layer_idx=layer_id,
        )

        decoder_layer = decoder_layer.to(
            device="cuda",
            dtype=torch.bfloat16,
        )

        # -----------------------------------------------------
        # Copy real dense checkpoint weights
        # -----------------------------------------------------
        with torch.no_grad():
            decoder_layer.input_layernorm.weight.copy_(
                weights.input_layernorm.cuda()
            )

            decoder_layer.self_attn.q_proj.weight.copy_(
                weights.q_proj.cuda()
            )

            decoder_layer.self_attn.k_proj.weight.copy_(
                weights.k_proj.cuda()
            )

            decoder_layer.self_attn.v_proj.weight.copy_(
                weights.v_proj.cuda()
            )

            decoder_layer.self_attn.o_proj.weight.copy_(
                weights.o_proj.cuda()
            )

            decoder_layer.post_attention_layernorm.weight.copy_(
                weights.post_attention_layernorm.cuda()
            )

        # -----------------------------------------------------
        # Construct our router using the REAL Mixtral gate
        # -----------------------------------------------------
        router = ExpertRouter(
            gate_weight=weights.moe_gate.cuda(),
            num_experts_per_tok=config.num_experts_per_tok,
        )

        # -----------------------------------------------------
        # Construct our MoE layer
        #
        # Same CacheManager and TransferScheduler are reused
        # across every layer.
        # -----------------------------------------------------
        moe_layer = MoELayer(
            layer_id=layer_id,
            router=router,
            cache_manager=cache_manager,
            model_loader=loader,
            transfer_scheduler=transfer_scheduler,
        )

        # -----------------------------------------------------
        # Inject our MoE implementation
        # -----------------------------------------------------
        decoder_layer.set_moe(moe_layer)

        # -----------------------------------------------------
        # Run this layer
        # -----------------------------------------------------
        with torch.no_grad():
            output = decoder_layer(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=None,
                position_ids=position_ids,
                cache_position=cache_position,
                past_key_values=None,
            )

        torch.cuda.synchronize()

        # -----------------------------------------------------
        # Per-layer correctness checks
        # -----------------------------------------------------
        assert output.shape == (
            batch_size,
            sequence_length,
            hidden_size,
        )

        assert output.device.type == "cuda"
        assert output.dtype == torch.bfloat16
        assert torch.isfinite(output).all()

        # -----------------------------------------------------
        # GPU cache must NEVER exceed one expert
        # -----------------------------------------------------
        stats = cache_manager.stats()

        assert stats.gpu_slots_used <= 1

        print(
            f"output shape: {tuple(output.shape)}, "
            f"cache misses: {stats.misses}, "
            f"GPU slots: {stats.gpu_slots_used}"
        )

        # -----------------------------------------------------
        # Feed this layer's output into the next layer
        # -----------------------------------------------------
        hidden_states = output

        layer_outputs.append(output)

        # -----------------------------------------------------
        # Release the layer-specific objects.
        #
        # The cache is intentionally NOT destroyed.
        # We want the next layer to exercise the same shared
        # cache/eviction system.
        # -----------------------------------------------------
        del decoder_layer
        del moe_layer
        del router
        del weights

        gc.collect()
        torch.cuda.empty_cache()

    # ---------------------------------------------------------
    # 7. Final synchronization
    # ---------------------------------------------------------
    torch.cuda.synchronize()

    # ---------------------------------------------------------
    # 8. Final correctness checks
    # ---------------------------------------------------------
    assert len(layer_outputs) == 32

    assert hidden_states.shape == (
        batch_size,
        sequence_length,
        hidden_size,
    )

    assert hidden_states.device.type == "cuda"
    assert hidden_states.dtype == torch.bfloat16
    assert torch.isfinite(hidden_states).all()

    # The network must actually transform the input.
    assert not torch.equal(
        initial_hidden_states,
        hidden_states,
    )

    # ---------------------------------------------------------
    # 9. Final cache invariants
    # ---------------------------------------------------------
    stats = cache_manager.stats()

    assert stats.gpu_slots_used <= 1
    assert stats.misses > 0

    print("\n========================================")
    print("32-LAYER TEST PASSED")
    print("========================================")
    print(f"Final hidden shape: {tuple(hidden_states.shape)}")
    print(f"Final dtype: {hidden_states.dtype}")
    print(f"Total cache misses: {stats.misses}")
    print(f"Total cache hits: {stats.hits}")
    print(f"GPU slots used: {stats.gpu_slots_used}")
    print(f"CPU slots used: {stats.cpu_slots_used}")
    print(f"Total evictions: {stats.evictions}")