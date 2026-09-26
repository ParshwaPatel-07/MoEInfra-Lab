import pytest
import torch
from transformers import MixtralConfig
from transformers.models.mixtral.modeling_mixtral import MixtralRotaryEmbedding
from cache.manager import CacheManager
from cache.types import EvictionPolicy
from transfer.scheduler import TransferScheduler
from cache.manager import CacheManager
from transfer.scheduler import TransferScheduler
from transfer.pinned_memory import PinnedMemoryBudget
from transfer.reusable_staging import ReusablePinnedStagingPool

MODEL_PATH = (
    "/kaggle/input/models/mistral-ai/mixtral/"
    "pytorch/8x7b-instruct-v0.1-hf/1"
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_real_mixtral_decoder_layer_on_t4():
    from model.loader import ModelLoader
    from model.moe_layer import MoELayer
    from engine.decoder_layer import MoEInfraDecoderLayer
    from engine.router import ExpertRouter
    from cache.manager import CacheManager
    from transfer.scheduler import TransferScheduler

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
    # 2. Load real layer-0 dense weights
    # ---------------------------------------------------------
    weights = loader.load_layer(0)

    assert weights.input_layernorm.device.type == "cpu"
    assert weights.q_proj.device.type == "cpu"
    assert weights.k_proj.device.type == "cpu"
    assert weights.v_proj.device.type == "cpu"
    assert weights.o_proj.device.type == "cpu"
    assert weights.post_attention_layernorm.device.type == "cpu"
    assert weights.moe_gate.device.type == "cpu"

    # ---------------------------------------------------------
    # 3. Load the real Mixtral configuration
    # ---------------------------------------------------------
    config = MixtralConfig.from_pretrained(MODEL_PATH)
    config._attn_implementation = "eager"

    assert config.hidden_size == 4096
    assert config.intermediate_size == 14336
    assert config.num_local_experts == 8
    assert config.num_experts_per_tok == 2

    # ---------------------------------------------------------
    # 4. Construct HF decoder layer in BF16
    # ---------------------------------------------------------
    decoder_layer = MoEInfraDecoderLayer(
        config,
        layer_idx=0,
    )

    decoder_layer = decoder_layer.to(
        device="cuda",
        dtype=torch.bfloat16,
    )

    # ---------------------------------------------------------
    # 5. Copy the REAL checkpoint dense weights
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # 6. Build OUR router from the REAL Mixtral gate
    # ---------------------------------------------------------
    router = ExpertRouter(
        gate_weight=weights.moe_gate.cuda(),
        num_experts_per_tok=config.num_experts_per_tok,
    )

    # ---------------------------------------------------------
    # 7. Construct the cache + transfer infrastructure
    #
    # For Step 3 we deliberately use the existing synchronous
    # path. No async prefetching or benchmarking yet.
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
    # 8. Construct OUR real MoE layer
    # ---------------------------------------------------------
    moe_layer = MoELayer(
        layer_id=0,
        router=router,
        cache_manager=cache_manager,
        model_loader=loader,
        transfer_scheduler=transfer_scheduler,
    )

    # ---------------------------------------------------------
    # 9. Inject OUR MoE into the HF decoder layer
    # ---------------------------------------------------------
    decoder_layer.set_moe(moe_layer)

    # ---------------------------------------------------------
    # 10. Create real hidden states
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

    # ---------------------------------------------------------
    # 11. Build real Mixtral rotary embeddings
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
    # 12. Run the COMPLETE real decoder layer
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # 13. Verify decoder output
    # ---------------------------------------------------------
    assert output.shape == (
        batch_size,
        sequence_length,
        hidden_size,
    )

    assert output.device.type == "cuda"
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()

    # ---------------------------------------------------------
    # 14. Verify that the MoE actually populated the cache
    # ---------------------------------------------------------
    stats = cache_manager.stats()

    assert stats.misses > 0
    assert stats.gpu_slots_used > 0

    print(f"Output shape: {tuple(output.shape)}")
    print(f"Output dtype: {output.dtype}")
    print(f"Output device: {output.device}")
    print(f"MoE cache misses: {stats.misses}")
    print(f"GPU cache slots used: {stats.gpu_slots_used}")