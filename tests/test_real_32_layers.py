import gc
import os
import time

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




class _TimingProfiler:
    """Lightweight timing hooks for the real Mixtral integration test."""

    def __init__(self):
        self.enabled = os.getenv("MOEINFRA_TIMING", "0") == "1"
        self.layer_count = int(os.getenv("MOEINFRA_PROFILE_LAYERS", "32"))
        self.file_open_ms = 0.0
        self.tensor_read_ms = 0.0
        self.quant_construct_ms = 0.0
        self.load_expert_ms = 0.0
        self.transfer_ms = 0.0
        self.stage_ms = 0.0
        self.h2d_ms = 0.0
        self.reconstruct_ms = 0.0
        self.compute_ms = 0.0
        self.ensure_ms = 0.0
        self.ensure_count = 0
        self.gpu_hits = 0
        self.cpu_hits = 0
        self.cpu_misses = 0
        self.load_expert_calls = 0
        self.per_expert = []
        self.current_expert = None

    def summary(self):
        print("\\n================ TIMING PROFILE ================")
        print(f"Profiled layers: {self.layer_count}")
        print(f"file_open:       {self.file_open_ms:10.2f} ms")
        print(f"tensor_read:     {self.tensor_read_ms:10.2f} ms")
        print(f"NF4 construct:   {self.quant_construct_ms:10.2f} ms")
        print(f"load_expert:     {self.load_expert_ms:10.2f} ms")
        print(f"stage/pinning:   {self.stage_ms:10.2f} ms")
        print(f"H2D transfer:    {self.h2d_ms:10.2f} ms")
        print(f"reconstruction:  {self.reconstruct_ms:10.2f} ms")
        print(f"transfer total:  {self.transfer_ms:10.2f} ms")
        print(f"decoder compute: {self.compute_ms:10.2f} ms")
        print("-------------------------------------------------")
        print(f"GPU hits:        {self.gpu_hits}")
        print(f"CPU hits:        {self.cpu_hits}")
        print(f"CPU misses:      {self.cpu_misses}")
        print(f"load_expert calls:{self.load_expert_calls}")
        print("=================================================\\n")


def _install_timing_hooks(profiler):
    """Install non-invasive timing wrappers around the existing implementation."""
    if not profiler.enabled:
        return lambda: None

    import safetensors
    import model.loader as loader_module
    import model.expert as expert_module
    import transfer.scheduler as scheduler_module
    import model.moe_layer as moe_module

    originals = []

    # -------------------------------------------------------------
    # safetensors: split file-open from get_tensor() time.
    # -------------------------------------------------------------
    original_safe_open = safetensors.safe_open

    class _TimedSafeOpen:
        def __init__(self, *args, **kwargs):
            started = time.perf_counter()
            self._inner = original_safe_open(*args, **kwargs)
            profiler.file_open_ms += (time.perf_counter() - started) * 1000.0

        def __enter__(self):
            self._inner.__enter__()
            return self

        def __exit__(self, exc_type, exc, tb):
            return self._inner.__exit__(exc_type, exc, tb)

        def get_tensor(self, name):
            started = time.perf_counter()
            result = self._inner.get_tensor(name)
            profiler.tensor_read_ms += (time.perf_counter() - started) * 1000.0
            return result

        def __getattr__(self, name):
            return getattr(self._inner, name)

    def timed_safe_open(*args, **kwargs):
        return _TimedSafeOpen(*args, **kwargs)

    safetensors.safe_open = timed_safe_open
    originals.append((safetensors, "safe_open", original_safe_open))
    if hasattr(loader_module, "safe_open"):
        originals.append((loader_module, "safe_open", loader_module.safe_open))
        loader_module.safe_open = timed_safe_open

    # -------------------------------------------------------------
    # QuantizedMixtralExpert construction.
    # -------------------------------------------------------------
    original_init = expert_module.QuantizedMixtralExpert.__init__

    def timed_init(self, *args, **kwargs):
        started = time.perf_counter()
        original_init(self, *args, **kwargs)
        profiler.quant_construct_ms += (time.perf_counter() - started) * 1000.0

    expert_module.QuantizedMixtralExpert.__init__ = timed_init
    originals.append((expert_module.QuantizedMixtralExpert, "__init__", original_init))

    # -------------------------------------------------------------
    # ModelLoader.load_expert total time.
    # -------------------------------------------------------------
    original_load_expert = loader_module.ModelLoader.load_expert

    def timed_load_expert(self, layer_id, expert_id):
        started = time.perf_counter()
        try:
            return original_load_expert(self, layer_id, expert_id)
        finally:
            profiler.load_expert_ms += (time.perf_counter() - started) * 1000.0
            profiler.load_expert_calls += 1

    loader_module.ModelLoader.load_expert = timed_load_expert
    originals.append((loader_module.ModelLoader, "load_expert", original_load_expert))

    # -------------------------------------------------------------
    # MoE residency path: distinguish GPU hit / CPU hit / CPU miss.
    # -------------------------------------------------------------
    original_ensure = moe_module.MoELayer._ensure_expert_on_gpu

    def timed_ensure(self, expert_id):
        gpu_present = self.cache_manager.peek_gpu(self.layer_id, expert_id)
        cpu_present = self.cache_manager.peek_cpu(self.layer_id, expert_id)

        if gpu_present is not None:
            profiler.gpu_hits += 1
        elif cpu_present is not None:
            profiler.cpu_hits += 1
        else:
            profiler.cpu_misses += 1

        started = time.perf_counter()
        try:
            return original_ensure(self, expert_id)
        finally:
            profiler.ensure_ms += (time.perf_counter() - started) * 1000.0
            profiler.ensure_count += 1

    moe_module.MoELayer._ensure_expert_on_gpu = timed_ensure
    originals.append((moe_module.MoELayer, "_ensure_expert_on_gpu", original_ensure))

    # -------------------------------------------------------------
    # Transfer internals. These hooks are optional because the exact
    # implementation may change while the profiling test remains useful.
    # -------------------------------------------------------------
    if hasattr(scheduler_module, "transfer_staged_expert_to_gpu"):
        original_h2d = scheduler_module.transfer_staged_expert_to_gpu

        def timed_h2d(*args, **kwargs):
            started = time.perf_counter()
            try:
                return original_h2d(*args, **kwargs)
            finally:
                profiler.h2d_ms += (time.perf_counter() - started) * 1000.0

        scheduler_module.transfer_staged_expert_to_gpu = timed_h2d
        originals.append((scheduler_module, "transfer_staged_expert_to_gpu", original_h2d))

    if hasattr(scheduler_module, "ReconstructedNF4Expert"):
        original_reconstruct = scheduler_module.ReconstructedNF4Expert

        class _TimedReconstructedNF4Expert(original_reconstruct):
            def __init__(self, *args, **kwargs):
                started = time.perf_counter()
                try:
                    super().__init__(*args, **kwargs)
                finally:
                    profiler.reconstruct_ms += (time.perf_counter() - started) * 1000.0

        scheduler_module.ReconstructedNF4Expert = _TimedReconstructedNF4Expert
        originals.append((scheduler_module, "ReconstructedNF4Expert", original_reconstruct))

    def restore():
        for obj, name, original in reversed(originals):
            setattr(obj, name, original)

    return restore



def _install_transfer_instance_hooks(profiler, staging_pool, transfer_scheduler):
    if not profiler.enabled:
        return lambda: None

    originals = []

    original_stage = staging_pool.stage_expert

    def timed_stage(*args, **kwargs):
        started = time.perf_counter()
        try:
            return original_stage(*args, **kwargs)
        finally:
            profiler.stage_ms += (time.perf_counter() - started) * 1000.0

    staging_pool.stage_expert = timed_stage
    originals.append((staging_pool, "stage_expert", original_stage))

    original_execute = transfer_scheduler.execute_next

    def timed_execute(*args, **kwargs):
        started = time.perf_counter()
        try:
            return original_execute(*args, **kwargs)
        finally:
            profiler.transfer_ms += (time.perf_counter() - started) * 1000.0

    transfer_scheduler.execute_next = timed_execute
    originals.append((transfer_scheduler, "execute_next", original_execute))

    def restore():
        for obj, name, original in reversed(originals):
            setattr(obj, name, original)

    return restore

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

    profiler = _TimingProfiler()
    restore_timing = _install_timing_hooks(profiler)
    restore_transfer = lambda: None

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

    restore_transfer = _install_transfer_instance_hooks(
        profiler, staging_pool, transfer_scheduler
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

    layer_count = min(config.num_hidden_layers, profiler.layer_count)

    try:
        for layer_id in range(layer_count):

            print(f"\n===== Layer {layer_id} =====")

            # -----------------------------------------------------
            # Load this layer's real dense weights from checkpoint
            # -----------------------------------------------------
            _t_load_layer = time.perf_counter()
            weights = loader.load_layer(layer_id)
            _load_layer_ms = (time.perf_counter() - _t_load_layer) * 1000.0
            if profiler.enabled:
                print(f"load_layer total: {_load_layer_ms:.2f} ms")

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

    finally:
        restore_transfer()
        restore_timing()

    # ---------------------------------------------------------
    # 7. Final synchronization
    # ---------------------------------------------------------
    torch.cuda.synchronize()

    # ---------------------------------------------------------
    # 8. Final correctness checks
    # ---------------------------------------------------------
    assert len(layer_outputs) == layer_count

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
    print(f"{layer_count}-LAYER TEST PASSED")
    print("========================================")
    print(f"Final hidden shape: {tuple(hidden_states.shape)}")
    print(f"Final dtype: {hidden_states.dtype}")
    print(f"Total cache misses: {stats.misses}")
    print(f"Total cache hits: {stats.hits}")
    print(f"GPU slots used: {stats.gpu_slots_used}")
    print(f"CPU slots used: {stats.cpu_slots_used}")
    print(f"Total evictions: {stats.evictions}")

    if profiler.enabled:
        profiler.summary()