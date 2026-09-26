"""Model loader for MoEInfra.

:class:`ModelLoader` downloads / loads a quantised Mixtral-8x7B checkpoint
from HuggingFace Hub and provides helpers to retrieve individual expert
tensors on demand.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch

from model.expert import QuantizedMixtralExpert
from model.types import LayerWeights
from safetensors import safe_open


class ModelLoader:
    """Loads and caches Mixtral-8x7B model weights for expert offloading.

    Responsible for downloading or finding a local INT4-quantised checkpoint
    and exposing individual expert layers to the inference engine.

    Attributes:
        model_name: HuggingFace model identifier or local path.
        num_layers: Total number of transformer layers.
        num_experts: Total number of experts per layer.
        hidden_size: Model hidden dimension.
        intermediate_size: FFN intermediate dimension.
        quantization: Quantisation scheme (e.g. ``"int4"``).
    """

    def __init__(
        self,
        model_name: str,
        num_layers: int,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        quantization: str = "int4",
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Initialise the loader.

        Args:
            model_name: HuggingFace Hub model name or local directory path.
            num_layers: Number of transformer layers in the model.
            num_experts: Number of experts per MoE layer.
            hidden_size: Model hidden dimension (e.g. 4096).
            intermediate_size: FFN intermediate dimension (e.g. 14336).
            quantization: Quantisation scheme to apply when loading
                (``"int4"`` or ``"int8"``).
            logger: Optional pre-configured logger.
        """
        self.model_name: str = model_name
        self.num_layers: int = num_layers
        self.num_experts: int = num_experts
        self.hidden_size: int = hidden_size
        self.intermediate_size: int = intermediate_size
        self.quantization: str = quantization
        self._logger: logging.Logger = logger or logging.getLogger(__name__)

        self._checkpoint_path: Optional[Path] = None
        self._is_loaded: bool = False
        self._weight_map: dict[str, str] = {}
        self._checkpoint_path: Optional[Path] = None
    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def load(self) -> None:
        """Locate and index the Mixtral safetensors checkpoint.

        This does not load model weights into memory. It only discovers the
        checkpoint and builds a tensor-name -> shard-file mapping so that
        individual experts can be loaded on demand.
        """
        self._logger.info(
            "Loading model %r with quantization=%s ...",
            self.model_name,
            self.quantization,
        )

        checkpoint_path = Path(self.model_name)

        if not checkpoint_path.exists():
            raise RuntimeError(
                f"Checkpoint path does not exist: {checkpoint_path}"
            )

        if not checkpoint_path.is_dir():
            raise RuntimeError(
                f"Checkpoint path is not a directory: {checkpoint_path}"
            )

        index_path = checkpoint_path / "model.safetensors.index.json"

        if not index_path.exists():
            raise RuntimeError(
                f"Safetensors index not found: {index_path}"
            )

        try:
            import json

            with index_path.open("r", encoding="utf-8") as f:
                index = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Failed to read safetensors index: {index_path}"
            ) from exc

        weight_map = index.get("weight_map")

        if not isinstance(weight_map, dict) or not weight_map:
            raise RuntimeError(
                f"Invalid or empty weight_map in {index_path}"
            )

        # Verify that every referenced shard exists.
        missing_shards = {
            shard
            for shard in weight_map.values()
            if not (checkpoint_path / shard).exists()
        }

        if missing_shards:
            raise RuntimeError(
                f"Checkpoint index references missing shards: "
                f"{sorted(missing_shards)}"
            )

        self._checkpoint_path = checkpoint_path
        self._weight_map = weight_map
        self._is_loaded = True

        self._logger.info(
            "Checkpoint indexed: %d tensors across %d shards.",
            len(self._weight_map),
            len(set(self._weight_map.values())),
        )

    def _require_loaded(self) -> None:
        """Ensure the checkpoint has been indexed before reading weights."""

        if not self._is_loaded or self._checkpoint_path is None:
            raise RuntimeError(
                "Call ModelLoader.load() before requesting model weights."
            )

    def _load_tensors(
        self,
        tensor_names: dict[str, str],
    ) -> dict[str, torch.Tensor]:
        """Load named tensors from the indexed safetensors checkpoint.

        Tensors may live in different shards. Shards are opened once per
        shard rather than once per tensor.
        """

        self._require_loaded()

        assert self._checkpoint_path is not None

        # Resolve tensor -> shard.
        try:
            tensor_to_shard = {
                name: self._weight_map[name]
                for name in tensor_names.values()
            }
        except KeyError as exc:
            raise RuntimeError(
                f"Tensor not found in checkpoint index: {exc}"
            ) from exc

        # Group requested tensors by shard so each shard is opened once.
        tensors_by_shard: dict[str, list[str]] = {}

        for tensor_name, shard in tensor_to_shard.items():
            tensors_by_shard.setdefault(shard, []).append(tensor_name)

        loaded: dict[str, torch.Tensor] = {}

        for shard, names in tensors_by_shard.items():
            shard_path = self._checkpoint_path / shard

            with safe_open(
                shard_path,
                framework="pt",
                device="cpu",
            ) as f:
                for tensor_name in names:
                    loaded[tensor_name] = f.get_tensor(tensor_name)

        return loaded

    def load_embeddings(self) -> torch.Tensor:
        """Load the input token embedding matrix onto CPU."""

        tensors = self._load_tensors({
            "embedding": "model.embed_tokens.weight",
        })

        return tensors["model.embed_tokens.weight"]

    def load_layer(self, layer_id: int) -> LayerWeights:
        """Load the dense weights for one Mixtral transformer layer.

        Args:
            layer_id: Zero-based transformer layer index.

        Returns:
            LayerWeights containing the layer's dense BF16 tensors,
            resident on CPU.
        """

        self._require_loaded()

        if not (0 <= layer_id < self.num_layers):
            raise IndexError(
                f"layer_id {layer_id} out of range "
                f"[0, {self.num_layers})"
            )

        prefix = f"model.layers.{layer_id}"

        tensor_names = {
            "input_layernorm":
                f"{prefix}.input_layernorm.weight",

            "q_proj":
                f"{prefix}.self_attn.q_proj.weight",

            "k_proj":
                f"{prefix}.self_attn.k_proj.weight",

            "v_proj":
                f"{prefix}.self_attn.v_proj.weight",

            "o_proj":
                f"{prefix}.self_attn.o_proj.weight",

            "post_attention_layernorm":
                f"{prefix}.post_attention_layernorm.weight",

            "moe_gate":
                f"{prefix}.block_sparse_moe.gate.weight",
        }

        tensors = self._load_tensors(tensor_names)

        return LayerWeights(
            input_layernorm=tensors[tensor_names["input_layernorm"]],
            q_proj=tensors[tensor_names["q_proj"]],
            k_proj=tensors[tensor_names["k_proj"]],
            v_proj=tensors[tensor_names["v_proj"]],
            o_proj=tensors[tensor_names["o_proj"]],
            post_attention_layernorm=tensors[
                tensor_names["post_attention_layernorm"]
            ],
            moe_gate=tensors[tensor_names["moe_gate"]],
        )
    def load_final_norm(self) -> torch.Tensor:
        """Load the final transformer RMSNorm weights onto CPU."""

        tensors = self._load_tensors({
            "final_norm": "model.norm.weight",
        })

        return tensors["model.norm.weight"]

    def load_lm_head(self) -> torch.Tensor:
        """Load the language-model output projection onto CPU."""

        tensors = self._load_tensors({
            "lm_head": "lm_head.weight",
        })

        return tensors["lm_head.weight"]

    def load_expert(
        self,
        layer_id: int,
        expert_id: int,
    ) -> QuantizedMixtralExpert:
        """Load and NF4-quantize one Mixtral expert."""

        if not self._is_loaded:
            raise RuntimeError(
                "Call ModelLoader.load() before requesting individual experts."
            )

        if not (0 <= layer_id < self.num_layers):
            raise IndexError(
                f"layer_id {layer_id} out of range [0, {self.num_layers})"
            )

        if not (0 <= expert_id < self.num_experts):
            raise IndexError(
                f"expert_id {expert_id} out of range [0, {self.num_experts})"
            )

        if self._checkpoint_path is None:
            raise RuntimeError("Checkpoint path is not initialized.")

        prefix = (
            f"model.layers.{layer_id}."
            f"block_sparse_moe.experts.{expert_id}."
        )

        tensor_names = {
            "w1": prefix + "w1.weight",
            "w2": prefix + "w2.weight",
            "w3": prefix + "w3.weight",
        }

        # Load the three expert tensors.
        # They may be stored across multiple safetensors shards.
        tensors = self._load_tensors(tensor_names)

        w1 = tensors[tensor_names["w1"]]
        w2 = tensors[tensor_names["w2"]]
        w3 = tensors[tensor_names["w3"]]


        # Construct the real NF4 expert.
        expert = QuantizedMixtralExpert(
            w1=w1,
            w2=w2,
            w3=w3,
        )

        # bnb performs the actual packing/quantization when moved to CUDA.
        # Move back to CPU afterwards so the returned expert is already
        # quantized and ready for the CPU cache.
        expert = expert.cuda()
        torch.cuda.synchronize()
        expert = expert.cpu()

        self._logger.debug(
            "Loaded and NF4-quantized expert: layer=%d, expert=%d",
            layer_id,
            expert_id,
        )

        return expert

    def get_expert_size_bytes(self) -> int:
        """Return the estimated on-disk / in-memory size of a single expert.

        Assumes INT4 quantisation (0.5 bytes per parameter) for the three
        projection matrices.

        Returns:
            Estimated size in bytes.
        """
        params_per_expert = (
            self.hidden_size * self.intermediate_size  # w1
            + self.intermediate_size * self.hidden_size  # w2
            + self.hidden_size * self.intermediate_size  # w3
        )
        bytes_per_param = 0.5  # INT4
        return int(params_per_expert * bytes_per_param)

    def __repr__(self) -> str:
        return (
            f"ModelLoader("
            f"model={self.model_name!r}, "
            f"layers={self.num_layers}, experts={self.num_experts}, "
            f"quant={self.quantization}, loaded={self._is_loaded})"
        )
