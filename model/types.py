# model/types.py

"""Model weight data structures used by MoEInfra."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LayerWeights:
    """Dense weights required by one Mixtral transformer layer.

    All tensors are loaded from the checkpoint in their original dtype
    and remain CPU-resident until the inference engine explicitly moves
    them to the target device.
    """

    input_layernorm: torch.Tensor

    q_proj: torch.Tensor
    k_proj: torch.Tensor
    v_proj: torch.Tensor
    o_proj: torch.Tensor

    post_attention_layernorm: torch.Tensor

    moe_gate: torch.Tensor