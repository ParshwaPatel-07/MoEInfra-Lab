from __future__ import annotations

import copy

import bitsandbytes as bnb
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.expert import QuantizedMixtralExpert


def _reconstruct_quant_state(original_qs, state):
    qs = copy.copy(original_qs)

    qs.absmax = state["absmax"]
    qs.code = state["code"]
    qs.offset = state["offset"]

    if original_qs.state2 is not None:
        state2 = copy.copy(original_qs.state2)

        state2.absmax = state["state2_absmax"]
        state2.code = state["state2_code"]
        state2.offset = state["state2_offset"]

        qs.state2 = state2

    return qs


def _reconstruct_linear(
    original_linear,
    state,
):
    original_param = original_linear.weight

    new_param = bnb.nn.Params4bit(
        state["weight"],
        requires_grad=False,
        quant_type="nf4",
    )

    new_qs = _reconstruct_quant_state(
        original_param.quant_state,
        state,
    )

    new_param.quant_state = new_qs

    new_linear = bnb.nn.Linear4bit(
        input_features=original_param.quant_state.shape[1],
        output_features=original_param.quant_state.shape[0],
        bias=False,
        quant_type="nf4",
        compress_statistics=False,
        compute_dtype=torch.float16,
    )

    new_linear.weight = new_param

    return new_linear


class ReconstructedNF4Expert(nn.Module):
    def __init__(
        self,
        original_expert: QuantizedMixtralExpert,
        gpu_state,
    ):
        super().__init__()

        self.w1 = _reconstruct_linear(
            original_expert.w1,
            gpu_state["w1"],
        )

        self.w2 = _reconstruct_linear(
            original_expert.w2,
            gpu_state["w2"],
        )

        self.w3 = _reconstruct_linear(
            original_expert.w3,
            gpu_state["w3"],
        )

    def forward(self, x):
        gate = self.w1(x)
        up = self.w3(x)

        return self.w2(F.silu(gate) * up)

    @property
    def device(self):
        return self.w1.weight.device

    @property
    def size_bytes(self) -> int:
        return sum(
            layer.weight.numel() * layer.weight.element_size()
            for layer in (self.w1, self.w2, self.w3)
        )