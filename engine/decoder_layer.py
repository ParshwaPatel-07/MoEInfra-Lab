from __future__ import annotations

import torch
from transformers.models.mixtral.modeling_mixtral import MixtralDecoderLayer


class MoELayerAdapter(torch.nn.Module):
    """Adapt MoEInfra's token-level MoELayer to HF's 3D interface."""

    def __init__(self, moe_layer):
        super().__init__()
        self.moe_layer = moe_layer

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError(
                "hidden_states must have shape "
                "(batch, sequence, hidden)"
            )

        batch_size, sequence_length, hidden_size = hidden_states.shape

        flat = hidden_states.reshape(
            batch_size * sequence_length,
            hidden_size,
        )

        output = self.moe_layer.forward(flat)

        return output.reshape(
            batch_size,
            sequence_length,
            hidden_size,
        )


class MoEInfraDecoderLayer(MixtralDecoderLayer):
    """Mixtral decoder layer using MoEInfra for expert execution."""

    def set_moe(self, moe_layer) -> None:
        """Replace Hugging Face's native MoE block."""
        self.mlp = MoELayerAdapter(moe_layer)