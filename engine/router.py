"""Expert router for Mixtral-style MoE inference.

:class:`ExpertRouter` wraps the top-k gating logic and provides a heuristic
helper to predict which experts the *next* layer is likely to need, enabling
the :class:`~engine.prefetch.PrefetchEngine` to issue proactive transfers.
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn.functional as F


class ExpertRouter:
    """Top-k router for a Mixtral MoE layer."""

    def __init__(
        self,
        gate_weight: torch.Tensor,
        num_experts_per_tok: int = 2,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        if gate_weight.ndim != 2:
            raise ValueError(
                f"gate_weight must be 2D, got shape {tuple(gate_weight.shape)}"
            )

        self.num_experts = gate_weight.shape[0]
        self.hidden_size = gate_weight.shape[1]
        self.num_experts_per_tok = num_experts_per_tok
        self._logger = logger or logging.getLogger(__name__)

        if num_experts_per_tok <= 0:
            raise ValueError("num_experts_per_tok must be positive")

        if num_experts_per_tok > self.num_experts:
            raise ValueError(
                f"num_experts_per_tok ({num_experts_per_tok}) must be <= "
                f"num_experts ({self.num_experts})"
            )

        self.gate_weight = gate_weight

    @property
    def device(self) -> torch.device:
        return self.gate_weight.device

    def logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute Mixtral router logits."""

        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"hidden_states last dim {hidden_states.shape[-1]} != "
                f"hidden_size {self.hidden_size}"
            )

        if hidden_states.device != self.gate_weight.device:
            raise ValueError(
                f"hidden_states device {hidden_states.device} != "
                f"gate device {self.gate_weight.device}"
            )

        # Mixtral gate projection.
        return F.linear(
            hidden_states,
            self.gate_weight,
        )

    def route(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute top-k expert assignments."""

        router_logits = self.logits(hidden_states)

        routing_weights = F.softmax(router_logits, dim=-1)

        expert_weights, expert_indices = torch.topk(
            routing_weights,
            self.num_experts_per_tok,
            dim=-1,
        )

        expert_weights = expert_weights / expert_weights.sum(
            dim=-1,
            keepdim=True,
        )

        return expert_indices, expert_weights

    def predict_next_layer_experts(
        self,
        current_expert_indices: torch.Tensor,
    ) -> list[int]:
        """Predict next-layer experts using current-layer selections."""

        flat = current_expert_indices.reshape(-1)
        unique_ids = flat.unique().tolist()

        return sorted(int(x) for x in unique_ids)
