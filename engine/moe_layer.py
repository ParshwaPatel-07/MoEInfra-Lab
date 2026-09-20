from __future__ import annotations

import logging
import time
from typing import Optional

import torch

from cache.manager import CacheManager
from model.loader import ModelLoader
from engine.router import ExpertRouter
from transfer.scheduler import TransferScheduler
from transfer.types import (
    TransferDirection,
    TransferPriority,
    TransferRequest,
)


class MoELayer:
    """Single Mixtral MoE layer with expert offloading."""

    def __init__(
        self,
        layer_id: int,
        router: ExpertRouter,
        cache_manager: CacheManager,
        model_loader: ModelLoader,
        transfer_scheduler: TransferScheduler,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.layer_id = layer_id
        self.router = router
        self.cache_manager = cache_manager
        self.model_loader = model_loader
        self.transfer_scheduler = transfer_scheduler
        self._logger = logger or logging.getLogger(__name__)

    def _ensure_expert_on_gpu(
        self,
        expert_id: int,
    ):
        """Return an expert that is resident on GPU."""

        expert = self.cache_manager.get(
            self.layer_id,
            expert_id,
        )

        # Cache miss: load the quantized expert into CPU cache.
        if expert is None:
            self._logger.debug(
                "Expert miss: layer=%d expert=%d",
                self.layer_id,
                expert_id,
            )

            expert = self.model_loader.load_expert(
                self.layer_id,
                expert_id,
            )

            self.cache_manager.put(
                self.layer_id,
                expert_id,
                expert,
                device="cpu",
            )

        # Already on GPU.
        if expert.device.type == "cuda":
            return expert

        # CPU-resident: transfer through the scheduler.
        request = TransferRequest(
            request_id=(
                f"layer-{self.layer_id}-expert-{expert_id}"
                f"-{id(self)}"
            ),
            layer_id=self.layer_id,
            expert_id=expert_id,
            direction=TransferDirection.CPU_TO_GPU,
            priority=TransferPriority.HIGH,
            issued_at=time.monotonic(),
        )

        submitted = self.transfer_scheduler.submit(request)

        if not submitted:
            raise RuntimeError(
                f"Failed to submit transfer for "
                f"layer={self.layer_id}, expert={expert_id}"
            )

        result = self.transfer_scheduler.execute_next()

        if result is None or not result.success:
            error = result.error if result is not None else "no transfer result"
            raise RuntimeError(
                f"Failed to transfer expert "
                f"layer={self.layer_id}, expert={expert_id}: {error}"
            )

        expert = self.cache_manager.get(
            self.layer_id,
            expert_id,
        )

        if expert is None or expert.device.type != "cuda":
            raise RuntimeError(
                f"Expert layer={self.layer_id}, expert={expert_id} "
                "was not resident on GPU after transfer"
            )

        return expert

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Run one Mixtral MoE layer."""

        if hidden_states.ndim != 2:
            raise ValueError(
                "hidden_states must have shape "
                "(tokens, hidden_size)"
            )

        if hidden_states.shape[-1] != self.router.hidden_size:
            raise ValueError(
                f"hidden size {hidden_states.shape[-1]} != "
                f"router hidden size {self.router.hidden_size}"
            )

        expert_indices, expert_weights = self.router.route(
            hidden_states
        )

        output = torch.zeros_like(hidden_states)

        # Process each selected expert.
        for slot in range(self.router.num_experts_per_tok):
            expert_ids = expert_indices[:, slot]
            weights = expert_weights[:, slot]

            # Process each unique expert once.
            for expert_id in expert_ids.unique().tolist():
                expert_id = int(expert_id)

                expert = self._ensure_expert_on_gpu(
                    expert_id
                )

                token_mask = expert_ids == expert_id

                if not token_mask.any():
                    continue

                expert_input = hidden_states[token_mask]

                expert_output = expert(expert_input)

                weighted_output = (
                    expert_output
                    * weights[token_mask].unsqueeze(-1)
                )

                output[token_mask] += weighted_output

        return output