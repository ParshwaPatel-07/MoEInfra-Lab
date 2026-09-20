"""Prefetch engine for MoEInfra.

Schedules proactive CPU→GPU expert transfers ahead of the compute pipeline to
hide PCIe latency.

Design choices
--------------
* **Policy dispatch in ``schedule_prefetch``**: a single method handles all
  three policies (NONE / NEXT_LAYER / LOOKAHEAD) so callers never need to
  branch.  NONE is a deliberate no-op — not an error — to allow easy A/B
  comparison of prefetch strategies.
* **In-flight set for deduplication**: ``_in_flight`` tracks
  ``(layer_id, expert_id)`` pairs whose ``TransferRequest`` has already been
  submitted to the :class:`~transfer.scheduler.TransferScheduler`.  This
  prevents double-submission when the same expert appears in multiple lookahead
  layers.  We store request IDs alongside the pair so ``cancel_stale`` can
  call ``TransferScheduler.cancel`` by ID.
* **Router heuristic for target layer**: ``ExpertRouter.predict_next_layer_experts``
  returns the unique expert IDs from the *current* layer's routing as a proxy
  for the next layer.  For LOOKAHEAD depth > 1 we reuse this same prediction
  for each additional layer (same heuristic applied repeatedly).  A more
  accurate approach (run a partial forward pass) is left for a later
  optimisation.
* **GPU-residency check before submitting**: if the target expert is already
  in GPU cache (``cache_manager.get`` returns non-None) we skip the transfer
  to avoid redundant PCIe moves.  CPU-resident or absent experts are promoted /
  fetched respectively.
* **cancel_stale iterates ``_in_flight``**: the in-flight set contains at most
  ``depth * num_experts_per_tok`` entries (≤ ~32), so linear scan is fine.
  We collect stale IDs first, then cancel and remove them.
* **``on_layer_complete`` is the main integration point**: the inference loop
  calls this once after each layer.  It handles both the "schedule ahead" and
  "cancel stale" responsibilities in a single call, keeping the engine.py code
  minimal.
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import List, Optional

import torch

from engine.types import PrefetchPolicy
from transfer.types import TransferDirection, TransferPriority, TransferRequest


class PrefetchEngine:
    """Lookahead expert prefetch scheduler.

    Wraps a :class:`~transfer.scheduler.TransferScheduler` and drives it based
    on the active :class:`~engine.types.PrefetchPolicy`.  After each layer
    completes (:meth:`on_layer_complete`) new transfers for upcoming layers are
    issued, and stale requests for past layers are cancelled
    (:meth:`cancel_stale`).

    Attributes:
        policy: The prefetch strategy in use.
        depth: Number of layers to look ahead (only used for
            :attr:`~engine.types.PrefetchPolicy.LOOKAHEAD`).
    """

    def __init__(
        self,
        cache_manager,
        transfer_scheduler,
        policy: PrefetchPolicy,
        depth: int,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Initialise the prefetch engine.

        Args:
            cache_manager: A :class:`~cache.manager.CacheManager` used to
                check current residency before issuing redundant transfers.
            transfer_scheduler: A :class:`~transfer.scheduler.TransferScheduler`
                that will actually execute the PCIe moves.
            policy: The prefetch strategy to apply.
            depth: Look-ahead depth in layers (used only for
                :attr:`~engine.types.PrefetchPolicy.LOOKAHEAD`).
            logger: Optional pre-configured logger.
        """
        self._cache_manager = cache_manager
        self._transfer_scheduler = transfer_scheduler
        self.policy: PrefetchPolicy = policy
        self.depth: int = depth
        self._logger: logging.Logger = logger or logging.getLogger(__name__)

        # Maps (layer_id, expert_id) → request_id for in-flight prefetches
        # so we can cancel them by ID.
        self._in_flight: dict[tuple[int, int], str] = {}

        self._logger.debug(
            "PrefetchEngine initialised: policy=%s, depth=%d",
            policy.value,
            depth,
        )

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _is_gpu_resident(self, layer_id: int, expert_id: int) -> bool:
        """Check whether an expert is already in GPU cache.

        Uses ``cache_manager.get`` which returns ``None`` on a miss.  Note
        that this increments the hit counter — a minor over-count, but
        acceptable since the alternative (direct dict introspection) would
        couple this class to the cache internals.

        Args:
            layer_id: Transformer layer index.
            expert_id: Expert index within the layer.

        Returns:
            ``True`` if a tensor was returned (GPU or CPU resident).
        """
        return self._cache_manager.is_gpu_resident(layer_id, expert_id)

    def _submit_if_needed(
        self, layer_id: int, expert_id: int
    ) -> Optional[str]:
        """Submit a CPU→GPU prefetch request if not already in-flight or resident.

        Args:
            layer_id: Target layer index.
            expert_id: Target expert index.

        Returns:
            The ``request_id`` if a new request was submitted, else ``None``.
        """
        key = (layer_id, expert_id)

        if key in self._in_flight:
            return None  # Already submitted

        if self._is_gpu_resident(layer_id, expert_id):
            return None  # Already on GPU — no transfer needed

        request_id = str(uuid.uuid4())
        request = TransferRequest(
            request_id=request_id,
            layer_id=layer_id,
            expert_id=expert_id,
            direction=TransferDirection.CPU_TO_GPU,
            priority=TransferPriority.LOW,  # Prefetch = background priority
            issued_at=time.monotonic(),
        )
        self._transfer_scheduler.submit(request)
        self._in_flight[key] = request_id

        self._logger.debug(
            "_submit_if_needed: prefetch submitted  layer=%d expert=%d id=%s",
            layer_id, expert_id, request_id,
        )
        return request_id

    def _predict_experts(self, expert_indices: torch.Tensor) -> List[int]:
        """Extract the unique expert IDs from a routing result tensor.

        Args:
            expert_indices: Int tensor of shape ``(tokens, k)`` from the
                router's ``route()`` call.

        Returns:
            Sorted list of unique expert IDs.
        """
        flat = expert_indices.reshape(-1)
        return sorted(int(x) for x in flat.unique().tolist())

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def schedule_prefetch(
        self, current_layer: int, expert_indices: torch.Tensor
    ) -> List[str]:
        """Schedule prefetch transfers for layers ahead of *current_layer*.

        Behaviour depends on :attr:`policy`:

        * :attr:`~engine.types.PrefetchPolicy.NONE` — no-op, returns ``[]``.
        * :attr:`~engine.types.PrefetchPolicy.NEXT_LAYER` — prefetch experts
          predicted for layer ``current_layer + 1``.
        * :attr:`~engine.types.PrefetchPolicy.LOOKAHEAD` — prefetch experts
          for layers ``current_layer + 1`` … ``current_layer + depth``.  The
          same router heuristic (unique experts from *current_layer*) is reused
          as a proxy for each lookahead layer.

        Args:
            current_layer: Index of the layer currently executing.
            expert_indices: Int tensor of shape ``(tokens, k)`` from the
                current layer's router — used to predict upcoming expert sets.

        Returns:
            List of ``request_id`` strings for newly submitted
            :class:`~transfer.types.TransferRequest` objects.
        """
        if self.policy == PrefetchPolicy.NONE:
            return []

        predicted_experts = self._predict_experts(expert_indices)

        if self.policy == PrefetchPolicy.NEXT_LAYER:
            target_layers = [current_layer + 1]
        else:  # LOOKAHEAD
            target_layers = list(range(current_layer + 1, current_layer + self.depth + 1))

        submitted: List[str] = []
        for target_layer in target_layers:
            for expert_id in predicted_experts:
                req_id = self._submit_if_needed(target_layer, expert_id)
                if req_id is not None:
                    submitted.append(req_id)

        if submitted:
            self._logger.debug(
                "schedule_prefetch: current_layer=%d  submitted=%d requests  "
                "targets=%s",
                current_layer, len(submitted), target_layers,
            )
        return submitted

    def on_layer_complete(
        self, layer_id: int, expert_indices: torch.Tensor
    ) -> None:
        """Callback invoked when a transformer layer finishes its forward pass.

        Performs two actions in order:

        1. Cancels stale in-flight prefetches for layers ≤ *layer_id* (already
           executed or no longer needed).
        2. Schedules new prefetches for upcoming layers via
           :meth:`schedule_prefetch`.

        Args:
            layer_id: The layer that just completed.
            expert_indices: Expert assignments used during this layer (shape
                ``(tokens, k)``), forwarded to :meth:`schedule_prefetch`.
        """
        self.cancel_stale(current_layer=layer_id + 1)
        self.schedule_prefetch(
            current_layer=layer_id,
            expert_indices=expert_indices,
        )
        self._logger.debug(
            "on_layer_complete: layer=%d  in_flight=%d", layer_id, len(self._in_flight)
        )

    def cancel_stale(self, current_layer: int) -> int:
        """Cancel all in-flight prefetch requests for layers < *current_layer*.

        Any expert prefetch whose target layer is strictly less than
        *current_layer* is now stale: the compute pipeline has already passed
        (or is about to pass) that layer and the transfer is no longer useful.

        Args:
            current_layer: The layer now executing; requests for any layer
                index below this are considered stale.

        Returns:
            The number of requests successfully cancelled.
        """
        stale_keys = [
            (layer_id, expert_id)
            for (layer_id, expert_id) in self._in_flight
            if layer_id < current_layer
        ]

        cancelled = 0
        for key in stale_keys:
            request_id = self._in_flight.pop(key)
            if self._transfer_scheduler.cancel(request_id):
                cancelled += 1
                self._logger.debug(
                    "cancel_stale: cancelled  layer=%d expert=%d id=%s",
                    key[0], key[1], request_id,
                )
            else:
                # Already executed — that's fine, just bookkeeping
                self._logger.debug(
                    "cancel_stale: already gone  layer=%d expert=%d id=%s",
                    key[0], key[1], request_id,
                )

        if cancelled:
            self._logger.debug(
                "cancel_stale: current_layer=%d  cancelled=%d", current_layer, cancelled
            )
        return cancelled

