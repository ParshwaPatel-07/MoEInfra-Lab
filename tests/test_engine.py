"""Tests for the inference engine subsystem.

Covers :class:`~engine.router.ExpertRouter` and
:class:`~engine.prefetch.PrefetchEngine`.

The prefetch tests use a small fake async scheduler so these tests verify
PrefetchEngine's scheduling/bookkeeping logic without requiring CUDA,
pinned staging memory, or an actual DMA transfer.
"""

from __future__ import annotations

import pytest
import torch

from engine.prefetch import PrefetchEngine
from engine.router import ExpertRouter
from engine.types import PrefetchPolicy
from transfer.types import TransferStatus


class FakeAsyncScheduler:
    """Minimal scheduler double for PrefetchEngine unit tests."""

    def __init__(self) -> None:
        self.submitted = []
        self.cancelled = []

    def submit_async(self, request):
        self.submitted.append(request)

        return type(
            "FakeHandle",
            (),
            {
                "request_id": request.request_id,
                "layer_id": request.layer_id,
                "expert_id": request.expert_id,
                "slot": None,
                "event": None,
                "status": TransferStatus.IN_FLIGHT,
            },
        )()

    def cancel(self, request_id: str) -> bool:
        self.cancelled.append(request_id)
        return True


# ── ExpertRouter ──────────────────────────────────────────────────────────


class TestExpertRouter:
    """ExpertRouter tests."""

    def _make_router(self) -> ExpertRouter:
        gate_weight = torch.randn(8, 4096, dtype=torch.float32)

        return ExpertRouter(
            gate_weight=gate_weight,
            num_experts_per_tok=2,
        )

    def test_route_output_shapes(self) -> None:
        """route() should return tensors with shape (tokens, k)."""
        router = self._make_router()

        tokens = 16
        hidden = torch.randn(tokens, 4096)

        indices, weights = router.route(hidden)

        assert indices.shape == (tokens, 2)
        assert weights.shape == (tokens, 2)

    def test_route_weights_sum_to_one(self) -> None:
        """Selected expert weights for each token should sum to one."""
        router = self._make_router()

        tokens = 8
        hidden = torch.randn(tokens, 4096)

        _, weights = router.route(hidden)

        sums = weights.sum(dim=-1)

        assert torch.allclose(
            sums,
            torch.ones_like(sums),
            atol=1e-5,
        )

    def test_predict_next_layer_experts_sorted(self) -> None:
        """predict_next_layer_experts should return a sorted list."""
        router = self._make_router()

        indices = torch.tensor(
            [
                [3, 1],
                [5, 2],
                [1, 4],
            ]
        )

        predicted = router.predict_next_layer_experts(indices)

        assert predicted == sorted(predicted)
        assert predicted == [1, 2, 3, 4, 5]

    def test_route_invalid_hidden_dimension_raises(self) -> None:
        """route() should reject hidden states with the wrong dimension."""
        router = self._make_router()

        with pytest.raises(ValueError):
            router.route(torch.randn(4, 5))


# ── PrefetchEngine behaviour ──────────────────────────────────────────────


class TestPrefetchEngineBehaviour:
    """Behavioural tests for the implemented PrefetchEngine."""

    def _make_engine(
        self,
        cache_manager,
        transfer_scheduler,
        policy,
        depth=2,
    ):
        return PrefetchEngine(
            cache_manager=cache_manager,
            transfer_scheduler=transfer_scheduler,
            policy=policy,
            depth=depth,
        )

    def test_none_policy_returns_empty(self, cache_manager) -> None:
        """NONE policy should return [] and submit nothing."""
        scheduler = FakeAsyncScheduler()

        engine = self._make_engine(
            cache_manager,
            scheduler,
            PrefetchPolicy.NONE,
        )

        indices = torch.tensor([[0, 1]])

        result = engine.schedule_prefetch(
            current_layer=0,
            expert_indices=indices,
        )

        assert result == []
        assert scheduler.submitted == []

    def test_next_layer_submits_one_request_per_unique_expert(
        self,
        cache_manager,
    ) -> None:
        """NEXT_LAYER should submit one request for each unique expert."""
        scheduler = FakeAsyncScheduler()

        engine = self._make_engine(
            cache_manager,
            scheduler,
            PrefetchPolicy.NEXT_LAYER,
        )

        indices = torch.tensor(
            [
                [0, 1],
                [2, 3],
                [1, 2],
            ]
        )

        submitted = engine.schedule_prefetch(
            current_layer=5,
            expert_indices=indices,
        )

        assert len(submitted) == 4
        assert len(scheduler.submitted) == 4

        assert {
            (request.layer_id, request.expert_id)
            for request in scheduler.submitted
        } == {
            (6, 0),
            (6, 1),
            (6, 2),
            (6, 3),
        }

    def test_lookahead_submits_requests_for_each_target_layer(
        self,
        cache_manager,
    ) -> None:
        """LOOKAHEAD depth=2 should schedule both upcoming layers."""
        scheduler = FakeAsyncScheduler()

        engine = self._make_engine(
            cache_manager,
            scheduler,
            PrefetchPolicy.LOOKAHEAD,
            depth=2,
        )

        indices = torch.tensor([[0, 1]])

        submitted = engine.schedule_prefetch(
            current_layer=0,
            expert_indices=indices,
        )

        assert len(submitted) == 4

        assert {
            (request.layer_id, request.expert_id)
            for request in scheduler.submitted
        } == {
            (1, 0),
            (1, 1),
            (2, 0),
            (2, 1),
        }

    def test_duplicate_requests_are_not_submitted(
        self,
        cache_manager,
    ) -> None:
        """Repeated scheduling should not duplicate in-flight transfers."""
        scheduler = FakeAsyncScheduler()

        engine = self._make_engine(
            cache_manager,
            scheduler,
            PrefetchPolicy.NEXT_LAYER,
        )

        indices = torch.tensor([[0, 1]])

        first = engine.schedule_prefetch(
            current_layer=3,
            expert_indices=indices,
        )

        second = engine.schedule_prefetch(
            current_layer=3,
            expert_indices=indices,
        )

        assert len(first) == 2
        assert second == []
        assert len(scheduler.submitted) == 2

    def test_gpu_resident_expert_is_skipped(
        self,
        cache_manager,
        monkeypatch,
    ) -> None:
        """Experts already resident on GPU should not be prefetched."""
        scheduler = FakeAsyncScheduler()

        original = cache_manager.is_gpu_resident

        def is_gpu_resident(
            layer_id: int,
            expert_id: int,
        ) -> bool:
            if (layer_id, expert_id) == (6, 0):
                return True

            return original(layer_id, expert_id)

        monkeypatch.setattr(
            cache_manager,
            "is_gpu_resident",
            is_gpu_resident,
        )

        engine = self._make_engine(
            cache_manager,
            scheduler,
            PrefetchPolicy.NEXT_LAYER,
        )

        indices = torch.tensor([[0, 1]])

        submitted = engine.schedule_prefetch(
            current_layer=5,
            expert_indices=indices,
        )

        assert len(submitted) == 1
        assert len(scheduler.submitted) == 1

        assert scheduler.submitted[0].layer_id == 6
        assert scheduler.submitted[0].expert_id == 1

    def test_cancel_stale_cancels_requests_for_past_layers(
        self,
        cache_manager,
    ) -> None:
        """cancel_stale() should cancel requests for past layers."""
        scheduler = FakeAsyncScheduler()

        engine = self._make_engine(
            cache_manager,
            scheduler,
            PrefetchPolicy.NEXT_LAYER,
        )

        indices = torch.tensor([[0, 1]])

        engine.schedule_prefetch(
            current_layer=3,
            expert_indices=indices,
        )

        cancelled = engine.cancel_stale(
            current_layer=5,
        )

        assert cancelled == 2
        assert len(scheduler.cancelled) == 2

    def test_cancel_stale_keeps_future_requests(
        self,
        cache_manager,
    ) -> None:
        """cancel_stale() should preserve requests for future layers."""
        scheduler = FakeAsyncScheduler()

        engine = self._make_engine(
            cache_manager,
            scheduler,
            PrefetchPolicy.LOOKAHEAD,
            depth=3,
        )

        indices = torch.tensor([[0, 1]])

        engine.schedule_prefetch(
            current_layer=0,
            expert_indices=indices,
        )

        cancelled = engine.cancel_stale(
            current_layer=2,
        )

        assert cancelled == 2
        assert len(scheduler.cancelled) == 2

    def test_on_layer_complete_cancels_stale_and_schedules_next(
        self,
        cache_manager,
    ) -> None:
        """on_layer_complete() should cancel stale work and schedule ahead."""
        scheduler = FakeAsyncScheduler()

        engine = self._make_engine(
            cache_manager,
            scheduler,
            PrefetchPolicy.NEXT_LAYER,
        )

        indices = torch.tensor([[0, 1]])

        # Simulate a prefetch for layer 1.
        engine.schedule_prefetch(
            current_layer=0,
            expert_indices=indices,
        )

        # Layer 1 remains useful, so cancel_stale() must not remove it.
        # schedule_prefetch() also sees it already in-flight and must not
        # submit a duplicate.
        engine.on_layer_complete(
            layer_id=0,
            expert_indices=indices,
        )

        assert len(scheduler.submitted) == 2
        assert scheduler.cancelled == []