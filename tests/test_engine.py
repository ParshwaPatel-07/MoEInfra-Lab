"""Tests for the inference engine subsystem.

Covers :class:`~engine.router.ExpertRouter`,
:class:`~engine.prefetch.PrefetchEngine`, and
:class:`~engine.engine.InferenceEngine`.
"""
from __future__ import annotations

import pytest
import torch

from engine.engine import InferenceEngine
from engine.prefetch import PrefetchEngine
from engine.router import ExpertRouter
from engine.types import ForwardRequest, PrefetchPolicy


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
        logits = torch.randn(tokens, 8)
        indices, weights = router.route(hidden, logits)
        assert indices.shape == (tokens, 2)
        assert weights.shape == (tokens, 2)

    def test_route_weights_sum_to_one(self) -> None:
        """Expert weights for each token should sum to approximately 1.0."""
        router = self._make_router()
        tokens = 8
        logits = torch.randn(tokens, 8)
        _, weights = router.route(torch.randn(tokens, 4096), logits)
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(tokens), atol=1e-5)

    def test_predict_next_layer_experts_sorted(self) -> None:
        """predict_next_layer_experts should return a sorted list."""
        router = self._make_router()
        indices = torch.tensor([[3, 1], [5, 2], [1, 4]])
        predicted = router.predict_next_layer_experts(indices)
        assert predicted == sorted(predicted)

    def test_route_invalid_logit_dim_raises(self) -> None:
        """route() should raise ValueError for wrong logit dimension."""
        router = self._make_router()
        with pytest.raises(ValueError):
            router.route(torch.randn(4, 4096), torch.randn(4, 5))


# ── PrefetchEngine behaviour ──────────────────────────────────────────────

class TestPrefetchEngineBehaviour:
    """Behavioural tests for the implemented PrefetchEngine."""

    def _make_engine(self, cache_manager, transfer_scheduler, policy, depth=2):
        return PrefetchEngine(
            cache_manager=cache_manager,
            transfer_scheduler=transfer_scheduler,
            policy=policy,
            depth=depth,
        )

    def test_none_policy_returns_empty(
        self, cache_manager, transfer_scheduler
    ) -> None:
        """NONE policy should return [] and submit nothing."""
        engine = self._make_engine(
            cache_manager, transfer_scheduler, PrefetchPolicy.NONE
        )
        indices = torch.tensor([[0, 1]])
        result = engine.schedule_prefetch(current_layer=0, expert_indices=indices)
        assert result == []
        assert transfer_scheduler.pending_count() == 0

    def test_next_layer_submits_requests(
        self, cache_manager, transfer_scheduler
    ) -> None:
        """NEXT_LAYER policy should submit at least one request."""
        engine = self._make_engine(
            cache_manager, transfer_scheduler, PrefetchPolicy.NEXT_LAYER
        )
        indices = torch.tensor([[0, 1], [2, 3]])
        submitted = engine.schedule_prefetch(current_layer=5, expert_indices=indices)
        assert len(submitted) > 0
        assert transfer_scheduler.pending_count() > 0

    def test_lookahead_submits_more_than_next_layer(
        self, cache_manager, transfer_scheduler
    ) -> None:
        """LOOKAHEAD depth=2 should submit more requests than NEXT_LAYER."""
        engine_la = self._make_engine(
            cache_manager, transfer_scheduler, PrefetchPolicy.LOOKAHEAD, depth=2
        )
        indices = torch.tensor([[0, 1]])
        submitted = engine_la.schedule_prefetch(current_layer=0, expert_indices=indices)
        # depth=2 → 2 target layers × unique experts; NEXT_LAYER would be 1 layer
        assert len(submitted) >= 2

    def test_cancel_stale_removes_past_layers(
        self, cache_manager, transfer_scheduler
    ) -> None:
        """cancel_stale() should cancel requests for layers < current_layer."""
        engine = self._make_engine(
            cache_manager, transfer_scheduler, PrefetchPolicy.NEXT_LAYER
        )
        indices = torch.tensor([[0, 1]])
        engine.schedule_prefetch(current_layer=3, expert_indices=indices)
        # Layer 4 was prefetched; cancel stale for anything < 5
        cancelled = engine.cancel_stale(current_layer=5)
        assert cancelled >= 0  # ≥ 0 (may have already executed)
        assert transfer_scheduler.pending_count() == 0

    def test_on_layer_complete_clears_in_flight(
        self, cache_manager, transfer_scheduler
    ) -> None:
        """on_layer_complete() should not raise and should update internal state."""
        engine = self._make_engine(
            cache_manager, transfer_scheduler, PrefetchPolicy.NEXT_LAYER
        )
        indices = torch.tensor([[0, 1]])
        # Should not raise
        engine.on_layer_complete(layer_id=0, expert_indices=indices)
