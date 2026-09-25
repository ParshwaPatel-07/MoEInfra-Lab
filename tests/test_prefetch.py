import torch

from engine.prefetch import PrefetchEngine
from engine.types import PrefetchPolicy
from transfer.types import TransferStatus


class FakeCacheManager:
    def __init__(self, gpu_resident=None):
        self.gpu_resident = set(gpu_resident or [])

    def is_gpu_resident(self, layer_id, expert_id):
        return (layer_id, expert_id) in self.gpu_resident


class FakeScheduler:
    """Small scheduler double for PrefetchEngine unit tests."""

    def __init__(self, async_status=TransferStatus.IN_FLIGHT):
        self.async_status = async_status
        self.async_calls = []
        self.cancel_calls = []

    def submit_async(self, request):
        self.async_calls.append(request)

        class Handle:
            def __init__(self, status):
                self.status = status

        return Handle(self.async_status)

    def cancel(self, request_id):
        self.cancel_calls.append(request_id)
        return True


def routing_indices(*experts):
    return torch.tensor([experts], dtype=torch.long)


def make_prefetch(
    policy=PrefetchPolicy.NEXT_LAYER,
    depth=1,
    gpu_resident=None,
    scheduler_status=TransferStatus.IN_FLIGHT,
):
    scheduler = FakeScheduler(scheduler_status)
    cache = FakeCacheManager(gpu_resident)
    prefetch = PrefetchEngine(
        cache_manager=cache,
        transfer_scheduler=scheduler,
        policy=policy,
        depth=depth,
    )
    return prefetch, scheduler


def test_none_policy_submits_nothing():
    prefetch, scheduler = make_prefetch(policy=PrefetchPolicy.NONE, depth=2)

    submitted = prefetch.schedule_prefetch(
        current_layer=3,
        expert_indices=routing_indices(1, 4),
    )

    assert submitted == []
    assert scheduler.async_calls == []
    assert prefetch._in_flight == {}


def test_next_layer_submits_unique_predicted_experts_async():
    prefetch, scheduler = make_prefetch()

    submitted = prefetch.schedule_prefetch(
        current_layer=3,
        expert_indices=routing_indices(1, 4, 1, 4),
    )

    assert len(submitted) == 2
    assert len(scheduler.async_calls) == 2
    assert {
        (r.layer_id, r.expert_id)
        for r in scheduler.async_calls
    } == {(4, 1), (4, 4)}
    assert len(prefetch._in_flight) == 2


def test_lookahead_submits_each_predicted_expert_for_each_target_layer():
    prefetch, scheduler = make_prefetch(
        policy=PrefetchPolicy.LOOKAHEAD,
        depth=3,
    )

    submitted = prefetch.schedule_prefetch(
        current_layer=2,
        expert_indices=routing_indices(1, 5),
    )

    assert len(submitted) == 6
    assert {
        (r.layer_id, r.expert_id)
        for r in scheduler.async_calls
    } == {
        (3, 1), (3, 5),
        (4, 1), (4, 5),
        (5, 1), (5, 5),
    }


def test_gpu_resident_experts_are_skipped():
    prefetch, scheduler = make_prefetch(
        gpu_resident={(4, 1)},
    )

    submitted = prefetch.schedule_prefetch(
        current_layer=3,
        expert_indices=routing_indices(1, 4),
    )

    assert len(submitted) == 1
    assert len(scheduler.async_calls) == 1
    assert (
        scheduler.async_calls[0].layer_id,
        scheduler.async_calls[0].expert_id,
    ) == (4, 4)


def test_duplicate_prefetch_is_not_submitted_twice():
    prefetch, scheduler = make_prefetch()

    first = prefetch.schedule_prefetch(
        current_layer=3,
        expert_indices=routing_indices(1, 4),
    )
    second = prefetch.schedule_prefetch(
        current_layer=3,
        expert_indices=routing_indices(1, 4),
    )

    assert len(first) == 2
    assert second == []
    assert len(scheduler.async_calls) == 2
    assert len(prefetch._in_flight) == 2


def test_rejected_async_submission_is_not_tracked():
    prefetch, scheduler = make_prefetch(
        scheduler_status=TransferStatus.REJECTED,
    )

    submitted = prefetch.schedule_prefetch(
        current_layer=3,
        expert_indices=routing_indices(1, 4),
    )

    assert submitted == []
    assert len(scheduler.async_calls) == 2
    assert prefetch._in_flight == {}


def test_on_layer_complete_cancels_stale_then_schedules_next():
    prefetch, scheduler = make_prefetch()

    prefetch._in_flight[(1, 7)] = "stale-request"

    prefetch.on_layer_complete(
        layer_id=1,
        expert_indices=routing_indices(3, 6),
    )

    assert scheduler.cancel_calls == ["stale-request"]
    assert {
        (r.layer_id, r.expert_id)
        for r in scheduler.async_calls
    } == {(2, 3), (2, 6)}


def test_cancel_stale_removes_successfully_cancelled_requests():
    prefetch, scheduler = make_prefetch()

    prefetch._in_flight = {
        (1, 2): "req-a",
        (2, 3): "req-b",
        (4, 5): "req-c",
    }

    cancelled = prefetch.cancel_stale(current_layer=3)

    assert cancelled == 2
    assert scheduler.cancel_calls == ["req-a", "req-b"]
    assert prefetch._in_flight == {(4, 5): "req-c"}


def test_cancel_stale_cleans_bookkeeping_when_scheduler_rejects_cancel():
    prefetch, scheduler = make_prefetch()

    def reject_cancel(request_id):
        scheduler.cancel_calls.append(request_id)
        return False

    scheduler.cancel = reject_cancel
    prefetch._in_flight = {(1, 2): "req-a"}

    cancelled = prefetch.cancel_stale(current_layer=2)

    assert cancelled == 0
    assert scheduler.cancel_calls == ["req-a"]
    assert prefetch._in_flight == {}


def test_new_prefetch_can_be_submitted_after_bookkeeping_is_cleared():
    prefetch, scheduler = make_prefetch()

    first = prefetch.schedule_prefetch(
        current_layer=3,
        expert_indices=routing_indices(1),
    )
    assert len(first) == 1

    prefetch._in_flight.clear()

    second = prefetch.schedule_prefetch(
        current_layer=3,
        expert_indices=routing_indices(1),
    )

    assert len(second) == 1
    assert len(scheduler.async_calls) == 2
