import pytest

from transfer.pinned_memory import PinnedMemoryBudget
from transfer.staging_pool import PinnedStagingPool


def make_pool(budget=1000):
    return PinnedStagingPool(
        PinnedMemoryBudget(budget)
    )


def test_initial_state():
    pool = make_pool()

    assert pool.used_bytes == 0
    assert pool.available_bytes == 1000
    assert pool.active_slots == 0


def test_acquire():
    pool = make_pool()

    slot = pool.acquire("s1", 300)

    assert slot is not None
    assert slot.slot_id == "s1"
    assert slot.size_bytes == 300
    assert slot.in_use is True

    assert pool.used_bytes == 300
    assert pool.active_slots == 1


def test_multiple_slots():
    pool = make_pool()

    pool.acquire("s1", 300)
    pool.acquire("s2", 400)

    assert pool.used_bytes == 700
    assert pool.active_slots == 2


def test_budget_exhaustion():
    pool = make_pool(500)

    assert pool.acquire("s1", 400) is not None
    assert pool.acquire("s2", 200) is None

    assert pool.used_bytes == 400
    assert pool.active_slots == 1


def test_release():
    pool = make_pool()

    pool.acquire("s1", 400)
    pool.release("s1")

    assert pool.used_bytes == 0
    assert pool.available_bytes == 1000
    assert pool.active_slots == 0


def test_reuse_after_release():
    pool = make_pool(500)

    assert pool.acquire("s1", 400) is not None
    assert pool.acquire("s2", 200) is None

    pool.release("s1")

    assert pool.acquire("s2", 400) is not None


def test_duplicate_slot():
    pool = make_pool()

    pool.acquire("s1", 300)

    with pytest.raises(ValueError):
        pool.acquire("s1", 300)


def test_release_unknown_slot():
    pool = make_pool()

    with pytest.raises(KeyError):
        pool.release("unknown")


def test_get_slot():
    pool = make_pool()

    pool.acquire("s1", 300)

    slot = pool.get("s1")

    assert slot.slot_id == "s1"
    assert slot.size_bytes == 300
    assert slot.in_use is True


def test_get_unknown_slot():
    pool = make_pool()

    with pytest.raises(KeyError):
        pool.get("unknown")


def test_release_returns_budget():
    pool = make_pool(1000)

    pool.acquire("s1", 300)
    pool.acquire("s2", 400)

    pool.release("s1")

    assert pool.used_bytes == 400
    assert pool.available_bytes == 600
    assert pool.active_slots == 1


def test_three_expert_prefetch_budget():
    # Approximate size of one NF4 Mixtral expert.
    expert_size = 84 * 1024 * 1024

    pool = make_pool(
        300 * 1024 * 1024
    )

    assert pool.acquire("e1", expert_size) is not None
    assert pool.acquire("e2", expert_size) is not None
    assert pool.acquire("e3", expert_size) is not None

    # Fourth expert should exceed the 300 MiB budget.
    assert pool.acquire("e4", expert_size) is None

    assert pool.active_slots == 3