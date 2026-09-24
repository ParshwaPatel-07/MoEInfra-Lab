import pytest

from transfer.pinned_memory import PinnedMemoryBudget


def test_initial_state():
    budget = PinnedMemoryBudget(1000)

    assert budget.budget_bytes == 1000
    assert budget.used_bytes == 0
    assert budget.available_bytes == 1000


def test_reserve():
    budget = PinnedMemoryBudget(1000)

    assert budget.reserve(300) is True
    assert budget.used_bytes == 300
    assert budget.available_bytes == 700


def test_reserve_multiple():
    budget = PinnedMemoryBudget(1000)

    assert budget.reserve(300)
    assert budget.reserve(400)

    assert budget.used_bytes == 700
    assert budget.available_bytes == 300


def test_exact_budget_boundary():
    budget = PinnedMemoryBudget(1000)

    assert budget.reserve(1000) is True
    assert budget.used_bytes == 1000
    assert budget.available_bytes == 0


def test_over_budget_reservation():
    budget = PinnedMemoryBudget(1000)

    assert budget.reserve(1001) is False
    assert budget.used_bytes == 0


def test_failed_reservation_does_not_change_state():
    budget = PinnedMemoryBudget(1000)

    budget.reserve(800)

    assert budget.reserve(300) is False
    assert budget.used_bytes == 800
    assert budget.available_bytes == 200


def test_release():
    budget = PinnedMemoryBudget(1000)

    budget.reserve(700)
    budget.release(300)

    assert budget.used_bytes == 400
    assert budget.available_bytes == 600


def test_release_all():
    budget = PinnedMemoryBudget(1000)

    budget.reserve(1000)
    budget.release(1000)

    assert budget.used_bytes == 0
    assert budget.available_bytes == 1000


def test_release_underflow():
    budget = PinnedMemoryBudget(1000)

    with pytest.raises(ValueError):
        budget.release(1)


def test_release_more_than_used():
    budget = PinnedMemoryBudget(1000)

    budget.reserve(300)

    with pytest.raises(ValueError):
        budget.release(301)

    assert budget.used_bytes == 300


def test_negative_budget():
    with pytest.raises(ValueError):
        PinnedMemoryBudget(-1)


@pytest.mark.parametrize("size", [-1, -100])
def test_negative_reserve(size):
    budget = PinnedMemoryBudget(1000)

    with pytest.raises(ValueError):
        budget.reserve(size)


@pytest.mark.parametrize("size", [-1, -100])
def test_negative_can_reserve(size):
    budget = PinnedMemoryBudget(1000)

    with pytest.raises(ValueError):
        budget.can_reserve(size)


@pytest.mark.parametrize("size", [-1, -100])
def test_negative_release(size):
    budget = PinnedMemoryBudget(1000)

    with pytest.raises(ValueError):
        budget.release(size)


def test_zero_reservation():
    budget = PinnedMemoryBudget(1000)

    assert budget.reserve(0) is True
    assert budget.used_bytes == 0


def test_zero_release():
    budget = PinnedMemoryBudget(1000)

    budget.reserve(500)
    budget.release(0)

    assert budget.used_bytes == 500


def test_reuse_after_release():
    budget = PinnedMemoryBudget(1000)

    assert budget.reserve(800)
    assert not budget.reserve(300)

    budget.release(800)

    assert budget.reserve(900)
    assert budget.used_bytes == 900
    assert budget.available_bytes == 100


def test_can_reserve_does_not_modify_state():
    budget = PinnedMemoryBudget(1000)

    budget.reserve(600)

    assert budget.can_reserve(400)
    assert budget.used_bytes == 600

    assert not budget.can_reserve(401)
    assert budget.used_bytes == 600


def test_repr():
    budget = PinnedMemoryBudget(1000)
    budget.reserve(400)

    assert repr(budget) == (
        "PinnedMemoryBudget("
        "budget_bytes=1000, "
        "used_bytes=400, "
        "available_bytes=600)"
    )