import pytest

from transfer.pinned_memory import PinnedMemoryBudget
from transfer.staging import PinnedMemoryStaging


def make_staging(budget=1000):
    return PinnedMemoryStaging(
        PinnedMemoryBudget(budget)
    )


def test_initial_state():
    staging = make_staging()

    assert staging.used_bytes == 0
    assert staging.available_bytes == 1000
    assert staging.active_reservations == 0


def test_reserve():
    staging = make_staging()

    assert staging.reserve("e1", 300) is True

    assert staging.used_bytes == 300
    assert staging.available_bytes == 700
    assert staging.active_reservations == 1


def test_multiple_reservations():
    staging = make_staging()

    assert staging.reserve("e1", 300)
    assert staging.reserve("e2", 400)

    assert staging.used_bytes == 700
    assert staging.available_bytes == 300
    assert staging.active_reservations == 2


def test_over_budget_reservation():
    staging = make_staging(500)

    assert staging.reserve("e1", 501) is False

    assert staging.used_bytes == 0
    assert staging.active_reservations == 0


def test_failed_reservation_does_not_create_entry():
    staging = make_staging(500)

    assert staging.reserve("e1", 600) is False

    with pytest.raises(KeyError):
        staging.reservation_size("e1")


def test_duplicate_reservation():
    staging = make_staging()

    assert staging.reserve("e1", 300)

    with pytest.raises(ValueError):
        staging.reserve("e1", 200)

    assert staging.used_bytes == 300


def test_release():
    staging = make_staging()

    staging.reserve("e1", 300)
    staging.release("e1")

    assert staging.used_bytes == 0
    assert staging.available_bytes == 1000
    assert staging.active_reservations == 0


def test_release_one_of_multiple():
    staging = make_staging()

    staging.reserve("e1", 300)
    staging.reserve("e2", 400)

    staging.release("e1")

    assert staging.used_bytes == 400
    assert staging.available_bytes == 600
    assert staging.active_reservations == 1


def test_release_unknown_request():
    staging = make_staging()

    with pytest.raises(KeyError):
        staging.release("unknown")


def test_reservation_size():
    staging = make_staging()

    staging.reserve("e1", 350)

    assert staging.reservation_size("e1") == 350


def test_reservation_size_unknown():
    staging = make_staging()

    with pytest.raises(KeyError):
        staging.reservation_size("unknown")


def test_reuse_budget_after_release():
    staging = make_staging(500)

    assert staging.reserve("e1", 400)
    assert not staging.reserve("e2", 200)

    staging.release("e1")

    assert staging.reserve("e2", 400)
    assert staging.used_bytes == 400