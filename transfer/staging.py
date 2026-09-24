from __future__ import annotations

from transfer.pinned_memory import PinnedMemoryBudget


class PinnedMemoryStaging:
    """CPU-only bookkeeping for pinned transfer staging."""

    def __init__(self, budget: PinnedMemoryBudget):
        self.budget = budget
        self._reservations: dict[str, int] = {}

    @property
    def used_bytes(self) -> int:
        return self.budget.used_bytes

    @property
    def available_bytes(self) -> int:
        return self.budget.available_bytes

    def reserve(self, request_id: str, size_bytes: int) -> bool:
        if request_id in self._reservations:
            raise ValueError(
                f"Request {request_id!r} already has a reservation"
            )

        if not self.budget.reserve(size_bytes):
            return False

        self._reservations[request_id] = size_bytes
        return True

    def release(self, request_id: str) -> None:
        if request_id not in self._reservations:
            raise KeyError(
                f"No reservation found for {request_id!r}"
            )

        size_bytes = self._reservations.pop(request_id)
        self.budget.release(size_bytes)

    def reservation_size(self, request_id: str) -> int:
        if request_id not in self._reservations:
            raise KeyError(
                f"No reservation found for {request_id!r}"
            )

        return self._reservations[request_id]

    @property
    def active_reservations(self) -> int:
        return len(self._reservations)