from __future__ import annotations

from dataclasses import dataclass

from transfer.pinned_memory import PinnedMemoryBudget


@dataclass
class StagingSlot:
    slot_id: str
    size_bytes: int
    in_use: bool = False


class PinnedStagingPool:
    """Bounded ownership of pinned-memory staging capacity."""

    def __init__(self, budget: PinnedMemoryBudget):
        self.budget = budget
        self._slots: dict[str, StagingSlot] = {}

    @property
    def used_bytes(self) -> int:
        return self.budget.used_bytes

    @property
    def available_bytes(self) -> int:
        return self.budget.available_bytes

    @property
    def active_slots(self) -> int:
        return sum(
            slot.in_use
            for slot in self._slots.values()
        )

    def acquire(
        self,
        slot_id: str,
        size_bytes: int,
    ) -> StagingSlot | None:

        if slot_id in self._slots:
            raise ValueError(
                f"Slot {slot_id!r} already exists"
            )

        if not self.budget.reserve(size_bytes):
            return None

        slot = StagingSlot(
            slot_id=slot_id,
            size_bytes=size_bytes,
            in_use=True,
        )

        self._slots[slot_id] = slot

        return slot

    def release(self, slot_id: str) -> None:

        if slot_id not in self._slots:
            raise KeyError(
                f"Unknown staging slot {slot_id!r}"
            )

        slot = self._slots.pop(slot_id)

        if not slot.in_use:
            raise RuntimeError(
                f"Slot {slot_id!r} is already released"
            )

        self.budget.release(slot.size_bytes)

    def get(self, slot_id: str) -> StagingSlot:
        if slot_id not in self._slots:
            raise KeyError(
                f"Unknown staging slot {slot_id!r}"
            )

        return self._slots[slot_id]