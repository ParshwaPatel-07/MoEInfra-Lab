from __future__ import annotations

from dataclasses import dataclass

import torch

from transfer.pinned_memory import PinnedMemoryBudget


@dataclass
class StagingSlot:
    slot_id: int
    size_bytes: int
    tensors: dict[str, dict[str, torch.Tensor]]
    in_use: bool = False
    ready_event: torch.cuda.Event | None = None


class ReusablePinnedStagingPool:
    """
    Bounded reusable pinned-memory slots.

    A slot remains unavailable until the CUDA event associated with
    its previous transfer has completed.
    """

    def __init__(
        self,
        budget: PinnedMemoryBudget,
        slot_size_bytes: int,
    ):
        if slot_size_bytes <= 0:
            raise ValueError("slot_size_bytes must be positive")

        self.budget = budget
        self.slot_size_bytes = slot_size_bytes
        self._slots: dict[int, StagingSlot] = {}

    @property
    def slot_count(self) -> int:
        return len(self._slots)

    @property
    def active_slots(self) -> int:
        return sum(slot.in_use for slot in self._slots.values())

    def _create_slot(self, slot_id: int) -> StagingSlot | None:
        if not self.budget.reserve(self.slot_size_bytes):
            return None

        try:
            slot = StagingSlot(
                slot_id=slot_id,
                size_bytes=self.slot_size_bytes,
                tensors={},
            )

            self._slots[slot_id] = slot
            return slot

        except Exception:
            self.budget.release(self.slot_size_bytes)
            raise

    def acquire(self) -> StagingSlot | None:
        # First try to reuse a completed slot.
        for slot in self._slots.values():

            if slot.in_use:
                continue

            if slot.ready_event is not None:
                if not slot.ready_event.query():
                    continue
                slot.ready_event = None

            slot.in_use = True
            return slot

        # No reusable slot exists.
        slot_id = len(self._slots)

        slot = self._create_slot(slot_id)

        if slot is not None:
            slot.in_use = True

        return slot

    def mark_transfer_complete(
        self,
        slot: StagingSlot,
        stream: torch.cuda.Stream,
    ) -> None:
        if not slot.in_use:
            raise RuntimeError(
                "Cannot mark an inactive slot"
            )

        event = torch.cuda.Event()

        with torch.cuda.stream(stream):
            event.record(stream)

        slot.ready_event = event

    def release(self, slot: StagingSlot) -> None:
        if not slot.in_use:
            raise RuntimeError("Slot is not in use")

        slot.in_use = False

    def destroy(self) -> None:
        self._slots.clear()
        self.budget.used_bytes = 0