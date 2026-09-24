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
    expert_key: tuple[int, int] | None = None

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

    def _expert_staging_size(expert) -> int:
        total = 0

        for name in ("w1", "w2", "w3"):
            param = getattr(expert, name).weight
            qs = param.quant_state

            total += param.numel() * param.element_size()
            total += qs.absmax.numel() * qs.absmax.element_size()
            total += qs.code.numel() * qs.code.element_size()

            if qs.offset is not None:
                total += qs.offset.numel() * qs.offset.element_size()

            if qs.state2 is not None:
                total += qs.state2.absmax.numel() * qs.state2.absmax.element_size()
                total += qs.state2.code.numel() * qs.state2.code.element_size()

                if qs.state2.offset is not None:
                    total += (
                        qs.state2.offset.numel()
                        * qs.state2.offset.element_size()
                    )

        return total

    def _stage_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        pinned = torch.empty_like(tensor, pin_memory=True)
        pinned.copy_(tensor)
        return pinned

    def stage_expert(
        self,
        slot: StagingSlot,
        expert,
        layer_id: int,
        expert_id: int,
    ) -> None:
        if not slot.in_use:
            raise RuntimeError("Slot must be acquired before staging")

        old_tensors = slot.tensors

        staged = {}

        for name in ("w1", "w2", "w3"):
            param = getattr(expert, name).weight
            qs = param.quant_state

            old_state = old_tensors.get(name, {})

            state = {
                "weight": self._copy_into_pinned(
                    param.data,
                    old_state.get("weight"),
                ),
                "absmax": self._copy_into_pinned(
                    qs.absmax,
                    old_state.get("absmax"),
                ),
                "code": self._copy_into_pinned(
                    qs.code,
                    old_state.get("code"),
                ),
                "offset": self._copy_into_pinned(
                    qs.offset,
                    old_state.get("offset"),
                ),
            }

            if qs.state2 is not None:
                state["state2_absmax"] = self._copy_into_pinned(
                    qs.state2.absmax,
                    old_state.get("state2_absmax"),
                )
                state["state2_code"] = self._copy_into_pinned(
                    qs.state2.code,
                    old_state.get("state2_code"),
                )
                state["state2_offset"] = self._copy_into_pinned(
                    qs.state2.offset,
                    old_state.get("state2_offset"),
                )

            staged[name] = state

        slot.tensors = staged
        slot.expert_key = (layer_id, expert_id)

    def _copy_into_pinned(self, source, existing=None):
        if source is None:
            return None

        if existing is None:
            existing = torch.empty_like(source, pin_memory=True)

        if existing.shape != source.shape or existing.dtype != source.dtype:
            raise ValueError("Existing staging buffer does not match source tensor")

        existing.copy_(source)
        return existing