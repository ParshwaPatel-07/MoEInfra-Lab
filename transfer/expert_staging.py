from __future__ import annotations

from dataclasses import dataclass

import torch

from transfer.pinned_memory import PinnedMemoryBudget


@dataclass
class PinnedExpertStaging:
    """Pinned CPU copy of a complete NF4 expert."""

    expert_id: str
    tensors: dict[str, dict[str, torch.Tensor]]
    size_bytes: int


class PinnedExpertStagingPool:
    """
    Bounded pool of temporary pinned CPU copies of NF4 experts.

    The original CPU expert remains pageable. This pool owns only
    temporary pinned copies used for asynchronous H2D transfers.
    """

    def __init__(self, budget: PinnedMemoryBudget):
        self.budget = budget
        self._active: dict[str, PinnedExpertStaging] = {}

    @property
    def used_bytes(self) -> int:
        return self.budget.used_bytes

    @property
    def available_bytes(self) -> int:
        return self.budget.available_bytes

    @property
    def active_count(self) -> int:
        return len(self._active)

    @staticmethod
    def _tensor_bytes(tensor: torch.Tensor | None) -> int:
        if tensor is None:
            return 0

        return tensor.numel() * tensor.element_size()

    def _expert_size_bytes(self, expert) -> int:
        total = 0

        for name in ("w1", "w2", "w3"):
            param = getattr(expert, name).weight
            qs = param.quant_state

            total += self._tensor_bytes(param.data)
            total += self._tensor_bytes(qs.absmax)
            total += self._tensor_bytes(qs.code)
            total += self._tensor_bytes(qs.offset)

            if qs.state2 is not None:
                total += self._tensor_bytes(qs.state2.absmax)
                total += self._tensor_bytes(qs.state2.code)
                total += self._tensor_bytes(qs.state2.offset)

        return total

    def stage(self, expert_id: str, expert) -> PinnedExpertStaging | None:
        if expert_id in self._active:
            raise ValueError(
                f"Expert {expert_id!r} is already staged"
            )

        size_bytes = self._expert_size_bytes(expert)

        if not self.budget.reserve(size_bytes):
            return None

        try:
            tensors = {}

            for name in ("w1", "w2", "w3"):
                param = getattr(expert, name).weight
                qs = param.quant_state

                state = {
                    "weight": param.data.pin_memory(),
                    "absmax": qs.absmax.pin_memory(),
                    "code": qs.code.pin_memory(),
                    "offset": (
                        None
                        if qs.offset is None
                        else qs.offset.pin_memory()
                    ),
                }

                if qs.state2 is not None:
                    state["state2_absmax"] = (
                        qs.state2.absmax.pin_memory()
                    )
                    state["state2_code"] = (
                        qs.state2.code.pin_memory()
                    )
                    state["state2_offset"] = (
                        None
                        if qs.state2.offset is None
                        else qs.state2.offset.pin_memory()
                    )

                tensors[name] = state

            staging = PinnedExpertStaging(
                expert_id=expert_id,
                tensors=tensors,
                size_bytes=size_bytes,
            )

            self._active[expert_id] = staging

            return staging

        except Exception:
            # Pinning failed after budget reservation.
            self.budget.release(size_bytes)
            raise

    def release(self, expert_id: str) -> None:
        staging = self._active.pop(expert_id, None)

        if staging is None:
            raise KeyError(
                f"Expert {expert_id!r} is not staged"
            )

        self.budget.release(staging.size_bytes)

        # Drop references to the pinned tensors.
        staging.tensors.clear()

    def get(self, expert_id: str) -> PinnedExpertStaging:
        try:
            return self._active[expert_id]
        except KeyError:
            raise KeyError(
                f"Expert {expert_id!r} is not staged"
            ) from None