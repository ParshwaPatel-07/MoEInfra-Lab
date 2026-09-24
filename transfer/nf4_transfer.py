from __future__ import annotations

import torch

from transfer.reusable_staging import StagingSlot


def transfer_staged_expert_to_gpu(
    slot: StagingSlot,
    stream: torch.cuda.Stream,
) -> dict[str, dict[str, torch.Tensor | None]]:
    """
    Asynchronously copy a staged NF4 expert from pinned CPU memory to GPU.

    The caller is responsible for:
    - acquiring the staging slot
    - staging the expert into the slot
    - synchronizing/recording completion through the staging pool
    """
    if not slot.in_use:
        raise RuntimeError("Slot must be acquired before transfer")

    gpu_state: dict[str, dict[str, torch.Tensor | None]] = {}

    with torch.cuda.stream(stream):
        for name in ("w1", "w2", "w3"):
            state = slot.tensors[name]

            gpu_state[name] = {}

            for key, tensor in state.items():
                if tensor is None:
                    gpu_state[name][key] = None
                else:
                    gpu_state[name][key] = tensor.to(
                        device="cuda",
                        non_blocking=True,
                    )

    return gpu_state