"""Type definitions for the MoEInfra transfer subsystem.

Transfer requests identify an expert by layer/expert ID. The scheduler does not
own or directly manipulate expert objects; CacheManager owns expert residency.
"""

from __future__ import annotations

import dataclasses
import enum


class TransferDirection(enum.Enum):
    """Direction of an expert residency transition."""

    CPU_TO_GPU = "cpu_to_gpu"
    GPU_TO_CPU = "gpu_to_cpu"


class TransferPriority(enum.Enum):
    """Scheduling priority for transfer requests."""

    HIGH = 0
    NORMAL = 1
    LOW = 2


@dataclasses.dataclass(frozen=True)
class TransferRequest:
    """A request to move one expert between cache tiers.

    The request identifies the expert by (layer_id, expert_id). The expert
    object itself is intentionally not stored here; CacheManager owns it.
    """

    request_id: str
    layer_id: int
    expert_id: int
    direction: TransferDirection
    priority: TransferPriority
    issued_at: float


@dataclasses.dataclass(frozen=True)
class TransferResult:
    """Result of executing one transfer request."""

    request_id: str
    success: bool
    elapsed_ms: float
    bytes_transferred: int
    error: str | None = None

class TransferStatus(enum.Enum):
    IN_FLIGHT = "in_flight"
    READY = "ready"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


@dataclasses.dataclass
class TransferHandle:
    request_id: str
    layer_id: int
    expert_id: int
    slot: object
    event: object
    status: TransferStatus = TransferStatus.IN_FLIGHT
    gpu_state: object = None
    cpu_expert: object = None
