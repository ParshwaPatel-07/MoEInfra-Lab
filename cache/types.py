"""Type definitions for the MoEInfra cache subsystem.

Defines the core data structures used by :class:`cache.manager.CacheManager`:
:class:`EvictionPolicy`, :class:`CacheEntry`, and :class:`CacheStats`.
"""
from __future__ import annotations

import dataclasses
import enum
import time
from typing import Optional

from model.expert import QuantizedMixtralExpert
import torch


class EvictionPolicy(enum.Enum):
    """Cache eviction strategy.

    Attributes:
        LRU: Least Recently Used — evict the entry whose ``last_access`` is
            the oldest.
        LFU: Least Frequently Used — evict the entry with the lowest
            ``access_count``.
        ARC: Adaptive Replacement Cache — balances recency and frequency.
    """

    LRU = "lru"
    LFU = "lfu"
    ARC = "arc"


@dataclasses.dataclass
class CacheEntry:
    """A single expert tensor slot in either GPU or CPU memory.

    Attributes:
        expert_id: Index of the MoE expert (0-based, within a layer).
        layer_id: Index of the transformer layer this expert belongs to.
        device: Device string, e.g. ``"cuda:0"`` or ``"cpu"``.
        expert: The expert weight tensor, or ``None`` if not yet loaded.
        last_access: UNIX timestamp of the most recent cache hit.
        access_count: Total number of times this entry has been accessed.
        size_bytes: Memory footprint of the tensor in bytes.
    """

    expert_id: int
    layer_id: int
    device: str
    expert: Optional[QuantizedMixtralExpert]
    last_access: float
    access_count: int
    size_bytes: int

    @property
    def is_on_gpu(self) -> bool:
        """Return ``True`` if this entry resides in GPU memory.

        Returns:
            ``True`` when :attr:`device` starts with ``"cuda"``.
        """
        return self.device.startswith("cuda")

    @property
    def age(self) -> float:
        """Return the number of seconds since this entry was last accessed.

        Returns:
            Elapsed seconds as a float.
        """
        return time.monotonic() - self.last_access


@dataclasses.dataclass
class CacheStats:
    """Aggregated statistics for a :class:`cache.manager.CacheManager`.

    Attributes:
        hits: Total number of successful cache lookups.
        misses: Total number of cache misses.
        evictions: Total number of entries evicted from any device.
        gpu_slots_used: Current number of occupied GPU slots.
        cpu_slots_used: Current number of occupied CPU slots.
    """

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    gpu_slots_used: int = 0
    cpu_slots_used: int = 0

    @property
    def hit_rate(self) -> float:
        """Return the cache hit rate as a value in ``[0.0, 1.0]``.

        Returns:
            ``hits / (hits + misses)``, or ``0.0`` if no requests have been
            made yet.
        """
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
