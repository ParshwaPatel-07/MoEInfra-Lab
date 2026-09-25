"""Expert cache manager for MoEInfra.

Manages a two-level (GPU + CPU) cache of quantised Mixtral expert tensors.
Implements LRU, LFU, and a simplified ARC eviction policy across both tiers.

Design choices
--------------
* **Two-dict lookup**: separate ``_gpu_cache`` and ``_cpu_cache`` dicts keyed
  by ``(layer_id, expert_id)`` give O(1) lookup without an ordered structure.
  Ordering is only needed at eviction time, which is rare relative to lookups.
* **LFU tie-breaking via LRU**: when two entries have the same access_count
  the one with the older last_access is evicted — a standard "LFU+LRU" combo.
* **ARC approximation**: full ARC tracks four lists (T1, T2, B1, B2) and
  adapts the split parameter *p* online.  We implement the full ghost-list
  bookkeeping for correct adaptation, with ghost lists capped at max_slots to
  bound memory.
* **promote_to_gpu** is a best-effort fast path: it evicts one GPU entry if
  the GPU tier is full, then moves the tensor with ``tensor.to("cuda:0")``.
  When CUDA is unavailable (unit-test environment) it logs a warning and
  returns ``False`` rather than crashing.
* **Thread-safety**: single-threaded by design (documented); no locks added to
  avoid overhead in the hot inference path.
"""
from __future__ import annotations

import logging
import time
from typing import Optional
from model.expert import QuantizedMixtralExpert

import torch

from cache.types import CacheEntry, CacheStats, EvictionPolicy


# ---------------------------------------------------------------------------
# ARC ghost-list state (only instantiated when policy == ARC)
# ---------------------------------------------------------------------------

class _ArcState:
    """Mutable bookkeeping for the Adaptive Replacement Cache algorithm.

    Tracks the four ARC lists (T1 = recency, T2 = frequency, B1/B2 = ghosts)
    and the adaptation parameter *p* (target size of T1).
    """

    def __init__(self, max_slots: int) -> None:
        self.max_slots: int = max_slots
        self.p: int = 0
        # LRU order: index 0 = least-recently-used, -1 = most-recently-used
        self.t1: list[tuple[int, int]] = []
        self.t2: list[tuple[int, int]] = []
        self.b1: list[tuple[int, int]] = []
        self.b2: list[tuple[int, int]] = []

    def _move_to_mru(self, lst: list, key: tuple[int, int]) -> None:
        try:
            lst.remove(key)
        except ValueError:
            pass
        lst.append(key)

    def on_hit(self, key: tuple[int, int]) -> None:
        """Promote a live hit: T1 → T2, or refresh position in T2."""
        if key in self.t1:
            self.t1.remove(key)
            self.t2.append(key)
        elif key in self.t2:
            self._move_to_mru(self.t2, key)

    def on_miss(self, key: tuple[int, int]) -> None:
        """Record a miss and adapt *p* based on ghost-list membership."""
        if key in self.b1:
            # Ghost hit in B1: T1 was too small, grow its target
            delta = max(1, len(self.b2) // max(1, len(self.b1)))
            self.p = min(self.p + delta, self.max_slots)
            self.b1.remove(key)
        elif key in self.b2:
            # Ghost hit in B2: T2 was too small, shrink T1 target
            delta = max(1, len(self.b1) // max(1, len(self.b2)))
            self.p = max(self.p - delta, 0)
            self.b2.remove(key)
        # New key enters T1 (recency list)
        self.t1.append(key)
        # Bound ghost-list sizes
        while len(self.b1) > self.max_slots:
            self.b1.pop(0)
        while len(self.b2) > self.max_slots:
            self.b2.pop(0)

    def choose_evict(self) -> Optional[tuple[int, int]]:
        """Pick an LRU victim from T1 or T2 per ARC policy.

        Returns:
            ``(layer_id, expert_id)`` key, or ``None`` if both lists are empty.
        """
        t1_len, t2_len = len(self.t1), len(self.t2)
        if t1_len == 0 and t2_len == 0:
            return None
        # Evict from T1 when it exceeds its target p, otherwise from T2
        if t1_len > 0 and (t1_len > self.p or t2_len == 0):
            victim = self.t1.pop(0)
            self.b1.append(victim)
        else:
            victim = self.t2.pop(0)
            self.b2.append(victim)
        return victim

    def remove(self, key: tuple[int, int]) -> None:
        """Remove a key from all live ARC lists."""
        for lst in (self.t1, self.t2):
            try:
                lst.remove(key)
            except ValueError:
                pass


# ---------------------------------------------------------------------------
# CacheManager
# ---------------------------------------------------------------------------

class CacheManager:
    """Two-level LRU/LFU/ARC expert cache.

    Maintains separate slot budgets for GPU (fast) and CPU (slower) memory.
    All public mutating methods must be called from a single thread unless the
    caller provides external synchronisation.

    Attributes:
        gpu_slots: Maximum number of expert tensors that may reside on GPU.
        cpu_slots: Maximum number of expert tensors that may reside on CPU.
        policy: The eviction policy in use.
    """

    def __init__(
        self,
        gpu_slots: int,
        cpu_slots: int,
        policy: EvictionPolicy,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Initialise the cache manager.

        Args:
            gpu_slots: Maximum number of expert tensors to keep on GPU.
            cpu_slots: Maximum number of expert tensors to keep on CPU RAM.
            policy: Eviction strategy to apply when either tier is full.
            logger: Optional pre-configured logger; a default child logger is
                used when ``None``.
        """
        self.gpu_slots: int = gpu_slots
        self.cpu_slots: int = cpu_slots
        self.policy: EvictionPolicy = policy

        # Internal stores keyed by (layer_id, expert_id)
        self._gpu_cache: dict[tuple[int, int], CacheEntry] = {}
        self._cpu_cache: dict[tuple[int, int], CacheEntry] = {}

        self._stats: CacheStats = CacheStats()
        self._logger: logging.Logger = logger or logging.getLogger(__name__)

        # ARC state objects (one per tier; ignored for LRU/LFU)
        self._arc_gpu = _ArcState(gpu_slots)
        self._arc_cpu = _ArcState(cpu_slots)

        self._logger.debug(
            "CacheManager initialised: gpu_slots=%d, cpu_slots=%d, policy=%s",
            gpu_slots,
            cpu_slots,
            policy.value,
        )

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _cache_for(self, device: str) -> dict[tuple[int, int], CacheEntry]:
        """Return the internal dict for the given device tier."""
        return self._gpu_cache if device.startswith("cuda") else self._cpu_cache

    def _slots_for(self, device: str) -> int:
        """Return the slot limit for the given device tier."""
        return self.gpu_slots if device.startswith("cuda") else self.cpu_slots

    def _arc_for(self, device: str) -> _ArcState:
        """Return the ARC state for the given device tier."""
        return self._arc_gpu if device.startswith("cuda") else self._arc_cpu

    def _touch(self, key: tuple[int, int], entry: CacheEntry) -> None:
        """Update access metadata and ARC lists on a cache hit."""
        entry.last_access = time.monotonic()
        entry.access_count += 1
        if self.policy == EvictionPolicy.ARC:
            self._arc_for(entry.device).on_hit(key)

    def _lru_victim(
        self, cache: dict[tuple[int, int], CacheEntry]
    ) -> Optional[tuple[int, int]]:
        """Return the key with the oldest ``last_access``."""
        if not cache:
            return None
        return min(cache, key=lambda k: cache[k].last_access)

    def _lfu_victim(
        self, cache: dict[tuple[int, int], CacheEntry]
    ) -> Optional[tuple[int, int]]:
        """Return the key with the lowest frequency; ties broken by LRU."""
        if not cache:
            return None
        return min(cache, key=lambda k: (cache[k].access_count, cache[k].last_access))

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def get(self, layer_id: int, expert_id: int) -> Optional[QuantizedMixtralExpert]:
        """Look up an expert tensor in the cache.

        Checks the GPU tier first (O(1) dict lookup), then CPU.  Updates
        access metadata on a hit per the active :attr:`policy`.

        Args:
            layer_id: Transformer layer index.
            expert_id: Expert index within the layer.

        Returns:
            The cached :class:`torch.Tensor` on a hit, or ``None`` on a miss.
        """
        key = (layer_id, expert_id)

        # GPU hit (fastest path)
        if key in self._gpu_cache:
            entry = self._gpu_cache[key]
            self._touch(key, entry)
            self._stats.hits += 1
            self._logger.debug(
                "GPU cache HIT layer=%d expert=%d",
                layer_id,
                expert_id,
            )
            return entry.expert

        if key in self._cpu_cache:
            entry = self._cpu_cache[key]
            self._touch(key, entry)
            self._stats.hits += 1
            self._logger.debug(
                "CPU cache HIT layer=%d expert=%d",
                layer_id,
                expert_id,
            )
            return entry.expert

        # Miss
        self._stats.misses += 1
        self._logger.debug("Cache MISS layer=%d expert=%d", layer_id, expert_id)
        if self.policy == EvictionPolicy.ARC:
            # Inform the CPU-tier ARC of the miss so it can adapt p
            self._arc_cpu.on_miss(key)
        return None

    def put(
        self,
        layer_id: int,
        expert_id: int,
        expert: QuantizedMixtralExpert,
        device: str,
    ) -> None:
        """Insert or refresh an expert tensor in the cache.

        If the target tier is full, :meth:`evict` is called first to free a
        slot.  If the key already exists its tensor is refreshed in-place
        without consuming an extra slot.

        Args:
            layer_id: Transformer layer index.
            expert_id: Expert index within the layer.
            tensor: The expert weight tensor to cache.
            device: Target device string (``"cuda:0"`` or ``"cpu"``).
        """
        key = (layer_id, expert_id)
        cache = self._cache_for(device)
        limit = self._slots_for(device)

        if key in cache:
            # Refresh existing entry — no slot consumed
            entry = cache[key]
            entry.expert = expert
            entry.device = device
            self._touch(key, entry)
            self._logger.debug(
                "Cache UPDATE layer=%d expert=%d device=%s", layer_id, expert_id, device
            )
            return

        # Evict to make room if the tier is at capacity
        if len(cache) >= limit:
            evicted = self.evict(device)
            if evicted is None:
                self._logger.warning(
                    "put: eviction returned None despite tier being full "
                    "(layer=%d expert=%d device=%s)",
                    layer_id, expert_id, device,
                )

        now = time.monotonic()
        entry = CacheEntry(
            expert_id=expert_id,
            layer_id=layer_id,
            device=device,
            expert=expert,
            last_access=now,
            access_count=1,
            size_bytes=expert.size_bytes,
        )
        cache[key] = entry

        # Keep slot-used counters in sync
        if device.startswith("cuda"):
            self._stats.gpu_slots_used = len(self._gpu_cache)
        else:
            self._stats.cpu_slots_used = len(self._cpu_cache)

        self._logger.debug(
            "Cache INSERT layer=%d expert=%d device=%s size=%dB",
            layer_id, expert_id, device, entry.size_bytes,
        )

    def evict(self, device: str) -> Optional[CacheEntry]:
        """Select and remove one entry from the specified tier.

        The victim is chosen per :attr:`policy`:

        * ``LRU`` — oldest ``last_access``.
        * ``LFU`` — lowest ``access_count`` (ties broken by ``last_access``).
        * ``ARC`` — from the ARC T1/T2 lists via :class:`_ArcState`;
          falls back to LRU if the ARC-chosen key is no longer live.

        Args:
            device: ``"cuda"`` / ``"cuda:0"`` for GPU tier, ``"cpu"`` for CPU.

        Returns:
            The evicted :class:`CacheEntry`, or ``None`` if the tier is empty.
        """
        cache = self._cache_for(device)
        if not cache:
            return None

        if self.policy == EvictionPolicy.LRU:
            victim_key = self._lru_victim(cache)
        elif self.policy == EvictionPolicy.LFU:
            victim_key = self._lfu_victim(cache)
        else:  # ARC
            arc = self._arc_for(device)
            victim_key = arc.choose_evict()
            # ARC ghost lists may contain keys no longer in the live cache;
            # fall back to LRU if so.
            if victim_key is not None and victim_key not in cache:
                victim_key = self._lru_victim(cache)

        if victim_key is None:
            return None

        evicted = cache.pop(victim_key)

        # GPU cache entries are disposable working copies. The authoritative
        # CPU NF4 backing copy remains in _cpu_cache, so GPU eviction must not
        # call Params4bit.cpu() on the live GPU object.
        if device.startswith("cuda"):
            evicted.device = "cuda"

        if self.policy == EvictionPolicy.ARC:
            self._arc_for(device).remove(victim_key)

        self._stats.evictions += 1
        if device.startswith("cuda"):
            self._stats.gpu_slots_used = len(self._gpu_cache)
        else:
            self._stats.cpu_slots_used = len(self._cpu_cache)

        self._logger.debug(
            "Cache EVICT layer=%d expert=%d from %s (policy=%s)",
            evicted.layer_id, evicted.expert_id, device, self.policy.value,
        )
        return evicted

    def promote_to_gpu(self, layer_id: int, expert_id: int) -> bool:
        """Report that direct promotion is no longer performed by CacheManager.

        GPU residency is now established by TransferScheduler, which stages the
        CPU NF4 backing copy, performs the H2D transfer, reconstructs a separate
        GPU expert, and inserts that copy with :meth:`put`.

        This method is retained temporarily for API compatibility. It does not
        call ``Params4bit.cuda()`` because moving the authoritative CPU expert
        in-place would violate the two-copy residency model.
        """
        if self.is_gpu_resident(layer_id, expert_id):
            return True

        if self.peek_cpu(layer_id, expert_id) is None:
            self._logger.debug(
                "promote_to_gpu: (layer=%d, expert=%d) not in CPU cache",
                layer_id,
                expert_id,
            )
        else:
            self._logger.debug(
                "promote_to_gpu: deferred to TransferScheduler "
                "(layer=%d, expert=%d)",
                layer_id,
                expert_id,
            )

        return False

    def demote_to_cpu(self, layer_id: int, expert_id: int) -> bool:
        """Remove GPU residency while retaining the CPU backing copy.

        The CPU cache is authoritative in the Phase 1 architecture. Therefore
        GPU demotion simply removes the disposable GPU copy; it does not call
        ``Params4bit.cpu()``.
        """
        key = (layer_id, expert_id)

        entry = self._gpu_cache.pop(key, None)
        if entry is None:
            self._logger.debug(
                "demote_to_cpu: (layer=%d, expert=%d) not in GPU cache",
                layer_id,
                expert_id,
            )
            return False

        self._stats.gpu_slots_used = len(self._gpu_cache)

        if self.policy == EvictionPolicy.ARC:
            self._arc_gpu.remove(key)

        # Keep the CPU backing copy untouched. It should already exist under
        # the same key; no Params4bit device movement is performed here.
        self._logger.debug(
            "demote_to_cpu: removed GPU working copy "
            "layer=%d expert=%d; CPU backing retained",
            layer_id,
            expert_id,
        )
        return True

    def get_size_bytes(self, layer_id: int, expert_id: int) -> int:
        """Return the cached expert size in bytes without touching statistics."""
        key = (layer_id, expert_id)

        entry = self._gpu_cache.get(key)
        if entry is None:
            entry = self._cpu_cache.get(key)

        if entry is None:
            raise KeyError(
                f"Expert not present in cache: layer={layer_id}, expert={expert_id}"
            )

        return entry.size_bytes

    # ------------------------------------------------------------------ #
    # Fully-implemented helpers                                            #
    # ------------------------------------------------------------------ #

    def stats(self) -> CacheStats:
        """Return a snapshot of current cache statistics.

        Returns:
            The :class:`CacheStats` instance maintained by this manager.
        """
        return self._stats

    def __repr__(self) -> str:
        return (
            f"CacheManager("
            f"gpu={len(self._gpu_cache)}/{self.gpu_slots}, "
            f"cpu={len(self._cpu_cache)}/{self.cpu_slots}, "
            f"policy={self.policy.value})"
        )

    def is_gpu_resident(self, layer_id: int, expert_id: int) -> bool:
        """Return whether an expert currently has a GPU-resident copy."""
        return (layer_id, expert_id) in self._gpu_cache

    def peek_cpu(
        self,
        layer_id: int,
        expert_id: int,
    ) -> Optional[QuantizedMixtralExpert]:
        """Inspect the CPU backing copy without changing cache statistics."""
        entry = self._cpu_cache.get((layer_id, expert_id))
        return None if entry is None else entry.expert

    def peek_gpu(
        self,
        layer_id: int,
        expert_id: int,
    ) -> Optional[QuantizedMixtralExpert]:
        """Inspect the GPU-resident copy without changing cache statistics."""
        entry = self._gpu_cache.get((layer_id, expert_id))
        return None if entry is None else entry.expert
    