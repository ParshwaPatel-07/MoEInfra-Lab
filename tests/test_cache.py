"""Tests for the cache subsystem.

Covers :class:`~cache.manager.CacheManager`,
:class:`~cache.types.CacheEntry`, and :class:`~cache.types.CacheStats`.

All tests that exercise CPU-only paths (put, get, evict, LRU/LFU ordering,
slot counts, statistics) run without CUDA. Tests that require an actual
GPU move (promote_to_gpu) are guarded by
``pytest.mark.skipif(not torch.cuda.is_available(), ...)``.

A lightweight :class:`FakeExpert` stand-in is used throughout so that
``bitsandbytes`` NF4 quantisation is not triggered in CI.  ``FakeExpert``
satisfies every interface that :class:`~cache.manager.CacheManager` calls:
- ``.size_bytes`` property
- ``.cpu()`` — returns ``self`` (no-op on CPU)
- ``.cuda()`` — raises ``RuntimeError`` when CUDA is absent
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest
import torch

from cache.manager import CacheManager
from cache.types import CacheEntry, CacheStats, EvictionPolicy

CUDA_AVAILABLE = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(
    not CUDA_AVAILABLE,
    reason="CUDA device required for GPU-transfer tests",
)


# ── Lightweight fake expert ───────────────────────────────────────────────

class FakeExpert:
    """Drop-in replacement for QuantizedMixtralExpert in unit tests.

    Satisfies the interface that CacheManager inspects:
    - ``size_bytes``  — configurable at construction time.
    - ``cpu()``       — returns self (already on CPU).
    - ``cuda()``      — moves to cuda:0 when CUDA is available; raises otherwise.
    - ``device``      — tracks current device as a string.
    """

    def __init__(self, size_bytes: int = 1024, device: str = "cpu") -> None:
        self._size_bytes = size_bytes
        self._device = device

    @property
    def size_bytes(self) -> int:
        return self._size_bytes

    @property
    def device(self) -> str:
        return self._device

    def cpu(self) -> "FakeExpert":
        self._device = "cpu"
        return self

    def cuda(self, device: int = 0) -> "FakeExpert":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        self._device = f"cuda:{device}"
        return self

    def __repr__(self) -> str:
        return f"FakeExpert(size={self._size_bytes}, device={self._device!r})"


def make_expert(size_bytes: int = 1024) -> FakeExpert:
    """Return a new FakeExpert on CPU."""
    return FakeExpert(size_bytes=size_bytes)


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def cm_lru() -> CacheManager:
    """CacheManager with LRU policy (gpu_slots=2, cpu_slots=4)."""
    return CacheManager(gpu_slots=2, cpu_slots=4, policy=EvictionPolicy.LRU)


@pytest.fixture
def cm_lfu() -> CacheManager:
    """CacheManager with LFU policy (gpu_slots=2, cpu_slots=4)."""
    return CacheManager(gpu_slots=2, cpu_slots=4, policy=EvictionPolicy.LFU)


# ── CacheManager initialisation ───────────────────────────────────────────

class TestCacheManagerInit:
    """Verify slot counts, policy, __repr__, and zero-valued stats on init."""

    def test_slot_counts_stored(self, cache_manager: CacheManager) -> None:
        """gpu_slots and cpu_slots must match constructor arguments."""
        assert cache_manager.gpu_slots == 4
        assert cache_manager.cpu_slots == 8

    def test_policy_stored(self, cache_manager: CacheManager) -> None:
        """Policy must be stored correctly."""
        assert cache_manager.policy == EvictionPolicy.LRU

    def test_repr_contains_slot_info(self, cache_manager: CacheManager) -> None:
        """__repr__ must mention 'CacheManager', 'gpu=', and 'cpu='."""
        r = repr(cache_manager)
        assert "CacheManager" in r
        assert "gpu=" in r
        assert "cpu=" in r

    def test_stats_initially_zero(self, cache_manager: CacheManager) -> None:
        """All stats counters must be zero after construction."""
        s = cache_manager.stats()
        assert s.hits == 0
        assert s.misses == 0
        assert s.evictions == 0
        assert s.gpu_slots_used == 0
        assert s.cpu_slots_used == 0
        assert s.hit_rate == 0.0


# ── CacheEntry interface ──────────────────────────────────────────────────

class TestCacheEntry:
    """CacheEntry now stores a QuantizedMixtralExpert (or FakeExpert stand-in)."""

    def test_expert_field_stored(self) -> None:
        """CacheEntry.expert holds the provided expert object."""
        expert = make_expert()
        entry = CacheEntry(
            expert_id=3,
            layer_id=7,
            device="cpu",
            expert=expert,
            last_access=time.monotonic(),
            access_count=0,
            size_bytes=expert.size_bytes,
        )
        assert entry.expert is expert

    def test_is_on_gpu_false_for_cpu(self) -> None:
        """is_on_gpu must be False when device == 'cpu'."""
        entry = CacheEntry(
            expert_id=0, layer_id=0, device="cpu",
            expert=None, last_access=time.monotonic(),
            access_count=0, size_bytes=0,
        )
        assert entry.is_on_gpu is False

    def test_is_on_gpu_true_for_cuda(self) -> None:
        """is_on_gpu must be True when device starts with 'cuda'."""
        entry = CacheEntry(
            expert_id=0, layer_id=0, device="cuda:0",
            expert=None, last_access=time.monotonic(),
            access_count=0, size_bytes=0,
        )
        assert entry.is_on_gpu is True

    def test_age_grows_over_time(self) -> None:
        """age should be >= 0.05 s for an entry last accessed 0.05 s ago."""
        entry = CacheEntry(
            expert_id=0, layer_id=0, device="cpu",
            expert=None,
            last_access=time.monotonic() - 0.05,
            access_count=0,
            size_bytes=0,
        )
        assert entry.age >= 0.05


# ── CacheStats interface ──────────────────────────────────────────────────

class TestCacheStats:
    """CacheStats dataclass correctness."""

    def test_hit_rate_zero_when_no_requests(self) -> None:
        """hit_rate must be 0.0 with no hits or misses."""
        assert CacheStats().hit_rate == 0.0

    def test_hit_rate_calculation(self) -> None:
        """hit_rate == hits / (hits + misses)."""
        stats = CacheStats(hits=3, misses=1)
        assert stats.hit_rate == pytest.approx(0.75)

    def test_hit_rate_all_hits(self) -> None:
        """hit_rate must be 1.0 when there are no misses."""
        stats = CacheStats(hits=5, misses=0)
        assert stats.hit_rate == pytest.approx(1.0)


# ── put() + get() round-trip ──────────────────────────────────────────────

class TestPutAndGet:
    """Verify put/get contract, slot accounting, and hit/miss statistics."""

    def test_put_then_get_returns_expert(self, cm_lru: CacheManager) -> None:
        """get() must return the same expert object that was put()."""
        expert = make_expert()
        cm_lru.put(0, 0, expert, "cpu")
        result = cm_lru.get(0, 0)
        assert result is expert

    def test_get_miss_returns_none(self, cm_lru: CacheManager) -> None:
        """get() on an absent key must return None."""
        assert cm_lru.get(99, 99) is None

    def test_get_miss_increments_misses(self, cm_lru: CacheManager) -> None:
        """A cache miss must increment CacheStats.misses."""
        cm_lru.get(1, 1)
        assert cm_lru.stats().misses == 1

    def test_get_hit_increments_hits(self, cm_lru: CacheManager) -> None:
        """A cache hit must increment CacheStats.hits."""
        cm_lru.put(0, 0, make_expert(), "cpu")
        cm_lru.get(0, 0)
        assert cm_lru.stats().hits == 1

    def test_cpu_slot_count_increments_on_put(self, cm_lru: CacheManager) -> None:
        """cpu_slots_used must increase by 1 for each new CPU-tier insertion."""
        assert cm_lru.stats().cpu_slots_used == 0
        cm_lru.put(0, 0, make_expert(), "cpu")
        assert cm_lru.stats().cpu_slots_used == 1
        cm_lru.put(0, 1, make_expert(), "cpu")
        assert cm_lru.stats().cpu_slots_used == 2

    def test_put_refresh_does_not_add_slot(self, cm_lru: CacheManager) -> None:
        """Putting a key that already exists must not consume an extra slot."""
        cm_lru.put(0, 0, make_expert(), "cpu")
        cm_lru.put(0, 0, make_expert(), "cpu")  # refresh
        assert cm_lru.stats().cpu_slots_used == 1

    def test_put_refresh_updates_expert(self, cm_lru: CacheManager) -> None:
        """Putting the same key twice must replace the stored expert."""
        first = make_expert(size_bytes=100)
        second = make_expert(size_bytes=200)
        cm_lru.put(0, 0, first, "cpu")
        cm_lru.put(0, 0, second, "cpu")
        assert cm_lru.get(0, 0) is second

    def test_gpu_slot_count_requires_cuda(self, cm_lru: CacheManager) -> None:
        """gpu_slots_used increments only after a successful GPU insertion."""
        # We cannot call cm_lru.put(..., "cuda:0") without CUDA; this just
        # verifies the initial state is 0.
        assert cm_lru.stats().gpu_slots_used == 0

    def test_gpu_cache_hit_preferred_over_cpu(self, cm_lru: CacheManager) -> None:
        """When the same key exists in both tiers, GPU entry should win.

        We fake a GPU entry by directly inserting into _gpu_cache to avoid
        needing CUDA for this logic test.
        """
        cpu_expert = make_expert(size_bytes=100)
        gpu_expert = FakeExpert(size_bytes=200, device="cuda:0")

        cm_lru.put(0, 0, cpu_expert, "cpu")
        # Inject a fake GPU-tier entry directly
        from cache.types import CacheEntry as CE
        cm_lru._gpu_cache[(0, 0)] = CE(
            expert_id=0, layer_id=0, device="cuda:0",
            expert=gpu_expert,
            last_access=time.monotonic(),
            access_count=1,
            size_bytes=gpu_expert.size_bytes,
        )
        cm_lru._stats.gpu_slots_used = 1

        result = cm_lru.get(0, 0)
        assert result is gpu_expert


# ── evict() ───────────────────────────────────────────────────────────────

class TestEvict:
    """Verify evict() removes exactly one entry and updates counters."""

    def test_evict_empty_returns_none(self, cm_lru: CacheManager) -> None:
        """evict() on an empty tier must return None."""
        assert cm_lru.evict("cpu") is None

    def test_evict_removes_one_entry(self, cm_lru: CacheManager) -> None:
        """evict() must pop one entry and return it."""
        cm_lru.put(0, 0, make_expert(), "cpu")
        evicted = cm_lru.evict("cpu")
        assert evicted is not None
        assert cm_lru.stats().cpu_slots_used == 0

    def test_evict_increments_eviction_counter(self, cm_lru: CacheManager) -> None:
        """Each evict() call must increment CacheStats.evictions by 1."""
        cm_lru.put(0, 0, make_expert(), "cpu")
        cm_lru.evict("cpu")
        assert cm_lru.stats().evictions == 1

    def test_evict_sets_expert_device_to_cpu(self, cm_lru: CacheManager) -> None:
        """CacheManager.evict('cpu') should return the entry without a device move."""
        expert = make_expert()
        cm_lru.put(0, 0, expert, "cpu")
        evicted = cm_lru.evict("cpu")
        # evict from CPU tier: device should remain cpu
        assert evicted.device == "cpu"


# ── LRU eviction ordering ─────────────────────────────────────────────────

class TestLRUEviction:
    """Verify that LRU eviction policy selects the least-recently-used entry."""

    def test_lru_evicts_oldest_access(self) -> None:
        """LRU must remove the entry with the smallest last_access timestamp."""
        cm = CacheManager(gpu_slots=2, cpu_slots=3, policy=EvictionPolicy.LRU)
        # Insert three entries; sleep between inserts so access times differ.
        cm.put(0, 0, make_expert(), "cpu")
        time.sleep(0.01)
        cm.put(0, 1, make_expert(), "cpu")
        time.sleep(0.01)
        cm.put(0, 2, make_expert(), "cpu")

        # Touch (0,1) to make it the most-recently used
        cm.get(0, 1)

        evicted = cm.evict("cpu")
        # (0,0) was the oldest untouched entry
        assert evicted.expert_id == 0

    def test_full_cache_auto_evicts_lru(self) -> None:
        """put() into a full CPU tier must auto-evict the LRU entry."""
        cm = CacheManager(gpu_slots=2, cpu_slots=2, policy=EvictionPolicy.LRU)
        cm.put(0, 0, make_expert(), "cpu")
        time.sleep(0.01)
        cm.put(0, 1, make_expert(), "cpu")

        # Touch (0,1) so (0,0) is the LRU victim
        cm.get(0, 1)

        # Insert a new entry — must evict (0,0)
        cm.put(0, 2, make_expert(), "cpu")

        assert cm.stats().evictions == 1
        # (0,0) should have been evicted; (0,1) and (0,2) remain
        assert cm.get(0, 0) is None
        assert cm.get(0, 1) is not None
        assert cm.get(0, 2) is not None

    def test_cpu_slots_remain_at_limit_after_full_put(self) -> None:
        """After auto-eviction on a full tier, slot count stays at capacity."""
        cm = CacheManager(gpu_slots=2, cpu_slots=3, policy=EvictionPolicy.LRU)
        for i in range(3):
            cm.put(0, i, make_expert(), "cpu")
        assert cm.stats().cpu_slots_used == 3

        cm.put(0, 99, make_expert(), "cpu")  # triggers eviction
        assert cm.stats().evictions == 1
        assert cm.stats().cpu_slots_used == 3  # still at capacity


# ── LFU eviction ordering ─────────────────────────────────────────────────

class TestLFUEviction:
    """Verify that LFU eviction policy selects the least-frequently-used entry."""

    def test_lfu_evicts_lowest_access_count(self) -> None:
        """LFU must remove the entry with the fewest accesses."""
        cm = CacheManager(gpu_slots=2, cpu_slots=3, policy=EvictionPolicy.LFU)
        cm.put(0, 0, make_expert(), "cpu")
        cm.put(0, 1, make_expert(), "cpu")
        cm.put(0, 2, make_expert(), "cpu")

        # Access (0,1) and (0,2) multiple times; (0,0) stays at access_count=1
        for _ in range(3):
            cm.get(0, 1)
            cm.get(0, 2)

        evicted = cm.evict("cpu")
        assert evicted.expert_id == 0

    def test_lfu_tiebreaks_by_lru(self) -> None:
        """When access counts are tied LFU falls back to evicting the LRU entry."""
        cm = CacheManager(gpu_slots=2, cpu_slots=3, policy=EvictionPolicy.LFU)
        # Insert two entries with identical access counts (both have 1 from put)
        cm.put(0, 0, make_expert(), "cpu")
        time.sleep(0.01)
        cm.put(0, 1, make_expert(), "cpu")

        # Both have access_count == 1; (0,0) has the older last_access → LRU victim
        evicted = cm.evict("cpu")
        assert evicted.expert_id == 0


# ── cache-full behaviour ──────────────────────────────────────────────────

class TestCacheFull:
    """Verify auto-eviction triggers exactly once per put when at capacity."""

    def test_no_eviction_below_capacity(self, cm_lru: CacheManager) -> None:
        """Inserting up to cpu_slots must not trigger any eviction."""
        for i in range(cm_lru.cpu_slots):
            cm_lru.put(0, i, make_expert(), "cpu")
        assert cm_lru.stats().evictions == 0

    def test_eviction_exactly_at_capacity(self, cm_lru: CacheManager) -> None:
        """Inserting one more than cpu_slots must trigger exactly 1 eviction."""
        for i in range(cm_lru.cpu_slots):
            cm_lru.put(0, i, make_expert(), "cpu")
        cm_lru.put(0, 99, make_expert(), "cpu")
        assert cm_lru.stats().evictions == 1
        assert cm_lru.stats().cpu_slots_used == cm_lru.cpu_slots

    def test_multiple_overflow_inserts(self, cm_lru: CacheManager) -> None:
        """Each insert beyond capacity must trigger exactly one eviction."""
        for i in range(cm_lru.cpu_slots):
            cm_lru.put(0, i, make_expert(), "cpu")
        extra = 3
        for i in range(extra):
            cm_lru.put(1, i, make_expert(), "cpu")
        assert cm_lru.stats().evictions == extra


# ── get_size_bytes() ──────────────────────────────────────────────────────

class TestGetSizeBytes:
    """Verify get_size_bytes() returns the correct size without touching stats."""

    def test_returns_correct_size_for_cpu_entry(self, cm_lru: CacheManager) -> None:
        """get_size_bytes() must return the expert's size_bytes for a CPU entry."""
        expert = make_expert(size_bytes=4096)
        cm_lru.put(0, 0, expert, "cpu")
        hits_before = cm_lru.stats().hits
        assert cm_lru.get_size_bytes(0, 0) == 4096
        # get_size_bytes must NOT increment the hit counter
        assert cm_lru.stats().hits == hits_before

    def test_raises_for_absent_expert(self, cm_lru: CacheManager) -> None:
        """get_size_bytes() must raise KeyError when the expert is not cached."""
        with pytest.raises(KeyError, match="Expert not present"):
            cm_lru.get_size_bytes(99, 99)

    def test_returns_correct_size_for_gpu_injected_entry(
        self, cm_lru: CacheManager
    ) -> None:
        """get_size_bytes() must also work for entries in _gpu_cache."""
        gpu_expert = FakeExpert(size_bytes=8192, device="cuda:0")
        from cache.types import CacheEntry as CE
        cm_lru._gpu_cache[(5, 3)] = CE(
            expert_id=3, layer_id=5, device="cuda:0",
            expert=gpu_expert,
            last_access=time.monotonic(),
            access_count=1,
            size_bytes=gpu_expert.size_bytes,
        )
        cm_lru._stats.gpu_slots_used = 1
        assert cm_lru.get_size_bytes(5, 3) == 8192


# ── promote_to_gpu() ──────────────────────────────────────────────────────

class TestPromoteToGpu:
    """Tests for CacheManager.promote_to_gpu().

    Full GPU moves only run when CUDA is available.
    CPU-only behaviour (key not in CPU cache) is always testable.
    """

    def test_returns_false_when_not_in_cpu_cache(
        self, cm_lru: CacheManager
    ) -> None:
        """promote_to_gpu() must return False when key is absent from CPU tier."""
        assert cm_lru.promote_to_gpu(0, 99) is False

    def test_returns_false_when_cuda_unavailable(
        self, cm_lru: CacheManager
    ) -> None:
        """promote_to_gpu() must return False (not raise) when CUDA is absent."""
        cm_lru.put(0, 0, make_expert(), "cpu")
        if not CUDA_AVAILABLE:
            assert cm_lru.promote_to_gpu(0, 0) is False

    @requires_cuda
    def test_promotes_entry_to_gpu_tier(self, cm_lru: CacheManager) -> None:
        """promote_to_gpu() must move the entry into _gpu_cache."""
        from model.expert import QuantizedMixtralExpert

        # Build a minimal real QuantizedMixtralExpert (small dims) for GPU move
        w1 = torch.randn(16, 8, dtype=torch.bfloat16)
        w2 = torch.randn(8, 16, dtype=torch.bfloat16)
        w3 = torch.randn(16, 8, dtype=torch.bfloat16)
        real_expert = QuantizedMixtralExpert(w1, w2, w3)

        cm_lru.put(0, 0, real_expert, "cpu")
        assert (0, 0) in cm_lru._cpu_cache

        result = cm_lru.promote_to_gpu(0, 0)
        assert result is True
        assert (0, 0) in cm_lru._gpu_cache
        assert (0, 0) not in cm_lru._cpu_cache
        assert cm_lru.stats().gpu_slots_used == 1
        assert cm_lru.stats().cpu_slots_used == 0

    @requires_cuda
    def test_promote_evicts_gpu_entry_when_full(self) -> None:
        """promote_to_gpu() must evict an existing GPU entry when gpu tier is full."""
        from model.expert import QuantizedMixtralExpert

        cm = CacheManager(gpu_slots=1, cpu_slots=4, policy=EvictionPolicy.LRU)

        def _make_real():
            w1 = torch.randn(16, 8, dtype=torch.bfloat16)
            w2 = torch.randn(8, 16, dtype=torch.bfloat16)
            w3 = torch.randn(16, 8, dtype=torch.bfloat16)
            return QuantizedMixtralExpert(w1, w2, w3)

        # Fill the one GPU slot
        real0 = _make_real()
        cm.put(0, 0, real0, "cpu")
        assert cm.promote_to_gpu(0, 0) is True  # now gpu=1/1

        # Promote a second expert — must evict (0,0) from GPU first
        real1 = _make_real()
        cm.put(0, 1, real1, "cpu")
        result = cm.promote_to_gpu(0, 1)
        assert result is True
        assert cm.stats().gpu_slots_used == 1
        assert (0, 1) in cm._gpu_cache


# ── demote_to_cpu() ───────────────────────────────────────────────────────

class TestDemoteToCpu:
    """Tests for CacheManager.demote_to_cpu().

    Expert.cpu() is a no-op on CPU, so we can test device tracking with
    FakeExpert injected directly into _gpu_cache.
    """

    def _inject_gpu_entry(
        self, cm: CacheManager, layer_id: int, expert_id: int, size_bytes: int = 1024
    ) -> FakeExpert:
        """Directly insert a FakeExpert into the GPU-tier dict."""
        expert = FakeExpert(size_bytes=size_bytes, device="cuda:0")
        from cache.types import CacheEntry as CE
        cm._gpu_cache[(layer_id, expert_id)] = CE(
            expert_id=expert_id, layer_id=layer_id, device="cuda:0",
            expert=expert,
            last_access=time.monotonic(),
            access_count=1,
            size_bytes=size_bytes,
        )
        cm._stats.gpu_slots_used = len(cm._gpu_cache)
        return expert

    def test_returns_false_when_not_in_gpu_cache(
        self, cm_lru: CacheManager
    ) -> None:
        """demote_to_cpu() must return False when key is absent from GPU tier."""
        assert cm_lru.demote_to_cpu(0, 99) is False

    def test_demote_removes_gpu_residency_without_cuda(
        self, cm_lru: CacheManager
    ) -> None:
        """demote_to_cpu() should remove GPU residency without requiring CUDA."""
        self._inject_gpu_entry(cm_lru, 0, 0)

        result = cm_lru.demote_to_cpu(0, 0)

        assert result is True
        assert (0, 0) not in cm_lru._gpu_cache
        assert cm_lru.stats().gpu_slots_used == 0

    @requires_cuda
    def test_demotes_entry_to_cpu_tier(self) -> None:
        """demote_to_cpu() must move the entry into _cpu_cache."""
        from model.expert import QuantizedMixtralExpert

        cm = CacheManager(gpu_slots=4, cpu_slots=8, policy=EvictionPolicy.LRU)

        w1 = torch.randn(16, 8, dtype=torch.bfloat16)
        w2 = torch.randn(8, 16, dtype=torch.bfloat16)
        w3 = torch.randn(16, 8, dtype=torch.bfloat16)
        real_expert = QuantizedMixtralExpert(w1, w2, w3)

        # Promote to GPU first
        cm.put(0, 0, real_expert, "cpu")
        cm.promote_to_gpu(0, 0)
        assert (0, 0) in cm._gpu_cache

        # Now demote back to CPU
        result = cm.demote_to_cpu(0, 0)
        assert result is True
        assert (0, 0) in cm._cpu_cache
        assert (0, 0) not in cm._gpu_cache
        assert cm.stats().gpu_slots_used == 0
        assert cm.stats().cpu_slots_used == 1


# ── Statistics correctness ────────────────────────────────────────────────

class TestStatisticsCorrectness:
    """Verify CacheStats reflects all mutations accurately."""

    def test_hit_rate_after_mixed_accesses(self, cm_lru: CacheManager) -> None:
        """hit_rate must equal hits / (hits + misses) across a mixed workload."""
        cm_lru.put(0, 0, make_expert(), "cpu")
        cm_lru.put(0, 1, make_expert(), "cpu")
        cm_lru.get(0, 0)   # hit
        cm_lru.get(0, 1)   # hit
        cm_lru.get(0, 2)   # miss
        cm_lru.get(0, 3)   # miss
        s = cm_lru.stats()
        assert s.hits == 2
        assert s.misses == 2
        assert s.hit_rate == pytest.approx(0.5)

    def test_evictions_tracked_correctly(self) -> None:
        """Each automatic or manual eviction must increment stats.evictions."""
        cm = CacheManager(gpu_slots=2, cpu_slots=2, policy=EvictionPolicy.LRU)
        cm.put(0, 0, make_expert(), "cpu")
        cm.put(0, 1, make_expert(), "cpu")
        # Two manual evictions
        cm.evict("cpu")
        cm.evict("cpu")
        assert cm.stats().evictions == 2

    def test_cpu_slots_used_stays_zero_after_full_evict(
        self, cm_lru: CacheManager
    ) -> None:
        """cpu_slots_used must be 0 after evicting all CPU entries."""
        for i in range(3):
            cm_lru.put(0, i, make_expert(), "cpu")
        for _ in range(3):
            cm_lru.evict("cpu")
        assert cm_lru.stats().cpu_slots_used == 0

    def test_stats_object_is_same_instance(self, cm_lru: CacheManager) -> None:
        """stats() must return the same object on every call (mutated in-place)."""
        s1 = cm_lru.stats()
        s2 = cm_lru.stats()
        assert s1 is s2
