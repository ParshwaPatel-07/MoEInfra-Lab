class PinnedMemoryBudget:
    """Tracks and bounds the amount of host memory reserved for pinned transfers."""

    def __init__(self, budget_bytes: int):
        if budget_bytes < 0:
            raise ValueError("budget_bytes must be non-negative")

        self.budget_bytes = budget_bytes
        self.used_bytes = 0

    @property
    def available_bytes(self) -> int:
        return self.budget_bytes - self.used_bytes

    def can_reserve(self, size_bytes: int) -> bool:
        self._validate_size(size_bytes)
        return self.used_bytes + size_bytes <= self.budget_bytes

    def reserve(self, size_bytes: int) -> bool:
        self._validate_size(size_bytes)

        if not self.can_reserve(size_bytes):
            return False

        self.used_bytes += size_bytes
        return True

    def release(self, size_bytes: int) -> None:
        self._validate_size(size_bytes)

        if size_bytes > self.used_bytes:
            raise ValueError("Pinned memory accounting underflow")

        self.used_bytes -= size_bytes

    @staticmethod
    def _validate_size(size_bytes: int) -> None:
        if size_bytes < 0:
            raise ValueError("size_bytes must be non-negative")

    def __repr__(self) -> str:
        return (
            f"PinnedMemoryBudget("
            f"budget_bytes={self.budget_bytes}, "
            f"used_bytes={self.used_bytes}, "
            f"available_bytes={self.available_bytes})"
        )