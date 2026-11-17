"""Cost and call ledger: reserve before every live attempt, settle with actual usage."""

import math
import threading
from dataclasses import dataclass
from typing import Any


class BudgetExceeded(Exception):
    def __init__(self, what: str, spent: float, cap: float, calls: int):
        super().__init__(what)
        self.what, self.spent, self.cap, self.calls = what, spent, cap, calls


@dataclass(slots=True)
class _Reservation:
    estimate: float


class CostLedger:
    """Thread-safe. Cache hits cost nothing (``saved_usd`` tracks what they would have cost)."""

    def __init__(self, max_usd: float | None = None, max_calls: int | None = None):
        self._lock = threading.Lock()
        self.max_usd, self.max_calls = max_usd, max_calls
        self.spent_usd = 0.0
        self.calls = 0
        self.cache_hits = 0
        self.saved_usd = 0.0
        self.tokens = 0
        self._reserved = 0.0

    def reserve(self, estimate: float = 0.0) -> _Reservation:
        """Check the caps *before* a live attempt; raises ``BudgetExceeded``."""
        with self._lock:
            if self.max_calls is not None and self.calls >= self.max_calls:
                raise BudgetExceeded("max_calls", self.spent_usd, self.max_usd or 0.0, self.calls)
            if (
                self.max_usd is not None
                and self.spent_usd + self._reserved + estimate > self.max_usd
            ):
                raise BudgetExceeded("max_usd", self.spent_usd, self.max_usd, self.calls)
            self.calls += 1
            self._reserved += estimate
            return _Reservation(estimate)

    def settle(self, reservation: _Reservation, actual_usd: float, tokens: int = 0) -> None:
        with self._lock:
            self._reserved = max(0.0, self._reserved - reservation.estimate)
            if not math.isfinite(actual_usd) or actual_usd < 0:
                actual_usd = reservation.estimate  # a bad figure never lowers or NaNs the total
            self.spent_usd += actual_usd
            self.tokens += max(0, int(tokens))

    def release(self, reservation: _Reservation) -> None:
        """A reserved attempt that never reached the provider."""
        with self._lock:
            self._reserved = max(0.0, self._reserved - reservation.estimate)
            self.calls = max(0, self.calls - 1)

    def record_hit(self, recorded_usd: float = 0.0) -> None:
        with self._lock:
            self.cache_hits += 1
            self.saved_usd += recorded_usd

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "spent_usd": round(self.spent_usd, 6), "calls": self.calls,
                "cache_hits": self.cache_hits, "saved_usd": round(self.saved_usd, 6),
                "tokens": self.tokens, "max_usd": self.max_usd, "max_calls": self.max_calls,
            }  # fmt: skip
