"""Cost and call ledger: reserve before every live attempt, settle with actual usage.

A reservation holds the larger of the caller's estimate and the highest real cost already seen for
the model. ``try_reserve`` admits one unsettled call of a model at a time until that model has
reported a first real cost, and again whenever a new call only fits if the calls in flight settle
for less than is held for them, so concurrent samples cannot all leave on an estimate that proves
too low. What is left is bounded: a call can cost more than was held for it, and ``overrun_usd``
reports it.
"""

import math
import threading
from dataclasses import dataclass
from typing import Any


class BudgetExceeded(Exception):
    def __init__(self, what: str, spent: float, cap: float, calls: int):
        super().__init__(what)
        self.what, self.spent, self.cap, self.calls = what, spent, cap, calls


@dataclass(slots=True)
class Reservation:
    estimate: float  # the amount held against the cap
    model: str = ""


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
        self._highest: dict[str, float] = {}  # model -> highest real cost of one call
        self._unsettled: dict[str, int] = {}

    def reserve(self, estimate: float = 0.0, model: str = "") -> Reservation:
        """Check the caps *before* a live attempt; raises ``BudgetExceeded``. Never waits."""
        with self._lock:
            return self._admit(estimate, model)

    def try_reserve(self, estimate: float = 0.0, model: str = "") -> Reservation | None:
        """Like ``reserve``, but ``None`` means "ask again after the next ``settle`` or
        ``release``": under a dollar cap, a model whose real cost is still unknown has one
        unsettled call at a time."""
        with self._lock:
            if self.max_usd is not None:
                self._check_calls()
                if model not in self._highest and self._unsettled.get(model, 0) > 0:
                    return None
                fits_alone = self.spent_usd + self._hold(estimate, model) <= self.max_usd
                if fits_alone and self._reserved > 0 and not self._fits(estimate, model):
                    return None  # the calls in flight may settle for less than is held for them
            return self._admit(estimate, model)

    def _hold(self, estimate: float, model: str) -> float:
        return max(estimate, self._highest.get(model, 0.0))

    def _fits(self, estimate: float, model: str) -> bool:
        return (
            self.max_usd is None
            or self.spent_usd + self._reserved + self._hold(estimate, model) <= self.max_usd
        )

    def _check_calls(self) -> None:
        if self.max_calls is not None and self.calls >= self.max_calls:
            raise BudgetExceeded("max_calls", self.spent_usd, self.max_usd or 0.0, self.calls)

    def _admit(self, estimate: float, model: str) -> Reservation:
        self._check_calls()
        hold = self._hold(estimate, model)
        if not self._fits(estimate, model):
            raise BudgetExceeded("max_usd", self.spent_usd, self.max_usd or 0.0, self.calls)
        self.calls += 1
        self._reserved += hold
        self._unsettled[model] = self._unsettled.get(model, 0) + 1
        return Reservation(hold, model)

    def _close(self, reservation: Reservation) -> None:
        self._reserved = max(0.0, self._reserved - reservation.estimate)
        self._unsettled[reservation.model] = max(0, self._unsettled.get(reservation.model, 0) - 1)

    def settle(
        self, reservation: Reservation, actual_usd: float, tokens: int = 0, *, answered: bool = True
    ) -> None:
        """``answered=False`` is an attempt that failed in transit (a timeout, a 429): it counts
        as a call but says nothing about what a call to the model costs."""
        with self._lock:
            self._close(reservation)
            if not math.isfinite(actual_usd) or actual_usd < 0:
                actual_usd = reservation.estimate  # a bad figure never lowers or NaNs the total
            self.spent_usd += actual_usd
            self.tokens += max(0, int(tokens))
            if answered:
                model = reservation.model
                self._highest[model] = max(self._highest.get(model, 0.0), actual_usd)

    def release(self, reservation: Reservation) -> None:
        """A reserved attempt that never reached the provider."""
        with self._lock:
            self._close(reservation)
            self.calls = max(0, self.calls - 1)

    def overrun_usd(self) -> float:
        """How far the spend is past ``max_usd`` (0.0 when it is within the cap or uncapped)."""
        with self._lock:
            if self.max_usd is None:
                return 0.0
            return max(0.0, self.spent_usd - self.max_usd)

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
