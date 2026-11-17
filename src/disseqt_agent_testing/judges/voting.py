"""Voting: majority over valid samples with a quorum; agreement is the only confidence signal."""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..assertions.canonical import round6
from .codes import ORDER


@dataclass(slots=True, frozen=True)
class SampleOutcome:
    order: int
    index: int
    ok: bool
    label: str | None = None
    parsed: Mapping[str, Any] | None = None
    code: str | None = None
    detail: str = ""
    cached: bool = False
    tokens: int = 0
    cost: float = 0.0
    recorded_cost: float = 0.0
    dropped_params: tuple[str, ...] = ()

    def summary(self) -> dict[str, Any]:
        return {
            "order": self.order, "index": self.index, "ok": self.ok, "label": self.label,
            "code": self.code, "detail": self.detail, "cached": self.cached,
        }  # fmt: skip


@dataclass(slots=True, frozen=True)
class Vote:
    label: str | None
    agreement: float
    n_valid: int
    n: int
    error_code: str | None = None
    error_detail: str = ""

    @property
    def is_error(self) -> bool:
        return self.error_code is not None


def vote(outcomes: Sequence[SampleOutcome], n: int, min_agreement: float) -> Vote:
    """``n`` valid samples are expected; a strict majority (``n // 2 + 1``) must be valid. A tie
    for the mode or agreement below ``min_agreement`` is ``unsure``, never a coin-flip pass or
    fail."""
    valid = [o.label for o in outcomes if o.ok and o.label is not None]
    errors = [o.code for o in outcomes if not o.ok and o.code]
    if len(valid) < n // 2 + 1 or not valid:
        counts = Counter(errors)
        code = "insufficient_votes"
        if counts:
            top = counts.most_common()
            if len(top) == 1 or top[0][1] > top[1][1]:
                code = str(top[0][0])
        return Vote(None, 0.0, len(valid), n, code, ", ".join(sorted({str(e) for e in errors})))
    counts = Counter(valid)
    ranked = counts.most_common()
    top_label, top_count = ranked[0]
    if len(ranked) > 1 and ranked[1][1] == top_count:  # tie for the mode
        return Vote("unsure", round6(top_count / len(valid)), len(valid), n)
    agreement = round6(top_count / len(valid))
    label = top_label if agreement >= round6(min_agreement) else "unsure"
    return Vote(label, agreement, len(valid), n)


def combine_orders(a: Vote, b: Vote, positive: str, negative: str) -> tuple[Vote, bool]:
    """``semantic_arg``: both orders must agree on a decisive label; otherwise ``unsure``.
    Returns the combined vote and ``order_consistent``."""
    if a.is_error or b.is_error:
        bad = a if a.is_error else b
        return bad, False
    consistent = a.label == b.label
    agreement = round6(min(a.agreement, b.agreement))
    n_valid = min(a.n_valid, b.n_valid)
    if consistent and a.label in (positive, negative):
        return Vote(a.label, agreement, n_valid, a.n), True
    return Vote("unsure", agreement, n_valid, a.n), consistent


def error_priority(code: str) -> int:
    return ORDER.index(code) if code in ORDER else len(ORDER)
