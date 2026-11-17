"""Agreement statistics between judge labels and golden labels, with seeded bootstrap CIs."""

import math
import random
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any

ABSTAIN = "unsure"
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 0


def _div(a: float, b: float) -> float | None:
    return a / b if b else None


def kappa(pairs: Sequence[tuple[str, str]]) -> float | None:
    """Cohen's kappa over ``(truth, prediction)`` pairs; ``None`` when undefined."""
    n = len(pairs)
    if n == 0:
        return None
    observed = sum(1 for t, p in pairs if t == p) / n
    truth, pred = Counter(t for t, _ in pairs), Counter(p for _, p in pairs)
    expected = sum(truth[c] * pred[c] for c in truth.keys() | pred.keys()) / (n * n)
    if math.isclose(expected, 1.0):
        return 1.0 if math.isclose(observed, 1.0) else None
    return (observed - expected) / (1 - expected)


def binary_stats(pairs: Sequence[tuple[str, str]], positive: str) -> dict[str, float | None]:
    """``pairs`` exclude abstentions. Positive = the label that lets the test pass."""
    tp = sum(1 for t, p in pairs if t == positive and p == positive)
    fn = sum(1 for t, p in pairs if t == positive and p != positive)
    tn = sum(1 for t, p in pairs if t != positive and p != positive)
    fp = sum(1 for t, p in pairs if t != positive and p == positive)
    return {
        "tpr": _div(tp, tp + fn), "tnr": _div(tn, tn + fp), "precision": _div(tp, tp + fp),
        "accuracy": _div(tp + tn, len(pairs)), "kappa": kappa(pairs),
    }  # fmt: skip


def multiclass_stats(pairs: Sequence[tuple[str, str]]) -> dict[str, Any]:
    classes = sorted({t for t, _ in pairs})
    per: dict[str, dict[str, float | None]] = {}
    for c in classes:
        tp = sum(1 for t, p in pairs if t == c and p == c)
        fn = sum(1 for t, p in pairs if t == c and p != c)
        fp = sum(1 for t, p in pairs if t != c and p == c)
        prec, rec = _div(tp, tp + fp), _div(tp, tp + fn)
        f1 = (
            _div(2 * prec * rec, prec + rec)
            if prec is not None and rec is not None
            else (0.0 if tp + fn else None)
        )
        per[c] = {"precision": prec, "recall": rec, "f1": f1, "support": float(tp + fn)}
    f1s = [v["f1"] for v in per.values() if v["f1"] is not None]
    return {"per_class": per, "macro_f1": sum(f1s) / len(f1s) if f1s else None,
            "accuracy": _div(sum(1 for t, p in pairs if t == p), len(pairs)),
            "kappa": kappa(pairs)}  # fmt: skip


def bootstrap_ci(
    pairs: Sequence[tuple[str, str]],
    stat: Callable[[Sequence[tuple[str, str]]], float | None],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float] | None:
    """95% percentile interval; deterministic for a given input."""
    if not pairs:
        return None
    rng = random.Random(seed)
    n = len(pairs)
    values = sorted(v for v in (stat([pairs[rng.randrange(n)] for _ in range(n)]) for _ in range(resamples))
                    if v is not None)  # fmt: skip
    if not values:
        return None
    lo, hi = (
        values[int(0.025 * (len(values) - 1))],
        values[int(math.ceil(0.975 * (len(values) - 1)))],
    )
    return (round(lo, 4), round(hi, 4))


def split_abstentions(
    rows: Sequence[tuple[str, str]],
) -> tuple[list[tuple[str, str]], float | None]:
    """``(decided pairs, abstain rate)``; an abstention is a prediction of ``unsure``."""
    decided = [(t, p) for t, p in rows if p != ABSTAIN]
    return decided, _div(len(rows) - len(decided), len(rows))
