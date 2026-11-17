"""Pass-rate statistics and the bounds rule (plan 05 section 8). Pure functions, exact where it
matters (``decide`` uses ``Fraction`` with decimal thresholds)."""

from fractions import Fraction
from math import comb, sqrt

Z95 = 1.959963984540054


def pass_at_1(p: int, d: int) -> float | None:
    return None if d == 0 else p / d


def pass_hat_k(p: int, d: int, k: int) -> float | None:
    """The tau-bench estimator: probability that ``k`` repeats drawn from the ``d`` decided ones
    all pass. ``None`` when fewer than ``k`` repeats were decided (never a free 1.0)."""
    return None if d < k else comb(p, k) / comb(d, k)


def pass_at_k(p: int, d: int, k: int) -> float | None:
    """The unbiased pass@k estimator."""
    return None if d < k else 1 - comb(d - p, k) / comb(d, k)


def wilson_95(p: int, d: int) -> tuple[float, float] | None:
    """The Wilson score interval, clipped to [0, 1]; ``None`` without decided repeats."""
    if d == 0:
        return None
    phat = p / d
    denom = 1 + Z95**2 / d
    centre = (phat + Z95**2 / (2 * d)) / denom
    half = Z95 * sqrt(phat * (1 - phat) / d + Z95**2 / (4 * d**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def decide(p: int, f: int, u: int, k: int, threshold: float) -> str:
    """``passed`` | ``failed`` | ``undecided``.

    ``failed`` if pass^k could not reach the threshold even if every undecided repeat passed;
    ``passed`` if it would hold even if every undecided repeat failed. Zero decided repeats can
    never pass. ``threshold`` has decimal semantics: ``0.8`` means exactly 4/5.
    """
    n = p + f + u
    if n < k or k < 1:
        return "undecided"
    t = Fraction(str(threshold))
    if Fraction(comb(p + u, k), comb(n, k)) < t:
        return "failed"
    if p + f > 0 and Fraction(comb(p, k), comb(n, k)) >= t:
        return "passed"
    return "undecided"
