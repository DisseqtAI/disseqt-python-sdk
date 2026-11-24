"""``summarize_repeats``: the per-case statistics and verdict from the repeats' reports."""

from collections import Counter
from collections.abc import Sequence
from typing import Any

from ..assertions.codes import UNVERIFIABLE, code_class
from ..assertions.report import DecisionReport, RepeatReport
from .stats import decide, pass_at_1, pass_at_k, pass_hat_k, wilson_95


def failure_codes_of(report: DecisionReport) -> list[str]:
    """Codes a repeat is accountable for, each once: harness problems, failing checks' reason
    codes, ``error:<code>`` for errored checks and ``unavailable:<code>`` for unmeasurable ones
    (so a flaky judge can be told apart from a flaky agent)."""
    codes: list[str] = []
    harness = report.harness
    if harness.get("status") != "finished":
        codes.append("harness.not_finished")
    if harness.get("violations"):
        codes.append("harness.violation")
    for check in report.checks:
        if check.status == "ok" and check.passed is False and check.reason_code:
            codes.append(check.reason_code)
        elif check.status == "error" and check.reason_code:
            codes.append(f"error:{check.reason_code}")
        elif (
            check.status == "skipped"
            and check.reason_code
            and code_class(check.reason_code) == UNVERIFIABLE
        ):
            codes.append(f"unavailable:{check.reason_code}")
    return sorted(set(codes))


def summarize_repeats(
    reports: Sequence[DecisionReport | None],
    *,
    k: int,
    threshold: float,
    n_planned: int | None = None,
    case_id: str | None = None,
) -> RepeatReport:
    """``reports[i]`` is repeat ``i``'s report, or ``None`` if it did not run (undecided).

    ``P`` and ``F`` come only from ``DecisionReport.outcome``. Incomplete and not-run repeats are
    undecided: they are left out of every denominator, and the bounds rule decides the case.
    """
    planned = len(reports) if n_planned is None else n_planned
    ran = [(i, r) for i, r in enumerate(reports) if r is not None]
    passed = sum(1 for _, r in ran if r.outcome == "passed")
    failed = sum(1 for _, r in ran if r.outcome == "failed")
    incomplete = sum(1 for _, r in ran if r.outcome not in ("passed", "failed"))
    d = passed + failed
    verdict = decide(passed, failed, planned - passed - failed, k, threshold)
    worst: str | None = None
    for wanted in ("failed", "incomplete"):
        hit = next(
            (
                i
                for i, r in ran
                if (r.outcome == wanted)
                or (wanted == "incomplete" and r.outcome not in ("passed", "failed"))
            ),
            None,
        )
        if hit is not None:
            worst = str(hit)
            break
    if worst is None and ran:
        worst = str(ran[0][0])
    histogram: Counter[str] = Counter()
    codes: Counter[str] = Counter()
    for _, r in ran:
        if r.decisions:
            histogram[str(r.decisions[0].get("label"))] += 1
        codes.update(failure_codes_of(r))
    wilson = wilson_95(passed, d)
    return RepeatReport(
        len(ran), n_passed=passed, n_failed=failed, n_incomplete=incomplete, k=k, threshold=threshold,
        pass_at_1=pass_at_1(passed, d), pass_hat_k=pass_hat_k(passed, d, k), pass_at_k=pass_at_k(passed, d, k),
        wilson_95=wilson, decision_histogram=dict(sorted(histogram.items())),
        failure_codes=dict(sorted(codes.items())), outcome="incomplete" if verdict == "undecided" else verdict,
        worst=worst,
    )  # fmt: skip


def _unused(*_: Any) -> None:  # keeps ``Any`` import honest for type readers
    return None
