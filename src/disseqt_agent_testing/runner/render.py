"""Text rendering shared by the pytest plugin and the CLI: the repeat strip, the case failure
layout and the summary table."""

import sys
from collections import Counter

from ..assertions.render import render as render_report
from .models import CaseResult, RepeatRecord, SuiteResult

SYMBOLS = {"passed": "✓", "failed": "✗", "incomplete": "?", "not_run": "·"}
ASCII = {"passed": "P", "failed": "F", "incomplete": "?", "not_run": "."}
ORDER = {"error": 0, "failed": 1, "incomplete": 2, "skipped": 3, "passed": 4}


def can_unicode(stream: object = None) -> bool:
    encoding = getattr(stream or sys.stdout, "encoding", None) or "ascii"
    try:
        "✓✗·".encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return False
    return True


def strip_of(
    repeats: list[RepeatRecord] | tuple[RepeatRecord, ...], ascii_only: bool = False
) -> str:
    table = ASCII if ascii_only else SYMBOLS
    return "".join(table.get(r.status, "?") for r in sorted(repeats, key=lambda r: r.index))


def _word(status: str) -> str:
    return {
        "failed": "FAILED",
        "incomplete": "INCOMPLETE",
        "error": "ERROR",
        "skipped": "SKIPPED",
        "passed": "PASSED",
    }[status]


def _counts(d: dict[str, int] | object) -> str:
    items = sorted(dict(d).items()) if d else []  # type: ignore[call-overload]
    return " ".join(f"{k}×{v}" for k, v in items)


def case_failure_text(
    result: CaseResult, *, ascii_only: bool = False, workers: int | None = None, early_stop: bool = False,
    max_lines: int | None = None, replay: str | None = None,
) -> str:  # fmt: skip
    """At most 8 header lines, then the assertions report of the worst repeat, then a replay hint."""
    where = (
        f" ({result.source.path}:{result.source.line})"
        if result.source.path and result.source.line
        else ""
    )
    lines = [f"agent case '{result.case_id}' {_word(result.status)}{where}"]
    rr = result.repeat_report
    if result.status == "error" or rr is None:
        if result.reason:
            lines.append(result.reason)
        return "\n".join(lines)
    d = rr.n_passed + rr.n_failed
    wilson = f"  Wilson95 [{rr.wilson_95[0]:.2f}, {rr.wilson_95[1]:.2f}]" if rr.wilson_95 else ""
    p1 = f"  pass@1={rr.pass_at_1:.2f}" if rr.pass_at_1 is not None else ""
    if rr.pass_hat_k is not None:
        cmp = "<" if rr.pass_hat_k < result.threshold else ">="
        hat = f"  pass^{result.k}={rr.pass_hat_k:.2f} {cmp} threshold {result.threshold:.2f}"
    else:
        hat = f"  pass^{result.k}=n/a (needs {result.k} decided repeats) threshold {result.threshold:.2f}"
    lines.append(
        f"repeats   {strip_of(result.repeats, ascii_only)}   c={rr.n_passed}/{d} decided{p1}{hat}{wilson}"
    )
    not_run = sum(1 for r in result.repeats if r.status == "not_run")
    lines.append(
        f"plan      n={result.n_planned} k={result.k} incomplete={rr.n_incomplete} not_run={not_run} "
        f"early_stop={'on' if early_stop else 'off'} seed={result.repeats[0].seed if result.repeats else 0}"
        + (f" workers={workers}" if workers else "")
    )  # fmt: skip
    if rr.decision_histogram:
        lines.append(f"decisions {_counts(rr.decision_histogram)}")
    if rr.failure_codes:
        lines.append(f"codes     {_counts(rr.failure_codes)}")
    worst = next((r for r in result.repeats if str(r.index) == rr.worst), None)
    if worst is not None:
        ms = f", {worst.duration_ms:.0f} ms" if worst.duration_ms is not None else ""
        trace = f", trace {worst.trace_id[:8]}" if worst.trace_id else ""
        lines.append(f"worst     repeat #{worst.index} (seed {worst.seed}{trace}{ms})")
    if result.reason:
        lines.append(result.reason)
    if worst is not None and worst.report is not None:
        lines.append(render_report(worst.report.to_dict(), max_lines=max_lines))
    if replay and worst is not None:
        lines.append(f"replay    {replay.format(index=worst.index, seed=worst.seed)}")
    return "\n".join(lines)


def summary_table(result: SuiteResult, ascii_only: bool = False) -> list[str]:
    """One row per case, worst first, then totals and a judge line."""
    rows = sorted(result.cases, key=lambda c: (ORDER.get(c.status, 9), c.source_key))
    out = []
    for c in rows:
        rr = c.repeat_report
        stats = ""
        if rr is not None:
            p1 = "n/a" if rr.pass_at_1 is None else f"{rr.pass_at_1:.2f}"
            hk = "n/a" if rr.pass_hat_k is None else f"{rr.pass_hat_k:.2f}"
            stats = f"  pass@1={p1} pass^{c.k}={hk}"
        out.append(
            f"{_word(c.status):<10} {c.case_id:<32} {strip_of(c.repeats, ascii_only):<10}{stats}"
        )
    t = result.totals
    out.append(
        f"{t.cases} cases: {t.passed} passed, {t.failed} failed, {t.incomplete} incomplete, "
        f"{t.skipped} skipped, {t.error} error; repeats run {t.repeats_run}, not run {t.repeats_not_run}"
    )  # fmt: skip
    u = result.judge_usage
    if u.calls or u.error_repeats:
        out.append(
            f"judge: {u.calls} calls, {u.tokens} tokens, ${u.cost_usd:.4f}, {u.error_repeats} error repeats"
        )
    return out


_ = Counter
