"""Render a report dict as text. Consumes only ``to_dict()`` output, so any tool can do it."""

import os
from collections.abc import Callable, Mapping
from typing import Any

from .canonical import clip
from .codes import lookup
from .metrics.base import MetricResult

DEFAULT_MAX_LINES = 40
LABEL = 10


def _cap(value: object) -> str:
    """A value for a message: strings as is (they are already descriptions), others clipped."""
    return (
        clip(value, 80)
        if not isinstance(value, str)
        else (value if len(value) <= 80 else f"{value[:77]}...(len={len(value)})")
    )


def _row(label: str, text: str, status: str = "") -> str:
    line = f"{label:<{LABEL}} {text}"
    return f"{line}   {status}" if status else line


def _status(r: MetricResult) -> str:
    if r.status == "ok":
        return "ok" if r.passed else "FAILED"
    return r.status


def _decision(r: MetricResult) -> tuple[str, ...]:
    lines = []
    for c in r.components or (r,):
        d = c.details
        if c.status != "ok":
            lines.append(_row("decision", f"turn {c.turn}: {c.reason_code}", c.status))
            continue
        actual = d.get("actual")
        exp = "/".join(d.get("expected", []))
        extra = ""
        if actual is not None:
            extra = f" ({d.get('rule')}, conf {float(d.get('confidence', 0)):.2f})"
        text = f"turn {c.turn}: expected {exp}, got {actual}{extra}"
        if not c.passed:
            text += f"  [{c.reason_code}]"
        lines.append(_row("decision", text, _status(c)))
    return tuple(lines)


def _tools(r: MetricResult) -> tuple[str, ...]:
    d = r.details
    scores = d.get("scores", {})
    matched = len(d.get("matched", []))
    req = d.get("required", 0)
    unexpected = d.get("unexpected", [])
    allowed = d.get("allowed_reads", [])
    text = f"matched {matched}/{req} required; unexpected {len(unexpected)}"
    if allowed:
        text += f" ({', '.join(a['name'] + ' ' + str(a['effect']) + ' allowed' for a in allowed)})"
    lines = [_row("tools", text, _status(r))]
    if not r.passed:
        for key, label in (
            ("missing", "missing"),
            ("unexpected", "unexpected"),
            ("hallucinated", "hallucinated"),
            ("not_allowed", "not allowed"),
            ("wrong_count", "wrong count"),
        ):
            items = d.get(key, [])
            if items:
                names = [
                    i if isinstance(i, str) else f"{i['name']} ({i['call_id']})" for i in items
                ]
                lines.append(f"  {label}: {', '.join(names)}  [{r.reason_code}]")
    _ = scores
    return tuple(lines)


def _args(r: MetricResult) -> tuple[str, ...]:
    pairs = r.details.get("pairs", [])
    if r.status == "skipped":
        return (_row("args", f"skipped: {r.reason_code}", "skipped"),)
    if r.passed:
        return (_row("args", f"{len(r.details.get('scores', {}))} scores", "ok"),)
    lines = []
    last = None
    for p in pairs:
        key = (p["call_id"], p["expect"])
        if key != last:
            lines.append(f"  args  {p['call_id']} vs expected {p['expect']!r}   [tool_args]")
            last = key
        lines.append(
            f"    {p['path_str']}  {p['code']}  Expected: {_cap(p['expected'])}"
            f"  Actual: {clip(p['actual'])}"
        )
    return tuple(lines) or (_row("args", r.reason, _status(r)),)


def _order(r: MetricResult) -> tuple[str, ...]:
    mode = r.details.get("mode", "")
    if r.status != "ok":
        return (_row("order", f"{mode}: {r.reason_code}", r.status),)
    if r.passed:
        return (_row("order", f"{mode}: satisfied", "ok"),)
    return (_row("order", f"{mode}: {r.reason}  [{r.reason_code}]", "FAILED"),)


def _forbidden(r: MetricResult) -> tuple[str, ...]:
    if r.status != "ok":
        return (_row("forbidden", f"{r.reason_code}: {r.reason}".strip(": "), r.status),)
    if r.passed:
        return (_row("forbidden", "no forbidden call made", "ok"),)
    return (_row("forbidden", f"{r.reason}  [{r.reason_code}]", "FAILED (veto)"),)


def _budgets(r: MetricResult) -> tuple[str, ...]:
    lines = []
    for c in r.components:
        if c.status == "ok":
            tail = "" if c.passed else f"  [{c.reason_code}] {c.reason}"
            lines.append(_row("budget", f"{c.name.split('.')[-1]}{tail}", _status(c)))
        else:
            lines.append(
                _row("budget", f"{c.name.split('.')[-1]}: {c.reason_code} ({c.reason})", c.status)
            )
    return tuple(lines) or (_row("budget", r.reason, _status(r)),)


def _answer(r: MetricResult) -> tuple[str, ...]:
    if r.status != "ok":
        return (_row("answer", f"{r.reason_code}: {r.reason}".strip(": "), r.status),)
    if r.passed:
        return (_row("answer", "expected text present", "ok"),)
    return (_row("answer", f"{r.reason}  [{r.reason_code}]", "FAILED"),)


def _simple(label: str) -> Callable[[MetricResult], tuple[str, ...]]:
    def lines(r: MetricResult) -> tuple[str, ...]:
        if r.status != "ok":
            return (_row(label, f"{r.reason_code}: {r.reason}".strip(": "), r.status),)
        if r.passed:
            return (_row(label, r.reason or "ok", "ok"),)
        veto = " (veto)" if r.vetoed else ""
        return (_row(label, f"{r.reason}  [{r.reason_code}]", f"FAILED{veto}"),)

    return lines


def _clarify(r: MetricResult) -> tuple[str, ...]:
    out = []
    for c in r.components or (r,):
        if c.status != "ok":
            out.append(_row("clarify", f"turn {c.turn}: {c.reason_code}", c.status))
        elif c.passed:
            out.append(_row("clarify", f"turn {c.turn}: asked", "ok"))
        else:
            out.append(_row("clarify", f"turn {c.turn}: {c.reason}  [{c.reason_code}]", "FAILED"))
    return tuple(out)


LINES: dict[str, Callable[[MetricResult], tuple[str, ...]]] = {
    "routing": _simple("routing"),
    "loops": _simple("loops"),
    "clarification": _clarify,
    "refusal": _simple("refusal"),
    "scope": _simple("scope"),
    "intent": _simple("intent"),
    "plan": _simple("plan"),
    "decision_match": _decision,
    "tool_selection": _tools,
    "tool_args": _args,
    "order": _order,
    "forbidden_calls": _forbidden,
    "budgets": _budgets,
    "answer": _answer,
}


def default_lines(r: MetricResult) -> tuple[str, ...]:
    """Used for metrics without their own renderer, and for codes with no template."""
    fn = LINES.get(r.name)
    if fn is not None:
        return fn(r)
    spec = lookup(r.reason_code) if r.reason_code else None
    detail = r.reason or (spec.template if spec else "")
    code = f"  [{r.reason_code}]" if r.reason_code else ""
    return (_row(r.name, f"{detail}{code}".strip(), _status(r)),)


def _lines_for(r: MetricResult) -> tuple[str, ...]:
    from .metrics.registry import get_metric

    try:
        metric = get_metric(r.name)
    except KeyError:
        return default_lines(r)
    return metric.render_lines(r)


def resolve_max_lines(explicit: int | None, config_value: int | None = None) -> int | None:
    """Explicit argument, else config, else env, else 40; ``0`` or ``CI`` set means unlimited."""
    if explicit is not None:
        return None if explicit == 0 else explicit
    if config_value is not None:
        return None if config_value == 0 else config_value
    env = os.environ.get("DISSEQT_REPORT_MAX_LINES")
    if env is not None:
        try:
            value = int(env)
            return None if value == 0 else value
        except ValueError:
            pass
    if os.environ.get("CI"):
        return None
    return DEFAULT_MAX_LINES


def header_lines(report: Mapping[str, Any]) -> list[str]:
    harness = report.get("harness", {})
    summary = report.get("summary", {})
    lines: list[str] = []
    status = harness.get("status", "finished")
    if status != "finished":
        error = harness.get("error")
        lines.append(
            f"agent run did not finish: status={status}" + (f" ({error})" if error else "")
        )
    violations = harness.get("violations", [])
    if violations and status == "finished":
        lines.append(f"{len(violations)} harness violation(s):")
        lines.extend(f"  {v['message']}" for v in violations)
    elif violations:
        lines.append(f"{len(violations)} harness violation(s):")
        lines.extend(f"  {v['message']}" for v in violations)
    outcome = report.get("outcome")
    case = report.get("case_id", "")
    counts = (
        f"{summary.get('n_failed', 0)} failed, {summary.get('n_incomplete', 0)} incomplete, "
        f"{summary.get('n_passed', 0)} passed, {summary.get('n_skipped', 0)} skipped"
    )
    if outcome == "incomplete" and not lines:
        pending = [
            f"{c['name']} ({c['reason_code']}: {c['reason']})"
            for c in report.get("checks", [])
            if c.get("status") != "ok" and _is_unverifiable(c)
        ]
        lines.append("could not verify: " + "; ".join(pending or ["see checks below"]))
    verdict = {"passed": "passed", "failed": "FAILED"}.get(str(outcome), "INCOMPLETE")
    lines.append(f"assert_agent {verdict} for case {case!r} ({counts})")
    return lines


def _is_unverifiable(check: Mapping[str, Any]) -> bool:
    from .codes import UNVERIFIABLE, code_class

    return code_class(check.get("reason_code")) == UNVERIFIABLE or check.get("status") == "error"


def render(report: Mapping[str, Any], max_lines: int | None = None, color: bool = False) -> str:
    """Header, harness block, each check's lines in rank order, scores, skips. Bounded output."""
    results = [MetricResult.from_dict(c) for c in report.get("checks", [])]
    blocks: list[tuple[bool, list[str]]] = []
    for r in results:
        failing = r.status == "ok" and r.passed is False
        blocks.append((failing, list(_lines_for(r))))
    lines = header_lines(report)
    tail: list[str] = []
    scores = report.get("scores", {})
    if scores:
        tail.append(
            _row("scores", " ".join(f"{k}={float(v):.2f}" for k, v in sorted(scores.items())))
        )
    for w in report.get("warnings", []):
        tail.append(f"warning: {w}")
    limit = resolve_max_lines(max_lines)
    body = [ln for _, block in blocks for ln in block]
    total = len(lines) + len(body) + len(tail)
    if limit is not None and total > limit:  # failures first when the budget is tight
        ordered = sorted(blocks, key=lambda b: not b[0])
        body = [ln for _, block in ordered for ln in block]
    out = lines + body + tail
    if limit is not None and len(out) > limit:
        hidden = len(out) - (limit - 1)
        out = out[: limit - 1] + [
            f"... {hidden} more lines (set DISSEQT_REPORT_MAX_LINES=0 to show all)"
        ]
    return "\n".join(out)
