"""Merge per-repeat ``CaseResult``s (split mode) into one ``CaseResult`` per case."""

from collections.abc import Sequence

from .errors import format_code
from .models import DELIBERATE_REASONS, CaseResult, RepeatRecord, cut_off_reason
from .summarize import summarize_repeats


def is_split_part(result: CaseResult) -> bool:
    """True for a result that ran only some repeats of its case (``only_repeat`` replay)."""
    return any(r.not_run_reason == "only_repeat" for r in result.repeats)


def _rank(rec: RepeatRecord) -> int:
    """Which record of one repeat index to keep: a repeat that ran, then one that was cut off,
    then any other not-run record; the ``only_repeat`` placeholder of another part comes last."""
    if rec.status != "not_run":
        return 3
    if cut_off_reason((rec.not_run_reason,)) is not None:
        return 2
    return 0 if rec.not_run_reason in DELIBERATE_REASONS else 1


def _breakdown(records: Sequence[RepeatRecord]) -> str:
    counts: dict[str, int] = {}
    for rec in records:
        if rec.status == "incomplete":
            counts["incomplete"] = counts.get("incomplete", 0) + 1
        elif rec.status == "not_run":
            label = f"not run: {rec.not_run_reason or 'cancelled'}"
            counts[label] = counts.get(label, 0) + 1
    return "; ".join(f"{n} {label}" for label, n in sorted(counts.items())) or "none"


def merge_repeat_results(parts: Sequence[CaseResult]) -> CaseResult:
    """Each part ran a subset of the repeats (typically one). The verdict is recomputed from the
    union with the same bounds rule as an internal-mode case, so both modes agree. Only split-repeat
    parts can be merged: two whole results with one id are a duplicate id, not repeats of a case
    (``ValueError``)."""
    import dataclasses

    base = parts[0]
    if len(parts) > 1 and not all(is_split_part(p) for p in parts):
        raise ValueError(f"duplicate case id '{base.case_id}': results are not split repeats")
    by_index: dict[int, RepeatRecord] = {}
    for part in parts:
        for rec in part.repeats:
            old = by_index.get(rec.index)
            if old is None or _rank(rec) > _rank(old):
                by_index[rec.index] = rec
    records = tuple(by_index[i] for i in sorted(by_index))
    cut = cut_off_reason(r.not_run_reason for r in records)
    if not any(r.status != "not_run" for r in records) and (
        base.status == "error" or (base.status == "skipped" and cut is None)
    ):
        return base
    reports = [r.report if r.status != "not_run" else None for r in records]
    rr = summarize_repeats(
        reports, k=base.k, threshold=base.threshold, n_planned=base.n_planned, case_id=base.case_id
    )
    reason: str = ""
    code: str | None = None
    d = rr.n_passed + rr.n_failed
    if rr.outcome == "incomplete":
        code = cut or "undecided"
        reason = format_code(
            "DSQ-I01",
            id=base.case_id,
            U=base.n_planned - d,
            n=base.n_planned,
            breakdown=_breakdown(records),
            k=base.k,
            t=base.threshold,
        )
    elif rr.outcome == "failed":
        code = "below_threshold"
        reason = format_code(
            "DSQ-F01",
            id=base.case_id,
            k=base.k,
            v=rr.pass_hat_k or 0.0,
            t=base.threshold,
            P=rr.n_passed,
            D=d,
            n=base.n_planned,
        )
    return dataclasses.replace(
        base, status=rr.outcome, repeat_report=rr, repeats=records, reason_code=code, reason=reason,
        comparable=all(r.status != "not_run" for r in records),
    )  # fmt: skip
