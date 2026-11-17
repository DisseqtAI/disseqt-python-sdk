"""Merge per-repeat ``CaseResult``s (split mode) into one ``CaseResult`` per case."""

from collections.abc import Sequence

from .models import CaseResult, RepeatRecord
from .summarize import summarize_repeats


def is_split_part(result: CaseResult) -> bool:
    """True for a result that ran only some repeats of its case (``only_repeat`` replay)."""
    return any(r.not_run_reason == "only_repeat" for r in result.repeats)


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
            if old is None or (old.status == "not_run" and rec.status != "not_run"):
                by_index[rec.index] = rec
    records = tuple(by_index[i] for i in sorted(by_index))
    if base.status in ("error", "skipped") and not any(r.status != "not_run" for r in records):
        return base
    reports = [r.report if r.status != "not_run" else None for r in records]
    rr = summarize_repeats(
        reports, k=base.k, threshold=base.threshold, n_planned=base.n_planned, case_id=base.case_id
    )
    reason, code = "", None
    if rr.outcome == "failed":
        code = "below_threshold"
        from .errors import format_code

        d = rr.n_passed + rr.n_failed
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
