"""Calibrate a judge: run it over a golden set and measure agreement with the labels."""

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from ...assertions.judge import JudgeTask
from ..api import JudgeEngine
from ..kinds import get_kind
from .golden import GoldenError, GoldenItem, validate_items
from .metrics import binary_stats, bootstrap_ci, multiclass_stats, split_abstentions
from .report import CalibrationReport, gate_failures
from .split import split_of


def _positive(kind: str, override: str | None) -> str | None:
    spec = get_kind(kind)
    if spec is None:
        raise GoldenError(f"unknown judge kind {kind!r}")
    if spec.family == "multiclass":
        return None
    if override or spec.positive:
        return override or spec.positive
    from ..templates import get_template

    labels = get_template(kind).label_set
    for candidate in ("pass", "equivalent"):
        if candidate in labels:
            return candidate
    raise GoldenError(f"kind {kind!r} has no positive label; pass positive=")


def _ledger_warning(
    root: Path | None, kind: str, test_ids: Sequence[str], identity: str
) -> list[str]:
    """The test split is for the final measurement. Running it again on the same items under a
    *different* identity means someone is tuning against it."""
    if root is None:
        return []
    digest = hashlib.sha256("\n".join(sorted(test_ids)).encode()).hexdigest()[:16]
    path = root / kind / "_test_ledger.json"
    try:
        ledger: dict[str, list[str]] = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        ledger = {}
    seen = ledger.setdefault(digest, [])
    warn = []
    if identity not in seen:
        if seen:
            warn.append(
                f"the test split ({len(test_ids)} items) was already evaluated under {len(seen)} other "
                "judge identit(y/ies); tune on the validation split and keep test for the final run"
            )
        seen.append(identity)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(ledger, indent=2, sort_keys=True), encoding="utf-8")
    return warn


def calibrate_kind(
    engine: JudgeEngine,
    kind: str,
    items: Sequence[GoldenItem],
    *,
    positive: str | None = None,
    samples: int = 1,
    report_root: str | Path | None = None,
    needs_order_consistency: bool | None = None,
) -> CalibrationReport:
    """Run ``engine`` over every item of ``kind`` and build the report. Only human-labelled items
    in the test split can make it gate; draft and silver items are measured separately."""
    problems = validate_items([i for i in items if i.kind == kind], kind)
    if problems:
        raise GoldenError("; ".join(problems[:5]))
    spec = get_kind(kind)
    assert spec is not None
    pos = _positive(kind, positive)
    rows: list[tuple[GoldenItem, str, Any]] = []
    identities: set[str] = set()
    cost, tokens, errors = 0.0, 0, 0
    for item in items:
        if item.kind != kind:
            continue
        verdict = engine.judge(JudgeTask(kind, inputs=dict(item.inputs)))
        cost += verdict.cost
        tokens += verdict.tokens
        if verdict.identity_hash:
            identities.add(verdict.identity_hash)
        if verdict.status != "ok" or verdict.label is None:
            errors += 1
            rows.append((item, "unsure", verdict))
        else:
            rows.append((item, str(verdict.label), verdict))
    if len(identities) != 1:
        raise GoldenError(
            f"expected one judge identity over the golden set, saw {sorted(identities)}"
        )
    identity_hash = identities.pop()
    identity = next((dict(v.raw.get("identity") or {}) for _, _, v in rows if v.raw), {})

    human_test = [
        (i.label, p) for i, p, _ in rows if i.source == "human" and split_of(kind, i.id) == "test"
    ]
    other = [(i.label, p) for i, p, _ in rows if i.source != "human"]
    decided, abstain = split_abstentions(human_test)

    def stat(
        fn: Callable[[Sequence[tuple[str, str]]], Any], key: str
    ) -> Callable[[Sequence[tuple[str, str]]], float | None]:
        return lambda pairs: fn(pairs).get(key)

    if spec.family == "multiclass":
        stats = multiclass_stats(decided)
        ci = {"macro_f1": bootstrap_ci(decided, stat(multiclass_stats, "macro_f1")),
              "kappa": bootstrap_ci(decided, stat(multiclass_stats, "kappa"))}  # fmt: skip
        silver_stats: Any = multiclass_stats(split_abstentions(other)[0]) if other else {}
    else:
        assert pos is not None
        stats = binary_stats(decided, pos)
        ci = {k: bootstrap_ci(decided, stat(lambda p: binary_stats(p, pos), k))
              for k in ("tpr", "tnr", "kappa")}  # fmt: skip
        silver_stats = binary_stats(split_abstentions(other)[0], pos) if other else {}
    consistency_flags = [v.raw.get("order_consistent") for i, _, v in rows
                         if i.source == "human" and split_of(kind, i.id) == "test" and v.raw]  # fmt: skip
    known = [c for c in consistency_flags if c is not None]
    order_consistency = (sum(1 for c in known if c) / len(known)) if known else None
    needs_order = spec.orders > 1 if needs_order_consistency is None else needs_order_consistency
    human_counts: dict[str, int] = dict(Counter(t for t, _ in human_test))
    counts: dict[str, Any] = {
        "total": len(rows),
        "by_split": dict(Counter(split_of(kind, i.id) for i, _, _ in rows)),
        "by_source": dict(Counter(i.source for i, _, _ in rows)),
        "human_test": human_counts,
    }
    if spec.never_gates:
        failures = ["advisory kind: never gates"]
    else:
        failures = gate_failures(
            spec.family,
            stats,
            abstain_rate=abstain,
            human_counts=human_counts,
            positive=pos,
            order_consistency=order_consistency,
            needs_order_consistency=needs_order,
        )
    warnings: list[str] = []
    if errors:
        warnings.append(f"{errors} judge call(s) failed and were counted as abstentions")
    if report_root is not None:
        test_ids = [i.id for i, _, _ in rows if split_of(kind, i.id) == "test"]
        warnings += _ledger_warning(Path(report_root), kind, test_ids, identity_hash)
    return CalibrationReport(
        kind, identity_hash, identity, family=spec.family, positive=pos, samples=samples,
        gating_allowed=not failures, failures=tuple(failures), stats=stats, ci=ci,
        abstain_rate=abstain, order_consistency=order_consistency, counts=counts,
        silver=silver_stats, errors=errors, cost_usd=cost, tokens=tokens, warnings=tuple(warnings),
    )  # fmt: skip


def calibrate(
    kind: str,
    items: Sequence[GoldenItem],
    make_engine: Callable[[int], JudgeEngine],
    *,
    samples: Sequence[int] = (3, 1),
    positive: str | None = None,
    report_root: str | Path | None = None,
) -> list[CalibrationReport]:
    """One report per ``samples`` value: ``n=3`` and ``n=1`` are different identities, and a
    calibration of one never licenses the other."""
    reports = []
    for n in samples:
        engine = make_engine(n)
        try:
            reports.append(calibrate_kind(engine, kind, items, positive=positive, samples=n,
                                          report_root=report_root))  # fmt: skip
        finally:
            engine.close()
    return reports
