"""Turn an ``Expect`` into the list of checks to run, apply thresholds, and gate the outcome."""

import dataclasses
from dataclasses import dataclass
from typing import Any

from ..models import Expect
from .canonical import short_hash
from .codes import BENIGN_SKIP, UNCALIBRATED, UNVERIFIABLE, code_class
from .config import EvalConfig, ExpectConfigError
from .metrics.base import BaseMetric, MetricResult
from .metrics.registry import get_metric, list_metrics

ALIASES = {
    "tool_recall": "tool_selection.recall",
    "tool_precision": "tool_selection.precision",
    "tool_f1": "tool_selection.f1",
    "arg_score": "tool_args.arg_score",
    "convergence": "plan.convergence",
}


@dataclass(slots=True, frozen=True)
class PlannedCheck:
    metric: BaseMetric
    params_hash: str
    required: bool = True
    threshold: float | None = None
    sub_thresholds: tuple[tuple[str, float], ...] = ()
    weight: float = 1.0


def valid_threshold_keys() -> list[str]:
    keys: set[str] = set()
    for m in list_metrics():
        keys.add(m.name)
        keys.update(f"{m.name}.{s}" for s in m.sub_scores)
    keys.update(a for a, target in ALIASES.items() if target in keys)
    return sorted(keys)


def parse_min_scores(expect: Expect) -> dict[str, tuple[float | None, dict[str, float]]]:
    """``min_scores`` as ``{metric: (threshold, {sub: threshold})}``; unknown keys are errors."""
    valid = valid_threshold_keys()
    out: dict[str, tuple[float | None, dict[str, float]]] = {}
    for raw, value in sorted(expect.min_scores.items()):
        key = ALIASES.get(raw, raw)
        if raw not in valid and key not in valid:
            raise ExpectConfigError(f"unknown min_scores key {raw!r}; valid keys: {valid}")
        metric_name, _, sub = key.partition(".")
        metric = get_metric(metric_name)
        if not metric.thresholdable:
            raise ExpectConfigError(
                f"min_scores[{raw!r}]: {metric_name!r} is a yes/no check and has no threshold"
            )
        main, subs = out.get(metric_name, (None, {}))
        if sub:
            subs[sub] = float(value)
        else:
            main = float(value)
        out[metric_name] = (main, subs)
    return out


def validate_expect(expect: Expect, config: EvalConfig) -> None:
    """Everything that can be wrong with a test is found here, before any scoring."""
    from ..models import DateEq, PlanExpect, Predicate
    from ..models.matchers import walk_matchers
    from .matching.impls3b import _zone
    from .predicates import lookup_predicate

    sources: list[Any] = [*expect.answer_contains, *expect.answer_excludes]
    for t in expect.tools:
        sources.extend([t.args, t.defaults])
    for m in walk_matchers(tuple(sources)):
        if isinstance(m, Predicate):
            lookup_predicate(m.name, config.predicates)
        elif isinstance(m, DateEq) and m.tz:
            _zone(m.tz)
    if expect.intent_labels:
        missing = [i for i in expect.intents if i not in expect.intent_labels]
        if missing:
            raise ExpectConfigError(
                f"intent {missing} is not in intent_labels {list(expect.intent_labels)}"
            )
    plan = expect.plan
    if isinstance(plan, PlanExpect):
        for name in plan.constraints:
            lookup_predicate(name, config.predicates)
        wants_convergence = {"plan.convergence", "convergence"} & set(expect.min_scores)
        if wants_convergence and plan.optimal_tool_calls is None:
            raise ExpectConfigError(
                "min_scores for plan.convergence needs PlanExpect.optimal_tool_calls"
            )


def plan_checks(expect: Expect, config: EvalConfig) -> list[PlannedCheck]:
    validate_expect(expect, config)
    thresholds = parse_min_scores(expect)
    chosen: dict[str, BaseMetric] = {}
    for metric in list_metrics():
        if metric.applicable(expect, config):
            chosen[metric.name] = metric
    for name in config.extra_metrics:
        try:
            chosen[name] = get_metric(name)
        except KeyError as exc:
            raise ExpectConfigError(str(exc.args[0])) from None
    for name in thresholds:
        if name not in chosen:
            raise ExpectConfigError(
                f"min_scores names {name!r} but this Expect does not enable that check, so it "
                "would silently pass; set the field it reads or remove the key"
            )
    planned = []
    for metric in sorted(chosen.values(), key=lambda m: (m.rank, m.name)):
        main, subs = thresholds.get(metric.name, (None, {}))
        if config.strict and metric.thresholdable:
            main = 1.0
        planned.append(
            PlannedCheck(
                metric,
                short_hash(metric.params(expect, config)),
                True,
                main,
                tuple(sorted(subs.items())),
            )
        )
    return planned


def _skipped_threshold(
    result: MetricResult, check: PlannedCheck, missing: list[str]
) -> MetricResult:
    names = ", ".join(f"{result.name}.{m}" for m in missing)
    return dataclasses.replace(
        result,
        status="skipped",
        value=None,
        passed=None,
        reason_code="data_unavailable",
        reason=f"min_scores names {names} but that sub-score was not measured",
        details={
            **dict(result.details),
            "thresholds": {"main": check.threshold, "subs": dict(check.sub_thresholds)},
        },
    )


def apply_threshold(result: MetricResult, check: PlannedCheck) -> MetricResult:
    """Recompute ``passed`` against a threshold. A threshold can never clear a veto, and a
    threshold on sub-scores alone clears only a failure those sub-scores measure."""
    if result.status != "ok" or not check.metric.thresholdable:
        return result
    if check.threshold is None and not check.sub_thresholds:
        return result
    ok = True
    reasons: list[str] = []
    if check.threshold is not None:
        value = result.value if result.value is not None else 0.0
        if value < check.threshold:
            ok = False
            reasons.append(f"{result.name}={value:.2f} < {check.threshold:.2f}")
    scores = result.sub_scores
    missing: list[str] = []
    for sub, limit in check.sub_thresholds:
        got = scores.get(sub)
        if got is None:
            missing.append(sub)
        elif float(got) < limit:
            ok = False
            reasons.append(f"{result.name}.{sub}={float(got):.2f} < {limit:.2f}")
    if missing and ok:
        # A threshold on a sub-score that was not measured is absent evidence: it can neither pass
        # nor fail the check. An already-failed result stays failed (the threshold must not
        # overwrite it with a pass); otherwise the check is incomplete.
        if result.passed is False:
            return result
        return _skipped_threshold(result, check, missing)
    cleared = (
        result.passed is not False
        or check.threshold is not None
        or check.metric.failure_within(result, frozenset(s for s, _ in check.sub_thresholds))
    )
    passed = ok and cleared and not result.vetoed
    code = None if passed else (result.reason_code or "threshold.below")
    if not ok:
        code = "threshold.below"
    applied = {"main": check.threshold, "subs": dict(check.sub_thresholds)}
    return dataclasses.replace(
        result,
        details={**dict(result.details), "thresholds": applied},
        passed=passed,
        threshold=check.threshold if check.threshold is not None else result.threshold,
        reason_code=code,
        reason="; ".join(reasons) if reasons else ("" if passed else result.reason),
    )


@dataclass(slots=True, frozen=True)
class Gate:
    outcome: str
    failed: bool
    incomplete: bool
    warnings: tuple[str, ...]
    n_passed: int
    n_failed: int
    n_skipped: int
    n_error: int
    n_incomplete: int


def gate(results: list[MetricResult], harness_failed: bool, config: EvalConfig) -> Gate:
    """AND gating: failure dominates incompleteness; absent evidence is never a pass."""
    failed, incomplete = harness_failed, False
    warnings: list[str] = []
    n_pass = n_fail = n_skip = n_err = n_inc = 0
    on_unavailable = "fail" if config.strict else config.on_unavailable
    for r in results:
        if r.status == "ok":
            if r.passed is False:
                n_fail += 1
                if r.required:
                    failed = True
            else:
                n_pass += 1
        elif r.status == "skipped":
            n_skip += 1
            cls = code_class(r.reason_code)
            if cls == BENIGN_SKIP:
                continue
            if cls == UNCALIBRATED:
                if config.strict or config.on_uncalibrated == "incomplete":
                    incomplete = r.required or incomplete
                    n_inc += 1
                else:
                    kind = r.details.get("judge_kind", "?")
                    label = r.details.get("advisory_label", "?")
                    warnings.append(
                        f"{r.name} not gated: judge {kind} uncalibrated (advisory label: {label})"
                    )
                continue
            if cls == UNVERIFIABLE or cls not in (BENIGN_SKIP,):
                if on_unavailable == "fail":
                    incomplete = incomplete or r.required
                    n_inc += 1
                elif on_unavailable == "warn":
                    warnings.append(f"{r.name}: {r.reason_code} ({r.reason})")
        else:
            n_err += 1
            incomplete = incomplete or r.required
            n_inc += 1
    outcome = "failed" if failed else "incomplete" if incomplete else "passed"
    return Gate(outcome, failed, incomplete, tuple(warnings), n_pass, n_fail, n_skip, n_err, n_inc)


def flat_scores(results: list[MetricResult]) -> dict[str, Any]:
    scores: dict[str, Any] = {}
    for r in results:
        if r.status != "ok" or r.value is None:
            continue
        scores[r.name] = r.value
        for sub, value in r.sub_scores.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                scores[f"{r.name}.{sub}"] = value
    return scores
