"""Matcher semantics for phase 3b: Normalized, TypeIs, Unordered, Contains, DictMatch, DateEq,
Predicate and Semantic."""

import unicodedata
from datetime import timezone, tzinfo
from fnmatch import fnmatchcase
from typing import Any

from ...models import (
    Contains,
    DateEq,
    DictMatch,
    Eq,
    Normalized,
    Predicate,
    Semantic,
    TypeIs,
    Unordered,
    thaw,
)
from ...models.matchers import parse_iso
from ..canonical import clip
from ..config import ExpectConfigError
from ..judge import JudgeTask, interpret_verdict
from ..predicates import lookup_predicate
from .impls import is_number, json_type
from .registry import (
    MatchContext,
    MatchResult,
    Segments,
    describe,
    failed,
    match_dict,
    match_value,
    passed,
    register_matcher_impl,
)


def _norm(text: str, strip: str, fold: bool) -> str:
    text = unicodedata.normalize("NFC", text)
    text = "".join(ch for ch in text if ch not in strip)
    return text.casefold() if fold else text


def _normalized(m: Normalized, actual: Any, path: Segments, ctx: MatchContext) -> list[MatchResult]:
    shown = describe(m)
    if not isinstance(actual, str):
        return [failed(path, shown, actual, f"type_error:string->{json_type(actual)}")]
    got = _norm(actual, m.strip_chars, m.casefold)
    if any(got == _norm(v, m.strip_chars, m.casefold) for v in m.values):
        return [passed(path, shown, actual)]
    return [failed(path, shown, actual, "value_error:normalized", f"expected {shown}")]


def _type_ok(name: str, value: Any) -> bool:
    if name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if name == "number":
        return is_number(value)
    return json_type(value) == name


def _type_is(m: TypeIs, actual: Any, path: Segments, ctx: MatchContext) -> list[MatchResult]:
    shown = describe(m)
    if any(_type_ok(t, actual) for t in m.types):
        return [passed(path, shown, actual)]
    return [failed(path, shown, actual, f"type_error:{'|'.join(m.types)}->{json_type(actual)}")]


def _element(
    matcher: Any, element: Any, ctx: MatchContext, path: Segments
) -> tuple[bool | None, list[MatchResult]]:
    """``True`` / ``False`` when the element was decided, else ``None`` with the skipped or error
    leaves that kept it from being decided. A real mismatch outweighs an undecided leaf."""
    results = match_value(matcher, element, ctx, path)
    if any(r.status == "ok" and r.ok is False for r in results):
        return False, []
    undecided = [r for r in results if r.status != "ok"]
    return (None, undecided) if undecided else (True, [])


def _unmeasured(
    undecided: list[MatchResult], path: Segments, shown: str, actual: Any
) -> list[MatchResult]:
    """One leaf for a container whose outcome hangs on leaves that could not be measured: an
    error if any of them errored, otherwise skipped, with the first such leaf's code."""
    errors = [r for r in undecided if r.status == "error"]
    first = (errors or undecided)[0]
    return [
        MatchResult(
            first.status, None, first.score, first.code, path, shown, actual, first.reason,
            any(r.judged for r in undecided), first.details,
        )
    ]  # fmt: skip


def _matching(
    weights: list[list[float]], n_elements: int
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    from ..assignment.hungarian import assign_max

    pairs = (
        [(i, j) for i, j in assign_max(weights) if weights[i][j] > 0]
        if weights and n_elements
        else []
    )
    used_items = {i for i, _ in pairs}
    used_elems = {j for _, j in pairs}
    missing = [i for i in range(len(weights)) if i not in used_items]
    extra = [j for j in range(n_elements) if j not in used_elems]
    return pairs, missing, extra


def _multiset(
    items: tuple[Any, ...], elements: list[Any], ctx: MatchContext, path: Segments
) -> tuple[list[int], list[int], list[int], list[int], list[MatchResult]]:
    """Match items to distinct elements by maximum bipartite matching (never ``sorted()``).
    Returns the missing items and extra elements counting decided matches only, the same two
    lists when every undecided cell is given the benefit of the doubt, and the undecided leaves.
    """
    cells = [
        [_element(item, el, ctx, (*path, j)) for j, el in enumerate(elements)] for item in items
    ]
    sure = [[1.0 if ok is True else 0.0 for ok, _ in row] for row in cells]
    maybe = [[0.0 if ok is False else 1.0 for ok, _ in row] for row in cells]
    undecided = [leaf for row in cells for _, leaves in row for leaf in leaves]
    _, missing, extra = _matching(sure, len(elements))
    if not undecided:
        return missing, extra, missing, extra, []
    _, could_miss, could_extra = _matching(maybe, len(elements))
    return missing, extra, could_miss, could_extra, undecided


def _unordered(m: Unordered, actual: Any, path: Segments, ctx: MatchContext) -> list[MatchResult]:
    shown = describe(m)
    if not isinstance(actual, (list, tuple)):
        return [failed(path, shown, actual, f"type_error:array->{json_type(actual)}")]
    missing, extra, could_miss, could_extra, undecided = _multiset(m.items, list(actual), ctx, path)
    if not missing and not extra:
        return [passed(path, shown, actual)]
    if undecided and not could_miss and not could_extra:
        return _unmeasured(undecided, path, shown, actual)
    return [
        failed(
            path, shown, actual, "value_error:unordered",
            f"{len(missing)} expected item(s) missing, {len(extra)} unexpected item(s)",
            {"missing": [describe(m.items[i]) for i in missing], "extra": [clip(actual[j]) for j in extra]},
        )
    ]  # fmt: skip


def _contains(m: Contains, actual: Any, path: Segments, ctx: MatchContext) -> list[MatchResult]:
    shown = describe(m)
    if isinstance(actual, str):
        folded = actual.casefold()
        missing = []
        unsure: list[MatchResult] = []
        for item in m.items:
            if isinstance(item, Eq) and isinstance(item.value, str):
                if item.value.casefold() not in folded:
                    missing.append(item.value)
                continue
            ok, leaves = _element(item, actual, ctx, path)
            unsure.extend(leaves)
            if ok is False:
                missing.append(describe(item))
        if unsure and not missing:
            return _unmeasured(unsure, path, shown, actual)
        if not missing:
            return [passed(path, shown, actual)]
        return [
            failed(
                path,
                shown,
                actual,
                "value_error:contains",
                f"missing {missing}",
                {"missing": missing},
            )
        ]
    if not isinstance(actual, (list, tuple)):
        return [failed(path, shown, actual, f"type_error:array->{json_type(actual)}")]
    missing_i, _, could_miss, _, undecided = _multiset(m.items, list(actual), ctx, path)
    if not missing_i:
        return [passed(path, shown, actual)]
    if undecided and not could_miss:
        return _unmeasured(undecided, path, shown, actual)
    names = [describe(m.items[i]) for i in missing_i]
    return [
        failed(path, shown, actual, "value_error:contains", f"missing {names}", {"missing": names})
    ]


def _dict_match(m: DictMatch, actual: Any, path: Segments, ctx: MatchContext) -> list[MatchResult]:
    shown = describe(m)
    if not isinstance(actual, dict):
        return [failed(path, shown, actual, f"type_error:object->{json_type(actual)}")]
    body = {k: v for k, v in actual.items() if not any(fnmatchcase(k, g) for g in m.ignore)}
    undecided: list[MatchResult] = []
    for key, matcher in m.defaults.items():
        if key in body and key not in m.fields:
            ok, leaves = _element(matcher, body[key], ctx, (*path, key))
            if ok is not False:  # at its default, or not measurable: never an unexpected key
                undecided.extend(leaves)
                del body[key]
    spec = {k: v for k, v in m.fields.items() if not any(fnmatchcase(k, g) for g in m.ignore)}
    return [*match_dict(spec, body, ctx, path, exact=m.mode == "exact"), *undecided]


def _zone(name: str | None) -> tzinfo:
    if name is None:
        return timezone.utc
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo(name)
    except Exception as exc:
        raise ExpectConfigError(
            f"DateEq(tz={name!r}) could not load the time zone ({exc}); install the 'tzdata' "
            "package on platforms without system zone data"
        ) from None


def _date_eq(m: DateEq, actual: Any, path: Segments, ctx: MatchContext) -> list[MatchResult]:
    shown = describe(m)
    if not isinstance(actual, str):
        return [failed(path, shown, actual, f"type_error:string->{json_type(actual)}")]
    try:
        got = parse_iso(actual)
    except ValueError:
        return [
            failed(path, shown, actual, "value_error:date_parse", "not an ISO date or datetime")
        ]
    zone = _zone(m.tz)
    want = parse_iso(m.value)
    if "T" not in m.value and " " not in m.value:  # a date-only expectation compares dates
        if got.tzinfo is not None:
            got = got.astimezone(zone)
        same = got.date() == want.date()
        return [
            passed(path, shown, actual) if same else failed(path, shown, actual, "value_error:date")
        ]
    if (got.tzinfo is None) != (want.tzinfo is None):
        if m.tz is None:
            return [
                failed(path, shown, actual, "type_error:naive_aware", "naive vs timezone-aware")
            ]
        got = got.replace(tzinfo=zone) if got.tzinfo is None else got
        want = want.replace(tzinfo=zone) if want.tzinfo is None else want
    elif got.tzinfo is None and m.tz is not None:
        got, want = got.replace(tzinfo=zone), want.replace(tzinfo=zone)
    return [
        (
            passed(path, shown, actual)
            if got == want
            else failed(path, shown, actual, "value_error:date")
        )
    ]


def _predicate(m: Predicate, actual: Any, path: Segments, ctx: MatchContext) -> list[MatchResult]:
    shown = describe(m)
    fn = lookup_predicate(m.name, ctx.predicates)
    try:
        out = fn(thaw(actual))
    except Exception as exc:
        return [
            MatchResult(
                "error",
                None,
                0.0,
                "predicate_error",
                path,
                shown,
                actual,
                f"{type(exc).__name__}: {exc}",
            )
        ]
    if isinstance(out, MatchResult):
        return [out]
    reason = ""
    if isinstance(out, tuple) and len(out) == 2 and isinstance(out[0], bool):
        out, reason = out
    if isinstance(out, bool):
        if out:
            return [passed(path, shown, actual)]
        return [
            failed(
                path,
                shown,
                actual,
                "value_error:predicate",
                reason or f"predicate {m.name} returned False",
            )
        ]
    return [
        MatchResult(
            "error", None, 0.0, "predicate_error", path, shown, actual,
            f"predicate {m.name} must return bool, (bool, reason) or MatchResult, got {type(out).__name__}",
        )
    ]  # fmt: skip


SEMANTIC_POSITIVE = ("equivalent",)
SEMANTIC_NEGATIVE = ("not_equivalent",)


def _semantic(m: Semantic, actual: Any, path: Segments, ctx: MatchContext) -> list[MatchResult]:
    from ..judge import call_record
    from ..judge_inputs import conversation, tool_schemas

    shown = describe(m)
    judging = ctx.judge_phase and ctx.judge is not None and ctx.use_judge
    if not judging:  # no judge, or still scoring candidate pairs (phase C judges matched pairs)
        return [
            MatchResult(
                "skipped", None, 1.0, "judge_unavailable", path, shown, actual, "needs a judge"
            )
        ]
    tool = ctx.consumer.name if ctx.consumer is not None else ""
    turn = ctx.consumer.turn if ctx.consumer is not None else 0
    inputs: dict[str, Any] = {
        "reference": m.reference, "actual": actual, "rubric": m.rubric, "tool": tool,
        "path": list(path),
    }  # fmt: skip
    if ctx.run is not None:
        inputs["conversation"] = conversation(ctx.run, turn)
        info = next((t for t in ctx.run.available_tools if t.name == tool), None)
        if info is not None and info.description:
            inputs["tool_description"] = info.description
        schema = tool_schemas(ctx.run).get(tool)
        props = (schema or {}).get("properties", {}) if isinstance(schema, dict) else {}
        if path and isinstance(path[0], str) and path[0] in props:
            inputs["param_schema"] = props[path[0]]
    task = JudgeTask("semantic_arg", inputs=inputs)
    verdict = ctx.judge.judge(task)
    if isinstance(ctx.judge_log, list):
        ctx.judge_log.append(call_record(verdict, "tool_args", task.kind))
    interp = interpret_verdict(verdict, SEMANTIC_POSITIVE, SEMANTIC_NEGATIVE)
    why = str(interp.details.get("reason", ""))
    if interp.status == "error":
        return [
            MatchResult("error", None, 0.0, interp.reason_code, path, shown, actual, why or "judge failed", True)
        ]  # fmt: skip
    if interp.status == "skipped":
        return [
            MatchResult(
                "skipped", None, 1.0, interp.reason_code, path, shown, actual, why or "judge unsure", True,
                {k: v for k, v in interp.details.items() if k != "reason"},
            )
        ]  # fmt: skip
    if interp.passed:
        return [MatchResult("ok", True, 1.0, None, path, shown, actual, judged=True)]
    return [
        MatchResult(
            "ok", False, 0.0, "value_error:semantic", path, shown, actual, f"expected {shown}", True
        )
    ]


def register_builtin_impls_3b() -> None:
    reg = register_matcher_impl
    reg(
        Normalized,
        _normalized,
        describe=lambda m: f"Normalized({clip(list(m.values))})",
        override=True,
    )
    reg(TypeIs, _type_is, describe=lambda m: f"TypeIs({'|'.join(m.types)})", override=True)
    reg(Unordered, _unordered, describe=lambda m: f"Unordered({len(m.items)} items)", override=True)
    reg(
        Contains,
        _contains,
        describe=lambda m: f"Contains({', '.join(describe(i) for i in m.items)})",
        override=True,
    )
    reg(
        DictMatch,
        _dict_match,
        describe=lambda m: f"DictMatch({', '.join(sorted(m.fields))})",
        override=True,
    )
    reg(
        DateEq,
        _date_eq,
        describe=lambda m: f"DateEq({m.value!r}" + (f", tz={m.tz!r})" if m.tz else ")"),
        override=True,
    )
    reg(Predicate, _predicate, describe=lambda m: f"Predicate({m.name!r})", override=True)
    reg(Semantic, _semantic, describe=lambda m: f"Semantic({clip(m.reference)})", override=True)


register_builtin_impls_3b()
