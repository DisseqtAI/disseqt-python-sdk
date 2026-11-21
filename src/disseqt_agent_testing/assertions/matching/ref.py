"""``Ref`` resolution: an argument must equal a value taken from an earlier call, the user input
or a fake. Two phases (see plan 03 section 8): provisional (every name-matching earlier call is a
candidate) and rebind (only the calls the assignment bound to the source expectation)."""

import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from typing import Any

from ...models import AgentRun, FakeSpec, Message, Ref, Step, parse_ref_source, thaw
from ..canonical import path_str
from ..config import ExpectConfigError
from .registry import MatchContext, MatchResult, Segments, failed, passed


@dataclass(slots=True, frozen=True)
class RefOutcome:
    values: tuple[Any, ...] = ()
    resolved_from: str = ""
    reason: str | None = None
    text: str | None = None  # for ``input`` refs: the searchable user text
    error: str | None = None  # an instrument error (ref_unresolvable)


def _walk(value: Any, path: Sequence[str | int]) -> tuple[bool, Any]:
    for segment in path:
        if isinstance(segment, int):
            if isinstance(value, (list, tuple)) and 0 <= segment < len(value):
                value = value[segment]
            else:
                return False, None
        else:
            if isinstance(value, Mapping) and segment in value:
                value = value[segment]
            else:
                return False, None
    return True, value


def user_texts(run: AgentRun) -> tuple[str, ...]:
    """The user's text per turn: turn 0 is the input, later turns are the follow-ups."""
    first: list[str] = []
    if isinstance(run.input, str):
        first.append(run.input)
    else:
        first.extend(m.content for m in run.input if isinstance(m, Message) and m.role == "user")
    return ("\n".join(first), *run.follow_ups)


@dataclass(slots=True)
class RefResolver:
    """Resolves sources for one evaluation. ``bound`` is ``None`` during the provisional phase."""

    run: AgentRun
    calls: tuple[Step, ...]
    expect_ids: Mapping[str, str]  # expect id -> tool name glob
    expect_turns: Mapping[str, int | None]
    fakes: tuple[FakeSpec, ...] = ()
    bound: Mapping[str, tuple[Step, ...]] | None = None
    _cache: dict[tuple[str, int], RefOutcome] = field(default_factory=dict)

    def candidates(self, target: str) -> tuple[list[Step], str]:
        """The calls a call-target stands for, and a label for the message."""
        if target in self.expect_ids:
            if self.bound is not None:
                return list(self.bound.get(target, ())), f"expectation {target!r}"
            name, turn = self.expect_ids[target], self.expect_turns.get(target)
            steps = [
                s
                for s in self.calls
                if fnmatchcase(s.name, name) and (turn is None or s.turn == turn)
            ]
            return steps, f"expectation {target!r}"
        if target.startswith("call-") and target[5:].isdigit():
            return [s for s in self.calls if s.call_id == target], target
        return [s for s in self.calls if fnmatchcase(s.name, target)], f"tool {target!r}"

    def resolve(self, source: str, consumer: Step | None) -> RefOutcome:
        kind, target, path = parse_ref_source(source)
        if kind == "input":
            texts = user_texts(self.run)
            if target is not None:
                turn = int(target)
                text = texts[turn] if turn < len(texts) else ""
            else:
                upto = consumer.turn if consumer is not None else len(texts) - 1
                text = "\n".join(texts[: upto + 1])
            return RefOutcome(text=text, resolved_from=source)
        if kind == "fake":
            return self._fake(source, target or "", path)
        return self._call(source, target or "", path, consumer)

    def _fake(self, source: str, name: str, path: tuple[str | int, ...]) -> RefOutcome:
        if not self.fakes:
            return RefOutcome(error="no fakes were supplied to evaluate()", resolved_from=source)
        spec = next((f for f in self.fakes if f.name == name), None)
        if spec is None:
            return RefOutcome(error=f"no fake named {name!r}", resolved_from=source)
        responses = [r for r in (*spec.responses, spec.default) if r is not None]
        values = []
        for response in responses:
            found, value = _walk(thaw(response.returns), path)
            if found:
                values.append(value)
        if not values:
            return RefOutcome(reason="path missing", resolved_from=f"fake {name!r}")
        return RefOutcome(values=tuple(values), resolved_from=f"fake {name!r}")

    def _call(
        self, source: str, target: str, path: tuple[str | int, ...], consumer: Step | None
    ) -> RefOutcome:
        steps, label = self.candidates(target)
        if not steps:
            return RefOutcome(reason="source not matched", resolved_from=label)
        limit = consumer.seq if consumer is not None else None
        before = [s for s in steps if limit is None or s.seq < limit]
        if not before:
            return RefOutcome(reason="source after use", resolved_from=label)
        part, rest = path[0], path[1:]
        values: list[Any] = []
        for step in before:
            base = step.result if part == "result" else step.args
            found, value = _walk(thaw(base), rest)
            if found:
                values.append(value)
        if not values:
            return RefOutcome(reason="path missing", resolved_from=label)
        return RefOutcome(values=tuple(values), resolved_from=label)


# Scripts written without spaces between words (Thai, Lao, Myanmar, Khmer; CJK is found by width).
_UNSPACED = ((0x0E00, 0x0EFF), (0x1000, 0x109F), (0x1780, 0x17FF))


def _joins(ch: str) -> bool:
    """Whether ``ch`` next to a value makes that value part of a longer token."""
    if unicodedata.category(ch)[0] not in "LMN":
        return False
    if unicodedata.east_asian_width(ch) in ("W", "F"):
        return False
    return not any(low <= ord(ch) <= high for low, high in _UNSPACED)


def _fold(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold()


def occurs_as_token(value: str, text: str) -> bool:
    """``value`` occurs in ``text`` as whole tokens: ``A12`` is in ``refund order A12.``, but
    ``1`` and ``a`` are not. A decimal point between digits is inside a token, so ``59`` is not in
    ``59.99``."""
    needle, hay = _fold(value).strip(), _fold(text)
    if not needle:
        return False
    start = hay.find(needle)
    while start != -1:
        end = start + len(needle)
        if not _extends(hay, start, -1, needle[0]) and not _extends(hay, end - 1, 1, needle[-1]):
            return True
        start = hay.find(needle, start + 1)
    return False


def _extends(hay: str, edge: int, direction: int, edge_char: str) -> bool:
    """Whether the token that holds ``hay[edge]`` continues past it in ``direction``."""
    if not _joins(edge_char):
        return False
    near = edge + direction
    if not 0 <= near < len(hay):
        return False
    if _joins(hay[near]):
        return True
    far = near + direction
    return edge_char.isdigit() and hay[near] == "." and 0 <= far < len(hay) and hay[far].isdigit()


def match_ref(ref: Ref, actual: Any, path: Segments, ctx: MatchContext) -> list[MatchResult]:
    from .impls import deep_equal

    shown = f"Ref({ref.source!r})"
    if ctx.refs is None:
        return [
            MatchResult(
                "error", None, 0.0, "ref_unresolvable", path, shown, actual, "no Ref resolver"
            )
        ]
    outcome = ctx.refs.resolve(ref.source, ctx.consumer)
    if outcome.error is not None:
        return [
            MatchResult("error", None, 0.0, "ref_unresolvable", path, shown, actual, outcome.error)
        ]
    details = {"source": ref.source, "resolved_from": outcome.resolved_from}
    if outcome.text is not None:
        scalar = actual is not None and not isinstance(actual, (dict, list, tuple))
        if scalar and occurs_as_token(str(actual), outcome.text):
            return [passed(path, shown, actual)]
        return [
            failed(
                path,
                shown,
                actual,
                "ungrounded_value",
                "value does not occur in the user input",
                {**details, "expected_value": None, "reason": "value differs"},
            )
        ]
    if outcome.reason is not None:
        return [
            failed(
                path,
                shown,
                actual,
                "ungrounded_value",
                outcome.reason,
                {**details, "expected_value": None, "reason": outcome.reason},
            )
        ]
    for value in outcome.values:
        if deep_equal(value, actual)[0]:
            return [passed(path, shown, actual)]
    expected_value = outcome.values[0] if len(outcome.values) == 1 else list(outcome.values)
    return [
        failed(
            path,
            f"{shown} = {expected_value!r}",
            actual,
            "ungrounded_value",
            "value differs",
            {**details, "expected_value": expected_value, "reason": "value differs"},
        )
    ]


def ref_sources(matchers: Any) -> list[str]:
    """Every Ref source inside a matcher map (used for cycle detection)."""
    from ...models.matchers import walk_matchers

    return [m.source for m in walk_matchers(matchers) if isinstance(m, Ref)]


def check_ref_cycles(deps: Mapping[str, set[str]], describe: Callable[[str], str] = str) -> None:
    """``deps[consumer]`` = the expectation ids its Refs read; a cycle is a test-authoring bug."""
    from graphlib import CycleError, TopologicalSorter

    try:
        tuple(TopologicalSorter({k: set(v) for k, v in deps.items()}).static_order())
    except CycleError as exc:
        cycle = " -> ".join(describe(x) for x in exc.args[1])
        raise ExpectConfigError(f"Ref dependency cycle: {cycle}") from None


__all__ = ["RefOutcome", "RefResolver", "match_ref", "path_str", "check_ref_cycles", "ref_sources"]
