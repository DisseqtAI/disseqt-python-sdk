"""One shared, lexicographic optimal assignment between expected tool slots and recorded calls."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from fractions import Fraction
from typing import Any

from ...models import AgentRun, FakeSpec, Step, ToolExpect, parse_ref_source
from ..config import EvalConfig
from ..matching import MatchContext, MatchResult, match_dict
from ..matching.ref import RefResolver, check_ref_cycles, ref_sources
from .backends import get_backend, register_assignment_backend, unregister_assignment_backend
from .hungarian import assign_max

CALL_KINDS = ("tool", "subagent")
POSITION_EPS = 1e-7
SCORE_TOL = 1e-12


def calls_of(run: AgentRun) -> tuple[Step, ...]:
    """Tool and sub-agent steps in ``seq`` order. MODEL steps never take part."""
    return tuple(s for s in run.steps if s.kind in CALL_KINDS)


def expect_id(te: ToolExpect, index: int) -> str:
    return te.id or f"{te.name}#{index}"


@dataclass(slots=True, frozen=True)
class Slot:
    index: int
    expect_index: int
    expect_id: str
    expect: ToolExpect
    required: bool


@dataclass(slots=True, frozen=True)
class PairEval:
    score: float
    leaves: tuple[MatchResult, ...]
    incomplete: bool = False
    errored: bool = False
    feasible: bool = True

    @property
    def failed_leaves(self) -> tuple[MatchResult, ...]:
        return tuple(r for r in self.leaves if r.status == "ok" and r.ok is False)


@dataclass(slots=True, frozen=True)
class MatchedPair:
    slot: Slot
    step: Step
    call_index: int
    eval: PairEval


@dataclass(slots=True, frozen=True)
class CallAssignment:
    slots: tuple[Slot, ...]
    calls: tuple[Step, ...]
    pairs: tuple[MatchedPair, ...]
    missing_required: tuple[Slot, ...]
    unmatched_calls: tuple[Step, ...]
    _bound: Mapping[str, tuple[Step, ...]] = field(default_factory=dict)

    def pair(self, slot_index: int, call_index: int) -> MatchedPair | None:
        for p in self.pairs:
            if p.slot.index == slot_index and p.call_index == call_index:
                return p
        return None

    def bound(self, expect_id: str) -> tuple[Step, ...]:
        """The calls assigned to an expectation, in ``seq`` order."""
        return self._bound.get(expect_id, ())

    @property
    def n_required(self) -> int:
        return sum(1 for s in self.slots if s.required)


def name_ok(te: ToolExpect, step: Step) -> bool:
    return fnmatchcase(step.name, te.name) and (te.turn is None or step.turn == te.turn)


def build_slots(tools: Sequence[ToolExpect], calls: Sequence[Step]) -> tuple[Slot, ...]:
    """Declared order; a count expands to its required slots first, then optional ones."""
    slots: list[Slot] = []
    for ei, te in enumerate(tools):
        eid = expect_id(te, ei)
        count = te.count
        low, high = count if isinstance(count, tuple) else (1, 1)
        n_match = sum(1 for c in calls if name_ok(te, c))
        optional = max(n_match - low, 0) if high is None else max(high - low, 0)
        for _ in range(low):
            slots.append(Slot(len(slots), ei, eid, te, True))
        for _ in range(optional):
            slots.append(Slot(len(slots), ei, eid, te, False))
    return tuple(slots)


def _strip(
    args: Mapping[str, Any], te: ToolExpect, ctx: MatchContext
) -> tuple[dict[str, Any], list[MatchResult]]:
    """Drop ignored keys and keys still at their default. A default whose matcher could not be
    decided is dropped too and its skipped/error leaves are returned, so it stays unmeasured
    instead of turning into an ``unexpected_param`` failure."""
    from ..matching import match_value

    out = dict(args)
    undecided: list[MatchResult] = []
    for key in list(out):
        if any(fnmatchcase(key, g) for g in te.ignore):
            del out[key]
    for key, matcher in te.defaults.items():
        if key in out and key not in te.args:
            results = match_value(matcher, out[key], ctx, (key,))
            if any(r.status == "ok" and r.ok is False for r in results):
                continue
            undecided.extend(r for r in results if r.status != "ok")
            del out[key]
    return out, undecided


def pair_eval(te: ToolExpect, step: Step, ctx: MatchContext, min_pair_score: float) -> PairEval:
    ctx.consumer = step
    actual, undecided = _strip(step.args, te, ctx)
    spec = {k: v for k, v in te.args.items() if not any(fnmatchcase(k, g) for g in te.ignore)}
    leaves = [*match_dict(spec, actual, ctx, (), exact=te.args_mode == "exact"), *undecided]
    decided = [r.score for r in leaves if r.status == "ok"]
    score = sum(decided) / len(decided) if decided else 1.0
    incomplete = any(r.status == "skipped" for r in leaves)
    errored = any(r.status == "error" for r in leaves)
    return PairEval(score, tuple(leaves), incomplete, errored, score >= min_pair_score)


def _has_semantic(te: ToolExpect) -> bool:
    from ...models import Semantic
    from ...models.matchers import walk_matchers

    return any(isinstance(m, Semantic) for m in walk_matchers(te.args))


def _has_expect_ref(te: ToolExpect, ids: Mapping[str, str]) -> bool:
    for source in ref_sources(te.args):
        kind, target, _ = parse_ref_source(source)
        if kind == "call" and target in ids:
            return True
    return False


def check_ref_graph(tools: Sequence[ToolExpect]) -> None:
    """Plan-time: a Ref between expectations must not form a cycle."""
    ids = {expect_id(te, i): i for i, te in enumerate(tools)}
    deps: dict[str, set[str]] = {}
    for i, te in enumerate(tools):
        eid = expect_id(te, i)
        deps.setdefault(eid, set())
        for source in ref_sources(te.args):
            kind, target, _ = parse_ref_source(source)
            if kind == "call" and target in ids:
                deps[eid].add(target)
    check_ref_cycles(deps)


Pairing = list[tuple[int, int]]


def _canonical(
    pairs: Pairing, slots: Sequence[Slot], n_calls: int, evals: Mapping[tuple[int, int], PairEval]
) -> Pairing:
    """The canonical form of an optimal pairing, so the result does not depend on how a solver
    breaks ties. Slots of one expectation are interchangeable: its calls go to its first slots
    (required ones come first) in ``seq`` order. Between expectations a change is taken only when
    it keeps the scores and the number of matched required slots and follows declared order more
    closely (exact arithmetic, the lower indices winning an exact tie): two expectations trade
    calls, or a call moves to another expectation, or an expectation takes a free call instead.
    """
    n = len(slots)
    members: dict[int, list[int]] = {}
    for slot in slots:
        members.setdefault(slot.expect_index, []).append(slot.index)

    def compact(current: Pairing) -> Pairing:
        held: dict[int, list[int]] = {}
        for i, j in current:
            held.setdefault(slots[i].expect_index, []).append(j)
        return sorted(
            pair for e, js in held.items() for pair in zip(members[e], sorted(js), strict=False)
        )

    def worth(current: Pairing) -> tuple[int, float] | None:
        if any(pair not in evals for pair in current):
            return None
        required = sum(1 for i, _ in current if slots[i].required)
        return required, sum(evals[pair].score for pair in current)

    def key(current: Pairing) -> tuple[Fraction, Pairing]:
        total = Fraction(0)
        for i, j in current:
            frac_i = Fraction(i, n - 1) if n > 1 else Fraction(0)
            frac_j = Fraction(j, n_calls - 1) if n_calls > 1 else Fraction(0)
            total += (frac_i - frac_j) ** 2
        return total, current

    def neighbours(current: Pairing) -> list[Pairing]:
        out: list[Pairing] = []
        used_slots = {i for i, _ in current}
        used_calls = {j for _, j in current}
        free_calls = [j for j in range(n_calls) if j not in used_calls]
        free_slot = {
            e: next(i for i in slot_ids if i not in used_slots)
            for e, slot_ids in members.items()
            if any(i not in used_slots for i in slot_ids)
        }
        for a, (i, j) in enumerate(current):
            rest = current[:a] + current[a + 1 :]
            out.extend([*rest, (i, j2)] for j2 in free_calls)
            out.extend(
                [*rest, (i2, j)] for e, i2 in free_slot.items() if e != slots[i].expect_index
            )
            for b in range(a + 1, len(current)):
                i2, j2 = current[b]
                if slots[i2].expect_index != slots[i].expect_index:
                    out.append([*rest[: b - 1], *rest[b:], (i, j2), (i2, j)])
        return out

    best = compact(pairs)
    target = worth(best)
    improved = True
    while improved and target is not None:  # the key strictly decreases over a finite set
        improved = False
        for candidate in map(compact, neighbours(best)):
            value = worth(candidate)
            if value is None or value[0] != target[0] or abs(value[1] - target[1]) > SCORE_TOL:
                continue
            if key(candidate) < key(best):
                best, improved = candidate, True
                break
    return best


def assign(
    tools: Sequence[ToolExpect],
    run: AgentRun,
    config: EvalConfig,
    fakes: tuple[FakeSpec, ...] = (),
    *,
    judge: Any = None,
    judge_log: list[dict[str, Any]] | None = None,
) -> CallAssignment:
    """Build slots, score every feasible (slot, call) pair, solve once, then rebind Refs."""
    check_ref_graph(tools)
    calls = calls_of(run)
    slots = build_slots(tools, calls)
    n, m = len(slots), len(calls)
    ids = {expect_id(te, i): te.name for i, te in enumerate(tools)}
    turns = {expect_id(te, i): te.turn for i, te in enumerate(tools)}
    resolver = RefResolver(run, calls, ids, turns, fakes)

    def new_ctx(refs: RefResolver) -> MatchContext:
        return MatchContext(
            run=run, refs=refs, fakes=fakes, judge=judge, use_judge=config.use_judge,
            predicates=config.predicates, judge_phase=False, judge_log=judge_log,
        )  # fmt: skip

    ctx = new_ctx(resolver)
    evals: dict[tuple[int, int], PairEval] = {}
    for slot in slots:
        for ci, call in enumerate(calls):
            if name_ok(slot.expect, call):
                ev = pair_eval(slot.expect, call, ctx, config.min_pair_score)
                if ev.feasible:
                    evals[(slot.index, ci)] = ev
    pairs_idx: list[tuple[int, int]] = []
    if n and m and evals:
        big_b = min(n, m) + 1
        weights = [[0.0] * m for _ in range(n)]
        for (i, j), ev in evals.items():
            frac_i = i / (n - 1) if n > 1 else 0.0
            frac_j = j / (m - 1) if m > 1 else 0.0
            # squared, so that of two equally scored pairings the non-crossing one weighs more
            align = POSITION_EPS / (big_b * n * m) * (1 - (frac_i - frac_j) ** 2)
            weights[i][j] = big_b * (1 if slots[i].required else 0) + 1 + ev.score / big_b + align
        backend = get_backend(config.assignment_backend)
        # The position term is far below float precision for large inputs and solvers break
        # exact ties differently, so the result is put into canonical form afterwards.
        solved = [(i, j) for i, j in backend(weights) if weights[i][j] > 0]
        pairs_idx = _canonical(solved, slots, m, evals)

    # Phase B: rebind expectation Refs to the calls actually assigned, without re-solving.
    bound_lists: dict[str, list[Step]] = {}
    for i, j in pairs_idx:
        bound_lists.setdefault(slots[i].expect_id, []).append(calls[j])
    bound = {k: tuple(sorted(v, key=lambda s: s.seq)) for k, v in bound_lists.items()}
    rebound = RefResolver(run, calls, ids, turns, fakes, bound=bound)
    rctx = new_ctx(rebound)
    matched: list[MatchedPair] = []
    for i, j in pairs_idx:
        slot, call = slots[i], calls[j]
        ev = evals[(i, j)]
        if _has_expect_ref(slot.expect, ids):
            ev = pair_eval(slot.expect, call, rctx, config.min_pair_score)
        if config.use_judge and judge is not None and _has_semantic(slot.expect):
            # Phase C: judge only the matched pairs (after the Ref rebind, without re-solving), so
            # the judge never runs on every slot x call combination.
            rctx.judge_phase = True
            ev = pair_eval(slot.expect, call, rctx, config.min_pair_score)
            rctx.judge_phase = False
        matched.append(MatchedPair(slot, call, j, ev))
    matched.sort(key=lambda p: p.slot.index)
    matched_slots = {p.slot.index for p in matched}
    matched_calls = {p.call_index for p in matched}
    return CallAssignment(
        slots=slots,
        calls=calls,
        pairs=tuple(matched),
        missing_required=tuple(s for s in slots if s.required and s.index not in matched_slots),
        unmatched_calls=tuple(c for k, c in enumerate(calls) if k not in matched_calls),
        _bound=bound,
    )


__all__ = [
    "CallAssignment",
    "MatchedPair",
    "PairEval",
    "Slot",
    "assign",
    "assign_max",
    "build_slots",
    "calls_of",
    "check_ref_graph",
    "expect_id",
    "get_backend",
    "name_ok",
    "pair_eval",
    "register_assignment_backend",
    "unregister_assignment_backend",
]
