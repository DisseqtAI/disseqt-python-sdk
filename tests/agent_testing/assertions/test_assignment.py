"""Assignment (A-01 ... A-12): optimal, lexicographic, backend-independent, Refs two-phase."""

import itertools
import random
from typing import Any

import pytest

from disseqt_agent_testing.assertions import (
    EvalConfig,
    ExpectConfigError,
    evaluate,
    register_assignment_backend,
)
from disseqt_agent_testing.assertions.assignment import (
    assign,
    assign_max,
    build_slots,
    unregister_assignment_backend,
)
from disseqt_agent_testing.assertions.assignment.backends import scipy_available
from disseqt_agent_testing.models import AnyValue, Expect, Ref, ToolExpect

from .helpers import run_of, step


def solve(tools: list[ToolExpect], *steps: Any, fakes: tuple[Any, ...] = (), **cfg: Any) -> Any:
    return assign(tuple(tools), run_of(*steps), EvalConfig(**cfg), fakes)


def test_a01_the_greedy_trap_passes_under_the_optimum() -> None:
    tools = [
        ToolExpect("search", {"q": AnyValue()}, id="e1"),
        ToolExpect("search", {"q": "paris"}, id="e2"),
    ]
    a = solve(tools, step(1, "search", {"q": "paris"}), step(2, "search", {"q": "rome"}))
    assert not a.missing_required
    assert {(p.slot.expect_id, p.step.call_id) for p in a.pairs} == {
        ("e1", "call-2"),
        ("e2", "call-1"),
    }
    assert all(p.eval.score == 1.0 for p in a.pairs)


def test_a02_a_wrong_argument_pair_stays_feasible() -> None:
    a = solve(
        [ToolExpect("issue_refund", {"order_id": "A"})],
        step(1, "issue_refund", {"order_id": "B"}),
    )
    assert len(a.pairs) == 1 and not a.missing_required and not a.unmatched_calls
    assert a.pairs[0].eval.score == 0.0


def test_required_matches_beat_better_scores() -> None:
    tools = [ToolExpect("a", {"x": 1}), ToolExpect("a", {"x": 2})]
    a = solve(tools, step(1, "a", {"x": 1}), step(2, "a", {"x": 9}))
    assert len(a.pairs) == 2 and not a.missing_required  # both slots matched, one imperfectly


def test_a03_count_two_with_three_calls_leaves_one_unexpected() -> None:
    a = solve([ToolExpect("w", count=2)], *(step(i, "w") for i in (1, 2, 3)))
    assert len(a.pairs) == 2 and len(a.unmatched_calls) == 1


def test_a04_open_ended_count_takes_every_identical_read() -> None:
    a = solve([ToolExpect("r", count=(1, None))], *(step(i, "r", effect="read") for i in (1, 2, 3)))
    assert len(a.pairs) == 3 and not a.unmatched_calls


def test_optional_slots_do_not_become_required() -> None:
    a = solve([ToolExpect("r", count=(0, 2))])
    assert a.n_required == 0 and not a.missing_required and len(a.slots) == 2


def test_a07_no_calls_means_everything_required_is_missing() -> None:
    a = solve([ToolExpect("x"), ToolExpect("y")])
    assert [s.expect_id for s in a.missing_required] == ["x#0", "y#1"] and not a.pairs


def test_slots_are_in_declared_order_with_required_first() -> None:
    slots = build_slots([ToolExpect("a", count=(1, 3)), ToolExpect("b")], (step(1, "a"),) * 3)
    assert [(s.expect_id, s.required) for s in slots] == [
        ("a#0", True),
        ("a#0", False),
        ("a#0", False),
        ("b#1", True),
    ]


def test_turn_restricts_which_calls_can_match() -> None:
    a = solve([ToolExpect("t", turn=1)], step(1, "t", turn=0))
    assert not a.pairs and a.missing_required
    a = solve([ToolExpect("t", turn=1)], step(1, "t", turn=0), step(2, "t", turn=1))
    assert a.pairs[0].step.call_id == "call-2"


def test_globs_match_tool_names() -> None:
    a = solve([ToolExpect("get_*")], step(1, "get_order"))
    assert len(a.pairs) == 1


def test_model_steps_never_take_part() -> None:
    a = solve([ToolExpect("t")], step(1, "m", kind="model"), step(2, "t"))
    assert [c.call_id for c in a.calls] == ["call-2"]


def test_exact_mode_and_ignore_and_defaults() -> None:
    a = solve([ToolExpect("t", {"a": 1}, args_mode="exact")], step(1, "t", {"a": 1, "b": 2}))
    assert (
        a.pairs[0].eval.score == 0.5 and a.pairs[0].eval.failed_leaves[0].code == "unexpected_param"
    )
    a = solve(
        [ToolExpect("t", {"a": 1}, args_mode="exact", ignore=("b",))],
        step(1, "t", {"a": 1, "b": 2}),
    )
    assert a.pairs[0].eval.score == 1.0
    a = solve(
        [ToolExpect("t", {"a": 1}, args_mode="exact", defaults={"b": 2})],
        step(1, "t", {"a": 1, "b": 2}),
    )
    assert a.pairs[0].eval.score == 1.0


def test_min_pair_score_makes_poor_pairs_infeasible() -> None:
    a = solve([ToolExpect("t", {"a": 1})], step(1, "t", {"a": 2}), min_pair_score=0.5)
    assert not a.pairs and a.missing_required


LOOKUP = ToolExpect("get_order", {"order_id": "A12"}, id="lookup")


def refund_expect(source: str = "call:lookup.result.order_id") -> ToolExpect:
    return ToolExpect("issue_refund", {"order_id": Ref(source)}, id="refund")


def test_a08_ref_to_an_earlier_result_is_checked_against_the_bound_call() -> None:
    steps = (
        step(1, "get_order", {"order_id": "A12"}, effect="read", result={"order_id": "A12"}),
        step(2, "issue_refund", {"order_id": "A12"}),
    )
    a = solve([LOOKUP, refund_expect()], *steps)
    assert all(p.eval.score == 1.0 for p in a.pairs)
    bad = solve(
        [LOOKUP, refund_expect()],
        steps[0],
        step(2, "issue_refund", {"order_id": "Z999"}),
    )
    leaf = bad.pairs[1].eval.leaves[0]
    assert leaf.code == "ungrounded_value" and leaf.details["reason"] == "value differs"
    assert leaf.details["expected_value"] == "A12"


def test_a09_a_source_used_after_the_consumer_is_reported() -> None:
    a = solve(
        [LOOKUP, refund_expect()],
        step(1, "issue_refund", {"order_id": "A12"}),
        step(2, "get_order", {"order_id": "A12"}, effect="read", result={"order_id": "A12"}),
    )
    refund_pair = next(p for p in a.pairs if p.slot.expect_id == "refund")
    assert refund_pair.eval.leaves[0].details["reason"] == "source after use"


def test_a10_a_direct_call_id_ref_and_a_missing_path() -> None:
    steps = (
        step(1, "get_order", {"order_id": "A12"}, effect="read", result={"order_id": "A12"}),
        step(2, "issue_refund", {"order_id": "A12"}),
    )
    ok = solve([refund_expect("call:call-1.result.order_id")], *steps)
    assert ok.pairs[0].eval.score == 1.0
    gone = solve([refund_expect("call:call-1.result.nope")], *steps)
    assert gone.pairs[0].eval.leaves[0].details["reason"] == "path missing"


def test_a10_ref_to_a_source_that_never_ran() -> None:
    a = solve([LOOKUP, refund_expect()], step(1, "issue_refund", {"order_id": "A12"}))
    leaf = a.pairs[0].eval.leaves[0]
    assert leaf.details["reason"] == "source not matched"


def test_ref_to_the_user_input() -> None:
    ok = solve(
        [ToolExpect("get_order", {"order_id": Ref("input")})],
        step(1, "get_order", {"order_id": "a12"}),
    )
    assert ok.pairs[0].eval.score == 1.0
    bad = solve(
        [ToolExpect("get_order", {"order_id": Ref("input")})],
        step(1, "get_order", {"order_id": "Q9"}),
    )
    assert bad.pairs[0].eval.leaves[0].code == "ungrounded_value"


def test_ref_to_a_fake_needs_fakes() -> None:
    from disseqt_agent_testing.models import FakeResponse, FakeSpec

    tools = [ToolExpect("pay", {"id": Ref("fake:get_order.order_id")})]
    unresolved = solve(tools, step(1, "pay", {"id": "A12"}))
    assert unresolved.pairs[0].eval.errored
    spec = FakeSpec("get_order", default=FakeResponse(returns={"order_id": "A12"}))
    assert solve(tools, step(1, "pay", {"id": "A12"}), fakes=(spec,)).pairs[0].eval.score == 1.0


def test_ref_cycles_are_a_config_error_naming_the_cycle() -> None:
    tools = [
        ToolExpect("a", {"x": Ref("call:b.result.v")}, id="a"),
        ToolExpect("b", {"x": Ref("call:a.result.v")}, id="b"),
    ]
    with pytest.raises(ExpectConfigError, match="Ref dependency cycle"):
        solve(tools, step(1, "a"), step(2, "b"))


def brute_force(w: list[list[float]]) -> float:
    n, m = len(w), len(w[0])
    best = 0.0
    for k in range(0, min(n, m) + 1):
        for rows in itertools.combinations(range(n), k):
            for cols in itertools.permutations(range(m), k):
                total = sum(w[r][c] for r, c in zip(rows, cols, strict=True))
                if all(w[r][c] > 0 for r, c in zip(rows, cols, strict=True)):
                    best = max(best, total)
    return best


def random_matrix(rng: random.Random) -> list[list[float]]:
    n, m = rng.randint(1, 5), rng.randint(1, 5)
    pool = [0, 0, 1, 1.0000001, 2, 3.5]
    return [[rng.choice(pool + [rng.random()]) for _ in range(m)] for _ in range(n)]


def test_a11_vendored_solver_equals_brute_force() -> None:
    rng = random.Random(7)
    for _ in range(1500):
        w = random_matrix(rng)
        pairs = [(i, j) for i, j in assign_max(w) if w[i][j] > 0]
        assert abs(sum(w[i][j] for i, j in pairs) - brute_force(w)) < 1e-9
        assert len({i for i, _ in pairs}) == len(pairs) == len({j for _, j in pairs})


def test_vendored_solver_edge_shapes() -> None:
    assert assign_max([]) == [] and assign_max([[]]) == []
    assert assign_max([[5.0]]) == [(0, 0)]
    assert assign_max([[1.0, 9.0]]) == [(0, 1)]
    assert assign_max([[1.0], [9.0]]) == [(1, 0)]


def bindings(tools: list[ToolExpect], steps: list[Any], backend: str) -> list[tuple[int, int]]:
    a = solve(tools, *steps, assignment_backend=backend)
    return [(p.slot.index, p.call_index) for p in a.pairs]


def order_outcome(tools: list[ToolExpect], steps: list[Any], backend: str) -> str:
    expect = Expect(tools=tuple(tools), order="in_order")
    return str(evaluate(run_of(*steps), expect, assignment_backend=backend).outcome)


def random_case(rng: random.Random) -> tuple[list[ToolExpect], list[Any]]:
    """Few tool names and argument shapes, so equally good pairings are common."""
    tools = []
    for _ in range(rng.randint(1, 3)):
        low = rng.randint(0, 4)
        high = rng.choice([low, low + 1, low + 2, None]) or None
        args = rng.choice([{}, {"x": 1}, {"x": AnyValue()}, {"x": 1, "y": 2}])
        tools.append(ToolExpect(rng.choice("abc"), args, count=(low, high)))
    shapes: list[dict[str, Any]] = [{}, {"x": 1}, {"x": 2}, {"x": 1, "y": 2}, {"x": 1, "y": 3}]
    steps = [step(i + 1, rng.choice("abc"), rng.choice(shapes)) for i in range(rng.randint(0, 9))]
    return tools, steps


# Four ``c`` calls bound to one expectation: its slots are interchangeable, so a solver is free to
# hand them out crossed, and ``in_order`` reads the calls in slot order.
FOUR_CALLS_ONE_EXPECTATION: tuple[list[ToolExpect], list[Any]] = (
    [
        ToolExpect("a", {"x": AnyValue()}, count=(0, 2)),
        ToolExpect("a", {"x": 1}, count=(0, 2)),
        ToolExpect("c", {"x": AnyValue()}, count=(2, None)),
    ],
    [
        step(1, "a", {"x": 1, "y": 2}),
        step(2, "c", {"x": 1, "y": 3}),
        step(3, "c", {"x": 2}),
        step(4, "c", {"x": 1, "y": 2}),
        step(5, "c", {"x": 1, "y": 2}),
    ],
)
FOUR_CALLS_BINDINGS = [(0, 0), (4, 1), (5, 2), (6, 3), (7, 4)]


def tie_shuffling_backend(weights: Any) -> list[tuple[int, int]]:
    """An optimal solver that ignores the position term: rows that weigh (nearly) the same in
    every column are interchangeable to it, and it hands their columns out in reverse."""
    pairs = dict(assign_max(weights))
    done: set[int] = set()
    for row in range(len(weights)):
        alike = [
            r
            for r in range(row, len(weights))
            if r not in done
            and all(abs(x - y) < 1e-6 for x, y in zip(weights[row], weights[r], strict=True))
        ]
        done.update(alike)
        columns = [pairs.pop(r) for r in alike if r in pairs]
        pairs.update(zip(reversed(alike), sorted(columns), strict=False))
    return sorted(pairs.items())


def test_calls_bound_to_one_expectation_take_its_slots_in_seq_order() -> None:
    tools, steps = FOUR_CALLS_ONE_EXPECTATION
    register_assignment_backend("shuffled", tie_shuffling_backend)
    try:
        assert bindings(tools, steps, "python") == FOUR_CALLS_BINDINGS
        assert bindings(tools, steps, "shuffled") == FOUR_CALLS_BINDINGS
        assert order_outcome(tools, steps, "python") == "passed"
        assert order_outcome(tools, steps, "shuffled") == "passed"
    finally:
        unregister_assignment_backend("shuffled")


def test_the_result_does_not_depend_on_how_a_solver_breaks_ties() -> None:
    rng = random.Random(23)
    register_assignment_backend("shuffled", tie_shuffling_backend)
    try:
        for _ in range(1500):
            tools, steps = random_case(rng)
            assert bindings(tools, steps, "shuffled") == bindings(tools, steps, "python")
    finally:
        unregister_assignment_backend("shuffled")


def test_an_expectation_that_ties_between_two_calls_takes_the_earlier_one() -> None:
    # The ``a`` slot sits exactly halfway between the two ``a`` calls.
    tools = [ToolExpect("p"), ToolExpect("a"), ToolExpect("q")]
    steps = [step(1, "p"), step(2, "a"), step(3, "x"), step(4, "a"), step(5, "q")]
    assert bindings(tools, steps, "python") == [(0, 0), (1, 1), (2, 4)]


@pytest.mark.skipif(not scipy_available(), reason="SciPy not installed")
def test_a12_python_and_scipy_backends_agree_on_totals_and_assignments() -> None:
    from disseqt_agent_testing.assertions.assignment.backends import get_backend

    py, sp = get_backend("python"), get_backend("scipy")
    rng = random.Random(11)
    for _ in range(3000):
        w = random_matrix(rng)
        a = [(i, j) for i, j in py(w) if w[i][j] > 0]
        b = [(i, j) for i, j in sp(w) if w[i][j] > 0]
        assert abs(sum(w[i][j] for i, j in a) - sum(w[i][j] for i, j in b)) < 1e-9


@pytest.mark.skipif(not scipy_available(), reason="SciPy not installed")
def test_python_scipy_and_auto_backends_give_the_same_bindings_and_verdict() -> None:
    tools, steps = FOUR_CALLS_ONE_EXPECTATION
    for backend in ("python", "scipy", "auto"):
        assert bindings(tools, steps, backend) == FOUR_CALLS_BINDINGS, backend
        assert order_outcome(tools, steps, backend) == "passed", backend
    rng = random.Random(5)
    for _ in range(4000):
        tools, steps = random_case(rng)
        assert bindings(tools, steps, "scipy") == bindings(tools, steps, "python")
    rng = random.Random(5)
    for _ in range(600):
        tools, steps = random_case(rng)
        assert order_outcome(tools, steps, "scipy") == order_outcome(tools, steps, "python")


def test_e03_a_temporary_backend_is_used_and_removed() -> None:
    calls: list[int] = []

    def backend(w: Any) -> list[tuple[int, int]]:
        calls.append(len(w))
        return assign_max(w)

    register_assignment_backend("spy", backend)
    try:
        solve([ToolExpect("t")], step(1, "t"), assignment_backend="spy")
        assert calls
        with pytest.raises(ValueError, match="already registered"):
            register_assignment_backend("spy", backend)
    finally:
        unregister_assignment_backend("spy")
    with pytest.raises(ExpectConfigError, match="unknown assignment backend"):
        solve([ToolExpect("t")], step(1, "t"), assignment_backend="spy")


def test_scipy_backend_without_scipy_is_a_clear_error() -> None:
    if scipy_available():
        pytest.skip("SciPy is installed")
    with pytest.raises(ExpectConfigError, match="needs SciPy"):
        solve([ToolExpect("t")], step(1, "t"), assignment_backend="scipy")


def test_identical_duplicates_assign_in_order() -> None:
    tools = [ToolExpect("a", id="x"), ToolExpect("a", id="y")]
    a = solve(tools, step(1, "a"), step(2, "a"))
    assert [(p.slot.expect_id, p.step.call_id) for p in a.pairs] == [
        ("x", "call-1"),
        ("y", "call-2"),
    ]


def test_bound_returns_assigned_calls_in_seq_order() -> None:
    a = solve([ToolExpect("a", id="x", count=(1, None))], step(3, "a"), step(5, "a"))
    assert [s.seq for s in a.bound("x")] == [3, 5] and a.bound("nope") == ()


def test_a_required_slot_wins_a_call_over_an_optional_slot_declared_earlier() -> None:
    tools = [ToolExpect("a", {"x": 1}, count=(0, 1), id="opt"), ToolExpect("a", {"x": 1}, id="req")]
    a = solve(tools, step(1, "a", {"x": 1}))
    assert not a.missing_required and [p.slot.expect_id for p in a.pairs] == ["req"]


def test_refs_are_rebound_to_the_calls_the_assignment_actually_chose() -> None:
    # Two lookups ran. The expectation asks for A99, so the refund must match A99's result, not
    # "any earlier lookup" (A12 happens to be an earlier candidate, which a provisional-only
    # resolution would wrongly accept).
    tools = [
        ToolExpect("get_order", {"order_id": "A99"}, id="lookup"),
        ToolExpect("issue_refund", {"order_id": Ref("call:lookup.result.order_id")}, id="refund"),
    ]
    steps = (
        step(1, "get_order", {"order_id": "A12"}, effect="read", result={"order_id": "A12"}),
        step(2, "get_order", {"order_id": "A99"}, effect="read", result={"order_id": "A99"}),
        step(3, "issue_refund", {"order_id": "A12"}),
    )
    a = solve(tools, *steps)
    refund = next(p for p in a.pairs if p.slot.expect_id == "refund")
    assert (
        refund.eval.leaves[0].ok is False
        and refund.eval.leaves[0].details["expected_value"] == "A99"
    )
