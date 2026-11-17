"""Judge contract (interpret_verdict, escalation) and language packs."""

import itertools
from typing import Any

import pytest

from disseqt_agent_testing.assertions import (
    EvalConfig,
    JudgeTask,
    JudgeVerdict,
    classify_turn,
    evaluate,
    interpret_verdict,
    register_lexicon,
)
from disseqt_agent_testing.assertions.config import Lexicons
from disseqt_agent_testing.assertions.judge import NOT_APPLICABLE, UNSURE
from disseqt_agent_testing.assertions.lexicons import LexiconPack, get_lexicon, unregister_lexicon
from disseqt_agent_testing.assertions.metrics import EvalContext, Memo
from disseqt_agent_testing.models import Decision, Expect

from .helpers import run_of

POS, NEG = ("yes",), ("no",)


def test_interpret_verdict_order_of_cases() -> None:
    assert interpret_verdict(None, POS, NEG).reason_code == "judge_unavailable"
    err = interpret_verdict(JudgeVerdict("error", None, error_code="timeout"), POS, NEG)
    assert (err.status, err.reason_code, err.details["error_code"]) == (
        "error",
        "judge_failed",
        "timeout",
    )
    assert interpret_verdict(JudgeVerdict("ok", UNSURE), POS, NEG).reason_code == "judge_unsure"
    na = interpret_verdict(JudgeVerdict("ok", NOT_APPLICABLE), POS, NEG)
    assert na.reason_code == "judge_unsure"
    assert (
        interpret_verdict(JudgeVerdict("ok", NOT_APPLICABLE), POS, NEG, na_benign=True).reason_code
        == "not_applicable"
    )
    adv = interpret_verdict(JudgeVerdict("ok", "yes", gating="advisory"), POS, NEG)
    assert (adv.reason_code, adv.details["advisory_label"]) == ("judge_uncalibrated", "yes")
    assert interpret_verdict(JudgeVerdict("ok", "yes"), POS, NEG).passed is True
    assert interpret_verdict(JudgeVerdict("ok", "no"), POS, NEG).passed is False
    odd = interpret_verdict(JudgeVerdict("ok", "maybe"), POS, NEG)
    assert (odd.status, odd.details["error_code"]) == ("error", "invalid_verdict")


def test_g05_passed_true_is_reachable_only_from_a_gating_positive_ok_verdict() -> None:
    for status, label, gating in itertools.product(
        ("ok", "error"), ("yes", "no", UNSURE, NOT_APPLICABLE, "weird", None), ("gate", "advisory")
    ):
        verdict = JudgeVerdict(status, label, gating=gating)  # type: ignore[arg-type]
        interp = interpret_verdict(verdict, POS, NEG)
        if interp.passed is True:
            assert (status, label, gating) == ("ok", "yes", "gate")
        if status == "error":
            assert interp.status == "error"


class FlipToAsk:
    def __init__(self, verdict: JudgeVerdict) -> None:
        self.verdict, self.tasks = verdict, []

    def judge(self, task: JudgeTask) -> JudgeVerdict:
        self.tasks.append(task)
        return self.verdict

    async def ajudge(self, task: JudgeTask) -> JudgeVerdict:
        return self.judge(task)


AMBIGUOUS = run_of(text="I'm sorry, I can't help with that. Which account do you mean?")


def test_d13_a_judge_resolves_a_low_confidence_label() -> None:
    judge = FlipToAsk(JudgeVerdict("ok", "ask"))
    ctx = EvalContext(None, AMBIGUOUS, Expect(), "c", EvalConfig(use_judge=True), judge, (), Memo())
    c = classify_turn(AMBIGUOUS, 0, ctx.config, ctx)
    assert (c.label, c.judged, c.rule) == ("ask", True, "judge")
    task = judge.tasks[0]
    assert task.kind == "decision" and task.inputs["turn"] == 0
    assert task.inputs["agent_text"].startswith("I'm sorry") and task.inputs["tool_calls"] == ()
    assert task.inputs["conversation"][0]["role"] == "user"
    rep = evaluate(AMBIGUOUS, Expect(decision=Decision.ASK), judge=judge, use_judge=True)
    assert rep.outcome == "passed" and rep.check("decision_match").judged  # type: ignore[union-attr]
    assert rep.judge_calls[0]["kind"] == "decision"


def test_without_a_judge_nothing_changes_and_the_low_confidence_is_reported() -> None:
    rep = evaluate(AMBIGUOUS, Expect(decision=Decision.REFUSE))
    comp = rep.check("decision_match").components[0]  # type: ignore[union-attr]
    assert rep.outcome == "passed" and comp.details["low_confidence"] is True
    judge = FlipToAsk(JudgeVerdict("ok", "ask"))
    off = evaluate(AMBIGUOUS, Expect(decision=Decision.REFUSE), judge=judge)  # use_judge is False
    assert off.outcome == "passed" and judge.tasks == []


@pytest.mark.parametrize(
    ("verdict", "status", "code"),
    [
        (JudgeVerdict("error", None, error_code="x"), "error", "judge_failed"),
        (JudgeVerdict("ok", UNSURE), "skipped", "judge_unsure"),
    ],
)
def test_j10_a_failed_or_unsure_judge_never_falls_back_to_the_heuristic(
    verdict: JudgeVerdict, status: str, code: str
) -> None:
    rep = evaluate(
        AMBIGUOUS, Expect(decision=Decision.REFUSE), judge=FlipToAsk(verdict), use_judge=True
    )
    comp = rep.check("decision_match").components[0]  # type: ignore[union-attr]
    assert (comp.status, comp.reason_code) == (status, code)
    assert rep.outcome == "incomplete"


def test_an_advisory_verdict_keeps_the_heuristic_label() -> None:
    judge = FlipToAsk(JudgeVerdict("ok", "ask", gating="advisory"))
    rep = evaluate(AMBIGUOUS, Expect(decision=Decision.REFUSE), judge=judge, use_judge=True)
    assert rep.outcome == "passed"


def test_uncalibrated_policy_warns_or_blocks_in_gating() -> None:
    from disseqt_agent_testing.assertions import BaseMetric, MetricResult, register_metric
    from disseqt_agent_testing.assertions.metrics import unregister_metric

    @register_metric("advisory_thing", "1")
    class Advisory(BaseMetric):
        def applicable(self, expect: Expect, config: EvalConfig) -> bool:
            return True

        def evaluate(self, ctx: Any) -> MetricResult:
            return self.skip(ctx, "judge_uncalibrated", "label=yes")

    try:
        warn = evaluate(run_of(), Expect(max_steps=1))
        assert warn.outcome == "passed" and any("not gated" in w for w in warn.warnings)
        assert (
            evaluate(run_of(), Expect(max_steps=1), on_uncalibrated="incomplete").outcome
            == "incomplete"
        )
        assert evaluate(run_of(), Expect(max_steps=1), strict=True).outcome == "incomplete"
    finally:
        unregister_metric("advisory_thing")


def test_e04_a_registered_language_pack_is_selected_by_lang() -> None:
    es = LexiconPack(
        lang="es",
        refusal_prefixes=("lo siento",),
        modal_prefixes=(),
        action_verbs=(),
        refusal_patterns=(),
        interrogatives=(),
        question_cues=(),
        closers=("algo mas",),
    )
    register_lexicon(es)
    try:
        run = run_of(text="Lo siento, no puedo con eso.")
        assert classify_turn(run, 0, EvalConfig(lang="es")).label == "refuse"
        assert classify_turn(run, 0, EvalConfig(lang="en")).label == "answer"
        with pytest.raises(ValueError, match="already registered"):
            register_lexicon(es)
    finally:
        unregister_lexicon("es")
    with pytest.raises(ValueError, match="unknown lexicon"):
        get_lexicon("es")


def test_lexicon_overrides_extend_and_exclude() -> None:
    run = run_of(text="Negative, that is not possible here.")
    assert classify_turn(run, 0, EvalConfig()).label == "answer"
    extra = EvalConfig(lexicons=Lexicons(refusal_extra=("negative,",)))
    assert classify_turn(run, 0, extra).label == "refuse"
    excluded = EvalConfig(lexicons=Lexicons(refusal_exclude=("is not possible",)))
    assert (
        classify_turn(run_of(text="I'm sorry, that is not possible."), 0, excluded).label
        == "answer"
    )
    closers = EvalConfig(lexicons=Lexicons(closers_extra=("so, yeah",)))
    assert classify_turn(run_of(text="Done. So, yeah?"), 0, closers).label == "answer"


def test_structured_outputs_are_never_lexicon_classified() -> None:
    run = run_of(text=None, outputs=({"msg": "I'm sorry, I can't help"},))
    assert classify_turn(run, 0, EvalConfig()).rule == "structured_answer"


def test_every_registered_code_is_documented_in_the_codes_module() -> None:
    from disseqt_agent_testing.assertions import codes

    doc = codes.__doc__ or ""
    missing = [c for c in codes.known_codes() if c not in doc]
    assert not missing, f"document these codes in codes.py's docstring: {missing}"


def test_every_reason_code_literal_in_the_source_is_registered() -> None:
    import re
    from pathlib import Path

    from disseqt_agent_testing.assertions import codes

    root = Path(codes.__file__).parent
    pattern = re.compile(
        r"""(?:reason_code=|code=|"code": |self\.(?:skip|error)\(ctx, |_fail\(\s*ctx, turn, |_blocked\(\s*[a-z_.\[\]"]+, )"([a-z_]+(?:[.:][a-z_*]+)?)\""""
    )
    found: set[str] = set()
    for path in root.rglob("*.py"):
        if path.name in {"codes.py", "test_judge_and_lexicons.py"}:
            continue
        found.update(pattern.findall(path.read_text()))
    unregistered = sorted(c for c in found if codes.lookup(c) is None)
    assert found and not unregistered, f"unregistered reason codes: {unregistered}"
