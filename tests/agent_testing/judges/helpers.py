"""Helpers for the judge tests: scripted engines, canned tasks, labelled responses."""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from disseqt_agent_testing.assertions import JudgeTask
from disseqt_agent_testing.judges import FakeJudgeModel, JudgeEngine, make_judge


def label(name: str, reasoning: str = "because") -> dict[str, Any]:
    return {"reasoning": reasoning, "label": name}


def clar(name: str) -> dict[str, Any]:
    """A valid ``clarification_relevance`` response."""
    return {**label(name), "items_requested": [], "asks_for_known_info": False}


def alt(name: str) -> dict[str, Any]:
    """A valid ``refusal_alternative`` response."""
    return {**label(name), "alternatives": []}


def decision_task(
    text: str = "Which order number is this about?", tool_calls: Any = ()
) -> JudgeTask:
    return JudgeTask(
        "decision",
        inputs={
            "turn": 0,
            "conversation": [{"role": "user", "turn": 0, "text": "I want a refund"}],
            "agent_text": text,
            "tool_calls": list(tool_calls),
        },
    )


def semantic_task(reference: Any = "NYC", actual: Any = "New York City", **extra: Any) -> JudgeTask:
    return JudgeTask(
        "semantic_arg",
        inputs={
            "reference": reference,
            "actual": actual,
            "tool": "book_trip",
            "path": ["city"],
            **extra,
        },
    )


def intent_task(labels: tuple[str, ...] = ("refund", "cancel")) -> JudgeTask:
    return JudgeTask(
        "intent",
        inputs={
            "labels": list(labels),
            "conversation": [{"role": "user", "turn": 0, "text": "my money back please"}],
            "turn": 0,
        },
    )


@contextmanager
def engine(script: Any, **options: Any) -> Iterator[tuple[JudgeEngine, FakeJudgeModel]]:
    """An engine over a scripted fake model; instant backoff unless told otherwise."""
    options.setdefault("backoff_base_s", 0.0)
    model_kw = {
        k: options.pop(k)
        for k in ("cost_per_call", "tokens", "price", "networked", "response_model_id", "model_id")
        if k in options
    }
    model = FakeJudgeModel(script, **model_kw)
    eng = make_judge(model, **options)
    try:
        yield eng, model
    finally:
        eng.close()
