"""A judge's identity must not change between its first and later calls.

Regression: for a model given as a string (``"service:judge-1"``) the engine builds the model lazily
on the first call, and the identity read ``base_url`` from the instance only when one existed. So
the identity hash before the first call differed from every later one: ``disseqt judges calibrate
--model service:...`` always died with "expected one judge identity over the golden set", and a
calibration report could never match the identity used at gate time."""

from typing import Any

import pytest

from disseqt_agent_testing.assertions import JudgeTask
from disseqt_agent_testing.judges import KindConfig, make_judge

INPUTS = {
    "labels": ["a", "b"], "conversation": [{"role": "user", "text": "hi", "turn": 0}], "turn": 0,
}  # fmt: skip


def test_identity_is_the_same_before_and_after_the_first_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The suite blocks sockets, so the call itself errors; the identity is still stamped on the verdict.
    monkeypatch.setenv("DISSEQT_JUDGE_SERVICE_URL", "https://judge.example.test/")
    monkeypatch.setenv("DISSEQT_JUDGE_SERVICE_KEY", "k")
    judge = make_judge(
        "service:judge-1", require_calibration="off",
        kinds={"intent": KindConfig(samples=1)}, cache_mode="off", max_transport_retries=0,
    )  # fmt: skip
    try:
        before = judge.identity("intent")
        first = judge.judge(JudgeTask("intent", inputs=INPUTS))
        after = judge.identity("intent")
        second: Any = judge.judge(JudgeTask("intent", inputs=INPUTS))
    finally:
        judge.close()
    assert (
        before.base_url == "https://judge.example.test"
    )  # trailing slash normalised like the transport
    assert before.hash == after.hash == first.identity_hash == second.identity_hash
