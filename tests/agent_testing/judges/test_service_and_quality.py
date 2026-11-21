"""The ``service:`` provider over a fake transport, and the phase-4b judge kinds."""

import json
from typing import Any

import pytest

from disseqt_agent_testing.assertions import JudgeTask
from disseqt_agent_testing.judges import (
    FakeServiceTransport,
    JudgeConfigError,
    KindConfig,
    ServiceModel,
    make_judge,
)
from disseqt_agent_testing.judges.kinds import get_kind
from disseqt_agent_testing.judges.templates import verify_lock

from .helpers import engine as _engine

ONE = {
    k: KindConfig(samples=1)
    for k in ("answer_quality", "groundedness", "plan_quality", "root_cause")
}


def engine(script: Any, **kw: Any) -> Any:
    return _engine(script, kinds=ONE, **kw)


pytestmark = pytest.mark.filterwarnings("ignore::disseqt_agent_testing.judges.api.JudgeWarning")
CONVO = [{"role": "user", "turn": 0, "text": "hi"}]


def _answer(met: list[bool], enough: bool = True) -> dict[str, Any]:
    return {"reasoning": "r", "criteria": [{"id": n, "met": m} for n, m in enumerate(met, 1)],
            "enough_data": enough}  # fmt: skip


def _atask(criteria: Any = ("polite", "short")) -> JudgeTask:
    return JudgeTask("answer_quality", inputs={"conversation": CONVO, "agent_text": "ok",
                                               "criteria": list(criteria)})  # fmt: skip


def _ground(verdicts: list[str], enough: bool = True) -> dict[str, Any]:
    return {"reasoning": "r", "claims": [{"claim": f"c{n}", "verdict": v}
                                          for n, v in enumerate(verdicts)], "enough_data": enough}  # fmt: skip


def _gtask() -> JudgeTask:
    return JudgeTask("groundedness", inputs={"agent_text": "x", "tool_calls": []})


def _plan(flaws: list[tuple[str, int]]) -> dict[str, Any]:
    return {"reasoning": "r", "flaws": [{"code": c, "step": s} for c, s in flaws],
            "enough_data": True}  # fmt: skip


def _ptask() -> JudgeTask:
    return JudgeTask("plan_quality", inputs={"goal": "refund", "plan": ["a", "b"]})


def test_lock_covers_the_new_templates() -> None:
    assert verify_lock() == []


def test_answer_quality_label_is_derived_from_criteria() -> None:
    with engine([_answer([True, True])]) as (eng, _):
        assert eng.judge(_atask()).label == "pass"
    with engine([_answer([True, False])]) as (eng, _):
        v = eng.judge(_atask())
    assert v.label == "fail" and v.value == 0.0
    assert v.raw["derived"] == {"criteria": ({"id": 1, "met": True}, {"id": 2, "met": False})}
    with engine([_answer([True, True], enough=False)]) as (eng, _):
        assert eng.judge(_atask()).label == "unsure"


def test_answer_quality_rejects_criteria_ids_that_do_not_match() -> None:
    with engine([_answer([True])] * 6) as (eng, _):
        v = eng.judge(_atask())
    assert v.status == "error" and v.error_code == "invalid_verdict"


def test_groundedness_derivation() -> None:
    cases = [(["supported", "not_checkable"], "grounded"), (["supported", "unsupported"], "ungrounded"),
             (["contradicted"], "ungrounded"), ([], "unsure")]  # fmt: skip
    for verdicts, want in cases:
        with engine([_ground(verdicts)]) as (eng, _):
            assert eng.judge(_gtask()).label == want, verdicts
    with engine([_ground(["supported"], enough=False)]) as (eng, _):
        assert eng.judge(_gtask()).label == "unsure"


def test_plan_quality_derivation() -> None:
    with engine([_plan([])]) as (eng, _):
        assert eng.judge(_ptask()).label == "sound"
    with engine([_plan([("missing_step", 2)])]) as (eng, _):
        v = eng.judge(_ptask())
    assert v.label == "flawed" and v.raw["derived"] == {
        "flaws": ({"code": "missing_step", "step": 2},)
    }


def test_root_cause_never_gates_and_validates_labels() -> None:
    spec = get_kind("root_cause")
    assert spec is not None and spec.never_gates and not spec.builtin_calibratable
    task = JudgeTask("root_cause", inputs={"failed_checks": ["tool_args"], "conversation": CONVO})
    with engine([{"reasoning": "r", "label": "wrong_args"}]) as (eng, _):
        assert eng.judge(task).label == "wrong_args"
    with engine([{"reasoning": "r", "label": "bogus"}] * 6) as (eng, _):
        assert eng.judge(task).status == "error"


# ----------------------------------------------------------------------------- service provider
def test_service_provider_round_trips_through_the_fake_transport() -> None:
    transport = FakeServiceTransport(script=[_plan([])])
    model = ServiceModel("judge-model-1", transport=transport)
    eng = make_judge(model, backoff_base_s=0.0, kinds=ONE)
    try:
        v = eng.judge(_ptask())
    finally:
        eng.close()
    assert v.label == "sound"
    sent = transport.requests[0]
    assert sent["model"] == "judge-model-1" and sent["kind"] == "plan_quality"
    assert json.dumps(sent["output_schema"]) and sent["messages"][0]["role"] == "system"
    assert not model.networked  # a fake transport never trips the network gate


def test_service_http_errors_map_to_provider_errors() -> None:
    transport = FakeServiceTransport(script=[
        (429, {"error": {"message": "slow", "retry_after_s": 0}}), (503, {}), _plan([]),
    ])  # fmt: skip
    eng = make_judge(ServiceModel("m-1", transport=transport), backoff_base_s=0.0, kinds=ONE)
    try:
        assert eng.judge(_ptask()).label == "sound"  # two transient failures were retried
        assert len(transport.requests) == 3
    finally:
        eng.close()
    for status, code in ((401, "auth_config"), (400, "bad_request")):
        t = FakeServiceTransport(script=[(status, {"error": {"message": "no"}})] * 3)
        e = make_judge(ServiceModel("m-1", transport=t), backoff_base_s=0.0, kinds=ONE)
        try:
            v = e.judge(_ptask())
        finally:
            e.close()
        assert v.status == "error", (status, code, v.error_code)


def test_service_without_transport_or_url_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DISSEQT_JUDGE_SERVICE_URL", raising=False)
    with pytest.raises(JudgeConfigError) as err:
        ServiceModel("m-1")
    assert err.value.code == "service_not_configured"
    monkeypatch.setenv("DISSEQT_JUDGE_SERVICE_URL", "https://svc.invalid/")
    live = ServiceModel("m-1")
    assert live.networked and live.base_url == "https://svc.invalid"


def test_service_provider_string_resolves_through_the_registry() -> None:
    t = FakeServiceTransport(script=[_plan([])])
    eng = make_judge("service:model-2", model_options={"model-2": {"transport": t}},
                     backoff_base_s=0.0, kinds=ONE)  # fmt: skip
    try:
        assert eng.judge(_ptask()).label == "sound"
    finally:
        eng.close()
