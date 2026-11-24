"""Review fix pass: secrets, redirects, transport hardening, cost clamping, quorum, identity."""

import email.message
import io
import json
import math
import urllib.request
import urllib.response
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from disseqt_agent_testing.assertions import JudgeTask
from disseqt_agent_testing.judges import (
    FakeError,
    FakeJudgeModel,
    FakeRaw,
    FakeServiceTransport,
    JudgeConfigError,
    JudgeWarning,
    KindConfig,
    Price,
    ServiceModel,
    make_judge,
    register_provider,
)
from disseqt_agent_testing.judges.api import _ModelSpec
from disseqt_agent_testing.judges.ledger import CostLedger
from disseqt_agent_testing.judges.models.base import (
    JudgeRequest,
    ProviderBadRequest,
    ProviderTransient,
)
from disseqt_agent_testing.judges.models.registry import unregister_provider
from disseqt_agent_testing.judges.models.service import HttpTransport
from disseqt_agent_testing.judges.voting import SampleOutcome, vote

from .helpers import decision_task, label, semantic_task
from .helpers import engine as _engine

pytestmark = pytest.mark.filterwarnings("ignore::disseqt_agent_testing.judges.api.JudgeWarning")
KEY = "SECRETKEY123"
PROJ = "PROJECT-ID-77"
ONE = {k: KindConfig(samples=1) for k in ("plan_quality", "decision", "semantic_arg")}


# ----------------------------------------------------------------------------- network fake
class Net:
    """Replaces the HTTP(S) handlers' open: records (url, headers) and answers from ``respond``."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, respond: Callable[[str], Any]):
        self.calls: list[tuple[str, dict[str, str]]] = []
        self._respond = respond

        def fake_open(handler: Any, req: urllib.request.Request) -> Any:
            self.calls.append((req.full_url, {k.lower(): v for k, v in req.header_items()}))
            code, headers, body = self._respond(req.full_url)
            msg = email.message.Message()
            for k, v in headers:
                msg[k] = v
            resp = urllib.response.addinfourl(io.BytesIO(body), msg, req.full_url, code)
            resp.msg = "msg"  # type: ignore[attr-defined]
            return resp

        monkeypatch.setattr(urllib.request.HTTPHandler, "http_open", fake_open)
        monkeypatch.setattr(urllib.request.HTTPSHandler, "https_open", fake_open)


def _send(t: HttpTransport) -> tuple[int, Any]:
    return t._send("/v1/judge/complete", {"a": 1}, 5.0)


# ----------------------------------------------------------------------------- finding 1
@pytest.mark.parametrize("bad", ["abc\nX-Evil: 1", "a\rb", "a\x00b", "a\x7fb", "tab\there"])
def test_key_and_project_with_control_chars_are_rejected_without_echo(bad: str) -> None:
    secret = "SECRET" + bad
    for kw in ({"api_key": secret}, {"api_key": "ok", "project_id": secret}):
        with pytest.raises(JudgeConfigError) as err:
            HttpTransport("https://svc.invalid", **kw)
        assert "SECRET" not in str(err.value) and "SECRET" not in err.value.detail


def test_key_is_stripped_of_surrounding_whitespace(monkeypatch: pytest.MonkeyPatch) -> None:
    net = Net(monkeypatch, lambda url: (200, [], b"{}"))
    _send(HttpTransport("https://svc.invalid", "  k1\n", project_id=" p1 "))
    assert net.calls[0][1]["x-api-key"] == "k1" and net.calls[0][1]["x-project-id"] == "p1"


def test_send_value_error_becomes_a_fixed_message(monkeypatch: pytest.MonkeyPatch) -> None:
    t = HttpTransport("https://svc.invalid", KEY)

    class Boom:
        def open(self, *a: Any, **k: Any) -> Any:
            raise ValueError(f"Invalid header value b'{KEY}'")

    monkeypatch.setattr(t, "_opener", Boom(), raising=False)
    with pytest.raises(ProviderBadRequest) as err:
        _send(t)
    assert KEY not in str(err.value)


class _Leaky:
    networked = True
    base_url = None

    def __init__(self, model_id: str, **opts: Any):
        self.id = model_id

    def price(self) -> None:
        return None

    async def complete(self, req: JudgeRequest) -> Any:
        raise ValueError(f"Invalid header value b'{KEY}' and {PROJ}")


def test_engine_scrubs_configured_and_env_secrets_from_provider_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    register_provider("leaky", _Leaky, override=True)
    monkeypatch.setenv("DISSEQT_API_KEY", "ENVKEY999")
    try:
        eng = make_judge(
            "leaky:m-1", kinds=ONE, require_calibration="off", cache_mode="off",
            allow_requests=True, max_transport_retries=0,
            model_options={"m-1": {"api_key": KEY, "project_id": PROJ}},
        )  # fmt: skip
        try:
            v = eng.judge(JudgeTask("plan_quality", inputs={"goal": "g", "plan": ["a"]}))
        finally:
            eng.close()
    finally:
        unregister_provider("leaky")
    blob = json.dumps([v.reason, dict(v.raw or {})], default=str)
    assert v.status == "error" and KEY not in blob and PROJ not in blob


def test_reprs_never_contain_the_key() -> None:
    spec = _ModelSpec("service", "m", None, {"api_key": KEY, "project_id": PROJ})
    t = HttpTransport("https://svc.invalid", KEY, project_id=PROJ)
    m = ServiceModel("m", transport=t)
    for obj in (spec, t, m, make_judge("service:m", model_options={"m": {"api_key": KEY}})):
        assert KEY not in repr(obj) and PROJ not in repr(obj)


# ----------------------------------------------------------------------------- finding 2
@pytest.mark.parametrize("code", [301, 302, 303, 307, 308])
def test_redirects_are_never_followed_and_secrets_not_forwarded(
    monkeypatch: pytest.MonkeyPatch, code: int
) -> None:
    net = Net(monkeypatch, lambda url: (code, [("Location", "https://evil.invalid/steal")], b""))
    t = HttpTransport("https://svc.invalid", KEY, project_id=PROJ)
    status, _ = _send(t)
    assert status == code
    assert [u for u, _h in net.calls] == ["https://svc.invalid/v1/judge/complete"]
    bearer = HttpTransport("https://svc.invalid", KEY)
    _send(bearer)
    assert all("evil" not in u for u, _h in net.calls)


def test_a_redirect_is_a_bad_request_for_the_model(monkeypatch: pytest.MonkeyPatch) -> None:
    Net(monkeypatch, lambda url: (302, [("Location", "https://evil.invalid/")], b""))
    model = ServiceModel("m", transport=HttpTransport("https://svc.invalid", KEY))
    import asyncio

    with pytest.raises(ProviderBadRequest):
        asyncio.run(model.complete(JudgeRequest(({"role": "user", "content": "x"},), {}, "n")))


# ----------------------------------------------------------------------------- finding 3
def test_http_to_a_remote_host_is_refused_with_or_without_credentials() -> None:
    for kw in ({"api_key": KEY}, {"project_id": PROJ}, {}):
        with pytest.raises(JudgeConfigError) as err:
            HttpTransport("http://svc.invalid", **kw)
        assert KEY not in str(err.value) and PROJ not in str(err.value)


@pytest.mark.parametrize(
    "url", ["http://localhost:8080", "http://127.0.0.1:9", "http://[::1]:9", "https://svc.invalid"]
)
def test_loopback_http_and_any_https_are_allowed_with_credentials(url: str) -> None:
    HttpTransport(url, KEY, project_id=PROJ)


@pytest.mark.parametrize("url", ["http://localhost:8080", "http://127.0.0.1:9", "http://[::1]:9"])
def test_loopback_http_without_credentials_is_allowed(url: str) -> None:
    assert HttpTransport(url).base_url == url


def test_a_service_model_with_a_remote_http_url_and_no_credentials_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for var in ("DISSEQT_JUDGE_SERVICE_KEY", "DISSEQT_PROJECT_ID", "DISSEQT_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(JudgeConfigError) as err:
        ServiceModel("m-1", base_url="http://svc.invalid")
    assert err.value.code == "service_url"
    eng = make_judge(
        "service:m-1", model_options={"m-1": {"base_url": "http://svc.invalid"}},
        backoff_base_s=0.0, kinds=ONE, cache_mode="off",
    )  # fmt: skip
    try:
        v = eng.judge(decision_task())  # refused when the model is built: nothing is sent
    finally:
        eng.close()
    assert v.status == "error" and v.error_code == "auth_config"


@pytest.mark.parametrize("url", ["ftp://svc.invalid", "file:///etc/passwd", "svc.invalid"])
def test_non_http_schemes_are_always_rejected(url: str) -> None:
    with pytest.raises(JudgeConfigError):
        HttpTransport(url)


def test_oversized_response_is_rejected_with_a_fixed_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    Net(monkeypatch, lambda url: (200, [], b"{" + b" " * (4 * 1024 * 1024 + 10) + b"}"))
    with pytest.raises((ProviderBadRequest, ProviderTransient)) as err:
        _send(HttpTransport("https://svc.invalid", KEY))
    assert "4" in str(err.value) and KEY not in str(err.value)


@pytest.mark.parametrize("status", [429, 503])
def test_retry_after_header_is_used_when_the_body_has_none(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    import asyncio

    Net(
        monkeypatch, lambda url: (status, [("Retry-After", "7")], b'{"error": {"message": "slow"}}')
    )
    model = ServiceModel("m", transport=HttpTransport("https://svc.invalid", KEY))
    req = JudgeRequest(({"role": "user", "content": "x"},), {}, "n")
    with pytest.raises(ProviderTransient) as err:
        asyncio.run(model.complete(req))
    assert err.value.retry_after_s == 7.0


def test_retry_after_header_is_clamped(monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    for header, want in (("9999", 60.0), ("-5", 0.0)):
        Net(monkeypatch, lambda url, h=header: (429, [("Retry-After", h)], b"{}"))
        model = ServiceModel("m", transport=HttpTransport("https://svc.invalid", KEY))
        req = JudgeRequest(({"role": "user", "content": "x"},), {}, "n")
        with pytest.raises(ProviderTransient) as err:
            asyncio.run(model.complete(req))
        assert err.value.retry_after_s == want


# ----------------------------------------------------------------------------- finding 4
def _complete(body: dict[str, Any], price: Price | None = None) -> Any:
    import asyncio

    t = FakeServiceTransport(script=[body])
    req = JudgeRequest(({"role": "user", "content": "x"},), {}, "n")
    return asyncio.run(ServiceModel("m", transport=t, price=price).complete(req))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -3.0, "abc", [1]])
def test_bad_service_cost_becomes_price_derived_or_zero(bad: Any) -> None:
    usage = {"input_tokens": 1_000_000, "output_tokens": 0}
    r = _complete({"text": "{}", "cost_usd": bad, "usage": usage}, Price(2.0, 0.0))
    assert r.cost_usd == 2.0
    assert _complete({"text": "{}", "cost_usd": bad}).cost_usd == 0.0


def test_non_numeric_usage_and_retry_after_do_not_raise() -> None:
    r = _complete({"text": "{}", "usage": {"input_tokens": "x", "output_tokens": None}})
    assert r.cost_usd == 0.0
    import asyncio

    for bad in ("abc", float("nan"), -4, [1]):
        t = FakeServiceTransport(script=[(429, {"error": {"message": "m", "retry_after_s": bad}})])
        req = JudgeRequest(({"role": "user", "content": "x"},), {}, "n")
        with pytest.raises(ProviderTransient) as err:
            asyncio.run(ServiceModel("m", transport=t).complete(req))
        ra = err.value.retry_after_s
        assert ra is None or 0.0 <= ra <= 60.0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -9.0])
def test_ledger_settle_never_lowers_or_nans_spent(bad: float) -> None:
    ledger = CostLedger()
    ledger.settle(ledger.reserve(0.25), bad, -50)
    assert math.isfinite(ledger.spent_usd) and ledger.spent_usd == 0.25 and ledger.tokens >= 0


def test_engine_settles_finite_cost_when_the_model_reports_nan() -> None:
    with _engine([label("ask")], kinds=ONE, cost_per_call=float("nan")) as (eng, _):
        eng.judge(decision_task())
        assert math.isfinite(eng.ledger.spent_usd) and eng.ledger.spent_usd >= 0


# ----------------------------------------------------------------------------- finding 5
def _o(i: int, lab: str | None, ok: bool = True) -> SampleOutcome:
    return SampleOutcome(0, i, ok, lab, None, None if ok else "timeout")


def test_quorum_is_a_strict_majority() -> None:
    assert vote([_o(0, "a"), _o(1, None, False)], 2, 0.5).is_error  # 1 of 2
    assert vote([_o(0, "a"), _o(1, "a"), _o(2, None, False), _o(3, None, False)], 4, 0.5).is_error
    ok = vote([_o(0, "a"), _o(1, "a"), _o(2, None, False)], 3, 0.5)
    assert not ok.is_error and ok.label == "a" and (ok.n_valid, ok.n) == (2, 3)
    assert not vote([_o(0, "a")], 1, 0.5).is_error


def test_tie_for_the_mode_is_unsure_even_when_agreement_passes_the_threshold() -> None:
    v = vote([_o(0, "a"), _o(1, "a"), _o(2, "b"), _o(3, "b")], 4, 0.5)
    assert v.label == "unsure" and v.agreement == 0.5


def test_verdict_surfaces_n_valid_over_n() -> None:
    script = [label("ask"), label("ask"), FakeError("provider_error")]
    with _engine(script, kinds={"decision": KindConfig(samples=3)}, max_transport_retries=0) as (
        eng,
        _,
    ):
        v = eng.judge(decision_task())
    assert v.status == "ok" and v.samples == 2
    assert v.raw["quorum"] == {"n_valid": 2, "n": 3} and "2 of 3" in v.reason


# ----------------------------------------------------------------------------- finding 6
class _SlashTransport(FakeServiceTransport):
    base_url = "https://svc.invalid/"


def _sound() -> dict[str, Any]:
    return {"reasoning": "r", "flaws": [], "enough_data": True}


@pytest.mark.parametrize("transport_cls", [FakeServiceTransport, _SlashTransport])
def test_transport_option_identity_is_stable_and_record_replays(
    tmp_path: Path, transport_cls: type[FakeServiceTransport]
) -> None:
    task = JudgeTask("plan_quality", inputs={"goal": "g", "plan": ["a"]})

    def build(mode: str, script: list[Any]) -> Any:
        t = transport_cls(script=script)
        return make_judge(
            "service:m-1", model_options={"m-1": {"transport": t}}, kinds=ONE,
            cache_mode=mode, cache_dir=str(tmp_path / "c"), require_calibration="off",
        )  # fmt: skip

    rec = build("record", [_sound()])
    try:
        before = rec.identity("plan_quality")
        v = rec.judge(task)
        after = rec.identity("plan_quality")
    finally:
        rec.close()
    assert before.hash == after.hash == v.identity_hash and not before.base_url.endswith("/")
    rep = build("replay", [])
    try:
        v2 = rep.judge(task)
    finally:
        rep.close()
    assert v2.status == "ok" and v2.cached and v2.identity_hash == v.identity_hash


def test_trailing_slash_is_normalised_for_every_provider_path() -> None:
    spec = _ModelSpec("other", "m", None, {"base_url": "https://x.invalid//"})
    from disseqt_agent_testing.judges.api import JudgeEngine

    assert JudgeEngine._base_url(spec) == "https://x.invalid"
    inst = FakeJudgeModel([], base_url="https://y.invalid/")
    assert JudgeEngine._base_url(_ModelSpec("fake", "m", inst, {})) == "https://y.invalid"


# ----------------------------------------------------------------------------- finding 7
TEMP_REJECT = FakeError("bad_request", detail="unsupported parameter: temperature")


def test_temperature_rejection_is_per_engine_and_cache_keys_use_requested_params(
    tmp_path: Path,
) -> None:
    kw: dict[str, Any] = {
        "kinds": ONE, "cache_dir": str(tmp_path / "c"), "require_calibration": "off",
    }  # fmt: skip
    t1, t2 = decision_task("First question?"), decision_task("Second question?")
    with _engine([TEMP_REJECT, label("ask"), label("ask")], cache_mode="record", **kw) as (e, m):
        v1 = e.judge(t1)
        e.judge(t2)
        assert "temperature" not in m.calls[2].params
    entries = [json.loads(p.read_text()) for p in sorted((tmp_path / "c").rglob("*.json"))]
    assert any(en.get("dropped_params") == ["temperature"] for en in entries)
    assert all(
        "temperature" in en["request"]["params"] or en.get("dropped_params") for en in entries
    )
    with _engine([], cache_mode="replay", **kw) as (e2, _m2):
        r1, r2 = e2.judge(t1), e2.judge(t2)
    assert v1.status == "ok" and r1.status == "ok" and r2.status == "ok" and r2.cached
    with _engine([label("ask")], cache_mode="off", **kw) as (e3, m3):
        e3.judge(decision_task("Third?"))
        assert "temperature" in m3.calls[0].params  # a fresh engine does not inherit the memory


# ----------------------------------------------------------------------------- finding 8
def _sem(ref: Any, act: Any, **extra: Any) -> tuple[Any, FakeJudgeModel]:
    script = [label("not_equivalent"), label("not_equivalent")]
    with _engine(script, kinds=ONE) as (eng, model):
        return eng.judge(semantic_task(ref, act, **extra)), model


@pytest.mark.parametrize(
    "ref,act", [("02134", "2134"), ("007", "7"), ("1e3", "1000"), (True, 1), ("1", 1)]
)
def test_numeric_looking_values_are_not_prechecked(ref: Any, act: Any) -> None:
    v, model = _sem(ref, act)
    assert len(model.calls) == 2 and v.label == "not_equivalent"


def test_a_rubric_disables_the_precheck() -> None:
    v, model = _sem("NYC", "NYC", rubric="case and spelling must match exactly")
    assert len(model.calls) == 2 and v.label == "not_equivalent"
    v, model = _sem("a", "A ", rubric="")
    assert len(model.calls) == 0 and v.label == "equivalent"  # an empty rubric changes nothing


def test_real_numbers_still_short_circuit() -> None:
    v, model = _sem(2, 2.0)
    assert len(model.calls) == 0 and v.label == "equivalent"


# ----------------------------------------------------------------------------- finding 9
def test_answer_quality_with_no_criteria_is_input_missing() -> None:
    task = JudgeTask(
        "answer_quality", inputs={"conversation": [], "agent_text": "ok", "criteria": []}
    )
    with _engine([], kinds={"answer_quality": KindConfig(samples=1)}) as (eng, model):
        v = eng.judge(task)
    assert v.status == "error" and v.error_code == "input_missing" and not model.calls


def _ground_resp(claims: list[dict[str, str]]) -> dict[str, Any]:
    return {"reasoning": "r", "claims": claims, "enough_data": True}


def test_groundedness_with_no_claims_is_unsure_not_a_confident_pass() -> None:
    task = JudgeTask("groundedness", inputs={"agent_text": "Your total is 59.99", "tool_calls": []})
    with _engine([_ground_resp([])], kinds={"groundedness": KindConfig(samples=1)}) as (eng, _):
        assert eng.judge(task).label == "unsure"


def test_groundedness_blank_answer_is_not_applicable_by_precheck() -> None:
    task = JudgeTask("groundedness", inputs={"agent_text": "  ", "tool_calls": []})
    with _engine([], kinds={"groundedness": KindConfig(samples=1)}) as (eng, model):
        v = eng.judge(task)
    assert v.label == "not_applicable" and not model.calls


# ----------------------------------------------------------------------------- finding 10
def test_invalid_cache_env_raises_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DISSEQT_JUDGE_CACHE", "recrod")
    with pytest.raises(JudgeConfigError) as err:
        make_judge(FakeJudgeModel([]))
    assert "recrod" in str(err.value)


def test_cache_write_failure_degrades_to_an_uncached_verdict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from disseqt_agent_testing.judges.cache import JsonDirStore

    def boom(*a: Any, **k: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(JsonDirStore, "put", boom)
    with _engine([label("ask")], kinds=ONE, cache_mode="record") as (eng, _):
        with pytest.warns(JudgeWarning, match="cache"):
            v = eng.judge(decision_task())
    assert v.status == "ok" and v.label == "ask"


def test_rpm_warns_once_that_it_is_not_enforced() -> None:
    with pytest.warns(JudgeWarning, match="rpm"):
        eng = make_judge(FakeJudgeModel([]), rpm=10)
    eng.close()


# ----------------------------------------------------------------------------- finding 11
def test_raw_model_output_in_reason_is_bounded() -> None:
    bad = {"reasoning": "r", "label": "Z" * 5000}
    with _engine([bad], kinds=ONE, max_parse_retries=0) as (eng, _):
        v = eng.judge(decision_task())
    assert v.error_code == "invalid_verdict" and len(v.reason) <= 900
    with _engine([FakeRaw("x" * 5000)], kinds=ONE, max_parse_retries=0) as (eng, _):
        v = eng.judge(decision_task())
    assert v.status == "error" and len(v.reason) <= 900
    wrong = {"reasoning": "r", "label": "ask", "extra": 1, "q" * 3000: 2}
    with _engine([wrong], kinds=ONE, max_parse_retries=0) as (eng, _):
        v = eng.judge(decision_task())
    assert len(v.reason) <= 900
