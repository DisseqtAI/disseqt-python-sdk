"""Test doubles: ``FakeJudgeModel`` runs the whole judge pipeline offline; ``StaticJudge``
implements the assertions package's ``Judge`` protocol directly."""

import asyncio
import json
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ...assertions.judge import JudgeTask, JudgeVerdict
from .base import (
    JudgeRequest,
    JudgeResponse,
    Price,
    ProviderAuth,
    ProviderBadRequest,
    ProviderTransient,
)


class FakeScriptExhausted(Exception):
    """The fake had no scripted response left: a test-authoring bug, never swallowed."""


@dataclass(slots=True, frozen=True)
class FakeRaw:
    """The model returned this text verbatim (use it to script malformed output)."""

    text: str


@dataclass(slots=True, frozen=True)
class FakeRefusal:
    pass


@dataclass(slots=True, frozen=True)
class FakeTruncated:
    pass


@dataclass(slots=True, frozen=True)
class FakeError:
    """A provider failure: ``code`` is ``rate_limited`` (429), ``provider_error`` (500),
    ``auth_config``, ``bad_request`` or ``content_filter``."""

    code: str
    retry_after_s: float | None = None
    detail: str = ""


@dataclass(slots=True, frozen=True)
class FakeSlow:
    """Wait ``seconds`` (in the engine's loop) and then behave like ``then``."""

    seconds: float
    then: Any = None


Script = "Sequence[Any] | Mapping[Any, Any] | Callable[[JudgeRequest], Any]"


class FakeJudgeModel:
    """A scripted judge model.

    ``script`` is a list consumed in order, a mapping ``{kind: [items]}`` (consumed per kind) or
    ``{(kind, order, sample_index): item}``, or ``fn(req) -> item``. Items are a ``dict`` (returned
    as JSON text), ``FakeRaw``, ``FakeRefusal``, ``FakeTruncated``, ``FakeError`` or ``FakeSlow``.
    Every request is recorded in ``.calls``.
    """

    def __init__(
        self,
        script: Any,
        *,
        model_id: str = "fake-judge-1",
        response_model_id: str | None = None,
        cost_per_call: float = 0.0,
        tokens: tuple[int, int] = (0, 0),
        price: Price | None = None,
        base_url: str | None = None,
        networked: bool = False,
    ):
        self.networked = networked  # a "networked" fake exercises the gate and the price rule
        self.provider = "fake"
        self._response_model_id = response_model_id or model_id
        self.id = model_id
        self.base_url = base_url
        self._script = script
        self._cost = cost_per_call
        self._tokens = tokens
        self._price = price
        self.calls: list[JudgeRequest] = []
        self._lock = threading.Lock()
        self._cursor: dict[Any, int] = {}

    def price(self) -> Price | None:
        return self._price

    def _next(self, req: JudgeRequest) -> Any:
        script = self._script
        with self._lock:
            self.calls.append(req)
            if callable(script):
                return script(req)
            if isinstance(script, Mapping):
                exact = (req.kind, req.order, req.sample_index)
                if exact in script:
                    return script[exact]
                if req.kind in script:
                    items = script[req.kind]
                    if isinstance(items, (list, tuple)):
                        i = self._cursor.get(req.kind, 0)
                        if i >= len(items):
                            raise FakeScriptExhausted(
                                f"no scripted response left for kind {req.kind!r}"
                            )
                        self._cursor[req.kind] = i + 1
                        return items[i]
                    return items
                raise FakeScriptExhausted(f"no scripted response for {exact}")
            i = self._cursor.get("*", 0)
            if i >= len(script):
                raise FakeScriptExhausted(f"script exhausted after {len(script)} responses")
            self._cursor["*"] = i + 1
            return script[i]

    async def complete(self, req: JudgeRequest) -> JudgeResponse:
        item = self._next(req)
        while isinstance(item, FakeSlow):
            await asyncio.sleep(item.seconds)
            item = item.then
        if isinstance(item, FakeError):
            if item.code == "rate_limited":
                raise ProviderTransient("429", status=429, retry_after_s=item.retry_after_s)
            if item.code == "provider_error":
                raise ProviderTransient(
                    item.detail or "500", status=500, retry_after_s=item.retry_after_s
                )
            if item.code == "auth_config":
                raise ProviderAuth(item.detail or "invalid key")
            if item.code == "bad_request":
                raise ProviderBadRequest(item.detail or "bad request")
            return (
                self._response("", "content_filter")
                if item.code == "content_filter"
                else self._response("", "other")
            )
        if isinstance(item, FakeRefusal):
            return self._response("", "refusal")
        if isinstance(item, FakeTruncated):
            return self._response('{"reasoning": "cut', "max_tokens")
        if isinstance(item, FakeRaw):
            return self._response(item.text)
        if isinstance(item, Mapping):
            return self._response(json.dumps(dict(item)))
        raise TypeError(f"unsupported fake script item {type(item).__name__}")

    def _response(self, text: str, stop: str = "stop") -> JudgeResponse:
        in_t, out_t = self._tokens
        return JudgeResponse(
            text, None, stop, {"input_tokens": in_t, "output_tokens": out_t} if (in_t or out_t) else None,
            self._cost, self._response_model_id,
        )  # fmt: skip


class StaticJudge:
    """A ``Judge`` that returns canned verdicts. ``verdicts`` is a verdict, a ``{kind: verdict}``
    mapping, a list consumed in order, or ``fn(task) -> verdict``. Records ``.tasks``."""

    def __init__(self, verdicts: Any):
        self._verdicts = verdicts
        self.tasks: list[JudgeTask] = []
        self._lock = threading.Lock()
        self._i = 0

    def judge(self, task: JudgeTask) -> JudgeVerdict:
        with self._lock:
            self.tasks.append(task)
            v = self._verdicts
            if callable(v):
                return v(task)  # type: ignore[no-any-return]
            if isinstance(v, JudgeVerdict):
                return v
            if isinstance(v, Mapping):
                return v[task.kind]  # type: ignore[no-any-return]
            item = v[self._i]
            self._i += 1
            return item  # type: ignore[no-any-return]

    async def ajudge(self, task: JudgeTask) -> JudgeVerdict:
        return self.judge(task)
