"""``make_judge`` / ``JudgeEngine``: the implementation of the assertions package's ``Judge``."""

import asyncio
import contextvars
import json
import math
import os
import random
import threading
import time
import warnings
import weakref
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ..assertions.judge import JudgeTask, JudgeVerdict
from ..models._frozen import freeze_mapping
from .cache import CacheCorrupt, JsonDirStore, cache_key, resolve_mode
from .calibration import lookup_calibration
from .codes import JudgeConfigError, message
from .config import JudgeConfig, JudgeOptions, KindConfig, resolve_config
from .identity import JudgeIdentity, sha256_hex
from .kinds import JudgeKindSpec, get_kind, list_kinds
from .ledger import BudgetExceeded, CostLedger
from .models.base import (
    ExtraMissing,
    JudgeModel,
    JudgeRequest,
    JudgeResponse,
    ProviderAuth,
    ProviderBadRequest,
    ProviderTransient,
    safe_int,
    total_tokens,
)
from .models.fake import FakeScriptExhausted
from .models.registry import build_model, check_provider, parse_model_string
from .network import requests_allowed
from .render import render_messages
from .runtime import JudgeRuntime
from .schema import validate
from .templates import JudgeTemplate, check_lock, get_template
from .verdict import error_verdict, vote_error_text
from .voting import SampleOutcome, combine_orders, vote

REASONING_CAP = 800
TEMPERATURE_KEYS = ("temperature", "top_p")


class JudgeWarning(UserWarning):
    """A judge setup or run condition worth a look (drift, ungated judges, odd temperature)."""


@dataclass(slots=True, frozen=True)
class _Parsed:
    """A validated model answer and the label decided from it."""

    data: Mapping[str, Any]
    label: str


@dataclass(slots=True)
class _ModelSpec:
    provider: str
    model_id: str  # the requested id (what the cache key and identity use)
    instance: JudgeModel | None = None
    options: dict[str, Any] = field(default_factory=dict, repr=False)  # may hold an api_key

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model_id}"


class _EvalScope:
    def __init__(self) -> None:
        self.count = 0
        self.lock = threading.Lock()


_EVAL: contextvars.ContextVar[_EvalScope | None] = contextvars.ContextVar(
    "judge_eval_scope", default=None
)
_SECRET_ENV = ("DISSEQT_API_KEY", "DISSEQT_JUDGE_SERVICE_KEY", "DISSEQT_PROJECT_ID")


def _cap(text: str) -> str:
    """Model-controlled text embedded in a reason never exceeds ``REASONING_CAP``."""
    return text if len(text) <= REASONING_CAP else text[: REASONING_CAP - 1] + "\u2026"


class JudgeEngine:
    """Thread-safe and shareable. Implements ``assertions.Judge`` (``judge`` and ``ajudge``)."""

    def __init__(
        self, grader: _ModelSpec, config: JudgeConfig, roles: Mapping[str, _ModelSpec] | None = None
    ):
        self.config = config
        self._models: dict[str, _ModelSpec] = {"grader": grader, **(roles or {})}
        self.ledger = CostLedger(config.max_usd, config.max_calls)
        self._rt = JudgeRuntime()
        self._mode = resolve_mode(config.cache_mode)
        self._store = JsonDirStore(
            config.cache_dir, track=os.environ.get("DISSEQT_JUDGE_CACHE_TRACK") == "1"
        )
        self._inflight: dict[str, asyncio.Future[SampleOutcome]] = {}  # loop thread only
        self._sems: dict[str, asyncio.Semaphore] = {}  # loop thread only
        self._circuit: set[str] = set()
        self._rejects_temp: set[str] = set()  # models that refused temperature/top_p (this engine)
        self._model_ids: dict[tuple[str, str], str] = {}
        self._rng = random.Random(0)
        self._sleep: Callable[[float], Any] = asyncio.sleep
        self.warnings: list[str] = []
        self._stats = {
            "judge_calls": 0,
            "samples": 0,
            "live_calls": 0,
            "cache_hits": 0,
            "errors": 0,
        }
        self._lock = threading.Lock()
        weakref.finalize(self, self._rt.close)

    # ------------------------------------------------------------------ public
    def judge(self, task: JudgeTask) -> JudgeVerdict:
        capped = self._eval_cap(task.kind)
        if capped is not None:
            return capped
        return self._rt.run(self._judge(task))

    async def ajudge(self, task: JudgeTask) -> JudgeVerdict:
        capped = self._eval_cap(task.kind)
        if capped is not None:
            return capped
        return await self._rt.arun(self._judge(task))

    @contextmanager
    def eval_scope(self) -> Any:
        """Bound the judge calls of one ``evaluate`` (``max_judge_calls_per_eval``)."""
        token = _EVAL.set(_EvalScope())
        try:
            yield
        finally:
            _EVAL.reset(token)

    def identity(self, kind: str, version: str = "") -> JudgeIdentity:
        spec = get_kind(kind)
        if spec is None:
            raise KeyError(message("unknown_kind", kind=kind, kinds=list(list_kinds())))
        kc = self.config.kind_config(kind)
        template = self._template(kc, kind, version)
        return self._identity(kind, template, kc, self._model(kc.role))

    def stats(self) -> dict[str, Any]:
        with self._lock:
            out: dict[str, Any] = dict(self._stats)
        out.update(
            ledger=self.ledger.summary(), cache_mode=self._mode, pending=self._rt.pending(),
            warnings=list(self.warnings),
        )  # fmt: skip
        return out

    def close(self) -> None:
        self._rt.close()

    def __enter__(self) -> "JudgeEngine":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ helpers
    def _warn(self, text: str) -> None:
        with self._lock:
            self.warnings.append(text)
        warnings.warn(text, JudgeWarning, stacklevel=3)

    def _bump(self, key: str, n: int = 1) -> None:
        with self._lock:
            self._stats[key] += n

    def _model(self, role: str) -> _ModelSpec:
        return self._models.get(role) or self._models["grader"]

    def _template(self, kc: KindConfig, kind: str, version: str) -> JudgeTemplate:
        from . import templates  # noqa: F401
        from .templates import builtin  # noqa: F401  (registers the v1.0 templates)

        return get_template(kc.template or kind, version or (kc.prompt_version or ""))

    @staticmethod
    def _base_url(model: _ModelSpec) -> str | None:
        """Never build the model here (replay needs no SDK and no keys), and never change with the
        lazy build: a string model's identity must be the same before and after its first call."""
        from .models.service import configured_base_url, normalize_base_url

        if model.instance is not None:
            return normalize_base_url(model.instance.base_url)
        if model.provider == "service":
            return configured_base_url(model.options)
        return normalize_base_url(model.options.get("base_url"))

    @staticmethod
    def _secrets(model: _ModelSpec) -> list[str]:
        """Every credential configured for ``model``: scrubbed from any provider error text."""
        found: list[Any] = [model.options.get("api_key"), model.options.get("project_id")]
        found += [os.environ.get(name) for name in _SECRET_ENV]
        getter = getattr(model.instance, "secrets", None)
        if callable(getter):
            found += list(getter())
        return sorted({s for s in found if isinstance(s, str) and s}, key=len, reverse=True)

    def _scrub(self, text: str, model: _ModelSpec) -> str:
        for secret in self._secrets(model):
            text = text.replace(secret, "[redacted]")
        return text

    def _identity(
        self, kind: str, template: JudgeTemplate, kc: KindConfig, model: _ModelSpec
    ) -> JudgeIdentity:
        return JudgeIdentity(
            kind=kind,
            prompt_version=template.version,
            prompt_hash=template.prompt_hash,
            model=model.label,
            base_url=self._base_url(model),
            params={"temperature": kc.resolved_temperature(), "max_tokens": kc.max_tokens},
            samples=kc.samples,
            min_agreement=kc.min_agreement,
            few_shot_hash=template.few_shot_hash,
            output_schema_hash=sha256_hex(dict(template.output_schema)),
        )

    def _eval_cap(self, kind: str) -> JudgeVerdict | None:
        scope = _EVAL.get()
        if scope is None:
            return None
        with scope.lock:
            scope.count += 1
            over = scope.count > self.config.max_judge_calls_per_eval
        if not over:
            return None
        s = self.ledger.summary()
        text = message("budget_exceeded", what="max_judge_calls_per_eval", spent=s["spent_usd"],
                       cap=self.config.max_usd or 0.0, calls=s["calls"])  # fmt: skip
        return error_verdict("budget_exceeded", text, kind=kind)

    def _use_cache(self, model: _ModelSpec) -> bool:
        if self._mode == "off":
            return False
        networked = True if model.instance is None else model.instance.networked
        explicit = self.config.cache_mode != "auto" or bool(os.environ.get("DISSEQT_JUDGE_CACHE"))
        return networked or explicit

    # ------------------------------------------------------------------ the judge flow
    async def _judge(self, task: JudgeTask) -> JudgeVerdict:
        self._bump("judge_calls")
        spec = get_kind(task.kind)
        if spec is None:
            text = message("unknown_kind", kind=task.kind, kinds=list(list_kinds()))
            return error_verdict("unknown_kind", text, kind=task.kind)
        inputs = task.inputs
        missing = [k for k in spec.required if k not in inputs]
        if missing:
            text = message("input_missing", kind=task.kind, detail=f"missing {missing}")
            return error_verdict("input_missing", text, kind=task.kind)
        if spec.check_inputs is not None:
            problem = spec.check_inputs(inputs)
            if problem:
                text = message("input_missing", kind=task.kind, detail=problem)
                return error_verdict("input_missing", text, kind=task.kind)
        kc = self.config.kind_config(task.kind)
        try:
            template = self._template(kc, task.kind, task.prompt_version)
        except KeyError as exc:
            return error_verdict("template_unknown", str(exc.args[0]), kind=task.kind)
        model = self._model(kc.role)
        identity = self._identity(task.kind, template, kc, model)
        status = lookup_calibration(spec, identity.hash, self.config)
        if status.status == "off":
            self._warn_once("require_calibration='off': every judge verdict can fail a case")
        common = {
            "kind": task.kind, "model": model.label, "identity_hash": identity.hash,
            "prompt_hash": template.prompt_hash, "prompt_version": template.version,
        }  # fmt: skip
        if spec.precheck is not None:
            pre = spec.precheck(inputs)
            if pre is not None:
                return self._verdict(
                    spec, template, identity, status.to_dict(), status.gating, [], pre.label, 1.0, 0,
                    precheck={"rule": pre.rule, "reason": pre.reason}, reason=pre.reason,
                )  # fmt: skip
        outcomes = await asyncio.gather(
            *(
                self._sample(spec, template, inputs, kc, model, identity, order, index)
                for order in range(spec.orders)
                for index in range(kc.samples)
            )
        )
        self._bump("samples", len(outcomes))
        per_order = [vote([o for o in outcomes if o.order == k], kc.samples, kc.min_agreement)
                     for k in range(spec.orders)]  # fmt: skip
        consistent: bool | None = None
        if spec.orders == 2:
            labels = template.label_set
            final, consistent = combine_orders(per_order[0], per_order[1], labels[0], labels[1])
        else:
            final = per_order[0]
        tokens = sum(o.tokens for o in outcomes)
        cost = sum(o.cost for o in outcomes)
        if final.is_error:
            self._bump("errors")
            codes = ", ".join(sorted({o.code for o in outcomes if o.code}))
            detail = next(
                (o.detail for o in outcomes if not o.ok and o.code == final.error_code), ""
            )
            text = (
                vote_error_text(
                    "insufficient_votes",
                    task.kind,
                    final.n_valid,
                    final.n,
                    codes,
                    model.label,
                    detail,
                )
                if final.error_code == "insufficient_votes"
                else detail
                or vote_error_text(
                    str(final.error_code), task.kind, final.n_valid, final.n, codes, model.label, ""
                )
            )
            return error_verdict(
                str(final.error_code), text, samples=final.n_valid, tokens=tokens, cost=cost,
                raw={"samples": [o.summary() for o in outcomes], "calibration": status.to_dict()}, **common,
            )  # fmt: skip
        return self._verdict(
            spec, template, identity, status.to_dict(), status.gating, list(outcomes), str(final.label),
            final.agreement, final.n_valid, consistent=consistent, tokens=tokens, cost=cost,
            n_total=kc.samples,
        )  # fmt: skip

    def _warn_once(self, text: str) -> None:
        with self._lock:
            seen = text in self.warnings
        if not seen:
            self._warn(text)

    def _verdict(
        self,
        spec: JudgeKindSpec,
        template: JudgeTemplate,
        identity: JudgeIdentity,
        calibration: dict[str, Any],
        gating: str,
        outcomes: list[SampleOutcome],
        label: str,
        agreement: float,
        n_valid: int,
        *,
        consistent: bool | None = None,
        precheck: dict[str, Any] | None = None,
        reason: str = "",
        tokens: int = 0,
        cost: float = 0.0,
        n_total: int = 0,
    ) -> JudgeVerdict:
        if not reason:
            matching = sorted(
                (o for o in outcomes if o.ok and o.label == label), key=lambda o: (o.order, o.index)
            )
            if matching and matching[0].parsed is not None:
                reason = str(matching[0].parsed.get("reasoning", ""))
            elif label == "unsure":
                reason = f"agreement {agreement:.2f} over {n_valid} samples" + (
                    "; orders disagree" if consistent is False else ""
                )
        reason = reason[:REASONING_CAP]
        if outcomes and n_valid < n_total:
            reason = f"{reason} [{n_valid} of {n_total} samples valid]".strip()
        derived: Any = None
        first = next((o for o in sorted(outcomes, key=lambda o: (o.order, o.index))
                      if o.ok and o.label == label and o.parsed is not None), None)  # fmt: skip
        if spec.summarize is not None and first is not None and first.parsed is not None:
            derived = spec.summarize(first.parsed)
        raw: dict[str, Any] = {
            "kind": spec.name, "prompt_version": template.version, "identity": identity.to_dict(),
            "samples": [o.summary() for o in outcomes], "order_consistent": consistent,
            "precheck": precheck, "derived": derived, "calibration": calibration,
            "quorum": {"n_valid": n_valid, "n": n_total} if outcomes else None,
            "dropped_params": sorted({p for o in outcomes for p in o.dropped_params}),
            "recorded_cost_usd": round(sum(o.recorded_cost for o in outcomes if o.cached), 6),
        }  # fmt: skip
        return JudgeVerdict(
            "ok", label, value=template.value_map.get(label), reason=reason,
            model=identity.model, prompt_hash=template.prompt_hash, samples=n_valid,
            agreement=agreement, cached=bool(outcomes) and all(o.cached for o in outcomes),
            tokens=tokens, cost=cost, raw=freeze_mapping(raw, "raw"), gating=gating,
            identity_hash=identity.hash,
        )  # fmt: skip

    # ------------------------------------------------------------------ one sample
    async def _sample(
        self, spec: JudgeKindSpec, template: JudgeTemplate, inputs: Mapping[str, Any], kc: KindConfig,
        model: _ModelSpec, identity: JudgeIdentity, order: int, index: int,
    ) -> SampleOutcome:  # fmt: skip
        blocks = spec.blocks(inputs, order, index)
        messages = render_messages(template, blocks, self.config.redact_patterns)
        schema = self._schema_for(spec, template, inputs)
        params: dict[str, Any] = {
            "temperature": kc.resolved_temperature(),
            "max_tokens": kc.max_tokens,
        }
        if model.options.get("seed") is not None:
            params["seed"] = int(model.options["seed"]) + index
        key = cache_key(  # always the REQUESTED params: dropping one must not change the key
            provider=model.provider, model=model.model_id, base_url=identity.base_url, messages=messages,
            output_schema=schema, schema_name=spec.name, params=params, sample_index=index + 1000 * order,
            kind=spec.name, prompt_hash=template.prompt_hash,
        )  # fmt: skip
        existing = self._inflight.get(key)
        if existing is not None:  # single-flight: identical concurrent requests share one result
            shared = await asyncio.shield(existing)
            return SampleOutcome(order, index, shared.ok, shared.label, shared.parsed, shared.code,
                                 shared.detail, shared.cached, 0, 0.0, shared.recorded_cost)  # fmt: skip
        future: asyncio.Future[SampleOutcome] = asyncio.get_running_loop().create_future()
        self._inflight[key] = future
        sent = (
            {k: v for k, v in params.items() if k not in TEMPERATURE_KEYS}
            if model.label in self._rejects_temp
            else params
        )
        try:
            outcome = await self._sample_uncached(
                spec,
                template,
                inputs,
                kc,
                model,
                schema,
                messages,
                sent,
                key,
                order,
                index,
                tuple(k for k in params if k not in sent),
            )
        except BaseException as exc:
            if not future.done():
                future.set_exception(exc)
                future.exception()  # mark retrieved: no "never retrieved" noise for a failed leader
            raise
        else:
            future.set_result(outcome)
            return outcome
        finally:
            self._inflight.pop(key, None)

    def _schema_for(
        self, spec: JudgeKindSpec, template: JudgeTemplate, inputs: Mapping[str, Any]
    ) -> dict[str, Any]:
        schema = json.loads(json.dumps(dict(template.output_schema)))
        if spec.labels_for is not None:
            schema["properties"]["label"]["enum"] = list(spec.labels_for(inputs))
        return schema  # type: ignore[no-any-return]

    async def _sample_uncached(
        self, spec: JudgeKindSpec, template: JudgeTemplate, inputs: Mapping[str, Any], kc: KindConfig,
        model: _ModelSpec, schema: dict[str, Any], messages: list[dict[str, str]], params: dict[str, Any],
        key: str, order: int, index: int, pre_dropped: tuple[str, ...] = (),
    ) -> SampleOutcome:  # fmt: skip

        def fail(code: str, text: str, **kw: Any) -> SampleOutcome:
            return SampleOutcome(order, index, False, None, None, code, text, **kw)

        use_cache = self._use_cache(model)
        mode = self._mode
        if use_cache and mode in ("replay", "record"):
            try:
                entry = self._store.get(model.provider, model.model_id, key)
            except CacheCorrupt as exc:
                if mode == "replay":
                    return fail("cache_corrupt", str(exc))
                self._warn(f"overwriting a corrupt judge cache entry: {exc}")
                entry = None
            if entry is not None:
                self._store.touch(model.provider, model.model_id, key)
                resp = entry["response"]
                self.ledger.record_hit(float(resp.get("cost_usd") or 0.0))
                self._bump("cache_hits")
                self._remember_model_id(model, str(resp.get("model_id") or ""), warn=True)
                interp = self._interpret(str(resp.get("text", "")), schema, spec, inputs)
                if isinstance(interp, tuple):
                    return fail("cache_corrupt", f"cached response is no longer valid: {interp[1]}")
                return SampleOutcome(order, index, True, interp.label, interp.data, cached=True,
                                     recorded_cost=float(resp.get("cost_usd") or 0.0),
                                     dropped_params=tuple(entry.get("dropped_params") or ()))  # fmt: skip
        if use_cache and mode == "replay":
            return fail(
                "cache_miss_offline",
                message("cache_miss_offline", kind=spec.name, i=index, key12=key[:12]),
            )
        return await self._live(
            spec,
            template,
            inputs,
            kc,
            model,
            schema,
            messages,
            params,
            key,
            order,
            index,
            use_cache,
            pre_dropped,
        )

    def _remember_model_id(self, model: _ModelSpec, seen: str, *, warn: bool) -> None:
        if not seen:
            return
        slot = (model.provider, model.model_id)
        with self._lock:
            before = self._model_ids.get(slot)
            self._model_ids.setdefault(slot, seen)
        if warn and before is not None and before != seen:
            self._warn(
                f"judge model drift for {model.label}: cached responses came from {before!r}, new from {seen!r}"
            )

    def _interpret(
        self, text: str, schema: dict[str, Any], spec: JudgeKindSpec, inputs: Mapping[str, Any]
    ) -> "_Parsed | tuple[str, str]":
        """A validated answer with its label, or ``(code, detail)`` for a re-ask / sample error."""
        out = self._interpret_raw(text, schema, spec, inputs)
        return out if isinstance(out, _Parsed) else (out[0], _cap(out[1]))

    def _interpret_raw(
        self, text: str, schema: dict[str, Any], spec: JudgeKindSpec, inputs: Mapping[str, Any]
    ) -> "_Parsed | tuple[str, str]":
        try:
            parsed = json.loads(text)
        except ValueError as exc:
            return "parse_error", f"not JSON ({exc})"
        if not isinstance(parsed, dict):
            return "parse_error", "the response is not a JSON object"
        problems = validate(parsed, schema)
        if problems:
            label_problem = [p for p in problems if p.startswith("$.label")]
            if label_problem and len(label_problem) == len(problems) and "label" in parsed:
                allowed = schema["properties"]["label"]["enum"]
                return "invalid_verdict", message(
                    "invalid_verdict", kind=spec.name, label=parsed["label"], labels=list(allowed)
                )
            return "parse_error", "; ".join(problems[:3])
        if spec.post_validate is not None:
            bad = spec.post_validate(parsed, inputs)
            if bad is not None:
                return bad
        if isinstance(parsed.get("reasoning"), str) and len(parsed["reasoning"]) > REASONING_CAP:
            parsed["reasoning"] = parsed["reasoning"][:REASONING_CAP]
        label = spec.derive(parsed, inputs) if spec.derive is not None else str(parsed["label"])
        return _Parsed(parsed, label)

    async def _live(
        self, spec: JudgeKindSpec, template: JudgeTemplate, inputs: Mapping[str, Any], kc: KindConfig,
        model: _ModelSpec, schema: dict[str, Any], messages: list[dict[str, str]], params: dict[str, Any],
        key: str, order: int, index: int, use_cache: bool, pre_dropped: tuple[str, ...] = (),
    ) -> SampleOutcome:  # fmt: skip
        tokens = 0
        cost = 0.0
        dropped: list[str] = list(pre_dropped)

        def fail(code: str, text: str) -> SampleOutcome:
            return SampleOutcome(order, index, False, None, None, code, text, tokens=tokens, cost=cost,
                                 dropped_params=tuple(dropped))  # fmt: skip

        if model.label in self._circuit:
            return fail(
                "auth_config",
                message(
                    "auth_config", model=model.label, detail="circuit open after an earlier failure"
                ),
            )
        try:
            instance = model.instance or build_model(
                model.provider, model.model_id, dict(model.options)
            )
            model.instance = instance
        except ExtraMissing as exc:
            return fail(
                "extra_missing", message("extra_missing", model=model.label, extra=exc.extra)
            )
        except JudgeConfigError as exc:
            return fail(
                "auth_config",
                message("auth_config", model=model.label, detail=self._scrub(exc.detail, model)),
            )
        except Exception as exc:  # a provider factory failed: never let its text leak a secret
            detail = self._scrub(f"{type(exc).__name__}: {exc}", model)
            return fail(
                "auth_config", message("auth_config", model=model.label, detail=_cap(detail))
            )
        if instance.networked and not requests_allowed(self.config.allow_requests, self._mode):
            return fail("network_forbidden", message("network_forbidden", model=model.label))
        convo = list(messages)
        reasks = 0
        while True:
            req = JudgeRequest(
                tuple(convo), schema, spec.name, dict(params), index, self.config.attempt_timeout_s,
                kind=spec.name, order=order,
            )  # fmt: skip
            got = await self._call(instance, model, req, params, dropped)
            if isinstance(got, SampleOutcome):  # an error from the transport layer
                return SampleOutcome(order, index, False, None, None, got.code, got.detail, tokens=tokens + got.tokens,
                                     cost=cost + got.cost, dropped_params=tuple(dropped))  # fmt: skip
            resp, call_cost = got
            tokens += total_tokens(resp.usage)
            cost += call_cost
            self._remember_model_id(model, resp.model_id, warn=True)
            if resp.stop_reason == "refusal":
                return fail("refusal", message("refusal", model=model.label))
            if resp.stop_reason == "content_filter":
                return fail("content_filter", message("content_filter", model=model.label))
            if resp.stop_reason == "max_tokens":
                return fail("truncated", message("truncated", model=model.label, n=kc.max_tokens))
            interp = self._interpret(resp.text, schema, spec, inputs)
            if isinstance(interp, _Parsed):
                if use_cache and self._mode in ("record", "refresh"):
                    self._write_cache(model, key, req, resp, call_cost, spec.name, index, dropped)
                return SampleOutcome(order, index, True, interp.label, interp.data, tokens=tokens, cost=cost,
                                     dropped_params=tuple(dropped))  # fmt: skip
            code, detail = interp
            if reasks >= self.config.max_parse_retries:
                text = (
                    detail
                    if code == "invalid_verdict"
                    else message("parse_error", kind=spec.name, n=reasks, detail=detail)
                )
                return fail(code, text)
            reasks += 1
            convo = [*convo, {"role": "assistant", "content": resp.text},
                     {"role": "user", "content": f"Your previous response was invalid: {detail}. "
                                                  "Respond again with only a valid JSON object."}]  # fmt: skip

    def _write_cache(
        self,
        model: _ModelSpec,
        key: str,
        req: JudgeRequest,
        resp: JudgeResponse,
        cost: float,
        kind: str,
        index: int,
        dropped: list[str] | None = None,
    ) -> None:
        try:
            from importlib.metadata import version

            _v = version("disseqt-ai-sdk")
        except Exception:  # not installed as a distribution (a source checkout)
            _v = "unknown"
        entry = {
            "cache_schema_version": 1, "key": key, "dropped_params": sorted(dropped or ()),
            "request": {
                "provider": model.provider, "model": model.model_id, "messages": [dict(m) for m in req.messages],
                "params": dict(req.params), "schema_name": req.schema_name, "sample_index": index, "kind": kind,
            },
            "response": {
                "text": resp.text, "parsed": None, "stop_reason": resp.stop_reason, "model_id": resp.model_id,
                "system_fingerprint": resp.system_fingerprint, "usage": dict(resp.usage) if resp.usage else None,
                "cost_usd": round(cost, 8),
            },
            "metadata": {"recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "disseqt_version": str(_v)},
        }  # fmt: skip
        try:
            self._store.put(model.provider, model.model_id, key, entry)
        except (OSError, CacheCorrupt) as exc:  # the verdict is good; only the recording failed
            self._warn_once(f"judge cache write failed ({type(exc).__name__}); verdict not cached")

    async def _call(
        self,
        instance: JudgeModel,
        model: _ModelSpec,
        req: JudgeRequest,
        params: dict[str, Any],
        dropped: list[str],
    ) -> "SampleOutcome | tuple[JudgeResponse, float]":
        """One request with transport retries (the only retry layer)."""
        cfg = self.config
        sem = self._sems.setdefault(model.label, asyncio.Semaphore(cfg.max_concurrency))
        attempts = 0
        started = time.monotonic()
        last_code, last_status, last_retry = "provider_error", "", None
        current = req
        price = instance.price()
        while True:
            attempts += 1
            est = (
                price.cost(
                    sum(len(m["content"]) for m in current.messages) // 4,
                    int(params.get("max_tokens", 0)),
                )
                if price
                else 0.0
            )
            try:
                reservation = self.ledger.reserve(est)
            except BudgetExceeded as exc:
                s = self.ledger.summary()
                return SampleOutcome(current.order, current.sample_index, False, None, None, "budget_exceeded",
                                     message("budget_exceeded", what=exc.what, spent=s["spent_usd"], cap=exc.cap, calls=s["calls"]))  # fmt: skip
            self._bump("live_calls")
            try:
                async with sem:
                    if model.label in self._circuit:  # another sample hit an auth error meanwhile
                        raise ProviderAuth("circuit open after an earlier authentication failure")
                    resp = await asyncio.wait_for(
                        instance.complete(current), timeout=cfg.attempt_timeout_s
                    )
            except (asyncio.TimeoutError, TimeoutError):
                self.ledger.settle(reservation, 0.0)
                last_code, last_status, last_retry = "timeout", "", None
            except ProviderTransient as exc:
                self.ledger.settle(reservation, 0.0)
                last_code = "rate_limited" if str(exc.status) == "429" else "provider_error"
                last_status, last_retry = str(exc.status), exc.retry_after_s
            except ProviderAuth as exc:
                self.ledger.release(reservation)
                self._circuit.add(model.label)
                return SampleOutcome(current.order, current.sample_index, False, None, None, "auth_config",
                                     message("auth_config", model=model.label, detail=_cap(self._scrub(str(exc), model))))  # fmt: skip
            except ProviderBadRequest as exc:
                self.ledger.release(reservation)
                named = [k for k in TEMPERATURE_KEYS if k in str(exc) and k in current.params]
                if named:  # one resend without the parameters the model refuses; remembered
                    with self._lock:
                        self._rejects_temp.add(model.label)
                    dropped.extend(named)
                    params = {k: v for k, v in params.items() if k not in named}
                    current = JudgeRequest(current.messages, current.output_schema, current.schema_name, params,
                                           current.sample_index, current.attempt_timeout_s, kind=current.kind, order=current.order)  # fmt: skip
                    attempts -= 1
                    continue
                return SampleOutcome(current.order, current.sample_index, False, None, None, "provider_error",
                                     message("provider_error", model=model.label, status=400, detail=_cap(self._scrub(str(exc), model))))  # fmt: skip
            except ExtraMissing as exc:
                self.ledger.release(reservation)
                return SampleOutcome(current.order, current.sample_index, False, None, None, "extra_missing",
                                     message("extra_missing", model=model.label, extra=exc.extra))  # fmt: skip
            except (FakeScriptExhausted, asyncio.CancelledError):
                self.ledger.release(reservation)
                raise
            except Exception as exc:
                self.ledger.release(reservation)
                return SampleOutcome(current.order, current.sample_index, False, None, None, "provider_error",
                                     message("provider_error", model=model.label, status="", detail=_cap(self._scrub(f"{type(exc).__name__}: {exc}", model))))  # fmt: skip
            else:
                reported = resp.cost_usd
                if not isinstance(reported, (int, float)) or not math.isfinite(reported):
                    reported = 0.0
                actual = max(reported, 0.0) or (
                    price.cost(
                        *(
                            safe_int((resp.usage or {}).get(k, 0))
                            for k in ("input_tokens", "output_tokens")
                        )
                    )
                    if price and resp.usage
                    else 0.0
                )
                self.ledger.settle(reservation, actual, total_tokens(resp.usage))
                return resp, actual
            elapsed = time.monotonic() - started
            delay = (
                min(max(float(last_retry), 0.0), cfg.backoff_cap_s)
                if isinstance(last_retry, (int, float)) and math.isfinite(last_retry)
                else self._rng.uniform(
                    0.0, min(cfg.backoff_cap_s, cfg.backoff_base_s * 2 ** (attempts - 1))
                )
            )
            if attempts > cfg.max_transport_retries or elapsed + delay > cfg.sample_timeout_s:
                text = (
                    message("rate_limited", model=model.label, n=attempts) if last_code == "rate_limited"
                    else message("timeout", model=model.label, secs=elapsed, n=attempts) if last_code == "timeout"
                    else message("provider_error", model=model.label, status=last_status, detail="transport failure")
                )  # fmt: skip
                return SampleOutcome(
                    current.order, current.sample_index, False, None, None, last_code, text
                )
            await self._sleep(delay)


def _resolve_spec(model: "str | JudgeModel", options: Mapping[str, Any]) -> _ModelSpec:
    if isinstance(model, str):
        provider, model_id = parse_model_string(model)
        check_provider(provider)
        return _ModelSpec(
            provider, model_id, None, dict(options.get(model_id, options.get(model, {})))
        )
    provider = getattr(model, "provider", "fake" if not model.networked else "custom")
    return _ModelSpec(provider, model.id, model, {})


def make_judge(
    model: "str | JudgeModel | None" = None, *, config: JudgeConfig | None = None, **options: Any
) -> JudgeEngine:
    """Validate the setup and build an engine. No SDK is imported and no API key is read here."""
    cfg = resolve_config(config, options)
    check_lock()
    grader_src = model if model is not None else os.environ.get("DISSEQT_JUDGE_MODEL")
    if grader_src is None or grader_src == "":
        raise JudgeConfigError("no_model", "pass a model or set DISSEQT_JUDGE_MODEL")
    grader = _resolve_spec(grader_src, cfg.model_options)
    roles: dict[str, _ModelSpec] = {}
    for role, text in cfg.roles.items():
        roles[role] = _resolve_spec(text, cfg.model_options)
    for key, value in os.environ.items():
        if key.startswith("DISSEQT_JUDGE_ROLE_") and value:
            roles.setdefault(
                key[len("DISSEQT_JUDGE_ROLE_") :].lower(), _resolve_spec(value, cfg.model_options)
            )
    for spec in (grader, *roles.values()):
        networked = True if spec.instance is None else spec.instance.networked
        if cfg.max_usd is not None and networked:
            price = spec.instance.price() if spec.instance else spec.options.get("price")
            if price is None:
                raise JudgeConfigError(
                    "price_unknown", f"max_usd is set but no price is known for {spec.label}"
                )
        if networked and not any(ch.isdigit() for ch in spec.model_id):
            warnings.warn(
                f"judge model id {spec.model_id!r} has no digits: pin a dated snapshot",
                JudgeWarning,
                stacklevel=2,
            )
    engine = JudgeEngine(grader, cfg, roles)
    if cfg.rpm is not None:
        engine._warn_once("JudgeConfig.rpm is accepted but not yet enforced: no rate limit applies")
    from .triage import enable_triage

    enable_triage()  # inert unless a case sets EvalConfig.triage=True
    for kind, kc in cfg.kinds.items():
        if kc.temperature == 0.0 and kc.samples > 1:
            engine._warn(
                f"{kind}: temperature 0.0 with {kc.samples} samples gives {kc.samples} near-identical votes"
            )
    return engine


def judge_from_env() -> JudgeEngine | None:
    """An engine from ``DISSEQT_JUDGE_MODEL``, or ``None`` (so plan 03 reports ``judge_unavailable``)."""
    value = os.environ.get("DISSEQT_JUDGE_MODEL", "").strip()
    return make_judge(value) if value else None


_ = JudgeOptions
