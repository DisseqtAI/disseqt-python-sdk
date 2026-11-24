"""The metric contract: ``MetricResult``, ``EvalContext``, ``Memo`` and ``BaseMetric``."""

import asyncio
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import KW_ONLY, dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from ...models import (
    AgentRun,
    Expect,
    FakeSpec,
    FrozenDict,
    RunResult,
    Step,
)
from ...models._frozen import JSONObject, freeze_json, freeze_mapping
from ..canonical import normalize, round6, short_hash
from ..config import EvalConfig
from ..judge import Judge, JudgeTask, JudgeVerdict, call_record

if TYPE_CHECKING:
    from ..assignment import CallAssignment
    from ..classify import Classification

STATUSES = ("ok", "skipped", "error")
SCOPES = ("run", "turn")


@dataclass(slots=True, frozen=True)
class MetricResult:
    """One check's outcome. ``skipped`` and ``error`` carry ``value=None``, ``passed=None`` and a
    reason code, so absent evidence can never read as a pass."""

    name: str
    _: KW_ONLY
    version: str = "1"
    params_hash: str = ""
    scope: str = "run"
    turn: int | None = None
    status: str = "ok"
    value: float | None = None
    passed: bool | None = None
    threshold: float | None = None
    vetoed: bool = False
    required: bool = True
    weight: float = 1.0
    reason_code: str | None = None
    reason: str = ""
    details: JSONObject = field(default_factory=FrozenDict)
    components: "tuple[MetricResult, ...]" = ()
    judged: bool = False
    extra: JSONObject = field(default_factory=FrozenDict)

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"MetricResult.status must be one of {list(STATUSES)}")
        if self.scope not in SCOPES:
            raise ValueError(f"MetricResult.scope must be one of {list(SCOPES)}")
        if self.status != "ok":
            if self.value is not None or self.passed is not None or not self.reason_code:
                raise ValueError(f"a {self.status} result needs value=None, passed=None, a code")
        elif self.value is not None:
            if not 0.0 <= self.value <= 1.0:
                raise ValueError(f"MetricResult.value must be in [0, 1], got {self.value}")
            object.__setattr__(self, "value", round6(float(self.value)))
        if self.threshold is not None:
            object.__setattr__(self, "threshold", round6(float(self.threshold)))
        object.__setattr__(self, "weight", round6(float(self.weight)))
        object.__setattr__(self, "details", freeze_json(normalize(self.details)))
        object.__setattr__(self, "components", tuple(self.components))
        object.__setattr__(self, "extra", freeze_mapping(self.extra, "MetricResult.extra"))

    @property
    def sub_scores(self) -> Mapping[str, Any]:
        scores = self.details.get("scores") if isinstance(self.details, Mapping) else None
        return scores if isinstance(scores, Mapping) else {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "params_hash": self.params_hash,
            "scope": self.scope,
            "turn": self.turn,
            "status": self.status,
            "value": self.value,
            "passed": self.passed,
            "threshold": self.threshold,
            "vetoed": self.vetoed,
            "required": self.required,
            "weight": self.weight,
            "reason_code": self.reason_code,
            "reason": self.reason,
            "details": normalize(self.details),
            "components": [c.to_dict() for c in self.components],
            "judged": self.judged,
            "extra": normalize(self.extra),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MetricResult":
        known = {
            "name", "version", "params_hash", "scope", "turn", "status", "value", "passed",
            "threshold", "vetoed", "required", "weight", "reason_code", "reason", "details",
            "components", "judged", "extra",
        }  # fmt: skip
        body = {k: v for k, v in data.items() if k in known and k != "components"}
        comps = tuple(cls.from_dict(c) for c in data.get("components", ()))
        return cls(components=comps, **body)


def rollup(components: Sequence[MetricResult]) -> tuple[str, float | None, bool | None, str | None]:
    """Parent status from children. A failed child dominates (``ok``/failed). Otherwise a child
    that is errored, or skipped for a real reason (``data_unavailable``, ``judge_unavailable``,
    ``judge_unsure``, uncalibrated, unknown codes), makes the parent error/skipped carrying that
    code, so one unmeasurable sub-check can never let the whole check pass. Benign skips
    (``not_applicable`` and friends) do not block. Returns (status, value, passed, reason_code)."""
    from ..codes import BENIGN_SKIP, code_class

    decided = [c for c in components if c.status == "ok"]
    if not components:
        return "skipped", None, None, "not_applicable"
    if any(c.passed is False for c in decided):
        values = [c.value for c in decided if c.value is not None]
        return "ok", (sum(values) / len(values) if values else 0.0), False, None
    errors = [c for c in components if c.status == "error"]
    if errors:
        return "error", None, None, errors[0].reason_code
    blocking = [
        c for c in components if c.status == "skipped" and code_class(c.reason_code) != BENIGN_SKIP
    ]
    if blocking:
        return "skipped", None, None, blocking[0].reason_code
    if not decided:
        first = components[0]
        return "skipped", None, None, first.reason_code
    values = [c.value for c in decided if c.value is not None]
    return "ok", (sum(values) / len(values) if values else 1.0), True, None


def first_reason(components: Sequence[MetricResult]) -> str:
    """The reason of the first child that was not ``ok`` (kept on a skipped or errored parent)."""
    return next((c.reason for c in components if c.status != "ok" and c.reason), "")


class Memo:
    """Compute-once cache for one ``evaluate`` call. Exceptions are cached too, so a failing
    assignment errors every dependent metric the same way."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._values: dict[Any, tuple[bool, Any]] = {}

    def get_or_compute(self, key: Any, fn: Callable[[], Any]) -> Any:
        with self._lock:
            if key not in self._values:
                try:
                    self._values[key] = (True, fn())
                except Exception as exc:  # cache the failure, re-raise on every read
                    self._values[key] = (False, exc)
            ok, value = self._values[key]
        if ok:
            return value
        raise value


@dataclass(slots=True, frozen=True)
class EvalContext:
    """Everything a metric may read. Frozen; the memo is the only shared mutable part."""

    result: RunResult | None
    run: AgentRun
    expect: Expect
    case_id: str
    config: EvalConfig
    judge: Judge | None
    fakes: tuple[FakeSpec, ...]
    memo: Memo
    judge_log: list[dict[str, Any]] = field(default_factory=list)

    def calls(self) -> tuple[Step, ...]:
        from ..assignment import calls_of

        return self.memo.get_or_compute(("calls",), lambda: calls_of(self.run))  # type: ignore[no-any-return]

    def assignment(self) -> "CallAssignment":
        from ..assignment import assign

        return self.memo.get_or_compute(  # type: ignore[no-any-return]
            ("assignment",),
            lambda: assign(
                self.expect.tools,
                self.run,
                self.config,
                self.fakes,
                judge=self.judge,
                judge_log=self.judge_log,
            ),
        )

    def classification(self, turn: int) -> "Classification":
        from ..classify import classify_turn

        return self.memo.get_or_compute(  # type: ignore[no-any-return]
            ("classification", turn), lambda: classify_turn(self.run, turn, self.config, self)
        )

    def request_judge(self, task: JudgeTask, metric: str = "") -> JudgeVerdict | None:
        """``None`` when there is no judge or ``use_judge`` is false; every other verdict is
        recorded for the report (tagged with the metric that asked)."""
        if self.judge is None or not self.config.use_judge:
            return None
        verdict = self.judge.judge(task)
        self.judge_log.append(call_record(verdict, metric, task.kind))
        return verdict

    @property
    def finished(self) -> bool:
        return self.result is None or self.result.status == "finished"


class BaseMetric:
    """Stateless base for metrics. Subclasses set the ClassVars and implement ``evaluate``."""

    name: ClassVar[str] = ""
    version: ClassVar[str] = "1"
    scope: ClassVar[Literal["run", "turn"]] = "run"
    rank: ClassVar[int] = 1000
    requires: ClassVar[frozenset[str]] = frozenset()
    sub_scores: ClassVar[tuple[str, ...]] = ()
    thresholdable: ClassVar[bool] = False
    run_on_unfinished: ClassVar[bool] = False  # forbidden_calls and loops still score a crash

    def params(self, expect: Expect, config: EvalConfig) -> dict[str, Any]:
        return {}

    def applicable(self, expect: Expect, config: EvalConfig) -> bool:
        return False

    def evaluate(self, ctx: EvalContext) -> MetricResult:
        raise NotImplementedError

    async def aevaluate(self, ctx: EvalContext) -> MetricResult:
        return await asyncio.to_thread(self.evaluate, ctx)

    def render_lines(self, result: MetricResult) -> tuple[str, ...]:
        from ..render import default_lines

        return default_lines(result)

    # -------------------------------------------------------------------- helpers
    def _hash(self, ctx: EvalContext) -> str:
        """The params hash, computed once per metric per evaluation."""
        return ctx.memo.get_or_compute(  # type: ignore[no-any-return]
            ("params_hash", self.name), lambda: short_hash(self.params(ctx.expect, ctx.config))
        )

    def make(self, ctx: EvalContext, **fields: Any) -> MetricResult:
        fields.setdefault("scope", self.scope)
        return MetricResult(self.name, version=self.version, params_hash=self._hash(ctx), **fields)

    def ok(
        self, ctx: EvalContext, value: float | None, passed: bool, **fields: Any
    ) -> MetricResult:
        return self.make(ctx, status="ok", value=value, passed=passed, **fields)

    def skip(self, ctx: EvalContext, code: str, reason: str = "", **fields: Any) -> MetricResult:
        if code == "judge_uncalibrated":
            fields["details"] = self._advisory_details(ctx, fields.get("details"))
        return self.make(ctx, status="skipped", reason_code=code, reason=reason, **fields)

    def _advisory_details(self, ctx: EvalContext, details: Any) -> dict[str, Any]:
        """Carry the judge kind and advisory label this check already asked for, so the
        uncalibrated warning is readable even when the judge has no identity hash."""
        out = dict(details or {})
        mine = [r for r in ctx.judge_log if r.get("metric") == self.name]
        if mine:
            out.setdefault("judge_kind", mine[-1].get("kind") or "?")
            out.setdefault("advisory_label", mine[-1].get("label") or "?")
        return out

    def error(self, ctx: EvalContext, code: str, reason: str = "", **fields: Any) -> MetricResult:
        return self.make(ctx, status="error", reason_code=code, reason=reason, **fields)

    def undecided(
        self, ctx: EvalContext, cls: "Classification", **fields: Any
    ) -> MetricResult | None:
        """The error/skipped result for a turn whose decision judge failed or was unsure, else
        ``None``. ``cls.label`` is then only the low-confidence heuristic guess, so no check may
        pass or fail on it."""
        if cls.judge_status == "judge_failed":
            return self.error(ctx, "judge_failed", "the judge failed", **fields)
        if cls.judge_status == "judge_unsure":
            return self.skip(ctx, "judge_unsure", "the judge was unsure", **fields)
        return None

    def failure_within(self, result: MetricResult, sub_scores: frozenset[str]) -> bool:
        """Whether every cause of this failed ``result`` is measured by one of ``sub_scores``.
        A ``min_scores`` threshold on a sub-score replaces the pass bar for that sub-score only,
        so it clears a failure only when this is true. The default is that it never does."""
        return False
