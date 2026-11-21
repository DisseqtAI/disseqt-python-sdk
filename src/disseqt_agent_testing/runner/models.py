"""Runner models: what ran (``CaseSpec``) and what happened (``CaseResult``, ``SuiteResult``).

All results are frozen, JSON-only, floats rounded to 6 decimals, with a ``schema_version`` and an
``extra`` bag. ``canonical_dict()`` removes the volatile fields (ids, times, trace ids, host info):
it is the basis of digests and golden tests, so the same deterministic suite yields byte-identical
canonical JSON whatever the worker count.
"""

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import KW_ONLY, dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from ..assertions.canonical import canonical_json, normalize, round6
from ..assertions.report import DecisionReport, RepeatReport
from ..models import AgentTestCase, ErrorInfo, FrozenDict, from_dict, to_dict
from ..models._frozen import JSONObject, freeze_mapping
from .config import SuiteConfig

SCHEMA_VERSION = "1"
CASE_STATUSES = ("passed", "failed", "incomplete", "skipped", "error", "xfailed")
REPEAT_STATUSES = ("passed", "failed", "incomplete", "not_run")
NOT_RUN_REASONS = (
    "early_stop",
    "case_timeout",
    "suite_timeout",
    "fail_fast",
    "cancelled",
    "only_repeat",
)
VOLATILE_KEYS = frozenset(
    {"suite_id", "created_at", "duration_ms", "environment", "trace_id", "max_workers"}
)
CASE_ERROR_REASONS_EXIT3 = (
    "harness_config",
    "expect_config",
    "no_agent_factory",
    "agent_import",
    "judge_required",
)


def _r(value: float | None) -> float | None:
    return None if value is None else round6(float(value))


@dataclass(frozen=True, slots=True)
class CaseSource:
    path: str | None = None
    _: KW_ONLY
    line: int | None = None
    index: int = 0
    nodeid: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "line": self.line, "index": self.index, "nodeid": self.nodeid}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CaseSource":
        return cls(
            data.get("path"),
            line=data.get("line"),
            index=int(data.get("index", 0)),
            nodeid=data.get("nodeid"),
        )


RUN_BLOCK_KEYS = ("agent", "llm", "timeout_s", "skip", "xfail", "requires", "extra")


@dataclass(frozen=True, slots=True)
class RunBlock:
    """Per-case runner settings (the ``run:`` block of a case file). A new runner setting is a new
    defaulted field here; plan 01's ``AgentTestCase`` never learns about it."""

    _: KW_ONLY
    agent: str | None = None  # "module:attr"
    llm: tuple[Any, ...] | None = None  # a FakeLLM script (JSON); handlers are Python-only
    timeout_s: float | None = None
    skip: str | None = None
    xfail: str | None = None
    requires: tuple[str, ...] = ()
    extra: JSONObject = field(default_factory=FrozenDict)

    def __post_init__(self) -> None:
        if self.timeout_s is not None and (
            isinstance(self.timeout_s, bool)
            or not isinstance(self.timeout_s, (int, float))
            or self.timeout_s <= 0
        ):
            raise ValueError(f"RunBlock.timeout_s must be > 0, got {self.timeout_s!r}")
        object.__setattr__(self, "requires", tuple(self.requires))
        if self.llm is not None:
            object.__setattr__(self, "llm", tuple(self.llm))
        object.__setattr__(self, "extra", freeze_mapping(self.extra, "RunBlock.extra"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent, "llm": None if self.llm is None else normalize(list(self.llm)),
            "timeout_s": self.timeout_s, "skip": self.skip, "xfail": self.xfail,
            "requires": list(self.requires), "extra": normalize(self.extra),
        }  # fmt: skip

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RunBlock":
        unknown = sorted(set(data) - set(RUN_BLOCK_KEYS))
        if unknown:
            raise ValueError(f"RunBlock: unknown field(s) {unknown}; valid: {list(RUN_BLOCK_KEYS)}")
        return cls(**{k: data[k] for k in data})


@dataclass(frozen=True, slots=True)
class CaseSpec:
    case: AgentTestCase
    source: CaseSource = field(default_factory=CaseSource)
    run: RunBlock = field(default_factory=RunBlock)


@dataclass(frozen=True, slots=True)
class CollectionErrorRecord:
    path: str | None
    message: str
    _: KW_ONLY
    line: int | None = None
    code: str = "DSQ-C02"

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "line": self.line, "code": self.code, "message": self.message}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CollectionErrorRecord":
        return cls(
            data.get("path"),
            str(data.get("message", "")),
            line=data.get("line"),
            code=str(data.get("code", "DSQ-C02")),
        )


@dataclass(frozen=True, slots=True)
class RepeatRecord:
    index: int
    seed: int
    status: str
    _: KW_ONLY
    not_run_reason: str | None = None
    run_status: str | None = None
    decision: str | None = None
    failure_codes: tuple[str, ...] = ()
    trace_id: str | None = None
    duration_ms: float | None = None
    judge_calls: int = 0
    judge_tokens: int = 0
    judge_cost_usd: float = 0.0
    judge_cost_unknown: int = 0
    judge_error: bool = False
    report: DecisionReport | None = None

    def __post_init__(self) -> None:
        if self.status not in REPEAT_STATUSES:
            raise ValueError(f"RepeatRecord.status must be one of {list(REPEAT_STATUSES)}")
        object.__setattr__(self, "failure_codes", tuple(sorted(set(self.failure_codes))))

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index, "seed": self.seed, "status": self.status,
            "not_run_reason": self.not_run_reason, "run_status": self.run_status,
            "decision": self.decision, "failure_codes": list(self.failure_codes),
            "trace_id": self.trace_id, "duration_ms": _r(self.duration_ms),
            "judge_calls": self.judge_calls, "judge_tokens": self.judge_tokens,
            "judge_cost_usd": round6(self.judge_cost_usd), "judge_cost_unknown": self.judge_cost_unknown,
            "judge_error": self.judge_error,
            "report": None if self.report is None else self.report.to_dict(),
        }  # fmt: skip

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RepeatRecord":
        report = data.get("report")
        return cls(
            int(data["index"]), int(data.get("seed", 0)), str(data["status"]),
            not_run_reason=data.get("not_run_reason"), run_status=data.get("run_status"),
            decision=data.get("decision"), failure_codes=tuple(data.get("failure_codes", ())),
            trace_id=data.get("trace_id"), duration_ms=data.get("duration_ms"),
            judge_calls=int(data.get("judge_calls", 0)), judge_tokens=int(data.get("judge_tokens", 0)),
            judge_cost_usd=float(data.get("judge_cost_usd", 0.0)),
            judge_cost_unknown=int(data.get("judge_cost_unknown", 0)),
            judge_error=bool(data.get("judge_error", False)),
            report=None if report is None else DecisionReport.from_dict(report),
        )  # fmt: skip


@dataclass(frozen=True, slots=True)
class CaseResult:
    case_id: str
    status: str
    _: KW_ONLY
    source: CaseSource = field(default_factory=CaseSource)
    tags: tuple[str, ...] = ()
    reason_code: str | None = None
    reason: str = ""
    error: ErrorInfo | None = None
    n_planned: int = 1
    k: int = 1
    threshold: float = 1.0
    repeat_report: RepeatReport | None = None
    repeats: tuple[RepeatRecord, ...] = ()
    stopped_early: bool = False
    comparable: bool = True
    case_digest: str = ""
    duration_ms: float | None = None
    extra: JSONObject = field(default_factory=FrozenDict)

    def __post_init__(self) -> None:
        if self.status not in CASE_STATUSES:
            raise ValueError(f"CaseResult.status must be one of {list(CASE_STATUSES)}")
        object.__setattr__(self, "tags", tuple(sorted(set(self.tags))))
        object.__setattr__(self, "extra", freeze_mapping(self.extra, "CaseResult.extra"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id, "status": self.status, "source": self.source.to_dict(),
            "tags": list(self.tags), "reason_code": self.reason_code, "reason": self.reason,
            "error": None if self.error is None else to_dict(self.error), "n_planned": self.n_planned,
            "k": self.k, "threshold": round6(self.threshold),
            "repeat_report": None if self.repeat_report is None else self.repeat_report.to_dict(),
            "repeats": [r.to_dict() for r in self.repeats], "stopped_early": self.stopped_early,
            "comparable": self.comparable, "case_digest": self.case_digest,
            "duration_ms": _r(self.duration_ms), "extra": normalize(self.extra),
        }  # fmt: skip

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CaseResult":
        err = data.get("error")
        rr = data.get("repeat_report")
        return cls(
            str(data["case_id"]), str(data["status"]), source=CaseSource.from_dict(data.get("source") or {}),
            tags=tuple(data.get("tags", ())), reason_code=data.get("reason_code"), reason=str(data.get("reason", "")),
            error=None if err is None else from_dict(ErrorInfo, err, unknown="ignore"),
            n_planned=int(data.get("n_planned", 1)), k=int(data.get("k", 1)),
            threshold=float(data.get("threshold", 1.0)),
            repeat_report=None if rr is None else RepeatReport.from_dict(rr),
            repeats=tuple(RepeatRecord.from_dict(r) for r in data.get("repeats", ())),
            stopped_early=bool(data.get("stopped_early", False)), comparable=bool(data.get("comparable", True)),
            case_digest=str(data.get("case_digest", "")), duration_ms=data.get("duration_ms"),
            extra=data.get("extra", {}),
        )  # fmt: skip

    @property
    def source_key(self) -> tuple[str, int, str]:
        return (self.source.path or "~", self.source.index, self.case_id)


@dataclass(frozen=True, slots=True)
class Totals:
    cases: int = 0
    passed: int = 0
    failed: int = 0
    incomplete: int = 0
    skipped: int = 0
    error: int = 0
    repeats_run: int = 0
    repeats_not_run: int = 0
    xfailed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {f: getattr(self, f) for f in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Totals":
        return cls(**{k: int(v) for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass(frozen=True, slots=True)
class Aggregate:
    mean_pass_at_1: float | None = None
    mean_pass_hat_k: float | None = None
    flaky_cases: tuple[str, ...] = ()
    consistently_failing: tuple[str, ...] = ()
    pass_hat_k_curve: tuple[tuple[int, float], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean_pass_at_1": _r(self.mean_pass_at_1), "mean_pass_hat_k": _r(self.mean_pass_hat_k),
            "flaky_cases": list(self.flaky_cases), "consistently_failing": list(self.consistently_failing),
            "pass_hat_k_curve": [[k, round6(v)] for k, v in self.pass_hat_k_curve],
        }  # fmt: skip

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Aggregate":
        return cls(
            data.get("mean_pass_at_1"), data.get("mean_pass_hat_k"), tuple(data.get("flaky_cases", ())),
            tuple(data.get("consistently_failing", ())),
            tuple((int(k), float(v)) for k, v in data.get("pass_hat_k_curve", ())),
        )  # fmt: skip


@dataclass(frozen=True, slots=True)
class JudgeUsage:
    calls: int = 0
    cached: int = 0
    tokens: int = 0
    cost_usd: float = 0.0
    cost_unknown_calls: int = 0
    error_repeats: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls, "cached": self.cached, "tokens": self.tokens,
            "cost_usd": round6(self.cost_usd), "cost_unknown_calls": self.cost_unknown_calls,
            "error_repeats": self.error_repeats,
        }  # fmt: skip

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "JudgeUsage":
        return cls(int(data.get("calls", 0)), int(data.get("cached", 0)), int(data.get("tokens", 0)),
                   float(data.get("cost_usd", 0.0)), int(data.get("cost_unknown_calls", 0)),
                   int(data.get("error_repeats", 0)))  # fmt: skip


CUT_OFF_REASONS = ("suite_timeout", "case_timeout", "cancelled")  # incomplete whatever the policy
DELIBERATE_REASONS = ("only_repeat", "early_stop")  # a repeat left out on purpose, never cut off


def cut_off_reason(reasons: Iterable[str | None]) -> str | None:
    """The cut-off reason among the not-run reasons of a case's repeats, if any repeat has one."""
    found = set(reasons)
    return next((r for r in CUT_OFF_REASONS if r in found), None)


def not_run_reason_of(reasons: Iterable[str | None]) -> str | None:
    """Why a case with no repeat run did not run: a cut-off first, then any reason that is not a
    deliberate omission (with ``only_repeat`` every other repeat carries that reason, which says
    nothing about the selected one)."""
    listed = [r for r in reasons if r]
    return (
        cut_off_reason(listed)
        or next((r for r in listed if r not in DELIBERATE_REASONS), None)
        or next(iter(listed), None)
    )


def compute_exit_code(
    cases: tuple[CaseResult, ...], errors: tuple[CollectionErrorRecord, ...], *,
    interrupted: bool, incomplete_policy: str,
) -> int:  # fmt: skip
    """130 interrupted > 4 internal error > 3 config/collection problem > 1 failure > 2
    incomplete (when it fails, and always for a case cut off by a timeout or cancellation) > 5 no
    cases ran (none, or every case skipped) > 0."""
    if interrupted:
        return 130
    if any(c.status == "error" and c.reason_code == "internal" for c in cases):
        return 4
    if errors or any(
        c.status == "error" and c.reason_code in CASE_ERROR_REASONS_EXIT3 for c in cases
    ):
        return 3
    if any(
        c.status == "failed"
        or (c.status == "error" and c.reason_code in ("agent_setup_failed", "worker_crashed"))
        for c in cases
    ):
        return 1
    if any(
        c.status == "incomplete"
        and (incomplete_policy == "fail" or c.reason_code in CUT_OFF_REASONS)
        for c in cases
    ):
        return 2
    if not cases or all(c.status == "skipped" for c in cases):
        return 5
    return 0


def _strip(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _strip(v) for k, v in value.items() if k not in VOLATILE_KEYS}
    if isinstance(value, list):
        return [_strip(v) for v in value]
    return value


@dataclass(frozen=True, slots=True)
class SuiteResult:
    SCHEMA_VERSION: ClassVar[str] = SCHEMA_VERSION

    cases: tuple[CaseResult, ...]
    _: KW_ONLY
    suite_id: str = ""
    run_label: str = ""
    suite_digest: str = ""
    created_at: str = ""
    duration_ms: float | None = None
    seed: int = 0
    config: SuiteConfig = field(default_factory=SuiteConfig)
    environment: JSONObject = field(default_factory=FrozenDict)
    totals: Totals = field(default_factory=Totals)
    aggregate: Aggregate = field(default_factory=Aggregate)
    judge_usage: JudgeUsage = field(default_factory=JudgeUsage)
    collection_errors: tuple[CollectionErrorRecord, ...] = ()
    interrupted: bool = False
    exit_code: int = 0
    extra: JSONObject = field(default_factory=FrozenDict)

    def __post_init__(self) -> None:
        ordered = tuple(sorted(self.cases, key=lambda c: c.source_key))
        object.__setattr__(self, "cases", ordered)
        object.__setattr__(
            self, "environment", freeze_mapping(self.environment, "SuiteResult.environment")
        )
        object.__setattr__(self, "extra", freeze_mapping(self.extra, "SuiteResult.extra"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION, "suite_id": self.suite_id, "run_label": self.run_label,
            "suite_digest": self.suite_digest, "created_at": self.created_at,
            "duration_ms": _r(self.duration_ms), "seed": self.seed, "config": self.config.to_dict(),
            "environment": normalize(self.environment), "cases": [c.to_dict() for c in self.cases],
            "totals": self.totals.to_dict(), "aggregate": self.aggregate.to_dict(),
            "judge_usage": self.judge_usage.to_dict(),
            "collection_errors": [e.to_dict() for e in self.collection_errors],
            "interrupted": self.interrupted, "exit_code": self.exit_code, "extra": normalize(self.extra),
        }  # fmt: skip

    def canonical_dict(self) -> dict[str, Any]:
        """Volatile fields removed: the basis of digests and golden tests."""
        out: dict[str, Any] = _strip(self.to_dict())
        return out

    def canonical_json(self) -> str:
        return json.dumps(
            self.canonical_dict(), sort_keys=True, allow_nan=False, ensure_ascii=False
        )

    def write(self, path: str | os.PathLike[str]) -> None:
        """Atomic: a temp file in the same directory, then ``os.replace``."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        text = (
            json.dumps(
                self.to_dict(), indent=2, sort_keys=True, allow_nan=False, ensure_ascii=False
            )
            + "\n"
        )
        fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=".disseqt-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(text)
            os.replace(tmp, target)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SuiteResult":
        """Unknown keys are ignored (a newer result loads in an older tool)."""
        return cls(
            tuple(CaseResult.from_dict(c) for c in data.get("cases", ())),
            suite_id=str(data.get("suite_id", "")), run_label=str(data.get("run_label", "")),
            suite_digest=str(data.get("suite_digest", "")), created_at=str(data.get("created_at", "")),
            duration_ms=data.get("duration_ms"), seed=int(data.get("seed", 0)),
            config=SuiteConfig.from_dict(data.get("config") or {}), environment=data.get("environment", {}),
            totals=Totals.from_dict(data.get("totals") or {}), aggregate=Aggregate.from_dict(data.get("aggregate") or {}),
            judge_usage=JudgeUsage.from_dict(data.get("judge_usage") or {}),
            collection_errors=tuple(CollectionErrorRecord.from_dict(e) for e in data.get("collection_errors", ())),
            interrupted=bool(data.get("interrupted", False)), exit_code=int(data.get("exit_code", 0)),
            extra=data.get("extra", {}),
        )  # fmt: skip


def case_digest(spec: CaseSpec, n: int, k: int, threshold: float) -> str:
    """Identity of what was run (so a baseline compares like with like)."""
    body = {
        "case": to_dict(spec.case),
        "n": n,
        "k": k,
        "threshold": threshold,
        "run": spec.run.to_dict(),
    }
    return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()[:16]


def suite_digest_of(cases: tuple[CaseResult, ...]) -> str:
    return hashlib.sha256(
        canonical_json([c.case_digest for c in cases]).encode("utf-8")
    ).hexdigest()[:16]
