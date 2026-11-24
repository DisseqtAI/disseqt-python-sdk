"""Turning votes into ``JudgeVerdict``s, and the per-call record kept in reports."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..assertions.judge import JudgeVerdict, call_record
from ..models._frozen import freeze_mapping
from .codes import message


@dataclass(slots=True, frozen=True)
class JudgeCallRecord:
    """One judge call as recorded in ``DecisionReport.judge_calls``."""

    metric: str
    kind: str
    identity_hash: str
    model: str
    prompt_version: str
    status: str
    label: str | None
    agreement: float | None
    samples: int
    cached: bool
    tokens: int
    cost: float
    error_code: str | None
    gating: str

    def to_dict(self) -> dict[str, Any]:
        return {f: getattr(self, f) for f in self.__dataclass_fields__}


def record_of(verdict: JudgeVerdict, metric: str = "", kind: str = "") -> dict[str, Any]:
    """The dict form of a ``JudgeCallRecord`` (the assertions package builds the same shape)."""
    return call_record(verdict, metric, kind)


def error_verdict(
    code: str,
    text: str,
    *,
    kind: str,
    model: str = "",
    identity_hash: str = "",
    prompt_hash: str = "",
    prompt_version: str = "",
    samples: int = 0,
    tokens: int = 0,
    cost: float = 0.0,
    raw: Mapping[str, Any] | None = None,
) -> JudgeVerdict:
    body = {"kind": kind, "prompt_version": prompt_version, **(raw or {})}
    return JudgeVerdict(
        "error", None, reason=f"{code}: {text}", model=model, prompt_hash=prompt_hash,
        samples=samples, tokens=tokens, cost=cost, raw=freeze_mapping(body, "raw"),
        error_code=code, identity_hash=identity_hash,
    )  # fmt: skip


def vote_error_text(
    code: str, kind: str, vote_valid: int, n: int, codes: str, model: str, detail: str
) -> str:
    """The exact message for a verdict-level failure."""
    if code == "insufficient_votes":
        return message(code, kind=kind, valid=vote_valid, n=n, codes=codes or "none")
    return message(code, kind=kind, model=model, detail=detail or codes, n=n, valid=vote_valid,
                   codes=codes, label="", labels="", secs=0, status="", i=0, key12="", path="",
                   what="", spent=0.0, cap=0.0, calls=0, extra="")  # fmt: skip
