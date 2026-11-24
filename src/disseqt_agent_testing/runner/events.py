"""Progress events for ``on_event`` consumers (they must ignore event types they do not know)."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CaseStarted:
    case_id: str
    n: int


@dataclass(frozen=True, slots=True)
class RepeatFinished:
    case_id: str
    index: int
    status: str  # passed | failed | incomplete | not_run


@dataclass(frozen=True, slots=True)
class CaseFinished:
    case_id: str
    status: str


@dataclass(frozen=True, slots=True)
class SuiteFinished:
    cases: int
    exit_code: int
