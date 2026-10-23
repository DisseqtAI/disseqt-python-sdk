"""Typed views of dataset-backend red-team responses (field names = Go json tags)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, cast

AggregateDecision = Literal["PASS", "BORDERLINE", "BLOCK"]
AGGREGATE_DECISIONS: frozenset[str] = frozenset({"PASS", "BORDERLINE", "BLOCK"})

# Run/job states after which polling stops; `stopped` = early stop, see stop_reason.
TERMINAL_RUN_STATES: frozenset[str] = frozenset(
    {
        "completed",
        "completed_with_errors",
        "complete",
        "failed",
        "cancelled",
        "canceled",
        "stopped",
        "error",
        "done",
    }
)


def has_more_pages(data: dict[str, Any], page: int, seen: int, list_key: str = "items") -> bool:
    """Next page exists: ``pagination.has_next`` capped by ``total_pages``
    (api/response.go PaginationMetadata), else ``total_count`` for mr-jailbreak lists."""
    pagination = data.get("pagination")
    if isinstance(pagination, dict):
        return bool(pagination.get("has_next")) and page < int(
            pagination.get("total_pages") or page + 1
        )
    return bool(data.get(list_key)) and seen < int(data.get("total_count") or 0)


@dataclass(frozen=True, slots=True)
class TestingRunStatus:
    """api/testing_types.go TestingRunResponse (subset)."""

    __test__ = False  # not a pytest class

    id: str
    status: str
    stop_reason: str | None = None
    progress: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TestingRunStatus:
        return cls(
            id=str(data.get("id", "")),
            status=str(data.get("status", "")),
            stop_reason=data.get("stop_reason") or None,
            progress=dict(data.get("progress") or {}),
            summary=dict(data.get("summary") or {}),
            error_message=data.get("error_message") or None,
        )

    @property
    def is_terminal(self) -> bool:
        return self.status.lower() in TERMINAL_RUN_STATES


@dataclass(frozen=True, slots=True)
class TestingValidateResult:
    """api/testing_validate_handler.go TestingValidateResponse."""

    __test__ = False

    aggregate_decision: AggregateDecision
    results: list[dict[str, Any]] = field(default_factory=list)
    processed_at: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TestingValidateResult:
        decision = str(data.get("aggregate_decision", ""))
        if decision not in AGGREGATE_DECISIONS:
            raise ValueError(f"unknown aggregate_decision {decision!r}")
        return cls(
            aggregate_decision=cast(AggregateDecision, decision),
            results=list(data.get("results") or []),
            processed_at=str(data.get("processed_at", "")),
        )

    @property
    def is_blocking(self) -> bool:
        return self.aggregate_decision == "BLOCK"
