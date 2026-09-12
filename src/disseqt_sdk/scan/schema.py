"""Data schema for the code scanner: findings + severity enum."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Severity = Literal["low", "medium", "high", "critical"]

SEVERITY_ORDER: dict[str, int] = {"low": 0, "medium": 1, "high": 2, "critical": 3}


@dataclass(frozen=True, slots=True)
class CodeFinding:
    """A single AI-security finding located in a source file."""

    file_path: str
    vulnerability: str
    vulnerability_type: str
    severity: Severity
    reason: str
    line_start: int = 1
    line_end: int | None = None
    recommendation: str | None = None
    code_snippet: str | None = None
    validator: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # keep the client payload small — drop the mirror of the raw envelope
        d.pop("raw", None)
        return d


def meets_min_severity(sev: str, minimum: str) -> bool:
    """Return True if ``sev`` is >= ``minimum`` on the severity ladder."""
    return SEVERITY_ORDER.get(sev, -1) >= SEVERITY_ORDER.get(minimum, 0)
